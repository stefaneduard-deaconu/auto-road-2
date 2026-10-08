"""The compiled grid engine: implicit-grid Dijkstra and height-area labelling.

`gridpath.cpp` is compiled on first use with the pinned `ziglang` wheel (or a system
`g++`/`clang++`), cached under `build/native/<source hash>/` and loaded with `ctypes`.
It reproduces `core.search.dijkstra` - same neighbour order, heap order, relaxation and
counters - without storing the graph, so memory is O(cells) instead of O(edges). That is
what makes the whole Idrija DEM at 3 m and the 1 m windows searchable.

A result row produced with it records `engine = 'native'`; timings of the two engines are
never compared (`core.summary.assert_one_build`).
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from core import costs as cost_models
from core.grid import Grid, offsets, step_lengths_m
from core.search import INF, SearchResult, SearchStats

SOURCE = Path(__file__).with_name('gridpath.cpp')
BUILD_ROOT = Path(__file__).resolve().parents[2] / 'build' / 'native'
ABI_VERSION = 1
FLAGS = ['-O2', '-std=c++17', '-ffp-contract=off', '-fno-fast-math', '-shared']
_KINDS = {'height': 0, 'height_tiebreak': 1, '3d': 2, 'weighted': 3, 'length': 4}

_lock = threading.Lock()
_lib: Optional[ctypes.CDLL] = None


class NativeUnavailable(RuntimeError):
    """No compiler, or the library failed to build or load."""


class NotNativeCost(ValueError):
    """A cost model the compiled engine cannot reproduce (a plain Python callable)."""


def _library_name() -> str:
    return {'win32': 'gridpath.dll', 'darwin': 'libgridpath.dylib'}.get(sys.platform,
                                                                        'libgridpath.so')


def _compiler() -> list[str]:
    try:
        import ziglang  # noqa: F401
        return [sys.executable, '-m', 'ziglang', 'c++']
    except ImportError:
        pass
    for name in ('clang++', 'g++'):
        if shutil.which(name):
            return [name]
    raise NativeUnavailable('no C++ compiler: `pip install --only-binary=:all: ziglang` '
                            '(pinned in requirements-dev.txt) or install g++/clang++')


def library_path() -> Path:
    digest = hashlib.sha256(SOURCE.read_bytes() + ' '.join(FLAGS).encode()).hexdigest()[:16]
    return BUILD_ROOT / f'{sys.platform}-{digest}' / _library_name()


def build(force: bool = False) -> Path:
    """Compile the library if it is missing; returns its path."""
    out = library_path()
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = _compiler() + FLAGS + ['-o', str(out), str(SOURCE)]
    if sys.platform != 'win32':
        cmd.insert(-3, '-fPIC')
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise NativeUnavailable(f'compiling {SOURCE.name} failed:\n{done.stderr}')
    return out


def load() -> ctypes.CDLL:
    global _lib
    with _lock:
        if _lib is not None:
            return _lib
        lib = ctypes.CDLL(str(build()))
        f64p = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
        u8p = ctypes.c_void_p
        i64 = ctypes.c_int64
        lib.gp_abi_version.restype = ctypes.c_int
        if lib.gp_abi_version() != ABI_VERSION:
            raise NativeUnavailable('stale native library; delete build/native and retry')
        lib.gp_count_edges.restype = i64
        lib.gp_count_edges.argtypes = [f64p, u8p, i64, i64, ctypes.c_int, ctypes.c_double,
                                       ctypes.c_double, ctypes.c_int, ctypes.c_double]
        lib.gp_dijkstra.restype = ctypes.c_int
        lib.gp_dijkstra.argtypes = [
            f64p, u8p, i64, i64, ctypes.c_int, ctypes.c_double, ctypes.c_double,
            ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double,
            i64, i64, ctypes.c_int, f64p,
            np.ctypeslib.ndpointer(np.uint8, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(np.int64, flags='C_CONTIGUOUS')]
        lib.gp_label_areas.restype = i64
        lib.gp_label_areas.argtypes = [np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS'),
                                       i64, i64, ctypes.c_int,
                                       np.ctypeslib.ndpointer(np.int64, flags='C_CONTIGUOUS')]
        _lib = lib
        return lib


def available() -> bool:
    try:
        load()
        return True
    except (NativeUnavailable, OSError):
        return False


def _cost_args(cost: cost_models.CostSpec) -> tuple:
    """`(kind, a, b, c, has_penalty, i_max, penalty)` for gp_dijkstra."""
    spec = cost_models.native_spec(cost)
    if spec is None:
        raise NotNativeCost(f'cost {cost_models.name_of(cost)!r} has no native form')
    has_penalty, i_max, penalty = 0, 0.0, 0.0
    if spec[0] == 'penalty':
        _, base, i_max, penalty = spec
        if base is None:
            raise NotNativeCost(f'the base of {cost_models.name_of(cost)!r} has no native form')
        spec, has_penalty = base, 1
    kind = _KINDS[spec[0]]
    a = b = c = 0.0
    if spec[0] == 'height_tiebreak':
        b = spec[1]
    elif spec[0] == 'weighted':
        a, b, c = spec[1:]
    return kind, a, b, c, has_penalty, i_max, penalty


def _lengths(grid: Grid) -> tuple[float, float]:
    return (float(step_lengths_m(1, 0, grid.cell_size_m)),
            float(step_lengths_m(1, 1, grid.cell_size_m)))


def _mask_arg(mask: Optional[np.ndarray], grid: Grid):
    if mask is None:
        return None, None
    m = np.ascontiguousarray(mask, dtype=np.uint8)
    if m.shape != grid.shape:
        raise ValueError(f'mask shape {m.shape} != grid shape {grid.shape}')
    return m, m.ctypes.data_as(ctypes.c_void_p)


def count_edges(grid: Grid, mask: Optional[np.ndarray] = None, connectivity: int = 8,
                max_abs_gradient_percent: Optional[float] = None) -> int:
    """Directed edges of the graph the search runs on, like `GridGraph.n_edges`."""
    lib = load()
    keep, mask_ptr = _mask_arg(mask, grid)
    straight, diag = _lengths(grid)
    cap = max_abs_gradient_percent
    return int(lib.gp_count_edges(grid.surf, mask_ptr, grid.n_rows, grid.n_cols,
                                  connectivity, straight, diag, int(cap is not None),
                                  (cap or 0.0) / 100.0))


def dijkstra(grid: Grid, start: int, target: int, *,
             cost: cost_models.CostSpec = cost_models.DEFAULT_COST,
             mask: Optional[np.ndarray] = None, connectivity: int = 8,
             max_abs_gradient_percent: Optional[float] = None,
             stop_at_target: bool = True, n_edges: int = 0) -> SearchResult:
    """`core.search.dijkstra(build_graph(grid, cost, mask, ...), start, target)`, compiled.

    `stats.n_edges` is `n_edges` (pass `count_edges(...)` when it is wanted; counting is a
    full pass, so it is kept out of the timed call).
    """
    lib = load()
    kind, a, b, c, has_penalty, i_max, penalty = _cost_args(cost)
    keep, mask_ptr = _mask_arg(mask, grid)
    straight, diag = _lengths(grid)
    n = grid.n_nodes
    distances = np.empty(n, dtype=np.float64)
    direction = np.empty(n, dtype=np.uint8)
    stats = np.zeros(4, dtype=np.int64)
    cap = max_abs_gradient_percent
    code = lib.gp_dijkstra(grid.surf, mask_ptr, grid.n_rows, grid.n_cols, connectivity,
                           straight, diag, kind, a, b, c, has_penalty, i_max, penalty,
                           int(cap is not None), (cap or 0.0) / 100.0, int(start),
                           int(target), int(stop_at_target), distances, direction, stats)
    if code == 1:
        raise MemoryError('native Dijkstra ran out of memory')
    if code != 0:
        raise ValueError('native Dijkstra rejected its arguments')
    searchable = n if mask is None else int(np.count_nonzero(mask))
    result_stats = SearchStats(nodes_expanded=int(stats[0]), nodes_pushed=int(stats[1]),
                               nodes_reached=int(stats[2]), n_nodes=searchable,
                               n_edges=int(n_edges))
    result_stats.native_heap_bytes = int(stats[3])
    if not np.isfinite(distances[int(target)]):
        return SearchResult(path=None, cost=INF, distances=distances, stats=result_stats)
    return SearchResult(path=_trace(direction, grid.n_cols, int(start), int(target)),
                        cost=float(distances[int(target)]), distances=distances,
                        stats=result_stats)


def _trace(direction: np.ndarray, n_cols: int, start: int, target: int) -> np.ndarray:
    steps = offsets(8)
    path = [target]
    while path[-1] != start:
        k = int(direction[path[-1]])
        if k == 255:
            raise AssertionError('broken predecessor chain in the native result')
        i, j = divmod(path[-1], n_cols)
        path.append((i - int(steps[k][0])) * n_cols + (j - int(steps[k][1])))
    path.reverse()
    return np.array(path, dtype=np.int64)


# -- the standalone executable (X9) --------------------------------------------------------

EXE_SOURCE = Path(__file__).with_name('gridpath_main.cpp')
EXE_FLAGS = ['-O2', '-std=c++17', '-ffp-contract=off', '-fno-fast-math']


def executable_path() -> Path:
    digest = hashlib.sha256(SOURCE.read_bytes() + EXE_SOURCE.read_bytes()
                            + ' '.join(EXE_FLAGS).encode()).hexdigest()[:16]
    name = 'gridpath_exe.exe' if sys.platform == 'win32' else 'gridpath_exe'
    return BUILD_ROOT / f'{sys.platform}-exe-{digest}' / name


def build_executable(force: bool = False) -> Path:
    """Compile the standalone executable if it is missing. Optional: only X9 needs it, and a
    failure raises NativeUnavailable, which X9 logs as `not_possible`."""
    out = executable_path()
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(_compiler() + EXE_FLAGS + ['-I', str(SOURCE.parent), '-o', str(out),
                                                     str(EXE_SOURCE)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise NativeUnavailable(f'compiling {EXE_SOURCE.name} failed:\n{done.stderr}')
    return out


def executable_available() -> bool:
    try:
        build_executable()
        return True
    except (NativeUnavailable, OSError):
        return False


def run_executable(grid_npy: Path, queries: list[tuple[int, int]], *, cell_size_m: float,
                   cost: cost_models.CostSpec = cost_models.DEFAULT_COST,
                   mask_npy: Optional[Path] = None, repeats: int = 1,
                   max_abs_gradient_percent: Optional[float] = None,
                   dist_out: Optional[Path] = None, workdir: Optional[Path] = None) -> dict:
    """Run a batch of queries in one process of the standalone executable.

    Returns `{'load_ms', 'wall_ms', 'queries': [{start, target, cost, expanded, pushed,
    reached, heap_bytes, search_ms: [...]}, ...]}`; `wall_ms` includes process start-up.
    """
    import json
    import tempfile
    import time
    exe = build_executable()
    kind, a, b, c, has_penalty, i_max, penalty = _cost_args(cost)
    straight = float(step_lengths_m(1, 0, cell_size_m))
    diag = float(step_lengths_m(1, 1, cell_size_m))
    cap = max_abs_gradient_percent
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        qfile = Path(tmp) / 'queries.txt'
        qfile.write_text(''.join(f'{int(s)} {int(t)}\n' for s, t in queries), encoding='ascii')
        args = [str(exe), str(grid_npy), str(mask_npy) if mask_npy else '-', str(qfile),
                str(kind), repr(a), repr(b), repr(c), str(has_penalty), repr(i_max),
                repr(penalty), str(int(cap is not None)), repr((cap or 0.0) / 100.0),
                repr(straight), repr(diag), str(max(1, int(repeats)))]
        if dist_out is not None:
            args.append(str(dist_out))
        t0 = time.perf_counter()
        done = subprocess.run(args, capture_output=True, text=True)
        wall_ms = (time.perf_counter() - t0) * 1000.0
    if done.returncode != 0:
        raise NativeUnavailable(f'gridpath executable failed: {done.stderr.strip()}')
    lines = [json.loads(line) for line in done.stdout.splitlines() if line.strip()]
    head, results = lines[0], lines[1:]
    for r in results:
        if 'error' in r:
            raise MemoryError if r['error'] == 1 else ValueError(f'query {r["query"]} rejected')
        if r['cost'] < 0:
            r['cost'] = INF
    return {'load_ms': head['load_ms'], 'wall_ms': wall_ms, 'queries': results}


def label_areas(buckets: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Height-area labels numbered exactly as `core.hag.build_hag` numbers them.

    `buckets` is an int32 bucket index per cell, -1 for nodata. Returns `(labels, n)`.
    """
    lib = load()
    buckets = np.ascontiguousarray(buckets, dtype=np.int32)
    labels = np.empty(buckets.shape, dtype=np.int64)
    n = lib.gp_label_areas(buckets, buckets.shape[0], buckets.shape[1], connectivity, labels)
    if n < 0:
        raise MemoryError('native area labelling ran out of memory')
    return labels, int(n)
