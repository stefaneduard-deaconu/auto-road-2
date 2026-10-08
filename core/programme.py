"""The experiment programme X1-X7: what each experiment runs, per tier.

Each experiment is a list of UNITS (one terrain or DEM window at one cell size and one
height band, plus its O-D pairs). A unit is the resume granule: `core.run` skips a unit
already logged as done. The cards in the study notes describe the
same experiments in prose; `python -m core.run list` prints this module's view.
"""
from __future__ import annotations

import ctypes
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from core import costs as cost_models
from core import dem as dem_mod
from core.experiment_hag import (Case, Pair, break_even_queries, hag_row, sample_pairs,
                                 search_rows, timed_hag)
from core.grid import Grid
from core.hag import HAG_EDGE_COSTS, area_graph, build_hag
from core.naming import DEFAULT_GRID_EDGE_COST, DEFAULT_HAG_EDGE_COST, grid_edge_cost
from core.terrain import TerrainSpec, generate_terrain

#: The Idrija Fault LiDAR DEM, OpenTopography, DOI 10.5069/G9QC01Q2 (core.dem.IDRIJA_DEM_DOI).
RASTER = Path('raster/Idrija_Fault_LiDAR_DEM.tif')
TIERS = ('quick', 'standard', 'full', 'hilly')
#: the `hilly` tier: synthetic terrains whose measured median slope is 20-25 gon (STAS 863-85's
#: hilly class). (Perlin period, relief m) pairs; median slope at 10 m cells, measured:
#: (4, 200) 24.3 gon, (6, 120) about 21 gon, (8, 100) 23.6 gon. Its results go to `*_hilly`
#: folders so they are never pooled with the other tiers. The DEM experiments (X6, X7) have
#: no units in it: the Idrija DEM is hilly already.
HILLY_TERRAINS = ((4, 200.0), (6, 120.0), (8, 100.0))
HILLY_RELIEF_M = 200.0
#: every DEM extent is a multiple of this many metres, so 20/10/5/3/1 m all tile it
DEM_ALIGN_M = 60
#: estimated bytes per cell of a HAG build plus one search (float64 heights, buckets,
#: distances, int64 labels, masks and numpy temporaries); an upper-side estimate
BYTES_PER_CELL = 96


# -- the units ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Unit:
    unit_id: str
    params: dict = field(hash=False, compare=False)


@dataclass(frozen=True)
class Experiment:
    key: str
    directory: str
    title: str
    question: str
    timing_sensitive: bool
    requires_raster: bool
    units: Callable[[str], list[Unit]]
    run: Callable[[Unit, 'RunOptions'], list[dict]]
    #: measures the machine itself (X9): never run beside other work, never with --workers
    exclusive: bool = False


@dataclass
class RunOptions:
    engine: str = 'auto'
    repeats: Optional[int] = None
    tif: Path = RASTER
    #: the tier being run, for experiments whose units depend on it at run time (X9)
    tier_hint: Optional[str] = None


def _tier(tier: str, quick, standard, full, hilly=None):
    """The value of one setting for `tier`; `hilly` defaults to the `standard` value."""
    return {'quick': quick, 'standard': standard, 'full': full,
            'hilly': standard if hilly is None else hilly}[tier]


def synthetic_specs(tier: str) -> list[dict]:
    seeds = _tier(tier, range(2), range(10), range(30))
    sizes = _tier(tier, (120,), (120, 240, 480), (120, 240, 480, 960))
    periods = _tier(tier, (4,), (2, 4, 6), (2, 3, 4, 6, 8))
    amplitudes = _tier(tier, (20.0,), (20.0, 60.0), (20.0, 60.0, 150.0))
    combos = (list(HILLY_TERRAINS) if tier == 'hilly'
              else [(p, a) for p in periods for a in amplitudes])
    return [{'seed': s, 'size': n, 'period': p, 'amplitude_m': a}
            for n in sizes for (p, a) in combos for s in seeds]


def synthetic_case(seed: int, size: int, period: int, amplitude_m: float,
                   cell_size_m: float = 10.0) -> Case:
    spec = TerrainSpec(seed=seed, grid_size=(size, size), periods=(period, period),
                       height_interval=(100.0, 100.0 + amplitude_m), cell_size_m=cell_size_m)
    return Case(case_id=f'syn_s{seed}_n{size}_p{period}_a{amplitude_m:g}', family='synthetic',
                grid=generate_terrain(spec),
                meta={'terrain_seed': seed, 'grid_rows': size, 'grid_cols': size,
                      'periods': period, 'relief_amplitude_m': amplitude_m,
                      'cell_size_m': cell_size_m})


def _default_repeats(options: RunOptions, fallback: int) -> int:
    return options.repeats if options.repeats is not None else fallback


# -- memory -----------------------------------------------------------------------------------

def _memory_status() -> Optional[tuple]:
    """(total, available) bytes of physical memory, or None when the OS does not say."""
    if sys.platform == 'win32':
        class Status(ctypes.Structure):
            _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32),
                        ('total', ctypes.c_uint64), ('available', ctypes.c_uint64),
                        ('pf_total', ctypes.c_uint64), ('pf_avail', ctypes.c_uint64),
                        ('v_total', ctypes.c_uint64), ('v_avail', ctypes.c_uint64),
                        ('ext', ctypes.c_uint64)]
        status = Status()
        status.length = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total), int(status.available)
        return None
    try:
        values = {}
        with open('/proc/meminfo') as handle:
            for line in handle:
                key, _, rest = line.partition(':')
                if key in ('MemTotal', 'MemAvailable'):
                    values[key] = int(rest.split()[0]) * 1024
        return values['MemTotal'], values['MemAvailable']
    except (OSError, KeyError, ValueError):
        return None


def available_memory_bytes() -> Optional[int]:
    status = _memory_status()
    return None if status is None else status[1]


def total_memory_bytes() -> Optional[int]:
    status = _memory_status()
    return None if status is None else status[0]


class NotPossible(RuntimeError):
    """A unit that cannot run here (no raster, not enough memory); logged, not an error."""


#: how long a unit waits for other workers to free memory before it is logged not_possible
MEMORY_WAIT_S = 1800.0


def require_memory(n_cells: int, what: str, *, wait_s: Optional[float] = None) -> None:
    """The machine limit is its TOTAL memory, so which units are possible does not depend on
    how many workers happen to be running or what else is open: a unit that needs more than
    80% of the physical memory is `not_possible`. One that fits waits (up to `wait_s`) for the
    memory it needs to be free, instead of being skipped because a neighbour holds it."""
    need = n_cells * BYTES_PER_CELL
    total = total_memory_bytes()
    if total is not None and need > 0.8 * total:
        raise NotPossible(f'{what}: needs about {need / 2**30:.1f} GiB for {n_cells:,} cells, '
                          f'over 80% of the {total / 2**30:.1f} GiB of this machine')
    deadline = time.monotonic() + (MEMORY_WAIT_S if wait_s is None else wait_s)
    while True:
        have = available_memory_bytes()
        if have is None or need <= 0.8 * have:
            return
        if time.monotonic() >= deadline:
            raise NotPossible(f'{what}: needs about {need / 2**30:.1f} GiB for {n_cells:,} '
                              f'cells, only {have / 2**30:.1f} GiB stayed available while '
                              f'waiting {MEMORY_WAIT_S / 60:.0f} min (other workers or programs)')
        time.sleep(5.0)


# -- the DEM -------------------------------------------------------------------------------

_EXTENT_CACHE: dict = {}


def dem_extent(tif_path: Path) -> dict:
    """Bounding box of the surveyed swath, aligned to `DEM_ALIGN_M`, from a 20 m scan."""
    key = str(tif_path)
    if key in _EXTENT_CACHE:
        return _EXTENT_CACHE[key]
    if not tif_path.exists():
        raise NotPossible(f'raster not found: {tif_path} (git-ignored; unzip raster.zip)')
    tif = dem_mod.read_geotiff_header(tif_path, sha256=False)
    stride = 20
    valid = dem_mod.coverage_mask(tif, stride=stride)
    rows, cols = np.nonzero(valid)
    r0 = (rows.min() * stride) // DEM_ALIGN_M * DEM_ALIGN_M
    c0 = (cols.min() * stride) // DEM_ALIGN_M * DEM_ALIGN_M
    r1 = min(tif.height, (rows.max() + 1) * stride)
    c1 = min(tif.width, (cols.max() + 1) * stride)
    n_rows = (r1 - r0) // DEM_ALIGN_M * DEM_ALIGN_M
    n_cols = (c1 - c0) // DEM_ALIGN_M * DEM_ALIGN_M
    out = {'tif': tif, 'row0': int(r0), 'col0': int(c0), 'n_rows': int(n_rows),
           'n_cols': int(n_cols), 'valid_share_20m': float(valid.mean()),
           'coverage_20m': valid, 'stride': stride}
    _EXTENT_CACHE[key] = out
    return out


def dem_case(tif, row0: int, col0: int, n_rows: int, n_cols: int, cell_m: int,
             label: str) -> Case:
    factor = int(round(cell_m / tif.pixel_scale_m[0]))
    require_memory((n_rows // factor) * (n_cols // factor), f'{label} at {cell_m} m')
    surf, audit = dem_mod.coarsen_stream(tif, row0=row0, col0=col0, n_rows=n_rows,
                                         n_cols=n_cols, factor=factor, strict=True)
    valid = np.isfinite(surf)
    if not valid.any():
        raise NotPossible(f'{label}: no valid cell at {cell_m} m')
    filled = np.where(valid, surf, np.nanmin(surf))
    return Case(case_id=f'{label}_c{cell_m}', family='dem',
                grid=Grid(surf=filled, cell_size_m=float(cell_m)), valid=valid,
                meta={'dem_label': label, 'dem_row0': row0, 'dem_col0': col0,
                      'dem_rows_1m': n_rows, 'dem_cols_1m': n_cols, 'cell_size_m': cell_m,
                      'grid_rows': surf.shape[0], 'grid_cols': surf.shape[1],
                      'void_policy': audit['method']})


def map_pair(pair: Pair, from_cell: float, to_cell: float) -> Pair:
    """The same physical O-D pair at another cell size (cell centres, block-aligned)."""
    def one(ij):
        return tuple(int(math.floor(((k + 0.5) * from_cell) / to_cell)) for k in ij)
    return Pair(pair.od_id, one(pair.start), one(pair.target), pair.straight_line_m, pair.band)


def _valid_pair(case: Case, pair: Pair) -> bool:
    ok = all(0 <= p[0] < case.grid.n_rows and 0 <= p[1] < case.grid.n_cols
             for p in (pair.start, pair.target))
    return ok and (case.valid is None or (case.valid[pair.start] and case.valid[pair.target]))


# -- X1: HAG descriptors ----------------------------------------------------------------------

DEM_CELLS = {'quick': (20,), 'standard': (20, 10, 5), 'full': (20, 10, 5, 3, 1), 'hilly': ()}
DEM_DELTAS = {'quick': (3,), 'standard': (1, 3, 5), 'full': (1, 2, 3, 5), 'hilly': ()}
SYN_DELTAS = {'quick': (3,), 'standard': (1, 3, 5), 'full': (1, 2, 3, 5), 'hilly': (1, 3, 5)}


def x1_units(tier: str) -> list[Unit]:
    units = [Unit(f'syn_s{s["seed"]}_n{s["size"]}_p{s["period"]}_a{s["amplitude_m"]:g}_d{d}',
                  {'kind': 'synthetic', **s, 'delta': d})
             for s in synthetic_specs(tier) for d in SYN_DELTAS[tier]]
    units += [Unit(f'dem_whole_c{c}_d{d}', {'kind': 'dem', 'cell': c, 'delta': d})
              for c in DEM_CELLS[tier] for d in DEM_DELTAS[tier]]
    return units


def _unit_case(unit: Unit, options: RunOptions) -> Case:
    p = unit.params
    if p['kind'] == 'synthetic':
        return synthetic_case(p['seed'], p['size'], p['period'], p['amplitude_m'])
    ext = dem_extent(options.tif)
    return dem_case(ext['tif'], ext['row0'], ext['col0'], ext['n_rows'], ext['n_cols'],
                    p['cell'], 'idrija_whole')


def x1_run(unit: Unit, options: RunOptions) -> list[dict]:
    case = _unit_case(unit, options)
    hag, timings = timed_hag(case, unit.params['delta'], use_native=True,
                             repeats=_default_repeats(options, 1))
    return [{**case.meta, 'case_id': case.case_id, 'family': case.family,
             'height_delta_m': unit.params['delta'], **hag_row(case, hag, timings)}]


# -- the search experiments ------------------------------------------------------------------

def _search_unit(case: Case, delta: float, pairs: list[Pair], spaces, grid_edges,
                 options: RunOptions, repeats: int, extra: dict,
                 engines: tuple[str, ...] = ('auto',), build_repeats: int = 1) -> list[dict]:
    hag, timings = timed_hag(case, delta, use_native=True, repeats=build_repeats)
    desc = hag_row(case, hag, timings)
    hag_columns = {'hag_n_areas': desc['n_areas'], 'hag_n_edges': desc['n_hag_edges'],
                   'hag_edges_per_area_mean': desc['edges_per_area_mean'],
                   'hag_hop_diameter_lb': desc['hop_diameter_lb'],
                   'hag_build_time_s': desc['hag_build_time_s'],
                   'n_valid_cells': desc['n_valid_cells'], 'relief_m': desc['relief_m'],
                   'terrain_slope_percent_p50': desc['terrain_slope_percent_p50']}
    terrain_edges = {e for _, e in spaces if e in HAG_EDGE_COSTS}
    rows = []
    for grid_edge in grid_edges:
        cost, _ = grid_edge_cost(grid_edge, extra.get('i_max_percent'))
        cache, graph_times = {}, {}
        for edge in sorted(terrain_edges):
            t0 = time.perf_counter()
            cache[(edge, cost_models.name_of(cost))] = area_graph(hag, edge, cost)
            graph_times[edge] = time.perf_counter() - t0
        for engine in engines:
            for pair in pairs:
                if not _valid_pair(case, pair):
                    rows.append({**case.meta, 'case_id': case.case_id, 'od_id': pair.od_id,
                                 'od_band': pair.band, 'engine': engine,
                                 'grid_edge_cost': grid_edge, 'path_found': False,
                                 'infeasible_reason': 'endpoint on nodata at this cell size'})
                    continue
                for row in search_rows(case, hag, pair, spaces=spaces, grid_edge=grid_edge,
                                       engine=engine, repeats=repeats,
                                       i_max_percent=extra.get('i_max_percent'),
                                       graph_cache=cache):
                    graph_s = graph_times.get(row.get('hag_edge_cost'), 0.0)
                    row['area_graph_time_s'] = graph_s
                    if row['search_space'] != 'full_grid' and 'query_time_saving_s' in row:
                        row['break_even_queries'] = break_even_queries(
                            desc['hag_build_time_s'], graph_s, row['query_time_saving_s'])
                    rows.append({**case.meta, 'case_id': case.case_id, 'family': case.family,
                                 'height_delta_m': delta, **hag_columns, **extra, **row})
    return rows


def _protocol_pairs(case: Case, hag_delta: float, seed: int, n: int) -> list[Pair]:
    hag = build_hag(case.grid, hag_delta, valid_mask=case.valid, native=True)
    diagonal = math.hypot(case.grid.n_rows, case.grid.n_cols) * case.grid.cell_size_m
    pairs, _ = sample_pairs(case, hag, seed=seed, n_pairs=n, min_m=0.6 * diagonal)
    return pairs


# X2 - which HAG edge cost
def x2_units(tier: str) -> list[Unit]:
    seeds = _tier(tier, range(2), range(10), range(30))
    sizes = _tier(tier, (120,), (120, 240), (120, 240, 480))
    amplitudes = _tier(tier, (20.0,), (20.0, 60.0), (20.0, 60.0, 150.0))
    combos = list(HILLY_TERRAINS) if tier == 'hilly' else [(4, a) for a in amplitudes]
    units = [Unit(f'syn_s{s}_n{n}_p{p}_a{a:g}_d3', {'kind': 'synthetic', 'seed': s, 'size': n,
                                                    'period': p, 'amplitude_m': a, 'delta': 3})
             for n in sizes for (p, a) in combos for s in seeds]
    # a synthetic HAG is often a tree (one chain between any two areas), so the edge cost
    # can only matter on the DEM, whose HAG has cycles
    per_size = _tier(tier, 1, 3, 5, hilly=0)
    units += [Unit(f'dem_win1920_k{k}_c{c}_d3', {'kind': 'dem', 'size': 1920, 'k': k,
                                                 'cell': c, 'delta': 3, 'per_size': per_size})
              for k in range(per_size) for c in (20, 10)]
    return units


def x2_run(unit: Unit, options: RunOptions) -> list[dict]:
    p = unit.params
    if p['kind'] == 'dem':
        case, pairs = _window_case_pairs(options, p)
    else:
        case = synthetic_case(p['seed'], p['size'], p['period'], p['amplitude_m'])
        pairs = _protocol_pairs(case, p['delta'], 1000 + p['seed'], 3)
    spaces = [(s, e) for e in HAG_EDGE_COSTS for s in ('hag_cta', 'hag_cta_ring1')]
    return _search_unit(case, p['delta'], pairs, spaces, ('climb_tiebreak', 'climb_plus_length'),
                        options, _default_repeats(options, 1), {})


# X3 - the reduction as a distribution
X3_SPACES = [('full_grid', ''), ('hag_cta', DEFAULT_HAG_EDGE_COST),
             ('hag_cta_ring1', DEFAULT_HAG_EDGE_COST)]


def x3_units(tier: str) -> list[Unit]:
    units = [Unit(f'syn_s{s["seed"]}_n{s["size"]}_p{s["period"]}_a{s["amplitude_m"]:g}_d{d}',
                   {**s, 'delta': d}) for s in synthetic_specs(tier) for d in SYN_DELTAS[tier]]
    return units


def x3_run(unit: Unit, options: RunOptions) -> list[dict]:
    p = unit.params
    case = synthetic_case(p['seed'], p['size'], p['period'], p['amplitude_m'])
    pairs = _protocol_pairs(case, p['delta'], 1000 + p['seed'], 3)
    return _search_unit(case, p['delta'], pairs, X3_SPACES, (DEFAULT_GRID_EDGE_COST,), options,
                        _default_repeats(options, 5), {})


# X4 - one terrain, five resolutions
RESOLUTIONS = (20, 10, 5, 3, 1)


def x4_units(tier: str) -> list[Unit]:
    seeds = _tier(tier, range(1), range(5), range(10))
    amplitudes = _tier(tier, (60.0,), (20.0, 60.0), (20.0, 60.0, 150.0), hilly=(HILLY_RELIEF_M,))
    cells = _tier(tier, (20, 10, 5), RESOLUTIONS, RESOLUTIONS)
    return [Unit(f'syn1200_s{s}_a{a:g}_c{c}_d3', {'seed': s, 'amplitude_m': a, 'cell': c,
                                                  'delta': 3})
            for s in seeds for a in amplitudes for c in cells]


def _fine_terrain(seed: int, amplitude_m: float) -> Grid:
    spec = TerrainSpec(seed=seed, grid_size=(1200, 1200), periods=(4, 4),
                       height_interval=(100.0, 100.0 + amplitude_m), cell_size_m=1.0)
    return generate_terrain(spec)


def x4_run(unit: Unit, options: RunOptions) -> list[dict]:
    p = unit.params
    fine = _fine_terrain(p['seed'], p['amplitude_m'])
    coarse20, _ = dem_mod.coarsen_to_cell_size(fine, 20.0)
    anchor = Case('anchor', 'synthetic', coarse20)
    pairs = _protocol_pairs(anchor, p['delta'], 1000 + p['seed'], 3)
    grid = fine if p['cell'] == 1 else dem_mod.coarsen_to_cell_size(fine, float(p['cell']))[0]
    case = Case(f'syn1200_s{p["seed"]}_a{p["amplitude_m"]:g}_c{p["cell"]}', 'synthetic', grid,
                meta={'terrain_seed': p['seed'], 'relief_amplitude_m': p['amplitude_m'],
                      'cell_size_m': p['cell'], 'grid_rows': grid.n_rows,
                      'grid_cols': grid.n_cols, 'extent_m': 1200})
    mapped = [map_pair(q, 20.0, float(p['cell'])) for q in pairs]
    return _search_unit(case, p['delta'], mapped, X3_SPACES[:3], (DEFAULT_GRID_EDGE_COST,),
                        options, _default_repeats(options, 3), {})


# X5 - computational scaling
def x5_units(tier: str) -> list[Unit]:
    sizes = _tier(tier, (120, 240), (120, 240, 480, 960), (120, 240, 480, 960, 1920))
    seeds = _tier(tier, range(1), range(3), range(5))
    amplitude = HILLY_RELIEF_M if tier == 'hilly' else 60.0
    return [Unit(f'scale_n{n}_s{s}', {'size': n, 'seed': s, 'amplitude_m': amplitude})
            for n in sizes for s in seeds]


def x5_run(unit: Unit, options: RunOptions) -> list[dict]:
    p = unit.params
    period = p['size'] // 60
    case = synthetic_case(p['seed'], p['size'], period, p.get('amplitude_m', 60.0))
    pairs = _protocol_pairs(case, 3, 1000 + p['seed'], 1)
    engines = ('native', 'python') if p['size'] <= 480 else ('native',)
    return _search_unit(case, 3, pairs, X3_SPACES[:3], (DEFAULT_GRID_EDGE_COST,), options,
                        _default_repeats(options, 5), {}, engines=engines, build_repeats=3)


# X6 - the whole Idrija DEM
DEM_BANDS = {'quick': (('2km', 1500, 2500),),
             'standard': (('1km', 750, 1250), ('2km', 1500, 2500), ('5km', 4000, 6000),
                          ('10km', 8000, 12000)),
             'full': (('1km', 750, 1250), ('2km', 1500, 2500), ('5km', 4000, 6000),
                      ('10km', 8000, 12000), ('20km', 16000, 24000))}


def x6_units(tier: str) -> list[Unit]:
    per_band = _tier(tier, 1, 2, 3)
    return [Unit(f'dem_whole_c{c}_d{d}', {'cell': c, 'delta': d, 'per_band': per_band,
                                           'tier': tier})
            for c in DEM_CELLS[tier] for d in DEM_DELTAS[tier]]


def _dem_pairs(ext: dict, delta: float, bands, per_band: int, seed: int,
               tif_label: str, row0=None, col0=None, n_rows=None, n_cols=None,
               min_hops: int = 3) -> tuple[list[Pair], dict]:
    anchor = dem_case(ext['tif'], row0 if row0 is not None else ext['row0'],
                      col0 if col0 is not None else ext['col0'],
                      n_rows or ext['n_rows'], n_cols or ext['n_cols'], 20, tif_label)
    hag = build_hag(anchor.grid, delta, valid_mask=anchor.valid, native=True)
    pairs, audit = [], {}
    for k, (name, lo, hi) in enumerate(bands):
        got, a = sample_pairs(anchor, hag, seed=seed + k, n_pairs=per_band, min_m=lo,
                              max_m=hi, band=name, min_hops=min_hops, margin_cells=2)
        pairs += got
        audit[name] = a
    return pairs, audit


def x6_run(unit: Unit, options: RunOptions) -> list[dict]:
    p = unit.params
    ext = dem_extent(options.tif)
    pairs, _ = _dem_pairs(ext, p['delta'], DEM_BANDS[p['tier']], p['per_band'], 2026,
                          'idrija_whole')
    case = dem_case(ext['tif'], ext['row0'], ext['col0'], ext['n_rows'], ext['n_cols'],
                    p['cell'], 'idrija_whole')
    mapped = [map_pair(q, 20.0, float(p['cell'])) for q in pairs]
    return _search_unit(case, p['delta'], mapped, X3_SPACES[:4], (DEFAULT_GRID_EDGE_COST,),
                        options, _default_repeats(options, 3 if p['cell'] >= 5 else 1), {})


# X7 - DEM windows, down to 1 m
WINDOW_SIZES = {'quick': (960,), 'standard': (960, 1920), 'full': (960, 1920, 3840), 'hilly': ()}


def dem_windows(ext: dict, size_m: int, n: int, seed: int, min_valid: float = 0.6) -> list[dict]:
    """`n` windows of `size_m`, at least `min_valid` surveyed, non-overlapping, by seed."""
    cov, stride = ext['coverage_20m'], ext['stride']
    win = size_m // stride
    sat = np.zeros((cov.shape[0] + 1, cov.shape[1] + 1), dtype=np.int64)
    sat[1:, 1:] = np.cumsum(np.cumsum(cov.astype(np.int64), axis=0), axis=1)
    total = sat[win:, win:] - sat[:-win, win:] - sat[win:, :-win] + sat[:-win, :-win]
    rows, cols = np.nonzero(total >= min_valid * win * win)
    order = np.random.default_rng(seed).permutation(rows.size)
    chosen: list[tuple[int, int]] = []
    for k in order:
        r, c = int(rows[k]), int(cols[k])
        if all(abs(r - r2) >= win or abs(c - c2) >= win for r2, c2 in chosen):
            chosen.append((r, c))
        if len(chosen) == n:
            break
    out = []
    for r, c in chosen:
        row0 = (r * stride) // DEM_ALIGN_M * DEM_ALIGN_M
        col0 = (c * stride) // DEM_ALIGN_M * DEM_ALIGN_M
        if row0 + size_m <= ext['tif'].height and col0 + size_m <= ext['tif'].width:
            out.append({'row0': row0, 'col0': col0, 'valid_share_20m':
                        float(total[r, c] / (win * win))})
    return out


def x7_units(tier: str) -> list[Unit]:
    per_size = _tier(tier, 1, 3, 5, hilly=0)
    return [Unit(f'win{s}_k{k}_c{c}_d3', {'size': s, 'k': k, 'cell': c, 'delta': 3,
                                          'per_size': per_size})
            for s in WINDOW_SIZES[tier] for k in range(per_size) for c in RESOLUTIONS]


def _window_case_pairs(options: RunOptions, p: dict) -> tuple[Case, list[Pair]]:
    """Window `k` of `size` m at `cell` m, and 3 O-D pairs drawn at 20 m and mapped."""
    ext = dem_extent(options.tif)
    windows = dem_windows(ext, p['size'], p['per_size'], seed=7000 + p['size'])
    if p['k'] >= len(windows):
        raise NotPossible(f'only {len(windows)} windows of {p["size"]} m are >= 60% surveyed')
    w = windows[p['k']]
    label = f'idrija_win{p["size"]}_k{p["k"]}'
    pairs, _ = _dem_pairs(ext, p['delta'], (('win', 0.4 * p['size'], math.inf),), 3,
                          8000 + p['k'], label, w['row0'], w['col0'], p['size'], p['size'],
                          min_hops=2)
    case = dem_case(ext['tif'], w['row0'], w['col0'], p['size'], p['size'], p['cell'], label)
    case.meta['window_valid_share_20m'] = w['valid_share_20m']
    return case, [map_pair(q, 20.0, float(p['cell'])) for q in pairs]


def x7_run(unit: Unit, options: RunOptions) -> list[dict]:
    p = unit.params
    case, pairs = _window_case_pairs(options, p)
    return _search_unit(case, p['delta'], pairs, X3_SPACES[:4], (DEFAULT_GRID_EDGE_COST,),
                        options, _default_repeats(options, 3), {})



# X9 - engines and parallelisation
#: the engines X9 compares: the Python CSR Dijkstra, the compiled library called in-process
#: through ctypes, and the same compiled search as a standalone executable (one process per
#: batch of queries, the grid loaded once from a memory-mapped .npy)
X9_ENGINES = ('python', 'native', 'exe')
#: the Python engine stores the whole graph (8 edges per cell); above this it is not run
X9_PYTHON_MAX_SIDE = 960
GRID_CACHE = Path('build') / 'cache' / 'grids'


def x9_worker_counts(tier: str) -> list[int]:
    from core.parallel import auto_workers
    top = auto_workers()
    counts, w = [], 1
    while w < top:
        counts.append(w)
        w *= 2
    counts.append(top)
    return counts if tier != 'quick' else counts[:2]


def x9_units(tier: str) -> list[Unit]:
    if tier == 'hilly':
        return []
    sizes = _tier(tier, (120, 240), (240, 480, 960, 1920), (240, 480, 960, 1920, 3840))
    return [Unit(f'eng_n{n}', {'size': n, 'seed': 0, 'period': 4, 'amplitude_m': 60.0})
            for n in sizes]


def cached_grid_npy(grid: Grid, name: str) -> Path:
    """The grid's heights as a float64 .npy, written once and reused (memory-mapped by readers)."""
    GRID_CACHE.mkdir(parents=True, exist_ok=True)
    path = GRID_CACHE / f'{name}.npy'
    if not path.exists():
        tmp = path.with_suffix('.tmp.npy')
        np.save(tmp, np.ascontiguousarray(grid.surf, dtype=np.float64))
        tmp.replace(path)
    return path


def _x9_queries(grid: Grid, n: int, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    side = grid.n_rows
    out = []
    while len(out) < n:
        i1, j1, i2, j2 = rng.integers(0, side, size=4)
        if math.hypot(i1 - i2, j1 - j2) >= 0.5 * side:
            out.append((int(i1 * grid.n_cols + j1), int(i2 * grid.n_cols + j2)))
    return out


def x9_run(unit: Unit, options: RunOptions) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor

    from core import native
    from core.parallel import _rss_bytes
    from core.search import build_graph, dijkstra
    p = unit.params
    n = p['size']
    require_memory(n * n, unit.unit_id)
    case = synthetic_case(p['seed'], n, p['period'], p['amplitude_m'])
    grid = case.grid
    worker_counts = x9_worker_counts(options.tier_hint or 'standard')
    queries = _x9_queries(grid, max(8, 2 * max(worker_counts)), 1000 + n)
    cost = 'height_tiebreak'  # the default grid edge cost (climb_tiebreak)
    npy = cached_grid_npy(grid, f'{case.case_id}')
    rows, reference = [], None
    for engine in X9_ENGINES:
        if engine == 'python' and n > X9_PYTHON_MAX_SIDE:
            continue
        if engine == 'native' and not native.available():
            continue
        if engine == 'exe' and not native.executable_available():
            continue
        prepare_s = 0.0
        if engine == 'python':
            t0 = time.perf_counter()
            graph = build_graph(grid, cost=cost)
            prepare_s = time.perf_counter() - t0

            def one(q):
                t = time.perf_counter()
                r = dijkstra(graph, q[0], q[1])
                return r.cost, r.stats.nodes_expanded, time.perf_counter() - t
        elif engine == 'native':
            def one(q):
                t = time.perf_counter()
                r = native.dijkstra(grid, q[0], q[1], cost=cost)
                return r.cost, r.stats.nodes_expanded, time.perf_counter() - t
        base_throughput = None
        for workers in worker_counts:
            t0 = time.perf_counter()
            if engine == 'exe':
                batches = [queries[k::workers] for k in range(workers)]
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    outs = list(pool.map(lambda b: native.run_executable(
                        npy, b, cell_size_m=grid.cell_size_m, cost=cost), batches))
                results = []
                for out in outs:
                    for q in out['queries']:
                        results.append((q['cost'], q['expanded'], q['search_ms'][0] / 1000.0))
                load_ms = sum(o['load_ms'] for o in outs) / len(outs)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results = list(pool.map(one, queries))
                load_ms = prepare_s * 1000.0
            wall = time.perf_counter() - t0
            costs = sorted(round(float(c), 9) for c, _, _ in results)
            if reference is None:
                reference = costs
            throughput = len(queries) / wall
            if workers == worker_counts[0]:
                base_throughput = throughput
            search = [s for _, _, s in results]
            rows.append({'case_id': case.case_id, 'family': 'synthetic', 'grid_rows': n,
                         'n_cells': n * n, 'engine_x9': engine, 'engine': 'x9',
                         'workers': workers, 'n_queries': len(queries), 'wall_s': wall,
                         'throughput_qps': throughput,
                         'speedup_vs_1': throughput / base_throughput,
                         'efficiency': throughput / base_throughput / workers,
                         'search_s_mean': float(np.mean(search)),
                         'search_s_median': float(np.median(search)),
                         'nodes_expanded_mean': float(np.mean([e for _, e, _ in results])),
                         'load_or_build_ms': load_ms, 'rss_bytes': _rss_bytes(),
                         'costs_match_reference': costs == reference})
    return rows

EXPERIMENTS: dict[str, Experiment] = {e.key: e for e in (
    Experiment('X1', 'x1_hag_descriptors', 'What HAGs look like',
               'How many areas, edges per area, area sizes and hop diameter do HAGs have, '
               'on synthetic terrains and on the whole Idrija DEM at 20-1 m?',
               False, False, x1_units, x1_run),
    Experiment('X2', 'x2_hag_edge_cost', 'Which HAG edge cost',
               'How do the two terrain HAG edge costs (shared_border, the study\'s, and '
               'area_means) compare in corridor size and objective?',
               False, False, x2_units, x2_run),
    Experiment('X3', 'x3_reduction_synthetic', 'The reduction as a distribution',
               'How much of the grid does the HAG remove, and at what cost in objective, '
               'across many synthetic terrains?',
               True, False, x3_units, x3_run),
    Experiment('X4', 'x4_resolution_synthetic', 'Reduction versus resolution',
               'Does the reduction and its quality loss depend on the cell size '
               '(20, 10, 5, 3, 1 m) of the same terrain?',
               True, False, x4_units, x4_run),
    Experiment('X5', 'x5_scaling', 'Computational scaling',
               'How do search time, nodes expanded and memory grow with the grid, full '
               'grid versus HAG, and after how many queries does the HAG build pay off?',
               True, False, x5_units, x5_run),
    Experiment('X6', 'x6_dem_whole', 'The whole Idrija DEM',
               'The reduction, its cost and the speed-up on the whole surveyed swath at '
               '20/10/5/3/1 m, for O-D pairs 1 to 20 km apart.',
               True, True, x6_units, x6_run),
    Experiment('X7', 'x7_dem_windows', 'DEM windows down to 1 m',
               'The same on 1-4 km windows of the DEM, which reach 1 m where the whole '
               'swath does not fit in memory.',
               True, True, x7_units, x7_run),
    Experiment('X9', 'x9_engines', 'Engines and parallelisation',
               'How fast is one search with the Python engine, the compiled library called '
               'in-process (ctypes) and the same compiled search as a standalone executable, '
               'and how does the throughput grow with the number of concurrent workers?',
               True, False, x9_units, x9_run, exclusive=True),
)}
