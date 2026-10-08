"""The terrain grid as numpy arrays, with ONE step-length function.

Every graph this package builds uses the single `step_lengths_m` (a diagonal step is
sqrt(2) cells long), so all baselines are measured alike.

Node ids are plain integers: `node = i * n_cols + j`, row-major, the same order as
`numpy.ravel_multi_index`. That is also the order `scipy.sparse.csgraph` expects, which
is what makes the oracle test in tests/test_core_search.py a real cross-check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

#: (di, dj) offsets. The first four are the 4-connected ones, so
#: `NEIGHBOR_OFFSETS[:4]` is the 4-connectivity and the whole array the 8-connectivity.
NEIGHBOR_OFFSETS = np.array([(-1, 0), (0, -1), (0, 1), (1, 0),
                             (-1, -1), (-1, 1), (1, -1), (1, 1)], dtype=np.int64)

Connectivity = int  # 4 or 8


def offsets(connectivity: Connectivity = 8) -> np.ndarray:
    if connectivity == 4:
        return NEIGHBOR_OFFSETS[:4]
    if connectivity == 8:
        return NEIGHBOR_OFFSETS
    raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")


def step_lengths_m(di: np.ndarray, dj: np.ndarray, cell_size_m: float) -> np.ndarray:
    """The ONE planar length function: Euclidean distance between cell centres.

    A straight step is `cell_size_m`, a diagonal step `sqrt(2) * cell_size_m`.
    """
    return np.hypot(np.asarray(di, dtype=float), np.asarray(dj, dtype=float)) * float(cell_size_m)


@dataclass(frozen=True)
class Grid:
    """A heightmap plus the physical size of one cell.

    `surf[i, j]` is the terrain height in meters at cell (i, j); the cell centre is at
    `(i * cell_size_m, j * cell_size_m)`.
    """

    surf: np.ndarray
    cell_size_m: float = 10.0

    def __post_init__(self):
        surf = np.ascontiguousarray(self.surf, dtype=float)
        if surf.ndim != 2:
            raise ValueError(f"surf must be 2D, got shape {surf.shape}")
        if float(self.cell_size_m) <= 0:
            raise ValueError("cell_size_m must be positive")
        if not np.isfinite(surf).all():
            # a NaN void from a real DEM otherwise propagates silently into
            # edge_arrays and into height_buckets, which takes surf.min()
            n_bad = int((~np.isfinite(surf)).sum())
            raise ValueError(f"surf has {n_bad} non-finite cells (NaN or inf). A DEM "
                             f"with voids must go through core.dem.fill_voids first.")
        object.__setattr__(self, 'surf', surf)
        object.__setattr__(self, 'cell_size_m', float(self.cell_size_m))

    def assert_plausible_heights(self, lo: float = -500.0, hi: float = 9000.0) -> None:
        """Fail loudly on a sentinel that was never converted.

        A surviving SRTM -32768 does not crash anything: `core.hag.height_buckets`
        takes `surf.min()`, so a 3 m bucket size silently becomes about 11000 buckets
        and the HAG is meaningless. This is opt-in and called by `core.dem`.
        """
        low, high = float(self.surf.min()), float(self.surf.max())
        if low < lo or high > hi:
            raise ValueError(f"heights run {low:.1f} to {high:.1f} m, outside the "
                             f"plausible {lo:.0f} to {hi:.0f} m. An unconverted nodata "
                             f"sentinel (SRTM uses -32768) is the usual cause.")

    # -- shape ---------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self.surf.shape

    @property
    def n_rows(self) -> int:
        return self.surf.shape[0]

    @property
    def n_cols(self) -> int:
        return self.surf.shape[1]

    @property
    def n_nodes(self) -> int:
        return self.surf.size

    # -- node ids ------------------------------------------------------------------
    def index(self, coord: Sequence[int]) -> int:
        i, j = int(coord[0]), int(coord[1])
        if not (0 <= i < self.n_rows and 0 <= j < self.n_cols):
            raise IndexError(f"coord {(i, j)} is outside a {self.shape} grid")
        return i * self.n_cols + j

    def indices(self, coords: Iterable[Sequence[int]]) -> np.ndarray:
        return np.array([self.index(c) for c in coords], dtype=np.int64)

    def coord(self, node: int) -> tuple[int, int]:
        node = int(node)
        if not 0 <= node < self.n_nodes:
            raise IndexError(f"node {node} is outside a {self.shape} grid")
        return divmod(node, self.n_cols)

    def coords(self, nodes: Iterable[int]) -> np.ndarray:
        nodes = np.asarray(list(nodes), dtype=np.int64)
        return np.stack(np.divmod(nodes, self.n_cols), axis=-1)

    # -- heights -------------------------------------------------------------------
    @property
    def heights_flat(self) -> np.ndarray:
        return self.surf.reshape(-1)

    def interpolate_height(self, xy: np.ndarray) -> np.ndarray:
        """Bilinear height at real-valued grid coordinates.

        `xy` is (..., 2) in grid units (not meters). Coordinates outside the grid are
        clamped to the border.
        """
        from scipy.ndimage import map_coordinates

        pts = np.atleast_2d(np.asarray(xy, dtype=float))
        rows = np.clip(pts[:, 0], 0, self.n_rows - 1)
        cols = np.clip(pts[:, 1], 0, self.n_cols - 1)
        out = map_coordinates(self.surf, np.stack([rows, cols]), order=1, mode='nearest')
        return out if np.ndim(xy) > 1 else out[0]

    def path_heights(self, path_xy: np.ndarray) -> np.ndarray:
        """Z re-sampled from the terrain for a smoothed XY path (research step 3)."""
        return self.interpolate_height(np.asarray(path_xy, dtype=float)[:, :2])

    def path_xyz_m(self, path_xy: np.ndarray) -> np.ndarray:
        """(x, y, z) in METERS for a path given in grid units."""
        path_xy = np.asarray(path_xy, dtype=float)[:, :2]
        z = self.path_heights(path_xy)
        return np.column_stack([path_xy * self.cell_size_m, z])


def edge_arrays(grid: Grid,
                mask: Optional[np.ndarray] = None,
                connectivity: Connectivity = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every directed edge of the grid graph, as four parallel arrays.

    Returns `(u, v, length_m, dh)` where `dh = height(v) - height(u)`. When `mask` is
    given (a boolean array shaped like `surf`) only edges with both ends inside the mask
    are produced; that is how the HAG-restricted graph is built.

    Both directions of every adjacency are present, so `u, v` and `v, u` both appear and
    `dh` is antisymmetric, and every length is positive.
    """
    n_rows, n_cols = grid.shape
    if mask is None:
        mask = np.ones(grid.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != grid.shape:
            raise ValueError(f"mask shape {mask.shape} != grid shape {grid.shape}")

    ii, jj = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing='ij')
    us, vs, lengths, dhs = [], [], [], []
    for di, dj in offsets(connectivity):
        i2, j2 = ii + di, jj + dj
        ok = (i2 >= 0) & (i2 < n_rows) & (j2 >= 0) & (j2 < n_cols) & mask
        ok &= np.where(ok, mask[np.clip(i2, 0, n_rows - 1), np.clip(j2, 0, n_cols - 1)], False)
        if not ok.any():
            continue
        u = (ii[ok] * n_cols + jj[ok]).astype(np.int64)
        v = (i2[ok] * n_cols + j2[ok]).astype(np.int64)
        us.append(u)
        vs.append(v)
        lengths.append(np.full(u.shape, step_lengths_m(di, dj, grid.cell_size_m)))
        dhs.append(grid.heights_flat[v] - grid.heights_flat[u])
    if not us:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=float)
        return empty_i, empty_i, empty_f, empty_f
    return (np.concatenate(us), np.concatenate(vs),
            np.concatenate(lengths), np.concatenate(dhs))
