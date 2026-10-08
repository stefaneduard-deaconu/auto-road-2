"""Height Area Graphs, built with `scipy.ndimage.label`.

A Height Area (HA) is a connected set of grid cells whose heights fall in the same
`height_delta` bucket. The Height Areas Graph (HAG) has one node per HA and an edge
between two HAs whose cells touch.

Design choices:

* **One connectivity.** A single `connectivity` argument drives both the labelling and
  the adjacency, so two diagonally touching cells of the *same* height are one area,
  never two neighbouring ones; `assert_no_equal_height_neighbours` checks it.
* **A breadth-first `vdist`.** The hop distance is a breadth-first sweep over the
  adjacency, which is the minimum hop count by construction.
* **The selection rule is the candidate terrain areas.** The selected areas are the HAG
  nodes on the cheapest start-to-target chain, every HAG edge priced by the terrain
  (`shared_border`, see `hag_edge_table`), and the grid search is restricted to them.
* **A k-hop dilation** (`dilate_areas`) as the sensitivity variant, because a
  one-area-wide chain can make the restricted grid path infeasible or poor.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra

from core import costs as cost_models
from core.grid import Connectivity, Grid

SELECTION_RULES = ('yellow', 'full')

#: how a HAG edge (two touching areas) is priced. Both price the edge with the SAME grid
#: cost model the detailed search uses, so both levels are in the same units. The study
#: always uses `shared_border`; `area_means` is kept as a configurable alternative.
HAG_EDGE_COSTS = ('shared_border', 'area_means')

#: keeps every terrain-priced edge strictly positive: nested areas can share a centroid.
HAG_MIN_EDGE_WEIGHT = 1e-12


def height_buckets(surf: np.ndarray, height_delta: float) -> np.ndarray:
    """Bucket heights into bands of `height_delta`.

    Returns the bucket's lower edge per cell: `min + height_delta * floor((h - min) / height_delta)`.
    NaN cells (DEM nodata) stay NaN and do not move the minimum.
    """
    surf = np.asarray(surf, dtype=float)
    height_delta = int(height_delta)
    if height_delta <= 0:
        raise ValueError('height_delta must be a positive integer')
    minimum = np.nanmin(surf)
    return minimum + height_delta * np.floor((surf - minimum) / height_delta)


def _structure(connectivity: Connectivity) -> np.ndarray:
    if connectivity == 4:
        return ndimage.generate_binary_structure(2, 1)
    if connectivity == 8:
        return ndimage.generate_binary_structure(2, 2)
    raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")


@dataclass(frozen=True)
class HAG:
    """Height Areas Graph over one terrain.

    Attributes:
        grid: the terrain.
        height_delta: the bucket size in meters.
        connectivity: 4 or 8, used for BOTH the labelling and the adjacency.
        labels: (n_rows, n_cols) int array, the area id of every cell, 0-based; -1 on a
            cell outside `valid_mask` (DEM nodata), which belongs to no area.
        area_height: (n_areas,) bucket height of every area.
        area_size: (n_areas,) number of cells.
        centroid: (n_areas, 2) cell-coordinate centroid of every area.
        neighbours: list of sets, the adjacency.
    """

    grid: Grid
    height_delta: float
    connectivity: Connectivity
    labels: np.ndarray
    area_height: np.ndarray
    area_size: np.ndarray
    centroid: np.ndarray
    neighbours: list[set[int]]

    @property
    def n_areas(self) -> int:
        return int(self.area_height.size)

    @property
    def n_valid_cells(self) -> int:
        """Cells that belong to an area: the full grid of a DEM excludes its nodata."""
        return int(self.area_size.sum())

    def area_of(self, coord: Sequence[int]) -> int:
        return int(self.labels[int(coord[0]), int(coord[1])])

    def cells(self, area_id: int) -> np.ndarray:
        """(k, 2) array of the cells of one area."""
        return np.argwhere(self.labels == int(area_id))

    def contour(self, area_id: int) -> np.ndarray:
        """(k, 2) cells of the area that touch another area or the grid border."""
        inside = self.labels == int(area_id)
        eroded = ndimage.binary_erosion(inside, structure=_structure(self.connectivity),
                                        border_value=0)
        return np.argwhere(inside & ~eroded)

    def mask(self, area_ids: Iterable[int]) -> np.ndarray:
        """Boolean cell mask for a set of areas, to restrict the grid search."""
        wanted = np.zeros(self.n_areas + 1, dtype=bool)
        ids = np.fromiter((int(a) for a in area_ids), dtype=np.int64)
        if ids.size:
            wanted[ids] = True
        # label -1 (nodata) indexes the extra last slot, which is never selected
        return wanted[self.labels]

    def n_cells_in(self, area_ids: Iterable[int]) -> int:
        return int(self.area_size[np.fromiter((int(a) for a in area_ids), dtype=np.int64)].sum()) \
            if area_ids else 0

    def assert_no_equal_height_neighbours(self) -> None:
        """Two areas of the same bucket height must never be adjacent."""
        for a, nbrs in enumerate(self.neighbours):
            for b in nbrs:
                if self.area_height[a] == self.area_height[b]:
                    raise AssertionError(
                        f"areas {a} and {b} have the same height {self.area_height[a]} "
                        f"and are adjacent; the connectivity is inconsistent")


def _area_sums(labels: np.ndarray, n_areas: int,
               band_rows: int = 2048) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell count, row-index sum and column-index sum per area, a band of rows at a time."""
    size = np.zeros(n_areas, dtype=np.int64)
    row_sum = np.zeros(n_areas)
    col_sum = np.zeros(n_areas)
    n_cols = labels.shape[1]
    cols = np.arange(n_cols, dtype=float)
    for r0 in range(0, labels.shape[0], band_rows):
        band = labels[r0:r0 + band_rows]
        inside = band >= 0
        ids = band[inside]
        size += np.bincount(ids, minlength=n_areas)
        rows = np.broadcast_to(np.arange(r0, r0 + band.shape[0], dtype=float)[:, None],
                               band.shape)
        row_sum += np.bincount(ids, weights=rows[inside], minlength=n_areas)
        col_sum += np.bincount(ids, weights=np.broadcast_to(cols, band.shape)[inside],
                               minlength=n_areas)
    return size, row_sum, col_sum


def build_hag(grid: Grid, height_delta: float, connectivity: Connectivity = 8,
              valid_mask: Optional[np.ndarray] = None, native: bool = False) -> HAG:
    """Label the height areas and their adjacency with one consistent connectivity.

    `valid_mask` (True = usable cell) excludes DEM nodata: those cells get label -1, join
    no area and create no adjacency. NaN heights are excluded the same way. `native=True`
    labels with `core.native` (same numbering, tested), which a large DEM needs.
    """
    surf = np.asarray(grid.surf, dtype=float)
    valid = np.isfinite(surf)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    if not valid.any():
        raise ValueError('no valid cell: the whole grid is nodata')
    buckets = height_buckets(np.where(valid, surf, np.nan), height_delta)
    structure = _structure(connectivity)

    if native:
        from core import native as native_engine
        minimum = np.nanmin(np.where(valid, surf, np.nan))
        index = np.where(valid, np.floor((np.where(valid, surf, minimum) - minimum)
                                         / int(height_delta)), -1).astype(np.int32)
        labels, n_areas = native_engine.label_areas(index, connectivity)
        cells = np.flatnonzero(labels.reshape(-1) >= 0)
        _, first_of = np.unique(labels.reshape(-1)[cells], return_index=True)
        heights = buckets.reshape(-1)[cells[first_of]].tolist()
    else:
        labels = np.full(grid.shape, -1, dtype=np.int64)
        next_id = 0
        heights = []
        for value in np.unique(buckets[valid]):
            component, n = ndimage.label(valid & (buckets == value), structure=structure)
            found = component > 0
            labels[found] = component[found] - 1 + next_id
            heights.extend([float(value)] * n)
            next_id += n
        n_areas = next_id
    if (labels[valid] < 0).any():
        raise AssertionError('some cells were not assigned to an area')

    area_height = np.array(heights, dtype=float)
    area_size, row_sum, col_sum = _area_sums(labels, n_areas)
    centroid = np.column_stack([row_sum, col_sum]) / area_size[:, None]

    neighbours: list[set[int]] = [set() for _ in range(n_areas)]
    from core.grid import offsets
    for di, dj in offsets(connectivity):
        a = labels[max(di, 0):labels.shape[0] + min(di, 0),
                   max(dj, 0):labels.shape[1] + min(dj, 0)]
        b = labels[max(-di, 0):labels.shape[0] + min(-di, 0),
                   max(-dj, 0):labels.shape[1] + min(-dj, 0)]
        differ = (a != b) & (a >= 0) & (b >= 0)
        pairs = np.unique(np.stack([a[differ], b[differ]], axis=1), axis=0)
        for u, v in pairs.tolist():
            neighbours[u].add(v)
            neighbours[v].add(u)

    return HAG(grid=grid, height_delta=float(height_delta), connectivity=connectivity,
               labels=labels, area_height=area_height, area_size=area_size,
               centroid=centroid, neighbours=neighbours)


# -- hop distance and selection ---------------------------------------------------------

def vdist(hag: HAG, start_area: int) -> np.ndarray:
    """Minimum number of area hops from `start_area` to every area.

    Unreached areas get -1. This is a breadth-first sweep with a `deque`, so the value
    is the minimum hop count by construction.
    """
    distances = np.full(hag.n_areas, -1, dtype=np.int64)
    distances[int(start_area)] = 0
    queue = deque([int(start_area)])
    while queue:
        area = queue.popleft()
        for neighbour in hag.neighbours[area]:
            if distances[neighbour] < 0:
                distances[neighbour] = distances[area] + 1
                queue.append(neighbour)
    return distances


def dilate_areas(hag: HAG, area_ids: Iterable[int], k_hops: int) -> set[int]:
    """Add every area within `k_hops` of the selection (sensitivity variant)."""
    selected = {int(a) for a in area_ids}
    for _ in range(int(k_hops)):
        selected |= {n for a in selected for n in hag.neighbours[a]}
    return selected


def select_areas(hag: HAG, start: Sequence[int], target: Sequence[int],
                 rule: str = 'yellow', k_hops: int = 0, *,
                 edge_cost: str = 'shared_border',
                 cost: cost_models.CostSpec = cost_models.DEFAULT_COST,
                 graph: Optional[csr_matrix] = None) -> set[int]:
    """Dispatch over the selection rules, with an optional k-ring dilation.

    `rule='yellow'` selects the candidate terrain areas: the areas on the cheapest HAG
    chain, with every HAG edge priced by `edge_cost` (one of `HAG_EDGE_COSTS`; the study
    uses `shared_border`). `rule='full'` selects everything, i.e. the full-grid baseline
    expressed in the same terms. Pass a prebuilt `graph` to reuse it over many O-D pairs
    of one HAG.
    """
    if rule == 'yellow':
        if graph is None:
            graph = area_graph(hag, edge_cost, cost)
        chain = cheapest_area_chain(graph, hag.area_of(start), hag.area_of(target))
        if chain is None:
            raise ValueError('start and target are in disconnected parts of the HAG')
        selected = set(chain)
    elif rule == 'full':
        selected = set(range(hag.n_areas))
    else:
        raise ValueError(f"unknown selection rule {rule!r}; known: {SELECTION_RULES}")
    return dilate_areas(hag, selected, k_hops) if k_hops else selected


# -- terrain-priced HAG edges --------------------------------------------------------------

def area_mean_heights(hag: HAG) -> np.ndarray:
    """(n_areas,) mean TERRAIN height of every area (not its bucket height)."""
    flat = hag.labels.reshape(-1)
    inside = flat >= 0
    total = np.bincount(flat[inside], weights=np.asarray(hag.grid.surf, float).reshape(-1)[inside],
                        minlength=hag.n_areas)
    return total / hag.area_size


def hag_edge_table(hag: HAG, edge_cost: str,
                   cost: cost_models.CostSpec = cost_models.DEFAULT_COST
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(u, v, weight)` for every undirected HAG edge, `u < v`.

    Both terrain rules feed the grid's own cost model with one `(dh, length_m)` per edge:

    * `area_means`: dh = difference of the two areas' mean terrain heights, length =
      distance between their centroids (at least one cell: nested areas share a centroid).
    * `shared_border`: dh = mean absolute height step over the cell pairs of the common
      contour, length = centroid -> contour midpoint -> centroid.
    """
    if edge_cost not in HAG_EDGE_COSTS:
        raise ValueError(f'unknown HAG edge cost {edge_cost!r}; known: {HAG_EDGE_COSTS}')
    low, high, node_low, node_high = crossing_arrays(hag)
    if low.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0)
    key = low * hag.n_areas + high
    keys, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    u, v = low[first], high[first]
    cell = hag.grid.cell_size_m

    if edge_cost == 'area_means':
        means = area_mean_heights(hag)
        dh = means[v] - means[u]
        length_m = np.maximum(np.linalg.norm(hag.centroid[u] - hag.centroid[v], axis=1) * cell,
                              cell)
    else:
        surf = np.asarray(hag.grid.surf, float).reshape(-1)
        n_cols = hag.labels.shape[1]
        counts = np.bincount(inverse, minlength=keys.size)
        dh = np.bincount(inverse, weights=np.abs(surf[node_high] - surf[node_low]),
                         minlength=keys.size) / counts
        mid_r = np.bincount(inverse, weights=(node_low // n_cols + node_high // n_cols) / 2.0,
                            minlength=keys.size) / counts
        mid_c = np.bincount(inverse, weights=(node_low % n_cols + node_high % n_cols) / 2.0,
                            minlength=keys.size) / counts
        mid = np.column_stack([mid_r, mid_c])
        length_m = np.maximum((np.linalg.norm(hag.centroid[u] - mid, axis=1)
                               + np.linalg.norm(mid - hag.centroid[v], axis=1)) * cell, cell)

    weight = np.asarray(cost_models.resolve(cost)(dh, length_m), dtype=float)
    weight = np.broadcast_to(weight, dh.shape).astype(float)
    if not np.all(np.isfinite(weight)) or (weight < 0).any():
        raise ValueError(f'HAG edge cost {edge_cost!r} under {cost_models.name_of(cost)} '
                         'produced a negative or non-finite weight')
    return u, v, np.maximum(weight, HAG_MIN_EDGE_WEIGHT)


def area_graph(hag: HAG, edge_cost: str,
               cost: cost_models.CostSpec = cost_models.DEFAULT_COST) -> csr_matrix:
    """The HAG as a symmetric sparse matrix of edge weights, for `cheapest_area_chain`."""
    u, v, w = hag_edge_table(hag, edge_cost, cost)
    n = hag.n_areas
    return csr_matrix((np.concatenate([w, w]), (np.concatenate([u, v]), np.concatenate([v, u]))),
                      shape=(n, n))


def cheapest_area_chain(graph: csr_matrix, start_area: int,
                        target_area: int) -> Optional[list[int]]:
    """The cheapest chain of areas from `start_area` to `target_area` (Dijkstra, in C)."""
    start_area, target_area = int(start_area), int(target_area)
    dist, previous = csgraph_dijkstra(graph, directed=True, indices=start_area,
                                      return_predecessors=True)
    if not np.isfinite(dist[target_area]):
        return None
    chain = [target_area]
    while chain[-1] != start_area:
        chain.append(int(previous[chain[-1]]))
    chain.reverse()
    return chain


def search_space_fraction(hag: HAG, area_ids: Iterable[int]) -> float:
    """Selected cells divided by all VALID cells, in [0, 1].

    Report it as "the search space is X% OF the full grid" and the reduction as
    "(1 - X)%"; results always carry both numbers, see `core.experiment_step1`. On a synthetic
    terrain every cell is valid, so the denominator is the grid size.
    """
    return float(hag.mask(area_ids).sum()) / float(hag.n_valid_cells)


# -- shared boundaries ------------------------------------------------------------------

def _forward_offsets(connectivity: Connectivity) -> np.ndarray:
    """Half of the offsets: the ones lexicographically after (0, 0).

    Every adjacency appears twice in `offsets`, once as `(di, dj)` and once as its
    negation. Iterating only the forward half visits every UNDIRECTED pair of touching
    cells exactly once, which is what a boundary wants: a boundary row is a crossing, not
    a directed edge.
    """
    from core.grid import offsets

    all_offsets = offsets(connectivity)
    keep = (all_offsets[:, 0] > 0) | ((all_offsets[:, 0] == 0) & (all_offsets[:, 1] > 0))
    return all_offsets[keep]


def _shift_slices(di: int, dj: int, shape: tuple[int, int]) -> tuple[tuple, tuple]:
    """The two aligned views `build_hag` uses, as explicit slices so ids can be taken too."""
    n_rows, n_cols = shape
    a = (slice(max(di, 0), n_rows + min(di, 0)), slice(max(dj, 0), n_cols + min(dj, 0)))
    b = (slice(max(-di, 0), n_rows + min(-di, 0)), slice(max(-dj, 0), n_cols + min(-dj, 0)))
    return a, b


def shared_boundaries(hag: HAG) -> dict[tuple[int, int], np.ndarray]:
    """The cells where two height areas touch, per pair of areas.

    `build_hag` already computes exactly these positions to fill `neighbours` and then
    throws them away, keeping only the fact that two areas touch. The portal overlay needs
    the positions, so this recomputes them with the same vectorised shift-and-compare.

    Returns a dict keyed by `(a, b)` with `a < b`, whose value is a `(k, 2)` array of
    global grid node ids: column 0 is the cell in area `a`, column 1 the cell in area `b`.
    One row per CROSSING, i.e. per undirected pair of touching cells, so a boundary has
    half as many rows as `core.grid.edge_arrays` has directed edges between the two areas.
    Rows are sorted by `(node_a, node_b)`, so the result never depends on set iteration
    order.
    """
    low, high, node_low, node_high = crossing_arrays(hag)
    if low.size == 0:
        return {}

    key = low * hag.n_areas + high
    order = np.lexsort((node_high, node_low, key))
    key, node_low, node_high = key[order], node_low[order], node_high[order]
    low, high = low[order], high[order]

    starts = np.flatnonzero(np.concatenate([[True], key[1:] != key[:-1]]))
    ends = np.concatenate([starts[1:], [key.size]])
    return {(int(low[s]), int(high[s])): np.column_stack([node_low[s:e], node_high[s:e]])
            for s, e in zip(starts, ends)}


def crossing_arrays(hag: HAG) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every undirected pair of touching cells in two different areas, unsorted.

    Returns `(low_area, high_area, node_low, node_high)` with `low_area < high_area`;
    `node_*` are global grid node ids. Nodata cells (label -1) never cross.
    """
    labels = hag.labels
    shape = labels.shape
    nodes = np.arange(labels.size, dtype=np.int64).reshape(shape)

    lows, highs, node_lows, node_highs = [], [], [], []
    for di, dj in _forward_offsets(hag.connectivity):
        sl_a, sl_b = _shift_slices(int(di), int(dj), shape)
        label_a, label_b = labels[sl_a], labels[sl_b]
        differ = (label_a != label_b) & (label_a >= 0) & (label_b >= 0)
        if not differ.any():
            continue
        la, lb = label_a[differ], label_b[differ]
        na, nb = nodes[sl_a][differ], nodes[sl_b][differ]
        swap = la > lb
        lows.append(np.where(swap, lb, la))
        highs.append(np.where(swap, la, lb))
        node_lows.append(np.where(swap, nb, na))
        node_highs.append(np.where(swap, na, nb))

    if not lows:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, empty
    return (np.concatenate(lows), np.concatenate(highs),
            np.concatenate(node_lows), np.concatenate(node_highs))
    order = np.lexsort((node_high, node_low, key))
    key, node_low, node_high = key[order], node_low[order], node_high[order]
    low, high = low[order], high[order]

    starts = np.flatnonzero(np.concatenate([[True], key[1:] != key[:-1]]))
    ends = np.concatenate([starts[1:], [key.size]])
    return {(int(low[s]), int(high[s])): np.column_stack([node_low[s:e], node_high[s:e]])
            for s, e in zip(starts, ends)}


def boundary_components(hag: HAG, crossings: np.ndarray) -> list[np.ndarray]:
    """Split one boundary into its arcs.

    Two areas can touch along several separate stretches - around a spur, or on both sides
    of a saddle - and each stretch needs its own portal, otherwise a single portal stands
    in for two crossings that are far apart on the ground.

    An arc is a connected component of the CONTACT ZONE: the union of both sides' cells,
    labelled with the HAG's own connectivity. A crossing's two cells are adjacent and both
    in that union, so they always land in the same component and the split does not depend
    on which area was called `a`. Components come back in raster-scan label order.
    """
    crossings = np.asarray(crossings, dtype=np.int64)
    if crossings.size == 0:
        return []
    contact = np.zeros(hag.labels.size, dtype=bool)
    contact[crossings.reshape(-1)] = True
    component, n_components = ndimage.label(contact.reshape(hag.labels.shape),
                                            structure=_structure(hag.connectivity))
    of_crossing = component.reshape(-1)[crossings[:, 0]]
    arcs = [crossings[of_crossing == c] for c in range(1, n_components + 1)]
    return [arc for arc in arcs if arc.size]
