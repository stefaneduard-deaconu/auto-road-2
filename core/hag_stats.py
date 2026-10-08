"""Descriptors of a Height Area Graph, for describing HAGs the way graphs are described.

Pure numpy/scipy, no search. `describe_hag` answers "what does the HAG of this terrain look
like": how many areas, how many edges per area, how the areas are sized, how far apart the
graph's ends are. `describe_selection` does the same for one selected chain (the CTA).

A HAG is planar: height areas are regions of the plane and two areas share an edge only if
they touch, so E <= 3N - 6. `planarity_ratio` = E / (3N - 6) says how close to a
triangulation the adjacency is; nested bands (rings round a hill) keep it low.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

from core.hag import HAG

#: 1 gon = 0.9 degree; STAS 863-85 states terrain classes in gon (hilly: 20-25 gon)
DEGREES_PER_GON = 0.9


def _csr(hag: HAG) -> csr_matrix:
    rows = np.repeat(np.arange(hag.n_areas), [len(s) for s in hag.neighbours])
    cols = np.fromiter((v for s in hag.neighbours for v in sorted(s)), dtype=np.int64,
                       count=rows.size)
    return csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)),
                      shape=(hag.n_areas, hag.n_areas))


def gini(values: np.ndarray) -> float:
    """0 when every area has the same size, towards 1 when one area holds everything."""
    x = np.sort(np.asarray(values, dtype=float))
    if x.size == 0 or x.sum() == 0:
        return 0.0
    ranks = np.arange(1, x.size + 1)
    return float((2 * (ranks * x).sum()) / (x.size * x.sum()) - (x.size + 1) / x.size)


def _eccentricity(graph: csr_matrix, source: int) -> tuple[int, int]:
    """(farthest node, its hop distance) from `source` by BFS."""
    order, predecessors = breadth_first_order(graph, source, directed=False,
                                              return_predecessors=True)
    depth = np.zeros(graph.shape[0], dtype=np.int64)
    for node in order[1:]:
        depth[node] = depth[predecessors[node]] + 1
    far = int(order[np.argmax(depth[order])])
    return far, int(depth[far])


def hop_diameter_lower_bound(hag: HAG, graph: Optional[csr_matrix] = None,
                             sweeps: int = 4) -> int:
    """Double-sweep lower bound of the hop diameter of the largest component.

    Exact on trees and usually exact on planar graphs; a lower bound in general. The exact
    diameter costs one BFS per area, which a 1 m DEM cannot afford.
    """
    graph = _csr(hag) if graph is None else graph
    if hag.n_areas < 2:
        return 0
    _, component = connected_components(graph, directed=False)
    largest = int(np.bincount(component).argmax())
    node = int(np.flatnonzero(component == largest)[0])
    best = 0
    for _ in range(sweeps):
        node, dist = _eccentricity(graph, node)
        best = max(best, dist)
    return best


def perimeters_cells(hag: HAG) -> np.ndarray:
    """Boundary cells of every area (cells touching another area, nodata or the border)."""
    inside = hag.labels >= 0
    eroded_same = np.ones(hag.labels.shape, dtype=bool)
    padded = np.pad(hag.labels, 1, constant_values=-2)
    n_rows, n_cols = hag.labels.shape
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if (di, dj) == (0, 0) or (hag.connectivity == 4 and di and dj):
                continue
            eroded_same &= padded[1 + di:1 + di + n_rows, 1 + dj:1 + dj + n_cols] == hag.labels
    boundary = inside & ~eroded_same
    return np.bincount(hag.labels[boundary], minlength=hag.n_areas)


def terrain_slope_percent(hag: HAG) -> np.ndarray:
    """Terrain slope in percent, central differences, on valid cells away from nodata."""
    surf = np.asarray(hag.grid.surf, dtype=float)
    gy, gx = np.gradient(surf, hag.grid.cell_size_m)
    slope = 100.0 * np.hypot(gx, gy)
    interior = ndimage.binary_erosion(hag.labels >= 0, border_value=1)
    return slope[interior]


def _percentiles(values: np.ndarray, prefix: str, qs=(50, 90)) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {f'{prefix}_mean': math.nan, f'{prefix}_max': math.nan,
                **{f'{prefix}_p{q}': math.nan for q in qs}}
    out = {f'{prefix}_mean': float(values.mean()), f'{prefix}_max': float(values.max())}
    out.update({f'{prefix}_p{q}': float(np.percentile(values, q)) for q in qs})
    return out


def describe_hag(hag: HAG, *, diameter: bool = True) -> dict:
    """Scalar descriptors of one HAG, for a result row."""
    n = hag.n_areas
    degree = np.array([len(s) for s in hag.neighbours], dtype=np.int64)
    n_edges = int(degree.sum() // 2)
    graph = _csr(hag)
    n_components, _ = connected_components(graph, directed=False)
    size_cells = hag.area_size.astype(float)
    cell_m2 = hag.grid.cell_size_m ** 2
    perimeter = perimeters_cells(hag).astype(float)
    compactness = 4 * math.pi * size_cells / np.maximum(perimeter, 1.0) ** 2
    slope = terrain_slope_percent(hag)
    slope_gon = np.degrees(np.arctan(slope / 100.0)) / DEGREES_PER_GON
    bands = np.unique(hag.area_height).size
    row = {
        'n_areas': n,
        'n_hag_edges': n_edges,
        'n_components': int(n_components),
        'n_height_bands': int(bands),
        'areas_per_band_mean': n / bands if bands else math.nan,
        'edges_per_area_mean': 2.0 * n_edges / n if n else math.nan,
        'degree_median': float(np.median(degree)) if n else math.nan,
        'degree_p90': float(np.percentile(degree, 90)) if n else math.nan,
        'degree_max': int(degree.max()) if n else 0,
        'share_degree_1': float((degree == 1).mean()) if n else math.nan,
        'planarity_ratio': n_edges / (3 * n - 6) if n > 2 else math.nan,
        'cyclomatic_number': int(n_edges - n + n_components),
        'area_cells_median': float(np.median(size_cells)),
        'area_m2_median': float(np.median(size_cells) * cell_m2),
        'area_m2_mean': float(size_cells.mean() * cell_m2),
        'area_size_gini': gini(size_cells),
        'largest_area_share': float(size_cells.max() / size_cells.sum()),
        'single_cell_area_share': float((size_cells == 1).mean()),
        'compactness_median': float(np.median(compactness)),
        'n_valid_cells': hag.n_valid_cells,
        'n_cells_total': int(hag.labels.size),
        'valid_share': hag.n_valid_cells / hag.labels.size,
        'relief_m': float(np.ptp(np.asarray(hag.grid.surf)[hag.labels >= 0])),
        **_percentiles(slope, 'terrain_slope_percent'),
        'terrain_slope_gon_p50': float(np.percentile(slope_gon, 50)) if slope.size else math.nan,
        'share_slope_20_25_gon': float(((slope_gon >= 20) & (slope_gon <= 25)).mean())
        if slope.size else math.nan,
    }
    if diameter:
        row['hop_diameter_lb'] = hop_diameter_lower_bound(hag, graph)
    return row


def describe_selection(hag: HAG, chain_or_set: Iterable[int], selected: Iterable[int]) -> dict:
    """Descriptors of one selection: the CTA chain and the (possibly dilated) area set."""
    chain = list(chain_or_set)
    selected = sorted({int(a) for a in selected})
    mask = hag.mask(selected)
    narrowest = math.nan
    if selected:
        # 2 x the largest inscribed distance of the thinnest selected area
        boxes = ndimage.find_objects(hag.labels + 1, max_label=hag.n_areas)
        widths = []
        for area in selected:
            box = boxes[area]
            inside = np.pad(hag.labels[box] == area, 1)
            widths.append(2.0 * ndimage.distance_transform_edt(inside).max())
        narrowest = float(min(widths)) * hag.grid.cell_size_m
    return {
        'cta_hops': max(len(chain) - 1, 0),
        'n_areas_selected': len(selected),
        'n_cells_selected': int(mask.sum()),
        'cta_narrowest_width_m': narrowest,
    }
