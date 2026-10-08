"""Shared machinery of the HAG experiment programme (X1-X7 in the study notes).

A `Case` is one terrain: a synthetic Perlin surface or a window of the Idrija DEM at one
cell size, with a validity mask for the DEM's nodata. For each case the runners call:

* `hag_row`     - build the HAG once and describe it (`core.hag_stats`), timed;
* `sample_pairs` - draw O-D pairs without running any search (scales to the whole DEM);
* `search_rows` - the full grid and the HAG selections on the same O-D pair, with the
  same Dijkstra (`engine`), timed as a median of repeats, memory in a separate run.

Everything here is pure computation: the runner in `core.run` owns files and resume.
"""
from __future__ import annotations

import math
import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

from core import costs as cost_models
from core import metrics as metrics_mod
from core import native
from core.grid import Grid
from core.hag import HAG, area_graph, build_hag, cheapest_area_chain, dilate_areas
from core.hag_stats import describe_hag, describe_selection
from core.naming import SEARCH_SPACES, grid_edge_cost
from core.search import build_graph, dijkstra


@dataclass
class Case:
    case_id: str
    family: str                      # 'synthetic' | 'dem'
    grid: Grid
    valid: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    @property
    def n_valid(self) -> int:
        return int(self.grid.n_nodes if self.valid is None else self.valid.sum())


@dataclass(frozen=True)
class Pair:
    od_id: str
    start: tuple[int, int]
    target: tuple[int, int]
    straight_line_m: float
    band: str = ''


# -- HAG ----------------------------------------------------------------------------------

def timed_hag(case: Case, height_delta: float, *, use_native: bool,
              repeats: int = 1) -> tuple[HAG, list[float]]:
    timings, hag = [], None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        hag = build_hag(case.grid, height_delta, valid_mask=case.valid, native=use_native)
        timings.append(time.perf_counter() - t0)
    return hag, timings


def hag_row(case: Case, hag: HAG, timings: Sequence[float]) -> dict:
    row = describe_hag(hag)
    row.update({'hag_build_time_s': float(statistics.median(timings)),
                'hag_build_repeats': len(timings)})
    return row


# -- O-D pairs ------------------------------------------------------------------------------

def sample_pairs(case: Case, hag: HAG, *, seed: int, n_pairs: int,
                 min_m: float, max_m: float = math.inf, min_hops: int = 3,
                 margin_cells: int = 5, band: str = '', max_candidates: int = 20_000
                 ) -> tuple[list[Pair], dict]:
    """O-D pairs by rule, no search: valid cells, one connected valid region, a distance
    band, different areas at least `min_hops` apart, pairwise distinct start and target
    areas. Draw order per candidate is i1, j1, i2, j2 from `default_rng(seed)`.

    On a connected valid region every 8-connected grid has a path, so connectivity is a
    component test instead of the protocol's full-grid Dijkstra, which a DEM cannot afford.
    """
    grid = case.grid
    valid = np.ones(grid.shape, dtype=bool) if case.valid is None else case.valid
    component, _ = ndimage.label(valid, structure=np.ones((3, 3), dtype=bool))
    rng = np.random.default_rng(seed)
    lo_i, hi_i = margin_cells, grid.n_rows - margin_cells
    lo_j, hi_j = margin_cells, grid.n_cols - margin_cells
    rejections = {k: 0 for k in ('invalid', 'distance', 'disconnected', 'same_area',
                                 'hops', 'duplicate_area')}
    pairs: list[Pair] = []
    used_start, used_target = set(), set()
    hop_cache: dict[int, np.ndarray] = {}
    drawn = 0
    from core.hag import vdist
    while len(pairs) < n_pairs and drawn < max_candidates:
        drawn += 1
        i1, j1 = int(rng.integers(lo_i, hi_i)), int(rng.integers(lo_j, hi_j))
        i2, j2 = int(rng.integers(lo_i, hi_i)), int(rng.integers(lo_j, hi_j))
        if not (valid[i1, j1] and valid[i2, j2]):
            rejections['invalid'] += 1
            continue
        dist = math.hypot(i1 - i2, j1 - j2) * grid.cell_size_m
        if not (min_m <= dist <= max_m):
            rejections['distance'] += 1
            continue
        if component[i1, j1] != component[i2, j2]:
            rejections['disconnected'] += 1
            continue
        a, b = hag.area_of((i1, j1)), hag.area_of((i2, j2))
        if a == b:
            rejections['same_area'] += 1
            continue
        if a not in hop_cache:
            hop_cache[a] = vdist(hag, a)
        if hop_cache[a][b] < min_hops:
            rejections['hops'] += 1
            continue
        if a in used_start or b in used_target:
            rejections['duplicate_area'] += 1
            continue
        used_start.add(a)
        used_target.add(b)
        pairs.append(Pair(f'{band or "od"}{len(pairs) + 1}', (i1, j1), (i2, j2), dist, band))
    return pairs, {'od_seed': seed, 'n_candidates': drawn, **{f'rejected_{k}': v
                                                             for k, v in rejections.items()}}


# -- search -----------------------------------------------------------------------------------

def pick_engine(engine: str, cost) -> str:
    if engine == 'auto':
        return 'native' if native.available() and cost_models.native_spec(cost) else 'python'
    return engine


def _run_search(case: Case, mask, start: int, target: int, cost, cap, engine: str):
    if engine == 'native':
        return native.dijkstra(case.grid, start, target, cost=cost, mask=mask,
                               max_abs_gradient_percent=cap)
    graph = build_graph(case.grid, cost=cost, mask=mask, max_abs_gradient_percent=cap)
    return dijkstra(graph, start, target)


def timed_search(case: Case, mask, start: int, target: int, cost, cap, *, engine: str,
                 repeats: int, memory: bool) -> tuple[object, list[float], Optional[int]]:
    timings, result = [], None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        result = _run_search(case, mask, start, target, cost, cap, engine)
        timings.append(time.perf_counter() - t0)
    peak = None
    if memory:
        tracemalloc.start()
        again = _run_search(case, mask, start, target, cost, cap, engine)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak = int(peak) + int(getattr(again.stats, 'native_heap_bytes', 0))
    return result, timings, peak


def _select(hag: HAG, space: str, hag_edge_cost: str, start, target, graph_cache: dict,
            cost) -> tuple[list[int], set[int], float, float]:
    """(chain, selected areas, area-graph build time, per-query selection time)."""
    spec = SEARCH_SPACES[space]
    build_s = 0.0
    t0 = time.perf_counter()
    if spec.rule == 'full':
        return [], set(range(hag.n_areas)), 0.0, 0.0
    a, b = hag.area_of(start), hag.area_of(target)
    key = (hag_edge_cost, cost_models.name_of(cost))
    if key not in graph_cache:
        tb = time.perf_counter()
        graph_cache[key] = area_graph(hag, hag_edge_cost, cost)
        build_s = time.perf_counter() - tb
        t0 = time.perf_counter()
    chain = cheapest_area_chain(graph_cache[key], a, b)
    if chain is None:
        raise ValueError('start and target are in disconnected parts of the HAG')
    chosen = dilate_areas(hag, chain, spec.k_hops) if spec.k_hops else set(chain)
    return chain, chosen, build_s, time.perf_counter() - t0


def _path_measures(case: Case, result) -> dict:
    coords = result.coords(case.grid)
    surf = case.grid.heights_flat
    steps = np.diff(coords.astype(float), axis=0)
    xyz = case.grid.path_xyz_m(coords.astype(float))
    measured = metrics_mod.measure(xyz)
    return {'path_cost': float(result.cost),
            'objective_sum_abs_dh_m': float(np.abs(np.diff(surf[result.path])).sum()),
            'path_length_m': float(np.hypot(steps[:, 0], steps[:, 1]).sum()
                                   * case.grid.cell_size_m),
            'path_i_max_percent': measured.i_max_percent,
            'path_r_min_m': measured.r_min_m,
            'n_path_cells': int(len(result.path))}, xyz


def search_rows(case: Case, hag: HAG, pair: Pair, *, spaces: Sequence[tuple[str, str]],
                grid_edge: str, engine: str, repeats: int, memory: bool = True,
                i_max_percent: Optional[float] = None,
                graph_cache: Optional[dict] = None) -> list[dict]:
    """One row per (search space, HAG edge cost), full grid first; quality vs the full grid."""
    cost, cap = grid_edge_cost(grid_edge, i_max_percent)
    engine = pick_engine(engine, cost)
    graph_cache = {} if graph_cache is None else graph_cache
    start, target = case.grid.index(pair.start), case.grid.index(pair.target)
    rows, baseline, baseline_xy = [], None, None
    ordered = sorted(spaces, key=lambda s: s[0] != 'full_grid')
    if not ordered or ordered[0][0] != 'full_grid':
        ordered = [('full_grid', '')] + list(ordered)
    for space, hag_edge in ordered:
        row = {'od_id': pair.od_id, 'od_band': pair.band, 'straight_line_m': pair.straight_line_m,
               'start_i': pair.start[0], 'start_j': pair.start[1],
               'target_i': pair.target[0], 'target_j': pair.target[1],
               'search_space': space, 'hag_edge_cost': hag_edge if space != 'full_grid' else '',
               'grid_edge_cost': grid_edge, 'cost_model': cost_models.name_of(cost),
               'engine': engine, 'timing_repeats': max(1, repeats)}
        chain, chosen, graph_s, select_s = _select(hag, space, hag_edge, pair.start,
                                                   pair.target, graph_cache, cost)
        if space == 'full_grid':
            mask = case.valid
        else:
            mask = hag.mask(chosen)
            mask.reshape(-1)[[start, target]] = True
        n_selected = case.n_valid if mask is None else int(mask.sum())
        row.update(describe_selection(hag, chain, chosen) if space != 'full_grid' else
                   {'cta_hops': 0, 'n_areas_selected': hag.n_areas,
                    'n_cells_selected': n_selected, 'cta_narrowest_width_m': math.nan})
        row.update({'n_cells_selected': n_selected,
                    'search_space_percent_of_full': 100.0 * n_selected / case.n_valid,
                    'search_space_reduction_percent': 100.0 * (1 - n_selected / case.n_valid),
                    'area_graph_time_s': graph_s, 'selection_time_s': select_s})
        result, timings, peak = timed_search(case, mask, start, target, cost, cap,
                                             engine=engine, repeats=repeats, memory=memory)
        timings.sort()
        row.update({'search_time_s_median': float(statistics.median(timings)),
                    'search_time_s_min': timings[0], 'search_time_s_max': timings[-1],
                    'peak_memory_bytes': peak, 'path_found': bool(result.reached),
                    **{f'search_{k}': v for k, v in result.stats.as_dict().items()}})
        if result.reached:
            measures, xyz = _path_measures(case, result)
            row.update(measures)
            if space == 'full_grid':
                baseline, baseline_xy = row, xyz
            elif baseline is not None and baseline.get('path_found'):
                row.update(_versus(row, baseline, xyz, baseline_xy))
        rows.append(row)
    return rows


def _versus(row: dict, base: dict, xyz: np.ndarray, base_xyz: np.ndarray) -> dict:
    t_full = base['search_time_s_median']
    t_query = row['search_time_s_median'] + row['selection_time_s']
    saving = t_full - t_query
    return {
        'objective_ratio_to_full': (row['objective_sum_abs_dh_m'] / base['objective_sum_abs_dh_m']
                                    if base['objective_sum_abs_dh_m'] else math.nan),
        'path_cost_ratio_to_full': (row['path_cost'] / base['path_cost']
                                    if base['path_cost'] else math.nan),
        'length_ratio_to_full': row['path_length_m'] / base['path_length_m'],
        'hausdorff_to_full_m': metrics_mod.hausdorff_m(xyz[:, :2], base_xyz[:, :2]),
        'nodes_expanded_ratio_to_full': (row['search_nodes_expanded']
                                         / base['search_nodes_expanded']),
        'search_time_ratio_to_full': row['search_time_s_median'] / t_full if t_full else math.nan,
        'query_time_saving_s': saving,
        'memory_ratio_to_full': ((row['peak_memory_bytes'] / base['peak_memory_bytes'])
                                 if row.get('peak_memory_bytes') and base.get('peak_memory_bytes')
                                 else math.nan),
    }


def break_even_queries(hag_build_s: float, area_graph_s: float, saving_s: float) -> float:
    """Queries on one HAG before building it pays off: build / saving per query."""
    if not saving_s or saving_s <= 0:
        return math.inf
    return (hag_build_s + area_graph_s) / saving_s
