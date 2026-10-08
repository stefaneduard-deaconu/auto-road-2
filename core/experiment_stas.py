"""Every STAS 863-85 parameter, at every point, across terrains, costs and smoothing arms.

    python -m core.experiment_stas --out results/stas

This is the runner for the article's three deliverables, measured together because
they share one pipeline:

  (a) HAG as a search-space reduction   the `full_grid` and `hag_cta` search spaces, with
                                        the nodes, the time and what the reduction costs
  (b) enhancements to Algorithm 1       the arms of `core.algorithm_1.ARMS`, plus `none`
  (c) STAS at every point               `core.checks.check_path_stations` on the axis each
                                        arm produced, against `STAS863_V40` / `STAS863_V25`

THE COST SWEEP IS MEMORYLESS, DELIBERATELY
------------------------------------------
An edge cost in this framework is `(dh, length_m) -> weight`. It cannot see the heading,
the last 20 m of path, or the radius a later smoothing pass would produce, so a
curvature-aware or look-ahead heuristic is NOT a cost model - it needs a turn-expanded
(edge-state) graph, which is a different search layer. The sweep below therefore covers
memoryless models only, and the article records the rest as a stated limitation. Every arm
names itself through `costs.name_of`, so a factory-built model is identified by its
parameters and not by a label typed into a table.

WHAT ONE ROW IS
---------------
One *(terrain, O-D, road class, search space, grid edge cost, Algorithm 1 arm)*. The search
is run once per *(terrain, O-D, class, search space, grid edge cost)* and its path is handed
to every Algorithm 1 arm, so the arms are compared on identical input. `hag_edge_cost` is
always `shared_border`, the HAG edge cost of the study (X8 does not sweep it).

The per-station series are written separately, to `series/`, because they are one row per
station and would otherwise swamp the CSV. By default only the reference scenario's series
are written; `--series all` writes them for every row and costs a lot of files.
"""
from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from core import costs as cost_models
from core import naming
from core.algorithm_1 import ARM_ORDER, ARMS, before_after_enhanced
from core.checks import STATION_CHECK_NAMES, check_path_stations
from core.experiment_matrix import (HEIGHT_DELTA_M, QUICK_TERRAINS, TERRAINS, TerrainCase,
                                    _assert_still_ours, _lock)
from core.hag import build_hag, select_areas
from core.metrics import DEFAULT_CHORD_M, DEFAULT_RESAMPLE_STEP_M, measure
from core.od_sampler import ODPair, sample_od_pairs
from core.provenance import provenance, write_jsonl
from core.results_io import read_jsonl, write_csv
from core.search import build_graph, dijkstra
from core.terrain import generate_terrain
from data.configs.road_classes import STAS_MATRIX_CLASSES, get

#: the search spaces. Deliverable (a) is `hag_cta` measured against `full_grid`. Values are
#: `core.naming.SEARCH_SPACES` names, not the old combined `search_arm` labels.
SEARCH_ARMS: tuple[tuple[str, str, int], ...] = (
    ('full_grid', 'full', 0),
    ('hag_cta', 'yellow', 0),
)

#: X8 never sweeps HAG-edge pricing: every selection uses the study's edge cost.
HAG_EDGE_COST: str = 'shared_border'

#: the memoryless grid-edge-cost sweep, in `core.naming`'s vocabulary. `climb_gradient_cut` is
#: not a cost model but a graph restriction, so it carries its own flag; every other arm is a
#: cost passed to `build_graph`.
COST_ARMS: tuple[str, ...] = (
    'climb',                    # the objective of Eq. 1 in both source articles
    'climb_tiebreak',           # the same, with a length tie-break: the matrix's default
    'length_3d',                # length along the terrain
    'climb_plus_length',        # equal weight on climb and on length
    'climb_gradient_penalty',   # steep edges cost more
    'climb_gradient_cut',       # steep edges are removed from the graph entirely
)

#: the arms of Algorithm 1, plus the unsmoothed search output.
SMOOTHING_ARMS: tuple[str, ...] = ARM_ORDER

#: Algorithm 1 is repeated this many times per arm (the draft's "iteration 1 versus 2")
ITERATIONS: tuple[int, ...] = (1, 2)

COLUMNS: tuple[str, ...] = (
    'scenario_id', 'terrain_id', 'terrain_seed', 'od_id', 'road_class',
    'road_class_status', 'road_class_standing', 'design_speed_kmh',
    'search_space', 'hag_edge_cost', 'grid_edge_cost', 'cost_model', 'smoothing_arm',
    'algorithm_1_iterations', 'connectivity',
    'timestamp_utc', 'git_sha', 'git_dirty', 'git_branch', 'python',
    'python_free_threaded', 'platform', 'numpy', 'scipy',
    # deliverable (a)
    'n_cells_selected', 'n_cells_total', 'search_space_percent_of_full',
    'n_graph_edges', 'search_nodes_expanded', 'search_wall_time_s', 'search_peak_bytes',
    'path_found', 'infeasible_reason', 'height_cost',
    # deliverable (b)
    'rough_length_m', 'rough_r_min_m', 'rough_i_max_percent',
    'length_m', 'r_min_m', 'i_max_percent', 'sinuosity',
    'deviation_hausdorff_m', 'deviation_mean_m',
    'selection_radius_m', 'deviation_budget_m', 'n_adjustments', 'hit_adjustment_cap',
    'gradient_recovered', 'radius_compliant', 'z_from_grade_line',
    'n_short_tangents_before', 'n_short_tangents_after', 's_value',
    # deliverable (c): one group per STAS parameter
    *[f'stas_{name}_{field}'
      for name in STATION_CHECK_NAMES
      for field in ('status', 'limit', 'worst', 'worst_station_m',
                    'n_applicable', 'n_violations', 'violation_length_m')],
    'stas_feasible', 'stas_verdict', 'stas_total_violation_length_m',
    'step_m', 'chord_m', 'straight_radius_m', 'series_file',
)


def resolve_cost(grid_edge_cost: str, road_class):
    """The cost model and the gradient cap one grid-edge-cost arm searches with.

    Returns `(cost, max_abs_gradient_percent)`. Only `climb_gradient_cut` sets the cap; it is
    a graph restriction, not a cost, so a found path holds `i_max` at cell resolution by
    construction, or no path is found at all.
    """
    if grid_edge_cost not in COST_ARMS:
        raise ValueError(f'unknown grid edge cost {grid_edge_cost!r}; known: {COST_ARMS}')
    return naming.grid_edge_cost(grid_edge_cost, i_max_percent=road_class.i_max_percent)


def _station_columns(report) -> dict:
    """The per-parameter result group of deliverable (c)."""
    out: dict = {}
    for name, check in report.by_name().items():
        out[f'stas_{name}_status'] = check.status
        out[f'stas_{name}_limit'] = check.limit
        out[f'stas_{name}_worst'] = check.worst_value
        out[f'stas_{name}_worst_station_m'] = check.worst_station_m
        out[f'stas_{name}_n_applicable'] = check.n_applicable
        out[f'stas_{name}_n_violations'] = check.n_violations
        out[f'stas_{name}_violation_length_m'] = check.violation_length_m
    out['stas_feasible'] = report.feasible
    out['stas_verdict'] = report.verdict
    out['stas_total_violation_length_m'] = report.total_violation_length_m
    return out


def write_series(path: Path, report) -> Path:
    """The four STAS parameters, one row per station, so a figure needs no search."""
    geometry = report.geometry
    by_name = report.by_name()
    limits = {name: by_name[name].limit for name in STATION_CHECK_NAMES}
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ('chainage_m,x_m,y_m,z_m,is_curve,r_horizontal_m,gradient_percent,'
              'tangent_length_m,r_vertical_m,vertical_kind,grade_percent,'
              'limit_R_H,limit_i,limit_L_alignment,limit_R_V')
    grade = (geometry.grade_percent if geometry.grade_percent is not None
             else np.full(len(geometry.chainage_m), np.nan))
    lines = [header]
    for k, s in enumerate(geometry.chainage_m):
        kind = geometry.vertical_kind[k]
        limit_rv = (limits['R_V_concave'] if kind == 'concave' else
                    limits['R_V_convex'] if kind == 'convex' else None)
        lines.append(','.join(_csv(v) for v in (
            s, geometry.stations_xyz_m[k, 0], geometry.stations_xyz_m[k, 1],
            geometry.stations_xyz_m[k, 2], bool(geometry.is_curve[k]),
            geometry.r_horizontal_m[k], geometry.gradient_percent[k],
            geometry.tangent_length_m[k], geometry.r_vertical_m[k], kind or '', grade[k],
            limits['R_H'], limits['i'], limits['L_alignment'], limit_rv)))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def _csv(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ''
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return 'inf' if np.isinf(value) else f'{value:.6g}'
    return str(value)


def run_stas_scenario(grid, hag, od: ODPair, class_name: str, *,
                      search_arms: Sequence[tuple[str, str, int]] = SEARCH_ARMS,
                      cost_arms: Sequence[str] = COST_ARMS,
                      smoothing_arms: Sequence[str] = SMOOTHING_ARMS,
                      series_dir: Optional[Path] = None,
                      series: str = 'none',
                      metric_step_m: float = DEFAULT_RESAMPLE_STEP_M,
                      metric_chord_m: float = DEFAULT_CHORD_M,
                      iterations: Sequence[int] = ITERATIONS) -> list[dict]:
    """One (terrain, O-D, class): every search space x grid edge cost x Algorithm 1 arm."""
    road_class = get(class_name).with_cell_size(grid.cell_size_m)
    n_cells_total = int(grid.surf.size)
    rows: list[dict] = []

    for search_space, rule, k_hops in search_arms:
        selected = select_areas(hag, od.start, od.target, rule=rule, k_hops=k_hops,
                                edge_cost=HAG_EDGE_COST)
        mask = hag.mask(selected).copy()
        mask[od.start] = True
        mask[od.target] = True
        n_selected = int(mask.sum())

        for grid_edge_cost in cost_arms:
            cost, cap = resolve_cost(grid_edge_cost, road_class)
            base = {
                'terrain_id': None, 'od_id': od.od_id, 'road_class': class_name,
                'road_class_status': road_class.status,
                'road_class_standing': road_class.standing,
                'design_speed_kmh': road_class.design_speed_kmh,
                'search_space': search_space, 'hag_edge_cost': HAG_EDGE_COST,
                'grid_edge_cost': grid_edge_cost, 'connectivity': 8,
                'n_cells_selected': n_selected, 'n_cells_total': n_cells_total,
                'search_space_percent_of_full': 100.0 * n_selected / n_cells_total,
                'step_m': metric_step_m, 'chord_m': metric_chord_m,
            }
            tracemalloc.start()
            t0 = time.perf_counter()
            graph = build_graph(grid, cost=cost, mask=mask, max_abs_gradient_percent=cap)
            result = dijkstra(graph, grid.index(od.start), grid.index(od.target))
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            base.update({
                'cost_model': graph.cost_name, 'n_graph_edges': graph.n_edges,
                'search_nodes_expanded': result.stats.nodes_expanded,
                'search_wall_time_s': elapsed, 'search_peak_bytes': int(peak),
                'path_found': bool(result.reached),
            })
            if not result.reached:
                base['infeasible_reason'] = (
                    'no path under this cost and mask; a gradient cut can disconnect the '
                    'selection, which a penalty cannot')
                base['smoothing_arm'] = 'none'
                base['scenario_id'] = (f'{od.od_id}__{class_name}__{search_space}__'
                                      f'{grid_edge_cost}__none')
                rows.append(base)
                continue

            path_xy = result.coords(grid).astype(float)
            rough_xyz = grid.path_xyz_m(path_xy)
            rough = measure(rough_xyz, step_m=metric_step_m, chord_m=metric_chord_m)
            base.update({
                # Equation 1: the sum of |height difference| along the axis
                'height_cost': rough.elevation_ascent_m + rough.elevation_descent_m,
                'rough_length_m': rough.length_m, 'rough_r_min_m': rough.r_min_m,
                'rough_i_max_percent': rough.i_max_percent,
            })

            for smoothing_arm, n_iter in [(arm, it) for arm in smoothing_arms
                                          for it in ((0,) if arm == 'none' else iterations)]:
                row = dict(base)
                row['smoothing_arm'] = smoothing_arm
                row['algorithm_1_iterations'] = n_iter
                row['scenario_id'] = (f'{od.od_id}__{class_name}__{search_space}__'
                                      f'{grid_edge_cost}__{smoothing_arm}'
                                      + (f'__it{n_iter}' if n_iter else ''))
                if smoothing_arm == 'none':
                    axis = rough_xyz
                    row.update({'length_m': rough.length_m, 'r_min_m': rough.r_min_m,
                                'i_max_percent': rough.i_max_percent,
                                'sinuosity': rough.sinuosity,
                                'deviation_hausdorff_m': 0.0, 'deviation_mean_m': 0.0})
                else:
                    report = before_after_enhanced(
                        grid, path_xy, road_class, ARMS[smoothing_arm], iterations=n_iter,
                        metric_step_m=metric_step_m, metric_chord_m=metric_chord_m)
                    after, enh = report['after'], report['enhancements']
                    row.update({
                        'length_m': after.get('length_m'), 'r_min_m': after.get('r_min_m'),
                        'i_max_percent': after.get('i_max_percent'),
                        'sinuosity': after.get('sinuosity'),
                        'deviation_hausdorff_m': report['deviation']['hausdorff_m'],
                        'deviation_mean_m': report['deviation']['mean_deviation_m'],
                        'selection_radius_m': enh['selection_radius_m'],
                        'deviation_budget_m': enh['budget_m'],
                        'n_adjustments': enh['n_adjustments'],
                        'hit_adjustment_cap': enh['hit_adjustment_cap'],
                        'gradient_recovered': enh['gradient_recovered'],
                        'radius_compliant': enh['radius_compliant'],
                        'z_from_grade_line': enh['z_from_grade_line'],
                        'n_short_tangents_before': enh['n_short_tangents_before'],
                        'n_short_tangents_after': enh['n_short_tangents_after'],
                        's_value': enh['s_search']['s_value'],
                    })
                    axis = _axis_of(grid, path_xy, road_class, smoothing_arm, n_iter)

                station_report = check_path_stations(axis, road_class,
                                                     step_m=metric_step_m,
                                                     chord_m=metric_chord_m)
                row.update(_station_columns(station_report))
                row['straight_radius_m'] = station_report.geometry.straight_radius_m
                if series_dir is not None and _wants_series(series, row):
                    name = f'{row["scenario_id"]}.csv'
                    write_series(series_dir / name, station_report)
                    row['series_file'] = name
                rows.append(row)
    return rows


def _axis_of(grid, path_xy, road_class, smoothing_arm: str, iterations: int = 1) -> np.ndarray:
    from core.algorithm_1 import algorithm_1_enhanced
    return algorithm_1_enhanced(grid, path_xy, road_class, ARMS[smoothing_arm],
                                iterations=iterations).smooth_xyz_m


def _wants_series(series: str, row: dict) -> bool:
    if series == 'all':
        return True
    if series == 'reference':
        return row['search_space'] == 'hag_cta' and row['grid_edge_cost'] == 'climb_tiebreak'
    return False


def run_matrix(out_dir: Path, *, terrains: Sequence[TerrainCase] = TERRAINS,
               class_names: Sequence[str] = STAS_MATRIX_CLASSES,
               cost_arms: Sequence[str] = COST_ARMS,
               smoothing_arms: Sequence[str] = SMOOTHING_ARMS,
               n_pairs: int = 3, series: str = 'reference',
               force: bool = False, step_m: float = DEFAULT_RESAMPLE_STEP_M,
               chord_m: float = DEFAULT_CHORD_M) -> dict:
    with _lock(out_dir) as lock:
        return _run_locked(out_dir, lock, terrains, class_names, cost_arms,
                           smoothing_arms, n_pairs, series, force, step_m, chord_m)


def _run_locked(out_dir: Path, lock: Path, terrains, class_names, cost_arms,
                smoothing_arms, n_pairs, series, force: bool,
                step_m: float = DEFAULT_RESAMPLE_STEP_M,
                chord_m: float = DEFAULT_CHORD_M) -> dict:
    jsonl = out_dir / 'stas.jsonl'
    if force and jsonl.exists():
        jsonl.unlink()
    done = {row.get('scenario_id') for row in read_jsonl(jsonl)}
    series_dir = out_dir / 'series'

    n_new = 0
    for case in terrains:
        grid = generate_terrain(case.spec)
        hag = build_hag(grid, height_delta=HEIGHT_DELTA_M)
        sample = sample_od_pairs(grid, hag, terrain_id=case.terrain_id,
                                 terrain_seed=case.spec.seed, n_pairs=n_pairs)
        for od in sample.pairs:
            for class_name in class_names:
                for row in run_stas_scenario(grid, hag, od, class_name,
                                             cost_arms=cost_arms,
                                             smoothing_arms=smoothing_arms,
                                             series_dir=series_dir, series=series,
                                             metric_step_m=step_m,
                                             metric_chord_m=chord_m):
                    row['scenario_id'] = f'{case.terrain_id}__{row["scenario_id"]}'
                    if row['scenario_id'] in done:
                        continue
                    row.update({'terrain_id': case.terrain_id,
                                'terrain_seed': case.spec.seed, **provenance()})
                    _assert_still_ours(lock)
                    write_jsonl(jsonl, row)
                    n_new += 1

    logged = read_jsonl(jsonl)
    csv_path = write_csv(out_dir / 'stas.csv', logged, COLUMNS)
    return {'jsonl': jsonl, 'csv': csv_path, 'n_new_rows': n_new,
            'n_total_rows': len(logged)}


#: the real-DEM mode (before-release task PR2): windows of the Idrija swath at the design cell
#: size, three O-D pairs each, drawn by the window protocol of the robustness study (X7).
DEM_WINDOW_SIZES_M: tuple[int, ...] = (960, 1920)
DEM_WINDOWS_PER_SIZE = 3
DEM_CELL_M = 10
DEM_HEIGHT_DELTA_M = 3.0
DEM_COST_ARMS: tuple[str, ...] = ('climb_tiebreak', 'climb_plus_length', 'climb_gradient_cut')


def run_dem_matrix(out_dir: Path, tif: Path, *, sizes: Sequence[int] = DEM_WINDOW_SIZES_M,
                   per_size: int = DEM_WINDOWS_PER_SIZE,
                   class_names: Sequence[str] = STAS_MATRIX_CLASSES,
                   cost_arms: Sequence[str] = DEM_COST_ARMS,
                   smoothing_arms: Sequence[str] = SMOOTHING_ARMS,
                   series: str = 'reference', force: bool = False,
                   step_m: float = DEFAULT_RESAMPLE_STEP_M,
                   chord_m: float = DEFAULT_CHORD_M) -> dict:
    """The STAS station check on real DEM terrain: Idrija windows at `DEM_CELL_M` cells."""
    from core import programme
    from core.hag import build_hag as build_dem_hag
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / 'stas.jsonl'
    if force and jsonl.exists():
        jsonl.unlink()
    done = {row.get('scenario_id') for row in read_jsonl(jsonl)}
    options = programme.RunOptions(tif=tif)
    n_new = 0
    for size in sizes:
        for k in range(per_size):
            case, pairs = programme._window_case_pairs(
                options, {'size': size, 'k': k, 'cell': DEM_CELL_M,
                          'delta': DEM_HEIGHT_DELTA_M, 'per_size': per_size})
            hag = build_dem_hag(case.grid, DEM_HEIGHT_DELTA_M, valid_mask=case.valid,
                                native=True)
            terrain_id = f'idrija_win{size}_k{k}'
            for pair in pairs:
                for class_name in class_names:
                    for row in run_stas_scenario(case.grid, hag, pair, class_name,
                                                 cost_arms=cost_arms,
                                                 smoothing_arms=smoothing_arms,
                                                 series_dir=out_dir / 'series', series=series,
                                                 metric_step_m=step_m, metric_chord_m=chord_m):
                        row['scenario_id'] = f'{terrain_id}__{row["scenario_id"]}'
                        if row['scenario_id'] in done:
                            continue
                        row.update({'terrain_id': terrain_id, 'terrain_seed': None,
                                    **provenance()})
                        write_jsonl(jsonl, row)
                        n_new += 1
    logged = read_jsonl(jsonl)
    csv_path = write_csv(out_dir / 'stas.csv', logged, COLUMNS)
    return {'jsonl': jsonl, 'csv': csv_path, 'n_new_rows': n_new, 'n_total_rows': len(logged)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='results/stas', type=Path)
    parser.add_argument('--quick', action='store_true',
                        help='a small subset on 40x40 terrains, for checking the plumbing')
    parser.add_argument('--series', default='reference',
                        choices=('none', 'reference', 'all'),
                        help='which scenarios get a per-station series file')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--step-m', type=float, default=DEFAULT_RESAMPLE_STEP_M,
                        help='station spacing along the axis, in metres (agreed item A8: 2)')
    parser.add_argument('--chord-m', type=float, default=DEFAULT_CHORD_M,
                        help='chord of the three-point radius, in metres')
    parser.add_argument('--dem', type=Path, default=None,
                        help='run on Idrija DEM windows from this GeoTIFF (writes to --out, '
                             'default results/stas_dem)')
    parser.add_argument('--summary-only', action='store_true',
                        help='rebuild stas.csv from the existing log only')
    args = parser.parse_args(argv)

    out = args.out
    if args.quick and args.out == Path('results/stas'):
        out = args.out / 'quick'       # a quick run must never poison the real log

    if args.dem is not None:
        out = Path('results/stas_dem') if args.out == Path('results/stas') else args.out
        result = run_dem_matrix(out, args.dem, series=args.series, force=args.force,
                                step_m=args.step_m, chord_m=args.chord_m,
                                per_size=1 if args.quick else DEM_WINDOWS_PER_SIZE,
                                sizes=DEM_WINDOW_SIZES_M[:1] if args.quick else
                                DEM_WINDOW_SIZES_M)
        for name, value in result.items():
            print(f'{name:>14}: {value}')
        return 0

    if args.summary_only:
        logged = read_jsonl(out / 'stas.jsonl')
        if not logged:
            raise SystemExit(f'no rows in {out / "stas.jsonl"}')
        csv_path = write_csv(out / 'stas.csv', logged, COLUMNS)
        print(f'csv: {csv_path} ({len(logged)} rows)')
        return 0

    result = run_matrix(
        out,
        terrains=QUICK_TERRAINS if args.quick else TERRAINS,
        cost_arms=COST_ARMS[:3] if args.quick else COST_ARMS,
        n_pairs=1 if args.quick else 3,
        series=args.series, force=args.force, step_m=args.step_m, chord_m=args.chord_m)
    for name, value in result.items():
        print(f'{name:>14}: {value}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
