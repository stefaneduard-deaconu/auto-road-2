"""Does giving the search `i_max` change what it finds, and does smoothing then undo it?

    python -m core.experiment_gradient --out results/gradient

Two open items of the study protocol measured together, on the SAME scenarios as T7's
committed matrix (same three terrains, same protocol O-D pairs, same three road classes --
imported from `core.experiment_matrix` rather than restated, so the two studies are directly
comparable row for row):

**B21 -- the search never saw `i_max`.** `core.experiment_step1`'s arms all minimise
`sum |dh|` (or its tie-broken form) and are blind to the road class; on a real crop the
unconstrained search returned an alignment at 20% where a 7%-holding alignment provably
existed (`test.py`). Two arms here give the search the class instead of checking it
afterwards, both already implemented in `core/`, neither wired into the T7 matrix because
doing so would make its search depend on the road class, breaking the "computed once per
O-D, reused across classes" design that keeps T7 affordable:

    hag_yellow          the matrix's own arm: `height_tiebreak`, blind to `i_max`
    hag_yellow_gradcut  `build_graph(max_abs_gradient_percent=class.i_max_percent)`: every
                        edge steeper than the limit is REMOVED, so a found path holds
                        `i_max` at cell resolution by construction, or none is found
    hag_yellow_gradpen  `costs.gradient_penalised(i_max_percent=class.i_max_percent)`: every
                        edge stays, steep ones cost more, so a path always exists

**B23 -- smoothing can destroy gradient feasibility.** `core.algorithm_1.before_after`
already reports when smoothing makes `i_max` worse; `before_after_guarded` adds the
decision that observation implies -- when smoothing takes a COMPLIANT `i_max` NON-compliant,
the rough axis is retained instead, and `checks` below are run on the RETAINED axis, not
blindly on the smoothed one. That is the point of running B21 and B23 together: a search
that already holds `i_max` (gradcut, gradpen) gives the guard less regression to catch, which
is itself the measurement -- does constraining the search reduce how often smoothing has to
be rejected?

This is a supplementary study, not an extension of T7: its own output directory, its own
(small, unaudited) column list, and it never touches `results/matrix.csv`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from core.algorithm_1 import before_after_guarded
from core.checks import check_path
from core.costs import DEFAULT_COST, gradient_penalised
from core.experiment_matrix import (HEIGHT_DELTA_M, QUICK_TERRAINS, TERRAINS, TerrainCase,
                                    _assert_still_ours, _lock)
from core.hag import build_hag, select_areas
from core.metrics import DEFAULT_CHORD_M, DEFAULT_RESAMPLE_STEP_M, measure
from core.od_sampler import ODPair, od_pairs_rows, sample_od_pairs
from core.provenance import provenance, write_jsonl
from core.results_io import read_jsonl, write_csv
from core.search import build_graph, dijkstra, path_height_cost, path_length_m
from core.terrain import generate_terrain
from data.configs.road_classes import PROPOSED_MATRIX_CLASSES, get

GRADIENT_ARMS: tuple[str, ...] = ('hag_yellow', 'hag_yellow_gradcut', 'hag_yellow_gradpen')

COLUMNS: tuple[str, ...] = (
    'scenario_id', 'terrain_id', 'terrain_seed', 'od_id', 'road_class', 'road_class_status',
    'arm', 'cost_model', 'connectivity',
    'timestamp_utc', 'git_sha', 'git_dirty', 'git_branch', 'python', 'python_free_threaded',
    'platform', 'numpy', 'scipy',
    'n_cells_selected', 'n_graph_edges_kept', 'n_graph_edges_total', 'edge_keep_fraction',
    'search_nodes_expanded', 'wall_time_s', 'path_found', 'infeasible_reason',
    'rough_length_m', 'rough_r_min_m', 'rough_i_max_percent',
    'after_length_m', 'after_r_min_m', 'after_i_max_percent',
    'smoothing_rejected', 'retained_source', 'retained_i_max_percent', 'retained_r_min_m',
    'check_r_min_passed', 'check_i_max_passed', 'feasible', 'verdict',
)


def _arm_graph(grid, mask, road_class, arm: str):
    """The graph one gradient arm searches, all sharing the SAME yellow mask (B21)."""
    if arm == 'hag_yellow':
        return build_graph(grid, cost=DEFAULT_COST, mask=mask)
    if arm == 'hag_yellow_gradcut':
        return build_graph(grid, cost=DEFAULT_COST, mask=mask,
                           max_abs_gradient_percent=road_class.i_max_percent)
    if arm == 'hag_yellow_gradpen':
        model = gradient_penalised(DEFAULT_COST, i_max_percent=road_class.i_max_percent)
        return build_graph(grid, cost=model, mask=mask)
    raise ValueError(f"unknown gradient arm {arm!r}; known: {GRADIENT_ARMS}")


def run_gradient_scenario(grid, hag, od: ODPair, class_name: str, *,
                          metric_step_m: float = DEFAULT_RESAMPLE_STEP_M,
                          metric_chord_m: float = DEFAULT_CHORD_M) -> list[dict]:
    """One (terrain, O-D, class): the three gradient arms, each searched, smoothed, checked."""
    road_class = get(class_name).with_cell_size(grid.cell_size_m)
    yellow = select_areas(hag, od.start, od.target, rule='yellow')
    mask = hag.mask(yellow).copy()
    mask[od.start] = True
    mask[od.target] = True

    full_edges = build_graph(grid, cost=DEFAULT_COST, mask=mask).n_edges
    rows: list[dict] = []
    for arm in GRADIENT_ARMS:
        row = {
            'scenario_id': f'{od.od_id}__{class_name}__{arm}',
            'road_class': class_name, 'road_class_status': road_class.status,
            'arm': arm, 'connectivity': 8,
            'n_cells_selected': int(mask.sum()),
        }
        t0 = time.perf_counter()
        graph = _arm_graph(grid, mask, road_class, arm)
        row['cost_model'] = graph.cost_name
        row['n_graph_edges_kept'] = graph.n_edges
        row['n_graph_edges_total'] = full_edges
        row['edge_keep_fraction'] = graph.n_edges / full_edges if full_edges else 0.0
        result = dijkstra(graph, grid.index(od.start), grid.index(od.target))
        row['wall_time_s'] = time.perf_counter() - t0
        row['search_nodes_expanded'] = result.stats.nodes_expanded
        row['path_found'] = bool(result.reached)
        if not result.reached:
            row['infeasible_reason'] = ('no path under this cost/mask; the gradient cut can '
                                        'disconnect the yellow selection, which gradpen '
                                        'cannot (every edge stays, only costs more)')
            rows.append(row)
            continue

        path_xy = result.coords(grid).astype(float)
        rough_xyz = grid.path_xyz_m(path_xy)
        rough = measure(rough_xyz, step_m=metric_step_m, chord_m=metric_chord_m)
        row.update({'rough_length_m': rough.length_m, 'rough_r_min_m': rough.r_min_m,
                   'rough_i_max_percent': rough.i_max_percent})

        report = before_after_guarded(grid, path_xy, road_class, iterations=1,
                                      metric_step_m=metric_step_m,
                                      metric_chord_m=metric_chord_m)
        row.update({'after_length_m': report['after'].get('length_m'),
                   'after_r_min_m': report['after'].get('r_min_m'),
                   'after_i_max_percent': report['after'].get('i_max_percent'),
                   'smoothing_rejected': report['smoothing_rejected'],
                   'retained_source': report['retained_source'],
                   'retained_i_max_percent': report['retained'].get('i_max_percent'),
                   'retained_r_min_m': report['retained'].get('r_min_m')})

        retained_xyz = rough_xyz if report['retained_source'] == 'before' else \
            grid.path_xyz_m(_smoothed_xy(grid, path_xy, road_class))
        checks = check_path(retained_xyz, road_class, step_m=metric_step_m,
                            chord_m=metric_chord_m)
        by_name = {c.name: c for c in checks.checks}
        row.update({
            'check_r_min_passed': by_name['R_min'].passed,
            'check_i_max_passed': by_name.get('i_max').passed if 'i_max' in by_name else None,
            'feasible': checks.feasible, 'verdict': checks.verdict,
        })
        rows.append(row)
    return rows


def _smoothed_xy(grid, path_xy, road_class):
    from core.algorithm_1 import algorithm_1
    return algorithm_1(path_xy, road_class, iterations=1).smooth_xy


def run_matrix(out_dir: Path, *, terrains: Sequence[TerrainCase] = TERRAINS,
               class_names: Sequence[str] = PROPOSED_MATRIX_CLASSES,
               force: bool = False) -> dict:
    with _lock(out_dir) as lock:
        return _run_locked(out_dir, lock, terrains, class_names, force)


def _run_locked(out_dir: Path, lock: Path, terrains, class_names, force: bool) -> dict:
    jsonl = out_dir / 'gradient.jsonl'
    if force and jsonl.exists():
        jsonl.unlink()
    done = {row.get('scenario_id') for row in read_jsonl(jsonl)}

    all_rows: list[dict] = []
    for case in terrains:
        grid = generate_terrain(case.spec)
        hag = build_hag(grid, height_delta=HEIGHT_DELTA_M)
        sample = sample_od_pairs(grid, hag, terrain_id=case.terrain_id,
                                 terrain_seed=case.spec.seed, n_pairs=3)
        for od in sample.pairs:
            for class_name in class_names:
                for row in run_gradient_scenario(grid, hag, od, class_name):
                    row['scenario_id'] = f'{case.terrain_id}__{row["scenario_id"]}'
                    if row['scenario_id'] in done:
                        continue
                    row.update({'terrain_id': case.terrain_id, 'terrain_seed': case.spec.seed,
                               'od_id': od.od_id, **provenance()})
                    _assert_still_ours(lock)
                    write_jsonl(jsonl, row)
                    all_rows.append(row)

    logged = read_jsonl(jsonl)
    csv_path = write_csv(out_dir / 'gradient.csv', logged, COLUMNS)
    return {'jsonl': jsonl, 'csv': csv_path, 'n_new_rows': len(all_rows),
           'n_total_rows': len(logged)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='results/gradient', type=Path)
    parser.add_argument('--quick', action='store_true',
                        help='a small subset on 40x40 terrains, for checking the plumbing')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--summary-only', action='store_true',
                        help='rebuild gradient.csv from the existing log only')
    args = parser.parse_args(argv)

    out = args.out
    if args.quick and args.out == Path('results/gradient'):
        out = args.out / 'quick'       # a quick run must never poison the real log

    if args.summary_only:
        logged = read_jsonl(out / 'gradient.jsonl')
        if not logged:
            raise SystemExit(f'no rows in {out / "gradient.jsonl"}')
        csv_path = write_csv(out / 'gradient.csv', logged, COLUMNS)
        print(f'csv: {csv_path} ({len(logged)} rows)')
        return 0

    terrains = QUICK_TERRAINS if args.quick else TERRAINS
    class_names = PROPOSED_MATRIX_CLASSES[:2] if args.quick else PROPOSED_MATRIX_CLASSES
    result = run_matrix(out, terrains=terrains, class_names=class_names, force=args.force)
    for name, value in result.items():
        print(f'{name:>12}: {value}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
