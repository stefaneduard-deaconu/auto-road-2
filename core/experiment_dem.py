"""Real-DEM experiments on the Idrija swath: census, swath, tiles, ladder.

The raster is the Idrija Fault LiDAR DEM from OpenTopography, DOI 10.5069/G9QC01Q2
(`core.dem.IDRIJA_DEM_DOI`).

    python -m core.experiment_dem --mode census --out results/dem
    python -m core.experiment_dem --mode swath  --out results/dem --cell 10
    python -m core.experiment_dem --mode tiles  --out results/dem --n 30 --cell 10
    python -m core.experiment_dem --mode ladder --out results/dem --row0 8825 --col0 5350 \
        --size 1200 --cells 1,2,5,10,20
    python -m core.experiment_dem --mode b23 --out results/dem
    python -m core.experiment_dem --mode tiles --out results/dem_tiles --summary-only

Five modes answer the questions research step 5 needs (`article.md` 5.5, the study protocol-3.4), none of which the T7 matrix (synthetic terrain) can answer:

  census   how much of the raster is usable, reproducibly -- replaces the ad hoc, unsaved
           "102 336 windows" figure of an earlier session (`core.dem.full_windows`)
  swath    the HAG on much larger terrain than any synthetic run uses: the whole surveyed
           corridor (or a stated sub-corridor), coarsened to one cell size, run through the
           same HAG/search chain as T7 -- the "Research 1: HAGs for much larger terrain"
           question, measured rather than assumed
  tiles    a POPULATION of real-DEM crops instead of one anecdote: N seeded, coverage-checked
           windows, each run through the same HAG/search/smoothing/checks chain as T7, so
           the real-DEM section gets a mean +/- SD like every synthetic one does
  ladder   one fixed crop at several cell sizes: the scale-dependence evidence of B20 --
           gradient-admissible cells barely change in NUMBER but their CONNECTIVITY collapses
           as the cell shrinks, measured here as connected components of
           the gradient-admissible subgraph, not asserted from one earlier exploration
  b23      the witness of a gradient certificate, smoothed by Algorithm 1: the one committed,
           reproducible instance of "smoothing destroys a feasible gradient" on real data, with the guard's verdict next to it

Every row carries `cell_size_m`, because none of these numbers mean anything without it
(`core.dem.coarsen_to_cell_size`'s own docstring). `swath` and `tiles` use the single-writer
lock and resumable JSONL log `core.experiment_matrix` already established, because both can
run for a long time: a row is logged, and the CSV refreshed, as soon as a scenario finishes,
so the results that exist are readable at any moment, and `--summary-only` rebuilds the CSV
from the log without the raster. `census`, `ladder` and `b23` are fast enough not to need it.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from core.algorithm_1 import before_after_guarded
from core.checks import check_path
from core.costs import DEFAULT_COST
from core.dem import (DEMError, coarsen_stream, coarsen_to_cell_size, full_windows,
                      load_geotiff_grid, read_geotiff_header)
from core.experiment_matrix import _assert_still_ours, _lock
from core.experiment_step1 import DEFAULT_ARMS, Step1Config, run_step1_full
from core.grid import Grid, edge_arrays
from core.hag import build_hag
from core.metrics import DEFAULT_CHORD_M, DEFAULT_RESAMPLE_STEP_M
from core.od_feasibility import find_feasible_pairs
from core.od_sampler import MIN_DIAGONAL_FRACTION
from core.provenance import provenance, write_jsonl
from core.results_io import read_jsonl, write_csv, write_path_csv
from data.configs.road_classes import RoadClass, get

DEFAULT_TIF = Path('raster/Idrija_Fault_LiDAR_DEM.tif')
DEFAULT_ROAD_CLASS = 'RO_CLASS_V_DEAL'
#: HAG bucket size for real relief; the synthetic matrix's 3 m is too fine for hundreds of
#: metres of real relief and would make a HAG of thousands of tiny areas
DEFAULT_HEIGHT_DELTA_M = 10.0
#: repetitions of every timed step on a real crop. A single run of the same crop varied by a
#: factor of three between two identical invocations on one desktop machine, so a single
#: timing is not a measurement; the median and the extremes of three are.
DEFAULT_TIMING_REPEATS = 3

CENSUS_COLUMNS = ('size_m', 'stride_m', 'min_relief_m', 'n_candidates', 'relief_min_m',
                  'relief_max_m', 'coverage_mask_shape', 'raster_height', 'raster_width',
                  'valid_fraction', 'n_exact_checked', 'n_exact_clean',
                  'over_approval_percent', 'sample_seed')

LADDER_COLUMNS = ('crop_row0', 'crop_col0', 'crop_size_m', 'road_class', 'i_max_percent',
                  'cell_size_m', 'n_cells', 'n_admissible_edges', 'n_total_edges',
                  'admissible_edge_fraction', 'n_components', 'largest_component_cells',
                  'largest_component_fraction', 'n_components_measured_note')

SWATH_TILE_COLUMNS = (
    'scenario_id', 'source', 'cell_size_m', 'height_delta_m', 'crop_row0', 'crop_col0', 'crop_n_rows',
    'crop_n_cols', 'relief_m', 'road_class', 'road_class_status', 'arm', 'rule', 'k_hops',
    'gradient_screened', 'start_i', 'start_j', 'target_i', 'target_j', 'straight_line_m',
    'n_areas_total', 'n_areas_selected', 'search_space_percent_of_full',
    'search_nodes_expanded', 'wall_time_s', 'wall_time_s_min', 'wall_time_s_max',
    'timing_repeats', 'hag_build_time_s', 'path_found',
    'infeasible_reason',
    'path_length_m', 'objective_cost_sum_abs_dh_m',
    'before_r_min_m', 'before_i_max_percent', 'after_r_min_m', 'after_i_max_percent',
    'smoothing_rejected', 'retained_source', 'check_r_min_passed', 'check_i_max_passed',
    'feasible', 'verdict',
    'timestamp_utc', 'git_sha', 'git_dirty', 'git_branch', 'python', 'python_free_threaded',
    'platform', 'numpy', 'scipy',
)


# -- census -------------------------------------------------------------------------------

CENSUS_SIZES_M: tuple[int, ...] = (300, 600, 1200, 2000, 3000, 4000, 5000)
#: sizes above this are not read exactly: 300 windows of 4 km would be 38 GB of reads, and
#: the point of the sample (how reliable the decimated scan is) is made by the smaller sizes
CENSUS_VERIFY_MAX_SIZE_M = 1200


def run_census(tif_path: Path, *, sizes_m: Sequence[int] = CENSUS_SIZES_M,
              stride_m: int = 25, min_relief_m: float = 30.0,
              verify_sample: int = 0, seed: int = 2026,
              verify_max_size_m: int = CENSUS_VERIFY_MAX_SIZE_M) -> list[dict]:
    """`full_windows` at several crop sizes: how many nodata-free candidates each gives.

    `full_windows` at a stride above 1 is a CANDIDATE list (its own docstring): a void that
    falls strictly between two samples is invisible to it. `verify_sample > 0` reads that
    many randomly chosen candidates per size EXACTLY (`load_geotiff_grid` with
    `max_void_fraction=0.0`) and records how many really were void-free, so the
    over-approval rate of the decimated scan is a measured, committed number rather than an
    assumption about how reliable the candidate counts are. Sizes above `verify_max_size_m`
    are not verified (their columns are empty), which is stated rather than hidden.

    The largest size with any candidate is the largest fully covered square: the swath is
    not aligned with the raster axes, so a bigger rectangle is mostly nodata, and a void is
    never filled over a large part of a domain.
    """
    tif = read_geotiff_header(tif_path, sha256=False)
    from core.dem import coverage_mask
    mask = coverage_mask(tif, stride=stride_m)
    rows = []
    for size_m in sizes_m:
        candidates = full_windows(tif, size_m, stride=stride_m, min_relief_m=min_relief_m)
        relief = [c.relief_m for c in candidates]

        n_checked = n_clean = None
        over_approval = None
        if verify_sample > 0 and candidates and size_m <= verify_max_size_m:
            rng = np.random.default_rng(seed)
            picked = rng.choice(len(candidates), size=min(verify_sample, len(candidates)),
                                replace=False)
            n_checked, n_clean = 0, 0
            for index in picked:
                candidate = candidates[int(index)]
                n_checked += 1
                try:
                    load_geotiff_grid(tif_path, row0=candidate.row0, col0=candidate.col0,
                                      size=size_m, sha256=False, tif=tif,
                                      max_void_fraction=0.0)
                    n_clean += 1
                except DEMError:
                    pass
            over_approval = 100.0 * (n_checked - n_clean) / n_checked

        rows.append({
            'size_m': size_m, 'stride_m': stride_m, 'min_relief_m': min_relief_m,
            'n_candidates': len(candidates),
            'relief_min_m': min(relief) if relief else None,
            'relief_max_m': max(relief) if relief else None,
            'coverage_mask_shape': f'{mask.shape[0]}x{mask.shape[1]}',
            'raster_height': tif.height, 'raster_width': tif.width,
            'valid_fraction': float(mask.mean()),
            'n_exact_checked': n_checked, 'n_exact_clean': n_clean,
            'over_approval_percent': over_approval,
            'sample_seed': seed if verify_sample > 0 else None,
        })
    return rows


# -- ladder -------------------------------------------------------------------------------

def run_ladder(tif_path: Path, *, row0: int, col0: int, size: int,
              cells_m: Sequence[float] = (1.0, 2.0, 5.0, 10.0, 20.0),
              road_class_name: str = DEFAULT_ROAD_CLASS) -> list[dict]:
    """One crop, several cell sizes: does the gradient-admissible subgraph stay connected?

    This is B20's evidence, measured rather than quoted from one earlier exploration: the FRACTION of admissible edges barely moves with cell size, but a
    `scipy.sparse.csgraph.connected_components` pass over the admissible subgraph shows how
    many pieces it is cut into, and how much of the crop the largest piece actually holds.
    """
    tif = read_geotiff_header(tif_path, sha256=False)
    fine, _, _ = load_geotiff_grid(tif_path, row0=row0, col0=col0, size=size, sha256=False,
                                   tif=tif, max_void_fraction=0.0)
    road_class = get(road_class_name)
    rows = []
    for cell_m in cells_m:
        grid = fine if cell_m == fine.cell_size_m else coarsen_to_cell_size(fine, cell_m)[0]
        i_max = road_class.i_max_percent
        u, v, lengths, dh = edge_arrays(grid, connectivity=8)
        admissible = np.abs(dh) <= (i_max / 100.0) * lengths
        n = grid.n_nodes

        graph = csr_matrix((np.ones(int(admissible.sum())), (u[admissible], v[admissible])),
                           shape=(n, n))
        n_components, labels = connected_components(graph, directed=False)
        sizes = np.bincount(labels, minlength=n_components)
        largest = int(sizes.max()) if sizes.size else 0

        rows.append({
            'crop_row0': row0, 'crop_col0': col0, 'crop_size_m': size,
            'road_class': road_class_name, 'i_max_percent': i_max,
            'cell_size_m': grid.cell_size_m, 'n_cells': n,
            'n_admissible_edges': int(admissible.sum()), 'n_total_edges': int(admissible.size),
            'admissible_edge_fraction': float(admissible.mean()) if admissible.size else 0.0,
            'n_components': int(n_components), 'largest_component_cells': largest,
            'largest_component_fraction': largest / n if n else 0.0,
            'n_components_measured_note':
                'components of the 8-connected ADMISSIBLE-EDGE subgraph over every grid '
                'cell, undirected; a cell with no admissible edge is its own component',
        })
    return rows


# -- swath / tiles: one scenario through the T7 chain --------------------------------------

def find_pair(grid: Grid, road_class: RoadClass, *, seed: int, margin: int,
             max_candidates: int = 20000) -> tuple[tuple[int, int], tuple[int, int], bool]:
    """A gradient-feasible pair if one exists in `max_candidates` tries; otherwise the first
    well-separated geometric candidate, flagged `False` (not gradient-screened).

    Real relief at swath/tile scale is often steeper than a hilly class's `i_max` can hold
    over a uniformly-random pair -- the survey's own description is "a gentle VALLEY", not
    a gentle square kilometre, and a 4 km square crop routinely has 500-800 m of relief
    (every `core.dem.full_windows` candidate at that size does). Requiring a
    gradient-feasible pair would make the scale demonstration depend on the random sampler
    finding the valley floor. The fallback keeps it honest instead of silently giving up:
    a geometrically-valid but gradient-infeasible pair is still run through the full
    chain, and `core.checks` reports it as infeasible under the class, exactly as any
    other infeasible scenario in this project is -- never dropped, never relaxed.
    """
    found = find_feasible_pairs(grid, road_class, n_pairs=1, seed=seed, margin=margin,
                                max_candidates=max_candidates)
    if found:
        return found[0].start, found[0].target, True
    from core.od_feasibility import _candidate_pairs
    rng = np.random.default_rng(seed)
    for start, target in _candidate_pairs(grid, rng, margin, MIN_DIAGONAL_FRACTION,
                                          max_candidates):
        return start, target, False
    raise DEMError(f"no pair at all satisfies the margin={margin}/diagonal-fraction "
                   f"criteria on a {grid.shape} grid")


def _postprocess_path(grid: Grid, road_class: RoadClass, path_xy, *, metric_step_m: float,
                      metric_chord_m: float) -> dict:
    """Algorithm 1 (guarded, B23) and the engineering checks on the axis it retains."""
    from core.algorithm_1 import algorithm_1
    report = before_after_guarded(grid, path_xy, road_class, iterations=1,
                                  metric_step_m=metric_step_m, metric_chord_m=metric_chord_m)
    smoothed_xyz = algorithm_1(path_xy, road_class, grid=grid, iterations=1).smooth_xyz_m
    retained_xyz = grid.path_xyz_m(path_xy) if report['retained_source'] == 'before' \
        else smoothed_xyz
    checks = check_path(retained_xyz, road_class, step_m=metric_step_m, chord_m=metric_chord_m)
    by_name = {c.name: c for c in checks.checks}
    return {
        'before_r_min_m': report['before'].get('r_min_m'),
        'before_i_max_percent': report['before'].get('i_max_percent'),
        'after_r_min_m': report['after'].get('r_min_m'),
        'after_i_max_percent': report['after'].get('i_max_percent'),
        'smoothing_rejected': report['smoothing_rejected'],
        'retained_source': report['retained_source'],
        'check_r_min_passed': by_name['R_min'].passed,
        'check_i_max_passed': by_name['i_max'].passed if 'i_max' in by_name else None,
        'feasible': checks.feasible, 'verdict': checks.verdict,
    }


def _run_dem_scenario(grid: Grid, *, scenario_id: str, source: str, crop: dict,
                      start, target, road_class_name: str, gradient_screened: bool = True,
                      height_delta_m: float = DEFAULT_HEIGHT_DELTA_M,
                      metric_step_m: float = DEFAULT_RESAMPLE_STEP_M,
                      metric_chord_m: float = DEFAULT_CHORD_M,
                      paths_out: Optional[dict] = None,
                      timing_repeats: int = DEFAULT_TIMING_REPEATS) -> list[dict]:
    """Every DEFAULT_ARMS arm, then the two gradient-aware arms (B21), on one real-DEM crop.

    `paths_out`, when given, is filled with `{arm: (n, 3) rough axis in metres}` for every arm
    that found a path, so a caller can commit the alignments and a figure can be drawn from
    them without re-searching (`core.figures` never re-runs a search).

    Mirrors `core.experiment_matrix.run_scenario`, at a scale the synthetic matrix never
    reaches, with the road class fixed per-scenario rather than swept (a real crop is
    expensive enough that the class the O-D was screened for, see `find_pair`, is the one
    worth reporting -- sweeping three classes per crop would triple the swath/tile cost for
    a question the synthetic matrix already answers).

    The four DEFAULT_ARMS all minimise `sum |dh|` and never see `i_max` (B21). On rugged real
    terrain that is not a footnote: an alignment of 93-243% gradient comes out of them on a
    4 km Idrija crop, physically absurd for any road class. `hag_yellow_gradcut` and
    `hag_yellow_gradpen` (`core.experiment_gradient`) give the search the class instead of
    checking it afterwards, on the SAME yellow mask, so the contrast is real-terrain evidence
    rather than a synthetic-only claim.

    `height_delta_m` is recorded on every row because it interacts with cell size and slope:
    on ground steeper than `height_delta_m` per cell, every cell falls in a different height
    bucket from its neighbour and the HAG degenerates toward one cell per area. That is a
    property of the parameter on steep ground, not of the terrain, and it has to be visible
    in the data rather than buried in a default.
    """
    from core.experiment_gradient import GRADIENT_ARMS, _arm_graph
    from core.hag import select_areas
    from core.search import dijkstra as dijkstra_search
    from core.search import path_height_cost, path_length_m

    road_class = get(road_class_name).with_cell_size(grid.cell_size_m)
    config = Step1Config(start=start, target=target, height_delta=height_delta_m,
                         arms=DEFAULT_ARMS, metric_step_m=metric_step_m,
                         metric_chord_m=metric_chord_m, timing_repeats=timing_repeats)
    run = run_step1_full(config, grid=grid, terrain_info={'source': source, **crop})
    builds = [run.row['hag_build_time_s']]
    for _ in range(max(1, timing_repeats) - 1):
        t0 = time.perf_counter()
        build_hag(grid, height_delta=height_delta_m, connectivity=config.connectivity)
        builds.append(time.perf_counter() - t0)
    hag_build_s = float(statistics.median(builds))

    def identity(label: str, rule: str, k_hops: int) -> dict:
        return {
            'scenario_id': f'{scenario_id}__{label}', 'source': source,
            'cell_size_m': grid.cell_size_m, 'height_delta_m': height_delta_m, **crop,
            'hag_build_time_s': hag_build_s,
            'road_class': road_class_name, 'road_class_status': road_class.status,
            'arm': label, 'rule': rule, 'k_hops': k_hops,
            'gradient_screened': gradient_screened,
            'start_i': start[0], 'start_j': start[1],
            'target_i': target[0], 'target_j': target[1],
            'straight_line_m': float(np.hypot(start[0] - target[0],
                                              start[1] - target[1])) * grid.cell_size_m,
        }

    rows = []
    for label, rule, k_hops in DEFAULT_ARMS:
        arm_row = run.row['arms'][label]
        row = identity(label, rule, k_hops)
        row.update({
            'n_areas_total': arm_row.get('n_areas_total'),
            'n_areas_selected': arm_row.get('n_areas_selected'),
            'search_space_percent_of_full': arm_row.get('search_space_percent_of_full'),
            'search_nodes_expanded': arm_row.get('search_nodes_expanded'),
            'wall_time_s': arm_row.get('wall_time_s'),
            'wall_time_s_min': arm_row.get('wall_time_s_min'),
            'wall_time_s_max': arm_row.get('wall_time_s_max'),
            'timing_repeats': arm_row.get('timing_repeats'),
            'path_found': arm_row.get('path_found'),
        })
        if not arm_row.get('path_found'):
            row['infeasible_reason'] = arm_row.get('infeasible_reason')
            rows.append(row)
            continue
        row.update({'path_length_m': arm_row.get('path_length_m'),
                   'objective_cost_sum_abs_dh_m': arm_row.get('objective_cost_sum_abs_dh_m')})
        if paths_out is not None:
            paths_out[label] = grid.path_xyz_m(run.paths_xy[label])
        row.update(_postprocess_path(grid, road_class, run.paths_xy[label],
                                     metric_step_m=metric_step_m,
                                     metric_chord_m=metric_chord_m))
        rows.append(row)

    # -- the two gradient-aware arms, on the SAME yellow mask ------------------------------
    yellow = select_areas(run.hag, start, target, rule='yellow')
    mask = run.hag.mask(yellow).copy()
    mask[tuple(start)] = True
    mask[tuple(target)] = True
    yellow_cells = int(mask.sum())
    for arm in (a for a in GRADIENT_ARMS if a != 'hag_yellow'):
        row = identity(arm, 'yellow', 0)
        row.update({'n_areas_total': run.hag.n_areas, 'n_areas_selected': len(yellow),
                   'search_space_percent_of_full': 100.0 * yellow_cells / grid.n_nodes})
        timings = []
        for _ in range(max(1, timing_repeats)):
            t0 = time.perf_counter()
            graph = _arm_graph(grid, mask, road_class, arm)
            result = dijkstra_search(graph, grid.index(start), grid.index(target))
            timings.append(time.perf_counter() - t0)
        row['wall_time_s'] = float(statistics.median(timings))
        row['wall_time_s_min'], row['wall_time_s_max'] = min(timings), max(timings)
        row['timing_repeats'] = len(timings)
        row['search_nodes_expanded'] = result.stats.nodes_expanded
        row['path_found'] = bool(result.reached)
        if not result.reached:
            row['infeasible_reason'] = ('no path under this cost/mask; the gradient cut can '
                                        'disconnect the yellow selection, which gradpen '
                                        'cannot (every edge stays, only costs more)')
            rows.append(row)
            continue
        row.update({'path_length_m': path_length_m(graph, result.path),
                   'objective_cost_sum_abs_dh_m': path_height_cost(graph, result.path)})
        if paths_out is not None:
            paths_out[arm] = grid.path_xyz_m(result.coords(grid).astype(float))
        row.update(_postprocess_path(grid, road_class, result.coords(grid).astype(float),
                                     metric_step_m=metric_step_m,
                                     metric_chord_m=metric_chord_m))
        rows.append(row)
    return rows


def run_swath(tif_path: Path, *, cell_m: float, row0: int, col0: int, n_rows: int,
             n_cols: int, road_class_name: str = DEFAULT_ROAD_CLASS,
             seed: int = 0, height_delta_m: float = DEFAULT_HEIGHT_DELTA_M,
             paths_out: Optional[dict] = None,
             timing_repeats: int = DEFAULT_TIMING_REPEATS) -> list[dict]:
    """The whole surveyed corridor (or the stated sub-corridor), one cell size, one O-D.

    `paths_out` is passed to `_run_dem_scenario`; the caller commits what it collects."""
    tif = read_geotiff_header(tif_path, sha256=False)
    coarse, audit = coarsen_stream(tif, row0=row0, col0=col0, n_rows=n_rows, n_cols=n_cols,
                                   factor=int(round(cell_m / tif.pixel_scale_m[0])))
    if np.isnan(coarse).any():
        from core.dem import fill_voids
        coarse, void_audit = fill_voids(coarse)
    else:
        void_audit = {'n_void': 0}
    grid = Grid(surf=coarse, cell_size_m=cell_m)

    road_class = get(road_class_name).with_cell_size(cell_m)
    start, target, screened = find_pair(grid, road_class, seed=seed,
                                        margin=max(5, int(50 / cell_m)))
    crop = {'crop_row0': row0, 'crop_col0': col0, 'crop_n_rows': n_rows, 'crop_n_cols': n_cols,
           'relief_m': float(np.nanmax(coarse) - np.nanmin(coarse))}
    sid = f'swath_cell{cell_m:g}' + ('' if height_delta_m == DEFAULT_HEIGHT_DELTA_M
                                     else f'_hd{height_delta_m:g}')
    return _run_dem_scenario(grid, scenario_id=sid, source='idrija_swath',
                             crop=crop, start=start, target=target,
                             road_class_name=road_class_name, gradient_screened=screened,
                             height_delta_m=height_delta_m, paths_out=paths_out,
                             timing_repeats=timing_repeats)


def run_tiles(tif_path: Path, *, n: int, cell_m: float, seed: int = 2026,
             size_m: int = 600, road_class_name: str = DEFAULT_ROAD_CLASS,
             min_relief_m: float = 30.0, stride_m: int = 25,
             height_delta_m: float = DEFAULT_HEIGHT_DELTA_M,
             skip: frozenset = frozenset(), on_tile=None,
             dropped_out: Optional[list] = None,
             timing_repeats: int = DEFAULT_TIMING_REPEATS) -> list[dict]:
    """N seeded, coverage-checked crops, each run through the same chain as a swath scenario.

    `skip` holds the scenario ids (`tile<row0>_<col0>_cell<cell>`) already logged, so a
    resumed run computes only what is missing; the candidates are still walked in the same
    order, because a tile's pair seed is `seed + its position in that order` and skipping
    must not move the pairs of the tiles after it. `on_tile(rows)` is called as soon as one
    tile has finished, so the caller can log it and partial results exist at any time.

    A candidate that is dropped (the exact read found nodata the decimated scan missed, or
    no pair fits the margin) is appended to `dropped_out` with the reason: the article
    states how many of the sampled windows were not used and why.
    """
    tif = read_geotiff_header(tif_path, sha256=False)
    candidates = full_windows(tif, size_m, stride=stride_m, min_relief_m=min_relief_m)
    if not candidates:
        raise DEMError(f"no candidate {size_m} m windows at stride {stride_m}")
    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)

    rows: list[dict] = []
    tried = 0
    for idx in chosen_idx:
        candidate = candidates[int(idx)]
        tried += 1
        tile_id = f'tile{candidate.row0}_{candidate.col0}_cell{cell_m:g}'
        if tile_id in skip:
            continue

        def drop(reason: str) -> None:
            if dropped_out is not None:
                dropped_out.append({'candidate_index': tried, 'row0': candidate.row0,
                                    'col0': candidate.col0, 'size_m': size_m, 'reason': reason})
        try:
            fine, _, _ = load_geotiff_grid(tif_path, row0=candidate.row0, col0=candidate.col0,
                                           size=size_m, sha256=False, tif=tif,
                                           max_void_fraction=0.0)
        except DEMError as error:
            drop(f'the exact read found nodata that the decimated scan missed ({error})')
            continue
        grid = coarsen_to_cell_size(fine, cell_m)[0] if cell_m != fine.cell_size_m else fine
        road_class = get(road_class_name).with_cell_size(cell_m)
        try:
            start, target, screened = find_pair(grid, road_class, seed=seed + tried,
                                                margin=max(5, int(50 / cell_m)))
        except DEMError as error:
            drop(f'no origin-destination pair fits the margin ({error})')
            continue
        crop = {'crop_row0': candidate.row0, 'crop_col0': candidate.col0,
               'crop_n_rows': size_m, 'crop_n_cols': size_m, 'relief_m': candidate.relief_m}
        tile_rows = _run_dem_scenario(
            grid, scenario_id=tile_id, source='idrija_tile', crop=crop, start=start,
            target=target, road_class_name=road_class_name, gradient_screened=screened,
            height_delta_m=height_delta_m, timing_repeats=timing_repeats)
        rows += tile_rows
        if on_tile is not None:
            on_tile(tile_rows)
    return rows


# -- a certified alignment, smoothed (bug B23) ----------------------------------------------

B23_COLUMNS = ('crop_row0', 'crop_col0', 'crop_size_m', 'cell_size_m', 'road_class',
               'i_max_limit_percent', 'r_min_limit_m', 'start_i', 'start_j', 'target_i',
               'target_j', 'certified', 'certified_length_m', 'certified_i_max_percent',
               'development_ratio', 'before_r_min_m', 'after_length_m', 'after_i_max_percent',
               'after_r_min_m', 'smoothing_rejected', 'retained_source')


def run_b23(tif_path: Path, *, row0: int = 1500, col0: int = 4200, size: int = 600,
           cell_m: float = 10.0, start=(53, 42), target=(19, 2),
           road_class_name: str = DEFAULT_ROAD_CLASS) -> list[dict]:
    """The witness of a gradient certificate, smoothed: smoothing can destroy feasibility.

    `certify_grade` returns, for a pair whose class limit CAN be held at cell resolution, the
    shortest alignment every step of which holds it (the witness). Algorithm 1 then smooths
    that alignment in x-y and re-reads the heights, and the gradient can rise above the limit
    although every step of the input was inside it: the regression `before_after_guarded`
    exists to catch. This is the one committed, reproducible instance of it on real data; the
    numbers of the study protocol (986 m at 6.89% smoothing to 669 m at 14.70%) came from an
    exploratory run and are not cited from there.

    The crop, cell size, pair and class are the ones of that exploratory run, recorded here.
    """
    from core.od_feasibility import certify_grade

    tif = read_geotiff_header(tif_path, sha256=False)
    fine, _, _ = load_geotiff_grid(tif_path, row0=row0, col0=col0, size=size, sha256=False,
                                   tif=tif, max_void_fraction=0.0)
    grid = coarsen_to_cell_size(fine, cell_m)[0] if cell_m != fine.cell_size_m else fine
    road_class = get(road_class_name).with_cell_size(grid.cell_size_m)

    certificate = certify_grade(grid, tuple(start), tuple(target), road_class)
    row = {'crop_row0': row0, 'crop_col0': col0, 'crop_size_m': size,
           'cell_size_m': grid.cell_size_m, 'road_class': road_class_name,
           'i_max_limit_percent': road_class.i_max_percent, 'r_min_limit_m': road_class.r_min_m,
           'start_i': start[0], 'start_j': start[1], 'target_i': target[0],
           'target_j': target[1], 'certified': bool(certificate.reachable)}
    if not certificate.reachable:
        return [row]

    report = before_after_guarded(grid, certificate.path_xy.astype(float), road_class,
                                  iterations=1)
    row.update({
        'certified_length_m': certificate.length_m,
        'certified_i_max_percent': certificate.witness_i_max_percent,
        'development_ratio': certificate.development_ratio,
        'before_r_min_m': report['before'].get('r_min_m'),
        'after_length_m': report['after'].get('length_m'),
        'after_i_max_percent': report['after'].get('i_max_percent'),
        'after_r_min_m': report['after'].get('r_min_m'),
        'smoothing_rejected': report['smoothing_rejected'],
        'retained_source': report['retained_source'],
    })
    return [row]


# -- CLI -----------------------------------------------------------------------------------

#: (row0, col0, size or n_rows=n_cols) in raster pixels, when the command line does not say.
#: ladder: the 1.2 km feature of the study protocol swath: the 4 km crop of the article. b23: the
#: 600 m crop of the study protocol
CROP_DEFAULTS = {'ladder': (8825, 5350, 1200), 'swath': (5100, 5525, 4000),
                 'b23': (1500, 4200, 600)}

TILES_DROPPED_COLUMNS = ('candidate_index', 'row0', 'col0', 'size_m', 'reason')


def _write_rows(out_dir: Path, name: str, rows: list[dict], columns: Sequence[str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return write_csv(out_dir / f'{name}.csv', rows, columns)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--mode', required=True,
                        choices=('census', 'swath', 'tiles', 'ladder', 'b23'))
    parser.add_argument('--tif', type=Path, default=DEFAULT_TIF)
    parser.add_argument('--out', type=Path, default=Path('results/dem'))
    parser.add_argument('--cell', type=float, default=10.0, help='cell size in metres')
    parser.add_argument('--cells', type=str, default='1,2,5,10,20',
                        help='comma-separated cell sizes for --mode ladder')
    parser.add_argument('--row0', type=int, default=None,
                        help=f'crop origin row, pixels; default per mode {CROP_DEFAULTS}')
    parser.add_argument('--col0', type=int, default=None)
    parser.add_argument('--size', type=int, default=None,
                        help='ladder / b23 crop size, metres (= pixels of the 1 m raster)')
    parser.add_argument('--n-rows', type=int, default=None, help='swath crop height, pixels')
    parser.add_argument('--n-cols', type=int, default=None, help='swath crop width, pixels')
    parser.add_argument('--n', type=int, default=30, help='tiles: how many crops')
    parser.add_argument('--tile-size', type=int, default=600, help='tiles: window side, metres')
    parser.add_argument('--census-sizes', type=str, default=None,
                        help=f'census: comma-separated window sizes; default {CENSUS_SIZES_M}')
    parser.add_argument('--verify-sample', type=int, default=0,
                        help='census: exact-verify this many candidates per size')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--road-class', default=DEFAULT_ROAD_CLASS)
    parser.add_argument('--height-delta', type=float, default=DEFAULT_HEIGHT_DELTA_M,
                        help='HAG bucket height in metres (swath/tiles); recorded per row')
    parser.add_argument('--repeats', type=int, default=DEFAULT_TIMING_REPEATS,
                        help='swath/tiles: repetitions of every timed step; the median is '
                             'recorded with the minimum and maximum')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--summary-only', action='store_true',
                        help='swath/tiles: rewrite the CSV from the JSONL log and stop; needs '
                             'neither the raster nor the lock')
    args = parser.parse_args(argv)

    if args.summary_only:
        if args.mode not in ('swath', 'tiles'):
            print('--summary-only applies to --mode swath or tiles', file=sys.stderr)
            return 2
        logged = read_jsonl(args.out / f'{args.mode}.jsonl')
        path = _write_rows(args.out, args.mode, logged, SWATH_TILE_COLUMNS)
        print(f'{args.mode}: {path} ({len(logged)} rows, rewritten from the log)')
        return 0

    if not args.tif.exists():
        print(f'{args.tif} not found (git-ignored local data; see README.md)', file=sys.stderr)
        return 1

    default_row0, default_col0, default_size = CROP_DEFAULTS.get(args.mode, (0, 0, 0))
    row0 = default_row0 if args.row0 is None else args.row0
    col0 = default_col0 if args.col0 is None else args.col0
    size = default_size if args.size is None else args.size
    n_rows = default_size if args.n_rows is None else args.n_rows
    n_cols = default_size if args.n_cols is None else args.n_cols

    if args.mode == 'census':
        sizes = (CENSUS_SIZES_M if args.census_sizes is None
                 else tuple(int(v) for v in args.census_sizes.split(',')))
        rows = run_census(args.tif, sizes_m=sizes, verify_sample=args.verify_sample,
                          seed=args.seed)
        path = _write_rows(args.out, 'census', rows, CENSUS_COLUMNS)
        print(f'census: {path} ({len(rows)} rows)')
        return 0

    if args.mode == 'ladder':
        cells = [float(c) for c in args.cells.split(',')]
        rows = run_ladder(args.tif, row0=row0, col0=col0, size=size,
                          cells_m=cells, road_class_name=args.road_class)
        path = _write_rows(args.out, 'ladder', rows, LADDER_COLUMNS)
        print(f'ladder: {path} ({len(rows)} rows)')
        return 0

    if args.mode == 'b23':
        rows = run_b23(args.tif, row0=row0, col0=col0, size=size, cell_m=args.cell,
                       road_class_name=args.road_class)
        path = _write_rows(args.out, 'b23_case', rows, B23_COLUMNS)
        print(f'b23: {path} ({len(rows)} rows)')
        return 0

    # swath / tiles: resumable, single-writer, like core.experiment_matrix
    jsonl = args.out / f'{args.mode}.jsonl'
    args.out.mkdir(parents=True, exist_ok=True)
    with _lock(args.out) as lock:
        if args.force and jsonl.exists():
            jsonl.unlink()
        done = {row.get('scenario_id', '').split('__')[0] for row in read_jsonl(jsonl)}

        def log(new_rows: list[dict]) -> None:
            """Append to the log as soon as a scenario finishes, then refresh the CSV, so the
            results that exist are readable at any moment of a run that takes hours."""
            for row in new_rows:
                _assert_still_ours(lock)
                write_jsonl(jsonl, {**row, **provenance()})
            _write_rows(args.out, args.mode, read_jsonl(jsonl), SWATH_TILE_COLUMNS)

        if args.mode == 'swath':
            sid = f'swath_cell{args.cell:g}' + ('' if args.height_delta ==
                                                DEFAULT_HEIGHT_DELTA_M
                                                else f'_hd{args.height_delta:g}')
            if sid not in done:
                collected: dict = {}
                rows = run_swath(args.tif, cell_m=args.cell, row0=row0, col0=col0,
                                 n_rows=n_rows, n_cols=n_cols,
                                 road_class_name=args.road_class, seed=args.seed,
                                 height_delta_m=args.height_delta, paths_out=collected,
                                 timing_repeats=args.repeats)
                for arm, xyz in collected.items():
                    write_path_csv(args.out / 'paths' / f'{sid}__{arm}__rough.csv', xyz)
                log(rows)
        else:
            dropped: list[dict] = []
            run_tiles(args.tif, n=args.n, cell_m=args.cell, seed=args.seed,
                      size_m=args.tile_size, road_class_name=args.road_class,
                      height_delta_m=args.height_delta, skip=frozenset(done), on_tile=log,
                      dropped_out=dropped, timing_repeats=args.repeats)
            _write_rows(args.out, 'tiles_dropped', dropped, TILES_DROPPED_COLUMNS)

    logged = read_jsonl(jsonl)
    path = _write_rows(args.out, args.mode, logged, SWATH_TILE_COLUMNS)
    print(f'{args.mode}: {path} ({len(logged)} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
