"""Research step 1: full grid versus HAG on the same terrain, O-D and parameters.

What the step has to show is not only "the HAG reduces the search space" but "the HAG
reduces the search space while keeping an acceptable solution". So every arm records
both sides:

  cost side     search-space fraction, nodes expanded / pushed / reached, wall time,
                tracemalloc peak
  quality side  path length, the Equation-1 objective cost, R_min, i_max, sinuosity,
                and the Hausdorff / mean deviation from the FULL-GRID path

The arms are the full grid, the candidate terrain areas (the cheapest shared-border HAG
chain) and the same dilated by k rings, all through one code path so they differ only in
the cell mask.

One `run_step1` call produces one JSON row (`core.provenance.write_jsonl`) carrying the
seed, the git SHA and the whole configuration, so the row is reproducible on its own.

Note on the two timings: `tracemalloc` slows allocation down measurably, so an arm is
run twice, once timed with tracemalloc off and once measured with it on. Both are
reported (`wall_time_s`, `peak_memory_bytes`) and neither is contaminated by the other.
"""
from __future__ import annotations

import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np

from core import costs as cost_models
from core import metrics as metrics_mod
from core.grid import Grid
from core.hag import HAG, build_hag, search_space_fraction, select_areas
from core.provenance import provenance
from core.search import build_graph, dijkstra, path_height_cost, path_length_m
from core.terrain import TerrainSpec, generate_terrain

#: the arms of the comparison: (label, selection rule, k-hop dilation)
DEFAULT_ARMS: tuple[tuple[str, str, int], ...] = (
    ('full_grid', 'full', 0),
    ('hag_yellow', 'yellow', 0),
    ('hag_yellow_k1', 'yellow', 1),
)


@dataclass(frozen=True)
class Step1Config:
    """Everything that decides the numbers of one step-1 run."""

    terrain: TerrainSpec = field(default_factory=TerrainSpec)
    start: tuple[int, int] = (5, 5)
    target: tuple[int, int] = (94, 94)
    height_delta: float = 3.0
    #: a registered name or any (dh, length_m) -> weights callable; the row
    #: records `cost_models.name_of(cost)`, never the object itself
    cost: cost_models.CostSpec = cost_models.DEFAULT_COST
    connectivity: int = 8
    arms: tuple[tuple[str, str, int], ...] = DEFAULT_ARMS
    baseline_arm: str = 'full_grid'
    metric_step_m: float = metrics_mod.DEFAULT_RESAMPLE_STEP_M
    metric_chord_m: float = metrics_mod.DEFAULT_CHORD_M
    #: how many times each search is timed. The median is reported: single-run timings
    #: on a 100x100 grid flipped the ordering between arms between runs (the study protocol
    #: 3a), because every arm finishes in tens of milliseconds.
    timing_repeats: int = 1
    #: a name for the terrain, so a matrix row says which one it came from
    terrain_label: str = ''

    def as_dict(self) -> dict:
        out = asdict(self)
        out['cost'] = cost_models.name_of(self.cost)
        out['terrain'] = self.terrain.as_dict()
        out['arms'] = [list(a) for a in self.arms]
        out['start'] = list(self.start)
        out['target'] = list(self.target)
        return out


def _run_arm(grid: Grid, hag: HAG, config: Step1Config,
             label: str, rule: str, k_hops: int) -> dict:
    selected = select_areas(hag, config.start, config.target, rule=rule, k_hops=k_hops)
    mask = None if rule == 'full' and k_hops == 0 else hag.mask(selected)
    if mask is not None:
        # the objective's endpoints must be searchable even if the rule left them out
        mask = mask.copy()
        mask[tuple(config.start)] = True
        mask[tuple(config.target)] = True
    fraction = 1.0 if mask is None else float(mask.sum()) / float(grid.n_nodes)

    start_node, target_node = grid.index(config.start), grid.index(config.target)

    def once():
        graph = build_graph(grid, cost=config.cost, mask=mask,
                            connectivity=config.connectivity)
        return graph, dijkstra(graph, start_node, target_node, stop_at_target=True)

    repeats = max(1, int(config.timing_repeats))
    timings = []
    graph = result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        graph, result = once()
        timings.append(time.perf_counter() - t0)
    timings.sort()
    wall_time_s = float(statistics.median(timings))

    tracemalloc.start()
    once()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    row = {
        'arm': label,
        'rule': rule,
        'k_hops': k_hops,
        'n_areas_selected': len(selected),
        'n_areas_total': hag.n_areas,
        'n_cells_selected': int(grid.n_nodes if mask is None else mask.sum()),
        'n_cells_total': int(grid.n_nodes),
        'search_space_fraction_of_full': fraction,
        'search_space_percent_of_full': 100.0 * fraction,
        'search_space_reduction_percent': 100.0 * (1.0 - fraction),
        'n_graph_edges': graph.n_edges,
        'wall_time_s': wall_time_s,           # the median; kept under the old name
        'wall_time_s_median': wall_time_s,
        'wall_time_s_min': float(timings[0]),
        'wall_time_s_max': float(timings[-1]),
        'timing_repeats': repeats,
        'peak_memory_bytes': int(peak),
        'path_found': bool(result.reached),
    }
    row.update({f'search_{k}': v for k, v in result.stats.as_dict().items()})
    if not result.reached:
        row['infeasible_reason'] = 'no path inside the selected areas'
        return row

    coords = result.coords(grid).astype(float)
    xyz = grid.path_xyz_m(coords)
    measured = metrics_mod.measure(xyz, step_m=config.metric_step_m,
                                   chord_m=config.metric_chord_m)
    row.update({
        'search_cost': result.cost,
        'objective_cost_sum_abs_dh_m': path_height_cost(graph, result.path),
        'path_length_m': path_length_m(graph, result.path),
        'n_path_cells': int(len(result.path)),
    })
    row.update({f'metric_{k}': v for k, v in measured.as_dict().items()})
    # both stripped before the row is written; `run_step1_full` hands them back
    row['_path_xyz_m'] = xyz
    row['_path_xy'] = coords
    return row


@dataclass(frozen=True)
class Step1Run:
    """The full outcome of one step-1 run: the JSON row plus the geometry it came from.

    `run_step1` returns only `row`, which is what goes into the JSON Lines log. The
    matrix runner (T7) and the DEM case (T8) also need the paths, to feed Algorithm 1
    and to write `results/paths/*.csv`, so they call `run_step1_full` instead.
    """

    row: dict
    paths_xyz_m: dict[str, np.ndarray]   # arm label -> (n, 3) in METRES
    paths_xy: dict[str, np.ndarray]      # arm label -> (n, 2) in GRID units
    grid: Grid
    hag: HAG


def run_step1_full(config: Step1Config = Step1Config(),
                   grid: Optional[Grid] = None,
                   terrain_info: Optional[dict] = None) -> Step1Run:
    """Run every arm on one terrain and O-D, keeping the paths.

    `grid=None` generates the terrain from `config.terrain`. A supplied `Grid` is used
    as-is and `terrain_info` replaces the `TerrainSpec` block in the recorded config,
    which is how the real-DEM case (T8) reuses this machinery: a DEM has no seed and no
    Perlin periods, so recording a fabricated `TerrainSpec` would be a lie.
    """
    grid = generate_terrain(config.terrain) if grid is None else grid
    if not (0 <= config.start[0] < grid.n_rows and 0 <= config.start[1] < grid.n_cols):
        raise ValueError(f"start {config.start} is outside the {grid.shape} terrain")
    if not (0 <= config.target[0] < grid.n_rows and 0 <= config.target[1] < grid.n_cols):
        raise ValueError(f"target {config.target} is outside the {grid.shape} terrain")

    t0 = time.perf_counter()
    hag = build_hag(grid, height_delta=config.height_delta,
                    connectivity=config.connectivity)
    hag_build_time_s = time.perf_counter() - t0

    arms = {label: _run_arm(grid, hag, config, label, rule, k)
            for label, rule, k in config.arms}

    paths_xyz: dict[str, np.ndarray] = {}
    paths_xy: dict[str, np.ndarray] = {}
    for label, arm in arms.items():
        xyz, xy = arm.pop('_path_xyz_m', None), arm.pop('_path_xy', None)
        if xyz is not None:
            paths_xyz[label], paths_xy[label] = xyz, xy

    baseline = arms.get(config.baseline_arm)
    baseline_path = paths_xyz.get(config.baseline_arm)
    for label, arm in arms.items():
        path = paths_xyz.get(label)
        if path is None or baseline_path is None or label == config.baseline_arm:
            continue
        arm.update({f'vs_baseline_{k}': v for k, v in metrics_mod.deviation(
            path, baseline_path, step_m=config.metric_step_m).items()})
        if baseline.get('objective_cost_sum_abs_dh_m'):
            arm['objective_cost_ratio_to_baseline'] = (
                arm['objective_cost_sum_abs_dh_m'] / baseline['objective_cost_sum_abs_dh_m'])
        if baseline.get('path_length_m'):
            arm['length_ratio_to_baseline'] = arm['path_length_m'] / baseline['path_length_m']
        if baseline.get('wall_time_s'):
            arm['time_reduction_percent'] = 100.0 * (1.0 - arm['wall_time_s'] / baseline['wall_time_s'])
    if baseline is not None and baseline_path is not None:
        baseline.setdefault('vs_baseline_hausdorff_m', 0.0)

    recorded_config = config.as_dict()
    if terrain_info is not None:
        recorded_config['terrain'] = terrain_info

    row = {
        'experiment': 'step1_hag_vs_full_grid',
        'config': recorded_config,
        'seed': None if terrain_info is not None else config.terrain.seed,
        'terrain_label': config.terrain_label,
        'hag_n_areas': hag.n_areas,
        'hag_build_time_s': hag_build_time_s,
        'arms': arms,
        'provenance': provenance(),
    }
    return Step1Run(row=row, paths_xyz_m=paths_xyz, paths_xy=paths_xy,
                    grid=grid, hag=hag)


def run_step1(config: Step1Config = Step1Config()) -> dict:
    """Run every arm on one terrain and O-D, and return one result row.

    The row carries no arrays, so it is JSON-serialisable as-is. Use `run_step1_full`
    when the paths are needed too.
    """
    return run_step1_full(config).row


def summarise(row: dict) -> str:
    """A short human-readable table of one run, for the terminal and the report."""
    header = (f"{'arm':<18}{'space %':>9}{'expanded':>10}{'time s':>9}{'peak MB':>9}"
              f"{'length m':>11}{'obj m':>9}{'haus m':>9}")
    lines = [header, '-' * len(header)]
    for label, arm in row['arms'].items():
        if not arm['path_found']:
            lines.append(f"{label:<18}{arm['search_space_percent_of_full']:>9.2f}"
                         f"{'':>10}{'':>9}{'':>9}   INFEASIBLE (no path in the selection)")
            continue
        lines.append(
            f"{label:<18}{arm['search_space_percent_of_full']:>9.2f}"
            f"{arm['search_nodes_expanded']:>10d}{arm['wall_time_s']:>9.3f}"
            f"{arm['peak_memory_bytes'] / 2 ** 20:>9.1f}{arm['path_length_m']:>11.1f}"
            f"{arm['objective_cost_sum_abs_dh_m']:>9.2f}"
            f"{arm.get('vs_baseline_hausdorff_m', float('nan')):>9.1f}")
    return '\n'.join(lines)
