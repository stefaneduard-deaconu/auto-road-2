"""Reading and writing the result files of the study protocol

`MATRIX_COLUMNS` is the machine-readable form of section 2's field list. A test compares
it against a hard-coded copy of that list, so the spec and the code cannot drift apart
silently.

Two grains, which the plan states inconsistently and which are resolved here:

* the **JSONL** grain is (terrain, O-D). Section 4 says "one JSON row per scenario", but
  section 1 also says the four search arms are computed once per (terrain, O-D) and
  reused across the three road classes -- so a per-class JSONL row would force the search
  to be re-run or duplicated three times. One line per (terrain, O-D) is also the natural
  restart unit.
* the **CSV** grain is (terrain, O-D, arm, road class, iterations). Section 4 omits
  `algorithm_1_iterations`, but section 1.4 records two iteration settings and section 2.1
  lists the column, so it has to be part of the key.

Non-finite values: `r_min_m` is genuinely `inf` for a straight alignment. It is written
as the literal text `inf` and, per section 3, counted separately and excluded from means
rather than coerced to a number. `objective_cost_ratio_to_baseline` and
`length_ratio_to_baseline` are left EMPTY for the `full_grid` arm rather than written as
1.0, so "1 by construction" is visible in the file instead of being a filtering rule the
reader has to know about.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence

INF_TEXT, NEG_INF_TEXT, NAN_TEXT = 'inf', '-inf', 'nan'

#: section 2.1 -- identity and provenance, on every row
IDENTITY_COLUMNS: tuple[str, ...] = (
    'scenario_id', 'terrain_id', 'terrain_seed', 'grid_size', 'periods',
    'height_interval', 'height_delta', 'od_id', 'od_seed', 'od_protocol',
    'start', 'target', 'road_class', 'road_class_status', 'road_class_source',
    'arm', 'rule', 'k_hops', 'cost_model', 'connectivity', 'algorithm_1_iterations',
    'metric_step_m', 'metric_chord_m',
    'timestamp_utc', 'git_sha', 'git_dirty', 'git_branch', 'python',
    'python_free_threaded', 'platform', 'numpy', 'scipy',
)

#: section 2.2 -- step 1, cost side
STEP1_COST_COLUMNS: tuple[str, ...] = (
    'n_areas_total', 'n_areas_selected', 'n_cells_selected', 'n_cells_total',
    'search_space_percent_of_full', 'search_space_reduction_percent', 'n_graph_edges',
    'search_nodes_expanded', 'search_nodes_pushed', 'search_nodes_reached',
    'wall_time_s_median', 'wall_time_s_min', 'wall_time_s_max', 'timing_repeats',
    'peak_memory_bytes', 'hag_build_time_s', 'time_reduction_percent',
)

#: section 2.3 -- step 1, quality side
STEP1_QUALITY_COLUMNS: tuple[str, ...] = (
    'path_found', 'infeasible_reason', 'n_path_cells', 'path_length_m',
    'objective_cost_sum_abs_dh_m', 'objective_cost_ratio_to_baseline',
    'length_ratio_to_baseline', 'vs_baseline_hausdorff_m',
    'vs_baseline_mean_deviation_m', 'vs_baseline_median_deviation_m',
    'vs_baseline_max_deviation_m',
)

_BEFORE_AFTER_METRICS: tuple[str, ...] = (
    'length_m', 'length_3d_m', 'sinuosity', 'r_min_m', 'i_max_percent',
    'i_min_signed_percent', 'i_max_signed_percent', 'elevation_net_m',
    'elevation_ascent_m', 'elevation_descent_m', 'elevation_span_m',
)

#: section 2.4 -- Algorithm 1 before/after
STEP3_COLUMNS: tuple[str, ...] = (
    ('n_representative_points', 's_value', 's_deviation_m', 's_budget_m',
     's_n_evaluations', 's_hit_cap', 's_degenerate')
    + tuple(f'before_{name}' for name in _BEFORE_AFTER_METRICS)
    + tuple(f'after_{name}' for name in _BEFORE_AFTER_METRICS)
    + ('deviation_hausdorff_m', 'deviation_mean_m', 'deviation_median_m',
       'deviation_max_m', 'delta_r_min_m', 'delta_i_max_percent', 'delta_length_m',
       'delta_sinuosity', 'smoothing_made_i_max_worse')
)

#: section 2.5 -- the engineering checks
STEP2_COLUMNS: tuple[str, ...] = (
    'check_r_min_value', 'check_r_min_limit', 'check_r_min_passed', 'check_r_min_note',
    'check_i_max_value', 'check_i_max_limit', 'check_i_max_passed', 'feasible', 'verdict',
)

MATRIX_COLUMNS: tuple[str, ...] = (
    IDENTITY_COLUMNS + STEP1_COST_COLUMNS + STEP1_QUALITY_COLUMNS
    + STEP3_COLUMNS + STEP2_COLUMNS
)

OD_PAIRS_COLUMNS: tuple[str, ...] = (
    'terrain_id', 'terrain_seed', 'od_seed', 'n_candidates_drawn', 'od_id',
    'start_i', 'start_j', 'target_i', 'target_j', 'start_area', 'target_area',
    'hag_hops', 'straight_line_m', 'diagonal_m', 'diagonal_fraction',
    'start_height_m', 'target_height_m', 'height_difference_m', 'protocol',
    'candidate_index',
    'rejected_too_short', 'rejected_same_height_area', 'rejected_hops_below_min',
    'rejected_no_full_grid_path', 'rejected_duplicate_start_area',
    'rejected_duplicate_target_area',
)


def fmt_value(value) -> str:
    """CSV text for one value. `None` is an empty cell; non-finite floats are words."""
    if value is None:
        return ''
    if isinstance(value, float):
        if math.isnan(value):
            return NAN_TEXT
        if value == math.inf:
            return INF_TEXT
        if value == -math.inf:
            return NEG_INF_TEXT
    if isinstance(value, (list, tuple)):
        return 'x'.join(str(v) for v in value)
    return str(value)


def parse_value(text: str):
    """Inverse of `fmt_value` for the numeric and boolean cells."""
    if text is None or text == '':
        return None
    if text == INF_TEXT:
        return math.inf
    if text == NEG_INF_TEXT:
        return -math.inf
    if text == NAN_TEXT:
        return math.nan
    if text in ('True', 'False'):
        return text == 'True'
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def write_csv(path, rows: Sequence[dict], columns: Sequence[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([fmt_value(row.get(name)) for name in columns])
    return path


def read_csv(path) -> list[dict]:
    """Every row of a CSV file, values parsed; a `.gz` file is read the same way."""
    path = Path(path)
    if path.suffix == '.gz':
        import gzip
        handle = gzip.open(path, 'rt', encoding='utf-8', newline='')
    else:
        handle = open(path, encoding='utf-8', newline='')
    with handle:
        return [{key: parse_value(value) for key, value in row.items()}
                for row in csv.DictReader(handle)]


def read_jsonl(path) -> list[dict]:
    """Every parseable line. A truncated last line (a crash mid-write) is skipped."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def write_path_csv(path, xyz_m) -> Path:
    """One alignment as `x_m,y_m,z_m`, so a figure can be redrawn without re-searching."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['x_m', 'y_m', 'z_m'])
        for point in xyz_m:
            writer.writerow([f'{float(v):.4f}' for v in point[:3]])
    return path


def path_csv_name(terrain_id: str, od_id: str, arm: str,
                  road_class: Optional[str] = None,
                  iterations: Optional[int] = None) -> str:
    """`<terrain>__<od>__<arm>__rough.csv`, or `...__<class>__it<n>.csv` when smoothed."""
    if road_class is None:
        return f'{terrain_id}__{od_id}__{arm}__rough.csv'
    return f'{terrain_id}__{od_id}__{arm}__{road_class}__it{iterations}.csv'


def _identity(scenario: dict, arm_row: dict, class_name: str,
              class_block: dict, iterations: int) -> dict:
    config, prov = scenario['config'], scenario['provenance']
    terrain = config.get('terrain', {})
    return {
        'scenario_id': scenario['scenario_id'],
        'terrain_id': scenario['terrain_id'],
        'terrain_seed': scenario.get('terrain_seed'),
        'grid_size': terrain.get('grid_size'),
        'periods': terrain.get('periods'),
        'height_interval': terrain.get('height_interval'),
        'height_delta': config.get('height_delta'),
        'od_id': scenario['od_id'],
        'od_seed': scenario.get('od_seed'),
        'od_protocol': scenario.get('od_protocol'),
        'start': config.get('start'),
        'target': config.get('target'),
        'road_class': class_name,
        'road_class_status': class_block.get('status'),
        'road_class_source': class_block.get('source'),
        'arm': arm_row.get('arm'),
        'rule': arm_row.get('rule'),
        'k_hops': arm_row.get('k_hops'),
        'cost_model': config.get('cost'),
        'connectivity': config.get('connectivity'),
        'algorithm_1_iterations': iterations,
        'metric_step_m': config.get('metric_step_m'),
        'metric_chord_m': config.get('metric_chord_m'),
        **{name: prov.get(name) for name in (
            'timestamp_utc', 'git_sha', 'git_dirty', 'git_branch', 'python',
            'python_free_threaded', 'platform', 'numpy', 'scipy')},
    }


def flatten_scenario(scenario: dict) -> list[dict]:
    """One nested (terrain, O-D) row -> one flat dict per (arm, class, iterations)."""
    rows: list[dict] = []
    hag_build = scenario.get('hag_build_time_s')

    for arm_name, arm_row in scenario['arms'].items():
        geometry = scenario.get('geometry', {}).get(arm_name, {})
        for class_name, per_class in geometry.items():
            for iterations_key, block in per_class.items():
                iterations = int(iterations_key)
                report = block.get('before_after') or {}
                checks = block.get('checks') or {}
                class_block = report.get('road_class', {}) or {}

                row = _identity(scenario, arm_row, class_name, class_block, iterations)
                row.update({name: arm_row.get(name) for name in STEP1_COST_COLUMNS})
                row['hag_build_time_s'] = hag_build
                row.update({name: arm_row.get(name) for name in STEP1_QUALITY_COLUMNS})
                rows.append(_fill_geometry(row, report, checks))

        if not geometry:      # an arm with no path still gets one row, per section 2.3
            row = _identity(scenario, arm_row, '', {}, 0)
            row.update({name: arm_row.get(name) for name in STEP1_COST_COLUMNS})
            row['hag_build_time_s'] = hag_build
            row.update({name: arm_row.get(name) for name in STEP1_QUALITY_COLUMNS})
            rows.append(row)
    return rows


def _fill_geometry(row: dict, report: dict, checks: dict) -> dict:
    if report:
        search = report.get('s_search', {})
        before, after = report.get('before', {}), report.get('after', {})
        deviation, delta = report.get('deviation', {}), report.get('delta', {})
        row.update({
            'n_representative_points': report.get('n_representative_points'),
            's_value': search.get('s_value'),
            's_deviation_m': search.get('deviation_m'),
            's_budget_m': search.get('budget_m'),
            's_n_evaluations': search.get('n_evaluations'),
            's_hit_cap': search.get('hit_cap'),
            's_degenerate': search.get('degenerate'),
            'deviation_hausdorff_m': deviation.get('hausdorff_m'),
            'deviation_mean_m': deviation.get('mean_deviation_m'),
            'deviation_median_m': deviation.get('median_deviation_m'),
            'deviation_max_m': deviation.get('max_deviation_m'),
            'delta_r_min_m': delta.get('r_min_m'),
            'delta_i_max_percent': delta.get('i_max_percent'),
            'delta_length_m': delta.get('length_m'),
            'delta_sinuosity': delta.get('sinuosity'),
        })
        row.update({f'before_{n}': before.get(n) for n in _BEFORE_AFTER_METRICS})
        row.update({f'after_{n}': after.get(n) for n in _BEFORE_AFTER_METRICS})
        worse = None
        if before.get('i_max_percent') is not None and after.get('i_max_percent') is not None:
            worse = bool(after['i_max_percent'] > before['i_max_percent'])
        row['smoothing_made_i_max_worse'] = worse

    if checks:
        per_check = checks.get('checks', {})
        r_min, i_max = per_check.get('R_min', {}), per_check.get('i_max', {})
        row.update({
            'check_r_min_value': r_min.get('value'),
            'check_r_min_limit': r_min.get('limit'),
            'check_r_min_passed': r_min.get('passed'),
            'check_r_min_note': r_min.get('note'),
            'check_i_max_value': i_max.get('value'),
            'check_i_max_limit': i_max.get('limit'),
            'check_i_max_passed': i_max.get('passed'),
            'feasible': checks.get('feasible'),
            'verdict': checks.get('verdict'),
        })
    return row


def matrix_rows(scenarios: Iterable[dict]) -> list[dict]:
    rows: list[dict] = []
    for scenario in scenarios:
        rows.extend(flatten_scenario(scenario))
    return rows
