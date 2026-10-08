"""Summary statistics for the robustness matrix.

Two populations, not one. Section 3 says the statistics are taken "over the 27 protocol
scenarios", but section 1 also says the four search arms are computed once per
(terrain, O-D) and reused across the three road classes. The search-space fraction,
the node counts, the runtime and the memory peak therefore do not depend on the road
class, and averaging them over 27 rows would count each of the 9 values three times:
`n` too large by 3x and the SD deflated. So:

* `P_search` -- the distinct (terrain, O-D, arm) rows, 9 x 4 = **36**. Used for the
  cost and quality statistics, section 3 items 1 to 7.
* `P_geom` -- (terrain, O-D, arm, class) at one iteration setting, 9 x 4 x 3 = **108**.
  Used for the geometry and check statistics, section 3 items 8 to 11.

Both populations, and why they differ, are printed in the summary header so a reader
never has to guess which `n` a table used.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from core.results_io import read_csv

#: below this many rows a statistic is reported with a warning, per section 3
MIN_ROWS_FOR_A_STATISTIC = 5

#: the guard of section 5: these must agree across every row before anything aggregates
BUILD_KEYS = ('git_sha', 'python', 'python_free_threaded')

#: r_min far beyond any real curve is a floating-point artefact, not a result (bug B25).
#: `core.metrics.menger_radii` returns literal `inf` only for EXACTLY collinear points;
#: near-collinear points (the common case once smoothing has flattened a curve) divide by
#: a tiny non-zero triangle area and return an astronomically large but technically FINITE
#: number instead -- an early run of this matrix recorded a class mean of 4.3e+14 m with a
#: max of 2.476e+15 m. `describe` already treats literal `inf` as "unbounded, exclude from
#: the mean" (`r_min_m` is genuinely infinite for a dead-straight alignment); this constant
#: extends the SAME treatment to values that are finite only by floating-point accident. No
#: `RoadClass` in the registry asks for a radius past 1000 m, so a bound three orders of
#: magnitude above that catches the artefact without ever touching a real result.
EFFECTIVELY_STRAIGHT_R_MIN_M = 1.0e6
R_MIN_QUANTITIES = ('before_r_min_m', 'after_r_min_m', 'delta_r_min_m')

#: below this baseline objective a ratio to it is noise, not a result (bug B22): a 19.3 m
#: difference over a 2.59 m baseline reads as +747%. `test.py` already applies this guard to
#: the real-DEM harness; the same threshold is used here so the two report consistently.
FLAT_BASELINE_OBJECTIVE_M = 25.0


def os_family(row: dict) -> str:
    """`'Windows-11-10.0.26200-SP0'` -> `'Windows'`; the OS half of the build guard.

    Derived from the `platform` string that `core.provenance.provenance()` already records,
    rather than stored as its own column, so the rows already committed in `results/` stay
    aggregatable and `core.results_io.COLUMNS` does not move.

    The OS matters because the same code does not give the same floats on both: numpy's
    `sin`/`cos` differ by up to 1 ULP between the Windows and the Linux build of one version, and wall times and memory peaks are even less comparable. The full
    `platform` string is deliberately NOT compared -- a kernel or build-number bump would
    then refuse an otherwise valid matrix.
    """
    return str(row.get('platform', '')).split('-')[0]


class ProvenanceMismatch(RuntimeError):
    """Rows from different builds. Timings from different interpreters are not comparable."""


@dataclass(frozen=True)
class Stat:
    name: str
    n: int
    mean: float
    sd: float
    minimum: float
    maximum: float
    median: float
    n_inf: int = 0
    n_nan: int = 0
    n_missing: int = 0
    note: str = ''

    def as_row(self) -> list[str]:
        def fmt(value: float) -> str:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return '-'
            return f'{value:.4g}'
        extra = []
        if self.n_inf:
            extra.append(f'{self.n_inf} inf')
        if self.n_missing:
            extra.append(f'{self.n_missing} missing')
        if self.note:
            extra.append(self.note)
        return [self.name, str(self.n), fmt(self.mean), fmt(self.sd), fmt(self.minimum),
                fmt(self.maximum), fmt(self.median), '; '.join(extra)]


def describe(values: Iterable, name: str) -> Stat:
    """mean / SD / min / max / median over the FINITE values only.

    Infinite values are counted and excluded rather than coerced: `r_min_m` is genuinely
    infinite for a straight alignment, and averaging that in would be meaningless.
    """
    raw = list(values)
    n_missing = sum(1 for v in raw if v is None)
    numbers = [float(v) for v in raw if isinstance(v, (int, float)) and not isinstance(v, bool)]
    n_inf = sum(1 for v in numbers if math.isinf(v))
    n_nan = sum(1 for v in numbers if math.isnan(v))
    finite = [v for v in numbers if math.isfinite(v)]

    if not finite:
        return Stat(name, 0, math.nan, math.nan, math.nan, math.nan, math.nan,
                    n_inf, n_nan, n_missing, 'no finite values')

    note = ''
    if len(finite) < MIN_ROWS_FOR_A_STATISTIC:
        note = f'n<{MIN_ROWS_FOR_A_STATISTIC}, see the study protocol'
    return Stat(name=name, n=len(finite), mean=statistics.fmean(finite),
                sd=statistics.stdev(finite) if len(finite) > 1 else math.nan,
                minimum=min(finite), maximum=max(finite),
                median=statistics.median(finite),
                n_inf=n_inf, n_nan=n_nan, n_missing=n_missing, note=note)


#: what makes two TIMINGS comparable on top of the build: how many workers shared the
#: machine and whether the run was a serial calibration. Counts are not affected by it.
TIMING_KEYS = ('timing_mode', 'n_workers')


def _timing_mode(row: dict):
    # rows written before the parallel runner existed were all serial
    return row.get('timing_mode') or 'serial'


def assert_one_build(rows: Sequence[dict], *, timing: bool = False) -> dict:
    """Refuse to aggregate rows from different builds. Returns the provenance block.

    With `timing=True` (whenever a time or memory column is summarised) it also refuses rows
    measured with a different number of concurrent workers or timing mode (`core/parallel.py`).
    """
    if not rows:
        raise ProvenanceMismatch('no rows to summarise')
    block = {}
    seen = {key: {row.get(key) for row in rows} for key in BUILD_KEYS}
    if timing:
        seen['timing_mode'] = {_timing_mode(row) for row in rows}
        seen['n_workers'] = {row.get('n_workers') or 1 for row in rows}
    seen['os_family'] = {os_family(row) for row in rows}
    if any('engine' in row for row in rows):
        # the compiled and the Python Dijkstra count the same work but not in the same time
        seen['engine'] = {row.get('engine') for row in rows}
    for key, values in seen.items():
        if len(values) > 1:
            raise ProvenanceMismatch(
                f"rows disagree on {key}: {sorted(str(v) for v in values)}. Timings and "
                f"counts from different builds must never share a table "
                f".")
        block[key] = values.pop()
    dirty = {row.get('git_dirty') for row in rows}
    block['git_dirty_values'] = sorted(str(d) for d in dirty)
    block['git_dirty_warning'] = len(dirty) > 1 or True in dirty
    for key in ('git_branch', 'platform', 'numpy', 'scipy', 'timestamp_utc'):
        block[key] = rows[0].get(key)
    return block


def _protocol_rows(rows: Sequence[dict]) -> list[dict]:
    """Protocol pairs only: the reference O-D pair and synthetic classes are excluded."""
    return [r for r in rows
            if r.get('od_protocol') is True
            and not str(r.get('road_class', '')).startswith('SYNTHETIC')]


def population_search(rows: Sequence[dict]) -> list[dict]:
    """One row per (terrain, O-D, arm): the cost statistics do not depend on the class."""
    seen, out = set(), []
    for row in _protocol_rows(rows):
        key = (row.get('terrain_id'), row.get('od_id'), row.get('arm'))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def population_geometry(rows: Sequence[dict], iterations: int = 1) -> list[dict]:
    """One row per (terrain, O-D, arm, class) at one iteration setting."""
    return [r for r in _protocol_rows(rows)
            if r.get('algorithm_1_iterations') == iterations and r.get('road_class')]


def _by(rows: Sequence[dict], key: str) -> dict:
    groups: dict = {}
    for row in rows:
        groups.setdefault(row.get(key), []).append(row)
    return dict(sorted(groups.items(), key=lambda item: str(item[0])))


def _censor_straight(value, bound: float = EFFECTIVELY_STRAIGHT_R_MIN_M):
    """`math.inf` for a real number past the straight-line artefact bound (B25); else as is."""
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and abs(value) > bound):
        return math.inf
    return value


def _baseline_objective_by_scenario(search_rows: Sequence[dict]) -> dict:
    """`{(terrain_id, od_id): full_grid's objective_cost_sum_abs_dh_m}`, the B22 flatness test."""
    return {(r.get('terrain_id'), r.get('od_id')): r.get('objective_cost_sum_abs_dh_m')
            for r in search_rows if r.get('arm') == 'full_grid'}


def _is_flat_baseline(row: dict, baseline_by_scenario: dict,
                      threshold: float = FLAT_BASELINE_OBJECTIVE_M) -> bool:
    baseline = baseline_by_scenario.get((row.get('terrain_id'), row.get('od_id')))
    return isinstance(baseline, (int, float)) and baseline < threshold


def _objective_abs_diff(row: dict, baseline_by_scenario: dict):
    """`objective - baseline`, meaningful even where the RATIO (B22) is not."""
    baseline = baseline_by_scenario.get((row.get('terrain_id'), row.get('od_id')))
    value = row.get('objective_cost_sum_abs_dh_m')
    if not isinstance(baseline, (int, float)) or not isinstance(value, (int, float)):
        return None
    return value - baseline


def _table(title: str, stats: Sequence[Stat]) -> list[str]:
    header = ['quantity', 'n', 'mean', 'SD', 'min', 'max', 'median', 'notes']
    lines = [f'### {title}', '', '| ' + ' | '.join(header) + ' |',
             '|' + '|'.join(['---'] * len(header)) + '|']
    for stat in stats:
        lines.append('| ' + ' | '.join(stat.as_row()) + ' |')
    lines.append('')
    return lines


COST_QUANTITIES = (
    'search_space_percent_of_full', 'search_space_reduction_percent',
    'search_nodes_expanded', 'wall_time_s_median', 'time_reduction_percent',
    'peak_memory_bytes',
)
QUALITY_QUANTITIES = (
    'objective_cost_ratio_to_baseline', 'length_ratio_to_baseline',
    'vs_baseline_hausdorff_m', 'vs_baseline_mean_deviation_m',
)
GEOMETRY_QUANTITIES = (
    'before_r_min_m', 'after_r_min_m', 'delta_r_min_m',
    'before_i_max_percent', 'after_i_max_percent', 'delta_i_max_percent',
    'deviation_hausdorff_m', 'deviation_mean_m',
    'before_sinuosity', 'after_sinuosity',
)


def render_markdown(rows: Sequence[dict], header_extra: Optional[dict] = None) -> str:
    """The whole of `results/summary.md`."""
    block = assert_one_build(rows)
    search_rows = population_search(rows)
    geom_rows = population_geometry(rows)
    header_extra = header_extra or {}

    lines: list[str] = ['# Robustness matrix: summary', '']
    lines += ['```',
              f"Produced: {block.get('timestamp_utc')}",
              f"Commit:   {block.get('git_sha')} on {block.get('git_branch')}, "
              f"{'DIRTY' if block.get('git_dirty_warning') else 'clean'}",
              f"Python:   {block.get('python')}, free-threaded: "
              f"{block.get('python_free_threaded')}",
              f"numpy {block.get('numpy')}, scipy {block.get('scipy')}, "
              f"platform {block.get('platform')}"]
    for key, value in header_extra.items():
        lines.append(f'{key:<10}{value}')
    lines += [f'Populations: P_search n={len(search_rows)} (terrain x O-D x arm; the '
              f'search does not depend on the road class, so averaging it per class '
              f'would triple-count), P_geom n={len(geom_rows)} (x road class, '
              f'iterations=1)',
              'Excluded from the statistics: the reference O-D pair; SYNTHETIC_* classes',
              '```', '']

    if block.get('git_dirty_warning'):
        lines += ['> **Warning:** at least one row was produced from a dirty working '
                  'tree. The table for the paper must come from a clean one '
                  '.', '']

    lines += ['## Cost of the search, per arm (P_search)', '']
    for arm, group in _by(search_rows, 'arm').items():
        lines += _table(f'arm `{arm}` (n={len(group)})',
                        [describe([r.get(q) for r in group], q) for q in COST_QUANTITIES])

    lines += ['## What the reduction costs, per arm (P_search)', '',
              'The `full_grid` arm is the baseline; its ratio columns are 1 by '
              'construction and are left empty in the CSV rather than averaged. '
              f'`objective_cost_ratio_to_baseline` excludes scenarios whose baseline is '
              f'below {FLAT_BASELINE_OBJECTIVE_M:g} m (bug B22: the ratio is noise on '
              f'near-flat ground); `objective_cost_abs_diff_m` reports every scenario, '
              f'flat ground included, because the absolute difference stays meaningful '
              f'there.', '']
    baseline_by_scenario = _baseline_objective_by_scenario(search_rows)
    for arm, group in _by(search_rows, 'arm').items():
        if arm == 'full_grid':
            continue
        flat = [r for r in group if _is_flat_baseline(r, baseline_by_scenario)]
        stats = []
        for q in QUALITY_QUANTITIES:
            if q == 'objective_cost_ratio_to_baseline':
                values = [r.get(q) for r in group
                         if not _is_flat_baseline(r, baseline_by_scenario)]
                stats.append(describe(values, f'{q} (baseline >= '
                                              f'{FLAT_BASELINE_OBJECTIVE_M:g} m only)'))
            else:
                stats.append(describe([r.get(q) for r in group], q))
        stats.append(describe([_objective_abs_diff(r, baseline_by_scenario) for r in group],
                              'objective_cost_abs_diff_m'))
        if flat:
            stats.append(Stat(name='(n excluded above: flat-baseline scenarios)',
                              n=len(flat), mean=math.nan, sd=math.nan, minimum=math.nan,
                              maximum=math.nan, median=math.nan,
                              note=f'baseline < {FLAT_BASELINE_OBJECTIVE_M:g} m (B22)'))
        lines += _table(f'arm `{arm}` (n={len(group)})', stats)

    lines += ['## Geometry before and after Algorithm 1, per road class (P_geom)', '',
              f'`before_r_min_m`, `after_r_min_m` and `delta_r_min_m` treat a value past '
              f'{EFFECTIVELY_STRAIGHT_R_MIN_M:.0g} m as the near-collinear floating-point '
              f'artefact it is (bug B25) and fold it into the same n_inf/exclusion '
              f'`describe` already applies to a genuinely straight alignment.', '']
    for road_class, group in _by(geom_rows, 'road_class').items():
        stats = []
        for q in GEOMETRY_QUANTITIES:
            values = ([_censor_straight(r.get(q)) for r in group] if q in R_MIN_QUANTITIES
                     else [r.get(q) for r in group])
            stats.append(describe(values, q))
        lines += _table(f'class `{road_class}` (n={len(group)})', stats)

    lines += ['## Engineering checks (P_geom)', '',
              'A class that cannot be met is reported, never relaxed or substituted.', '',
              '| class | n | R_min passes | i_max passes | feasible | smoothing made '
              'i_max worse | s degenerate |',
              '|---|---|---|---|---|---|---|']
    for road_class, group in _by(geom_rows, 'road_class').items():
        n = len(group)
        lines.append(
            f'| `{road_class}` | {n} '
            f'| {sum(1 for r in group if r.get("check_r_min_passed") is True)} '
            f'| {sum(1 for r in group if r.get("check_i_max_passed") is True)} '
            f'| {sum(1 for r in group if r.get("feasible") is True)} '
            f'| {sum(1 for r in group if r.get("smoothing_made_i_max_worse") is True)} '
            f'| {sum(1 for r in group if r.get("s_degenerate") is True)} |')
    lines.append('')

    no_path = [r for r in _protocol_rows(rows) if r.get('path_found') is False]
    lines += ['## Failures, reported rather than dropped', '',
              f'- rows where the selection produced no path at all: '
              f'{len({(r.get("terrain_id"), r.get("od_id"), r.get("arm")) for r in no_path})} '
              f'of {len(search_rows)} (P_search)',
              f'- rows with a degenerate s-value search (three or fewer representative '
              f'points, so no spline and no search): '
              f'{sum(1 for r in geom_rows if r.get("s_degenerate") is True)} '
              f'of {len(geom_rows)} (P_geom)', '']

    lines += ['---', '',
              'Every road class in this table is marked `TO CONFIRM`: the values come '
              'from Ordin MT 1296/2017 but have not been signed off by the road-design '
              'co-author. Nothing here is a validated result; the method is *evaluated* '
              'for computational efficiency, solution quality, geometric feasibility '
              'and robustness.', '']
    return '\n'.join(lines)


def summary_from_csv(csv_path, header_extra: Optional[dict] = None) -> str:
    return render_markdown(read_csv(csv_path), header_extra=header_extra)
