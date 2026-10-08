"""Algorithm 1 (path smoothing), driven entirely by a `RoadClass`.

One entry point, `algorithm_1(path_xy, road_class, grid=...)`. No prints, no figures, no
files, no literals: the minimum radius and the cell size come from the `RoadClass` and
from nothing else.

Design choices:

* the mid-point rule takes `road_class.r_min_m` and `math.pi`, never a fixed radius;
* the deviation budget is in meters and is converted with `road_class.cell_size_m`;
* the spline looseness is chosen by `search_s_value`, an exponential-then-bisection
  search on floats, so the deviation stays inside the budget;
* `remove_bad_points` is called with `strict=False`, so a path with two consecutive
  short lines is smoothed instead of raising `Exception('BAD1')`;
* smoothing is XY only, and Z is re-sampled from the terrain
  afterwards with bilinear interpolation (`Grid.path_xyz_m`), never carried over from
  the rough path. That is the `XY_smooth -> Z_new -> i_new` requirement: moving the
  alignment moves it to different ground, so the gradients must be recomputed.

Units: `path_xy` is in GRID units (the same (i, j) coordinates the search works in),
because that is what the interpolation helpers expect. Every reported quantity is in
meters.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.interpolate import interpolate_2d_path_v2, remove_bad_points
from core import metrics as metrics_mod
from core.grid import Grid
from data.configs.road_classes import RoadClass

logger = logging.getLogger(__name__)

#: points generated per representative point by the final B-spline
SPLINE_MULTIPLIER = 8
#: the deviation budget of the s-value search, as a fraction of `r_min_m`, in meters.
DEVIATION_FRACTION_OF_R_MIN = 0.1
#: the s-value search never goes above this, so a degenerate path cannot loop forever
MAX_S_VALUE = 2.0 ** 20
#: relative width at which the bisection stops
S_SEARCH_TOLERANCE = 1e-3


#: the B-spline degree the article uses; a short path has to drop below it
SPLINE_DEGREE = 3


def _interpolate(points_xy: np.ndarray, multiplier: int, s_value: float) -> np.ndarray:
    """B-spline through the points, with the degree lowered when there are too few.

    `splprep` needs strictly more points than the spline degree. A strict road class can
    collapse a path to three or fewer representative points, which used to raise
    `TypeError: m > k must hold` from deep inside scipy; the alignment is then reported
    as it is instead of crashing.
    """
    points_xy = np.asarray(points_xy, dtype=float)
    if len(points_xy) < 3:
        return points_xy
    degree = min(SPLINE_DEGREE, len(points_xy) - 1)
    return np.asarray(interpolate_2d_path_v2(points_xy, multiplier=multiplier,
                                             s_value=s_value, k=degree), dtype=float)


def insert_midpoints(path_xy: np.ndarray, road_class: RoadClass) -> np.ndarray:
    """Add the middle point of every long segment, so the spline has a point to bend around.

    A segment counts as long when its length exceeds `pi * r_min_m`, i.e. half the
    circumference of the smallest allowed curve. The radius is the class's.
    """
    path_xy = np.asarray(path_xy, dtype=float)
    if len(path_xy) < 3:
        return path_xy
    threshold_m = math.pi * road_class.r_min_m
    out = [path_xy[0]]
    for p1, p2 in zip(path_xy[1:], path_xy[2:]):
        out.append(p1)
        if np.linalg.norm(p2 - p1) * road_class.cell_size_m > threshold_m:
            out.append((p1 + p2) / 2)
    out.append(path_xy[-1])
    return np.array(out)


def select_representative_points(path_xy: np.ndarray, road_class: RoadClass) -> np.ndarray:
    """Algorithm 1, steps 0 to 2, plus the mid-point rule. XY only."""
    path_xy = np.asarray(path_xy, dtype=float)[:, :2]
    representative, removed = remove_bad_points(path_xy,
                                                minimal_radius=road_class.r_min_m,
                                                GRID_RATIO_TO_METERS=road_class.cell_size_m,
                                                strict=False)
    logger.debug('representative points: %d of %d, removed: %s', len(representative),
                 len(path_xy), {k: len(v) for k, v in removed.items()})
    return insert_midpoints(representative, road_class)


@dataclass(frozen=True)
class SValueSearch:
    """Outcome of the s-value search (Algorithm 1, part 2)."""

    s_value: float
    deviation_m: float
    budget_m: float
    n_evaluations: int
    hit_cap: bool
    #: True when there were too few representative points to fit a spline at all, so
    #: there was nothing to search and `s_value` is meaningless. It happens when a
    #: strict road class collapses the path to three points or fewer.
    degenerate: bool = False

    def as_dict(self) -> dict:
        return {'s_value': self.s_value, 'deviation_m': self.deviation_m,
                'budget_m': self.budget_m, 'n_evaluations': self.n_evaluations,
                'hit_cap': self.hit_cap, 'degenerate': self.degenerate}


def search_s_value(representative_xy: np.ndarray,
                   road_class: RoadClass,
                   max_deviation_m: Optional[float] = None,
                   multiplier: int = SPLINE_MULTIPLIER) -> SValueSearch:
    """The largest spline looseness whose deviation stays inside the budget.

    The reference is the tight interpolation (`s_value = 0`), which passes through the
    representative points. `s_value` is grown from 1 by doubling while the Hausdorff
    deviation from that reference is within budget, then bisected between the last good
    and the first bad value. The budget is
    `DEVIATION_FRACTION_OF_R_MIN * road_class.r_min_m` METERS.
    """
    representative_xy = np.asarray(representative_xy, dtype=float)
    budget_m = (DEVIATION_FRACTION_OF_R_MIN * road_class.r_min_m
                if max_deviation_m is None else float(max_deviation_m))
    if len(representative_xy) < SPLINE_DEGREE + 1:
        # nothing to fit: every s_value gives the same polyline, so the deviation is
        # identically zero and the search would run to the cap on meaningless values.
        logger.debug('only %d representative points, the s-value search is skipped',
                     len(representative_xy))
        return SValueSearch(s_value=0.0, deviation_m=0.0, budget_m=budget_m,
                            n_evaluations=0, hit_cap=False, degenerate=True)
    tight = _interpolate(representative_xy, multiplier, 0.0)
    evaluations = 0

    def deviation_m(s_value: float) -> float:
        nonlocal evaluations
        evaluations += 1
        loose = _interpolate(representative_xy, multiplier, s_value)
        return metrics_mod.hausdorff_m(np.asarray(tight) * road_class.cell_size_m,
                                       np.asarray(loose) * road_class.cell_size_m)

    good, good_dev = 0.0, 0.0
    candidate = 1.0
    hit_cap = False
    while candidate <= MAX_S_VALUE:
        dev = deviation_m(candidate)
        if dev > budget_m:
            break
        good, good_dev = candidate, dev
        candidate *= 2.0
    else:
        hit_cap = True

    if not hit_cap and candidate <= MAX_S_VALUE:
        low, high = good, candidate
        while high - low > S_SEARCH_TOLERANCE * max(high, 1.0):
            mid = 0.5 * (low + high)
            dev = deviation_m(mid)
            if dev > budget_m:
                high = mid
            else:
                low, good_dev = mid, dev
        good = low

    return SValueSearch(s_value=float(good), deviation_m=float(good_dev),
                        budget_m=budget_m, n_evaluations=evaluations, hit_cap=hit_cap)


@dataclass(frozen=True)
class Algorithm1Result:
    """XY smoothing, then Z re-sampled from the terrain."""

    representative_xy: np.ndarray
    smooth_xy: np.ndarray
    smooth_xyz_m: Optional[np.ndarray]
    s_search: SValueSearch
    iterations: int
    road_class_name: str

    @property
    def s_value(self) -> float:
        return self.s_search.s_value


def algorithm_1(path_xy: np.ndarray,
                road_class: RoadClass,
                grid: Optional[Grid] = None,
                iterations: int = 1,
                multiplier: int = SPLINE_MULTIPLIER,
                max_deviation_m: Optional[float] = None) -> Algorithm1Result:
    """Smooth a rough path in XY, then re-sample Z from the terrain.

    Args:
        path_xy: the rough path in grid coordinates, (n, 2) or (n, 3). Only XY is used.
        road_class: the only source of `r_min_m` and `cell_size_m`.
        grid: the terrain. Given, the result carries `smooth_xyz_m`; omitted, only XY.
        iterations: how many times steps 0 to 3 are repeated (the method repeats them).
        max_deviation_m: override the s-value search budget.
    """
    if iterations < 1:
        raise ValueError('iterations must be >= 1')
    current = np.asarray(path_xy, dtype=float)[:, :2]
    representative = current
    search = SValueSearch(0.0, 0.0, 0.0, 0, False)
    for _ in range(int(iterations)):
        representative = select_representative_points(current, road_class)
        search = search_s_value(representative, road_class,
                                max_deviation_m=max_deviation_m, multiplier=multiplier)
        current = _interpolate(representative, multiplier, search.s_value)
    xyz = grid.path_xyz_m(current) if grid is not None else None
    return Algorithm1Result(representative_xy=representative, smooth_xy=current,
                            smooth_xyz_m=xyz, s_search=search, iterations=int(iterations),
                            road_class_name=road_class.name)


def before_after(grid: Grid,
                 rough_path_xy: np.ndarray,
                 road_class: RoadClass,
                 iterations: int = 1,
                 metric_step_m: float = metrics_mod.DEFAULT_RESAMPLE_STEP_M,
                 metric_chord_m: float = metrics_mod.DEFAULT_CHORD_M) -> dict:
    """Research step 3: the before/after table for one case.

    "Before" is the rough search output with Z taken from the terrain; "after" is the
    smoothed axis with Z RE-SAMPLED from the terrain, so the gradients really are the
    gradients of the new alignment (`XY_smooth -> Z_new -> i_new`).
    """
    if not math.isclose(grid.cell_size_m, road_class.cell_size_m, rel_tol=1e-9):
        raise ValueError(
            f"grid cell {grid.cell_size_m} m != road class cell {road_class.cell_size_m} m. "
            f"Algorithm 1 converts between grid units and metres with the ROAD CLASS's "
            f"cell size, so the two must agree; call "
            f"road_class.with_cell_size({grid.cell_size_m}) first. On the 10 m synthetic "
            f"terrains they coincide, but a DEM grid does not.")

    rough_xy = np.asarray(rough_path_xy, dtype=float)[:, :2]
    before_xyz = grid.path_xyz_m(rough_xy)
    result = algorithm_1(rough_xy, road_class, grid=grid, iterations=iterations)
    after_xyz = result.smooth_xyz_m

    before = metrics_mod.measure(before_xyz, step_m=metric_step_m, chord_m=metric_chord_m)
    after = metrics_mod.measure(after_xyz, step_m=metric_step_m, chord_m=metric_chord_m)
    dev = metrics_mod.deviation(after_xyz[:, :2], before_xyz[:, :2], step_m=metric_step_m)

    return {
        'road_class': {'name': road_class.name, 'source': road_class.source,
                       'status': road_class.status, 'r_min_m': road_class.r_min_m,
                       'i_max_percent': road_class.i_max_percent,
                       'cell_size_m': road_class.cell_size_m},
        'iterations': result.iterations,
        's_search': result.s_search.as_dict(),
        'n_representative_points': int(len(result.representative_xy)),
        'before': before.as_dict(),
        'after': after.as_dict(),
        'deviation': dev,
        'delta': {
            'r_min_m': after.r_min_m - before.r_min_m,
            'i_max_percent': after.i_max_percent - before.i_max_percent,
            'length_m': after.length_m - before.length_m,
            'sinuosity': after.sinuosity - before.sinuosity,
        },
        'r_min_meets_class': bool(after.r_min_m >= road_class.r_min_m),
        'i_max_meets_class': bool(after.i_max_percent <= road_class.i_max_percent),
        'warnings': _warnings(before, after, road_class),
    }


def _warnings(before: metrics_mod.PathMetrics, after: metrics_mod.PathMetrics,
              road_class: RoadClass) -> list[str]:
    out: list[str] = []
    if after.r_min_m < road_class.r_min_m:
        out.append(
            f"R_min after smoothing is {after.r_min_m:.1f} m, below the {road_class.r_min_m:.0f} m "
            f"required by {road_class.name}; the alignment is infeasible under this class "
            f"(NOT relaxed)")
    if after.i_max_percent is not None and after.i_max_percent > road_class.i_max_percent:
        out.append(
            f"i_max after smoothing is {after.i_max_percent:.2f}%, above the "
            f"{road_class.i_max_percent}% of {road_class.name}")
    if (before.i_max_percent is not None and after.i_max_percent is not None
            and after.i_max_percent > before.i_max_percent):
        out.append(
            f"smoothing made the longitudinal geometry worse: i_max went from "
            f"{before.i_max_percent:.2f}% to {after.i_max_percent:.2f}% because the axis "
            f"moved onto different ground")
    return out


def before_after_guarded(grid: Grid,
                         rough_path_xy: np.ndarray,
                         road_class: RoadClass,
                         iterations: int = 1,
                         metric_step_m: float = metrics_mod.DEFAULT_RESAMPLE_STEP_M,
                         metric_chord_m: float = metrics_mod.DEFAULT_CHORD_M) -> dict:
    """`before_after`, plus a gradient-regression guard (bug B23).

    Algorithm 1 is free to move the axis onto different ground, and research step 3's own
    question is whether that ever makes the longitudinal profile WORSE. On the Idrija crop
    of the study protocol it does: a certified 986 m alignment holding 6.89% smooths to 669 m
    at 14.70%, more than twice the 7% limit, because the XY smoothing shortcuts the
    development length while Z is re-sampled from the terrain. `before_after`'s own
    `_warnings` already reports this per scenario ("smoothing made the longitudinal
    geometry worse"); this function adds the one decision that observation implies.

    When smoothing takes an `i_max` that MET the class and makes it NOT meet the class, the
    smoothed axis is flagged `smoothing_rejected` and `retained` holds the metrics of the
    axis that should actually be reported as this scenario's result -- the rough one, since
    trading a compliant longitudinal profile for a non-compliant one is a worse outcome by
    the class's own standard, whatever it did to the plan-view curvature. The original
    `before`/`after` pair is returned UNCHANGED, so this is a reported mitigation applied on
    top of the measurement, never a silent substitution inside it.

    A scenario that was already non-compliant before smoothing, or stays compliant after,
    is never rejected: the guard only fires on a compliant-to-non-compliant regression,
    which is the specific failure mode B23 names.
    """
    report = dict(before_after(grid, rough_path_xy, road_class, iterations=iterations,
                               metric_step_m=metric_step_m, metric_chord_m=metric_chord_m))
    before_i = report['before'].get('i_max_percent')
    after_i = report['after'].get('i_max_percent')
    was_compliant = before_i is not None and before_i <= road_class.i_max_percent
    now_nonCompliant = after_i is not None and after_i > road_class.i_max_percent
    rejected = bool(was_compliant and now_nonCompliant)

    report['smoothing_rejected'] = rejected
    report['retained_source'] = 'before' if rejected else 'after'
    report['retained'] = report['before'] if rejected else report['after']
    if rejected:
        report['warnings'] = list(report['warnings']) + [
            f"smoothing REJECTED: i_max was compliant at {before_i:.2f}% and smoothing "
            f"took it to {after_i:.2f}%, above {road_class.i_max_percent}%; the rough "
            f"axis is retained for this scenario's result (B23 guard)"]
    return report


def format_before_after(report: dict) -> str:
    """The before/after table as text."""
    rows = [('length (m)', 'length_m', '{:.1f}'),
            ('3D length (m)', 'length_3d_m', '{:.1f}'),
            ('sinuosity', 'sinuosity', '{:.3f}'),
            ('R_min (m)', 'r_min_m', '{:.1f}'),
            ('i_max (%)', 'i_max_percent', '{:.2f}'),
            ('ascent (m)', 'elevation_ascent_m', '{:.1f}'),
            ('descent (m)', 'elevation_descent_m', '{:.1f}')]
    rc = report['road_class']
    lines = [f"Algorithm 1 before/after - road class {rc['name']} ({rc['status']})",
             f"  requires R >= {rc['r_min_m']:.0f} m, i <= {rc['i_max_percent']}%, "
             f"cell {rc['cell_size_m']:.0f} m",
             (f"  the class left only {report['n_representative_points']} representative "
              f"points, too few for a cubic spline: no s-value search, the axis is the "
              f"polyline through them" if report['s_search']['degenerate'] else
              f"  s_value = {report['s_search']['s_value']:.4f} "
              f"(deviation {report['s_search']['deviation_m']:.2f} m of a "
              f"{report['s_search']['budget_m']:.2f} m budget, "
              f"{report['s_search']['n_evaluations']} evaluations)"),
             '',
             f"{'measure':<16}{'before':>12}{'after':>12}{'change':>12}",
             '-' * 52]
    for label, key, fmt in rows:
        b, a = report['before'].get(key), report['after'].get(key)
        if b is None or a is None:
            continue
        lines.append(f"{label:<16}{fmt.format(b):>12}{fmt.format(a):>12}"
                     f"{fmt.format(a - b):>12}")
    lines += ['-' * 52,
              f"deviation from the rough path: Hausdorff "
              f"{report['deviation']['hausdorff_m']:.1f} m, mean "
              f"{report['deviation']['mean_deviation_m']:.1f} m",
              f"meets the class: R_min {report['r_min_meets_class']}, "
              f"i_max {report['i_max_meets_class']}"]
    lines += ['WARNING: ' + w for w in report['warnings']]
    return '\n'.join(lines)


# ==========================================================================================
# Enhancements (deliverable (b) of the article)
#
# Everything above keeps its behaviour exactly. Each enhancement below is a separate,
# OPT-IN arm, so the committed results stay valid and so the article
# can report each one's benefit against its cost in deviation. The arms are:
#
#   none                the raw search output, unsmoothed (the runner's business, not this
#                       module's)
#   baseline            `before_after_guarded`: today's Algorithm 1 with the B23
#                       reject-on-gradient-regression guard
#   gradient_aware      E1: on a gradient regression, keep more of the rough axis and
#                       re-smooth, instead of discarding the smoothed axis. It aims at
#                       the class's i_max, or at the rough axis's own i_max where the
#                       ground was already steeper than the class allows - a smoothing
#                       pass should never be the reason the profile got worse.
#
# WHY E1 BACKS OFF THE SELECTION RADIUS AND NOT THE DEVIATION BUDGET. Smoothing raises
# i_max because it SHORTENS the axis: on one 100x100 scenario a 1742 m rough path becomes
# 1412 m, and the same elevation difference over a shorter development length is a steeper
# road. The deviation budget only controls how loose the final spline is (step 3); the
# shortening happens earlier, when `remove_bad_points` drops points (steps 0 to 2), and
# tightening the spline cannot put them back - measured: shrinking the budget eight times
# moved i_max from 12.12% to 12.19%, the wrong way. The knob that does control the
# development length is the radius the point selection is run at, so that is what backs
# off. The cost is curvature: keeping more points keeps sharper corners, which is exactly
# the trade-off this arm exists to quantify.
#   per_element_budget  E2: judge the deviation inside each tangent and each curve against
#                       its own budget, instead of one global Hausdorff distance
#   min_tangent         E3: merge two curves separated by a tangent shorter than the
#                       class's minimum alignment length
#   radius_iterate      E4: grow the budget until R_min meets the class, and report what
#                       that cost in deviation
#   smooth_z            E5: replace the terrain-following profile by the designed grade
#                       line, then re-derive the gradients from it
#
# WHAT E5 IMPLIES. Taking Z from the grade line rather than from the ground means earth is
# moved: the difference between the two profiles is cut and fill. This module reports the
# geometry, and does NOT cost the earthworks, which is outside the scope of this study. An E5 result is a vertical-geometry statement, not a buildability statement.
# ==========================================================================================

#: the deviation budget on a curve, as a fraction of `r_min_m` (E2)
CURVE_BUDGET_FRACTION = 0.1
#: the deviation budget on a tangent, as a fraction of `r_min_m` (E2). Looser: moving a
#: straight stretch sideways costs nothing geometrically, bending a curve does.
TANGENT_BUDGET_FRACTION = 0.3
#: how far the point-selection radius is cut on each gradient back-off (E1)
BACKOFF_FACTOR = 0.5
#: how far the budget is grown on each radius iteration (E4)
GROWTH_FACTOR = 2.0
#: cap on either loop, so a path that cannot be fixed stops instead of running forever
MAX_ADJUSTMENTS = 8


@dataclass(frozen=True)
class Enhancements:
    """Which Algorithm 1 enhancements are switched on. All default to off."""

    name: str = 'baseline'
    gradient_aware: bool = False
    per_element_budget: bool = False
    enforce_min_tangent: bool = False
    radius_iterate: bool = False
    smooth_z: bool = False
    max_adjustments: int = MAX_ADJUSTMENTS

    @property
    def any_enabled(self) -> bool:
        return any((self.gradient_aware, self.per_element_budget,
                    self.enforce_min_tangent, self.radius_iterate, self.smooth_z))


#: one arm per enhancement, plus the untouched baseline. The runner sweeps these.
ARMS: dict[str, Enhancements] = {
    'baseline': Enhancements(name='baseline'),
    'gradient_aware': Enhancements(name='gradient_aware', gradient_aware=True),
    'per_element_budget': Enhancements(name='per_element_budget', per_element_budget=True),
    'min_tangent': Enhancements(name='min_tangent', enforce_min_tangent=True),
    'radius_iterate': Enhancements(name='radius_iterate', radius_iterate=True),
    'smooth_z': Enhancements(name='smooth_z', smooth_z=True),
}

#: the order arms are reported in, everywhere.
ARM_ORDER: tuple[str, ...] = ('none', 'baseline', 'gradient_aware', 'per_element_budget',
                              'min_tangent', 'radius_iterate', 'smooth_z')


def _to_meters(path_xy_grid: np.ndarray, road_class: RoadClass) -> np.ndarray:
    return np.asarray(path_xy_grid, dtype=float)[:, :2] * road_class.cell_size_m


# -- E2: a deviation budget per alignment element ------------------------------------------

def _point_to_polyline_m(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest place on the polyline, not to its vertices.

    Measuring to the vertices instead would report a deviation of half the vertex spacing
    between a polyline and itself, which is several metres at spline resolution.
    """
    a, b = polyline[:-1], polyline[1:]
    ab = b - a
    denominator = (ab * ab).sum(axis=1)
    denominator[denominator == 0.0] = 1.0
    t = (((points[:, None, :] - a[None, :, :]) * ab[None, :, :]).sum(axis=2)
         / denominator[None, :])
    np.clip(t, 0.0, 1.0, out=t)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    diff = points[:, None, :] - closest
    return np.sqrt((diff * diff).sum(axis=2)).min(axis=1)


def element_deviations_m(tight_xy_grid: np.ndarray,
                         loose_xy_grid: np.ndarray,
                         road_class: RoadClass,
                         step_m: float = metrics_mod.DEFAULT_RESAMPLE_STEP_M,
                         chord_m: float = metrics_mod.DEFAULT_CHORD_M) -> list[tuple[str, float]]:
    """How far the loose spline moved away from the tight one, inside each element.

    The tight interpolation is decomposed into tangents and curves, and each element
    reports the largest distance from its own stations to the loose polyline. One global
    Hausdorff distance cannot tell a 2 m shift of a long straight (harmless) from a 2 m
    shift inside a curve (which changes its radius); this can.
    """
    from core import alignment as alignment_mod

    tight_m = _to_meters(tight_xy_grid, road_class)
    loose_m = _to_meters(loose_xy_grid, road_class)
    if len(tight_m) < 3 or len(loose_m) < 2:
        return []
    stations, chainage = alignment_mod.resample_xyz(
        np.column_stack([tight_m, np.zeros(len(tight_m))]), step_m)
    radii = alignment_mod._radii_on_stations(stations[:, :2], step_m, chord_m)
    is_curve = alignment_mod.classify_stations(radii, chainage, chord_m)
    elements = alignment_mod.horizontal_elements(is_curve, chainage, radii)
    nearest = _point_to_polyline_m(stations[:, :2], loose_m)
    return [(e.kind, float(nearest[e.i_start:e.i_end + 1].max())) for e in elements]


def search_s_value_per_element(representative_xy: np.ndarray,
                               road_class: RoadClass,
                               multiplier: int = SPLINE_MULTIPLIER,
                               curve_fraction: float = CURVE_BUDGET_FRACTION,
                               tangent_fraction: float = TANGENT_BUDGET_FRACTION
                               ) -> SValueSearch:
    """`search_s_value`, but every element must stay inside its own budget (E2).

    Same doubling-then-bisection search on the same reference; only the acceptance test
    changes, from one Hausdorff distance to the worst element-relative overrun.
    """
    representative_xy = np.asarray(representative_xy, dtype=float)
    curve_budget = curve_fraction * road_class.r_min_m
    tangent_budget = tangent_fraction * road_class.r_min_m
    if len(representative_xy) < SPLINE_DEGREE + 1:
        return SValueSearch(s_value=0.0, deviation_m=0.0, budget_m=curve_budget,
                            n_evaluations=0, hit_cap=False, degenerate=True)
    tight = _interpolate(representative_xy, multiplier, 0.0)
    evaluations = 0

    def overrun(s_value: float) -> tuple[float, float]:
        """(worst budget-relative overrun, worst absolute deviation) at this s_value."""
        nonlocal evaluations
        evaluations += 1
        loose = _interpolate(representative_xy, multiplier, s_value)
        per_element = element_deviations_m(tight, loose, road_class)
        if not per_element:
            return 0.0, 0.0
        ratios = [d / (curve_budget if kind == 'curve' else tangent_budget)
                  for kind, d in per_element]
        return max(ratios), max(d for _, d in per_element)

    good, good_dev = 0.0, 0.0
    candidate, hit_cap = 1.0, False
    while candidate <= MAX_S_VALUE:
        ratio, dev = overrun(candidate)
        if ratio > 1.0:
            break
        good, good_dev = candidate, dev
        candidate *= 2.0
    else:
        hit_cap = True

    if not hit_cap and candidate <= MAX_S_VALUE:
        low, high = good, candidate
        while high - low > S_SEARCH_TOLERANCE * max(high, 1.0):
            mid = 0.5 * (low + high)
            ratio, dev = overrun(mid)
            if ratio > 1.0:
                high = mid
            else:
                low, good_dev = mid, dev
        good = low

    return SValueSearch(s_value=float(good), deviation_m=float(good_dev),
                        budget_m=curve_budget, n_evaluations=evaluations, hit_cap=hit_cap)


# -- E3: merge curves separated by too short a tangent --------------------------------------

def short_interior_tangents(path_xy_grid: np.ndarray,
                            road_class: RoadClass,
                            step_m: float = metrics_mod.DEFAULT_RESAMPLE_STEP_M,
                            chord_m: float = metrics_mod.DEFAULT_CHORD_M) -> list[tuple[float, float]]:
    """Chainage spans of every tangent between two curves shorter than the class minimum.

    An empty list when the class sets no minimum alignment length, or when the alignment
    has no interior tangent at all.
    """
    from core import alignment as alignment_mod

    if road_class.l_alignment_min_m is None:
        return []
    xy_m = _to_meters(path_xy_grid, road_class)
    if len(xy_m) < 3:
        return []
    geometry = alignment_mod.measure_alignment(
        np.column_stack([xy_m, np.zeros(len(xy_m))]), road_class,
        step_m=step_m, chord_m=chord_m)
    elements = geometry.elements
    return [(e.s_start_m, e.s_end_m)
            for k, e in enumerate(elements)
            if e.kind == 'tangent' and 0 < k < len(elements) - 1
            and e.length_m < road_class.l_alignment_min_m]


def _drop_points_in_spans(representative_xy: np.ndarray,
                          spans_m: list[tuple[float, float]],
                          road_class: RoadClass) -> np.ndarray:
    """Remove the representative points holding a too-short tangent apart.

    Without a point inside it, the spline sweeps the two neighbouring curves into one,
    which is the merge the standard's minimum alignment length asks for. The two end
    points are always kept, and enough points are kept for a cubic spline.
    """
    xy_m = _to_meters(representative_xy, road_class)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy_m, axis=0), axis=1))])
    keep = np.ones(len(representative_xy), dtype=bool)
    for start, end in spans_m:
        keep &= ~((s > start) & (s < end))
    keep[0] = keep[-1] = True
    if keep.sum() < SPLINE_DEGREE + 1:
        return representative_xy
    return np.asarray(representative_xy, dtype=float)[keep]


# -- E5: take Z from the designed grade line rather than from the ground ---------------------

def grade_line_xyz_m(path_xyz_m: np.ndarray, road_class: RoadClass,
                     step_m: float = metrics_mod.DEFAULT_RESAMPLE_STEP_M) -> np.ndarray:
    """The alignment with Z replaced by the designed grade line (E5).

    Returns the path unchanged when the class defines no design step. The difference
    between the two profiles is cut and fill, which this module does not cost.
    """
    from core import alignment as alignment_mod

    if road_class.l_design_step_min_m is None:
        return np.asarray(path_xyz_m, dtype=float)
    stations, chainage = alignment_mod.resample_xyz(path_xyz_m, step_m)
    line = alignment_mod.grade_line(chainage, stations[:, 2], road_class.l_design_step_min_m)
    z = np.interp(chainage, line.s_design_m, line.z_design_m)
    return np.column_stack([stations[:, 0], stations[:, 1], z])


# -- the enhanced driver --------------------------------------------------------------------

@dataclass(frozen=True)
class EnhancedResult:
    """One enhanced smoothing run: the axis, and what the enhancements did to get it."""

    result: Algorithm1Result
    smooth_xyz_m: np.ndarray
    arm: str
    budget_m: float
    selection_radius_m: float
    n_adjustments: int
    hit_adjustment_cap: bool
    gradient_recovered: Optional[bool]
    radius_compliant: Optional[bool]
    n_short_tangents_before: Optional[int]
    n_short_tangents_after: Optional[int]
    z_from_grade_line: bool

    def as_dict(self) -> dict:
        return {'arm': self.arm, 'budget_m': self.budget_m,
                'selection_radius_m': self.selection_radius_m,
                'n_adjustments': self.n_adjustments,
                'hit_adjustment_cap': self.hit_adjustment_cap,
                'gradient_recovered': self.gradient_recovered,
                'radius_compliant': self.radius_compliant,
                'n_short_tangents_before': self.n_short_tangents_before,
                'n_short_tangents_after': self.n_short_tangents_after,
                'z_from_grade_line': self.z_from_grade_line,
                's_search': self.result.s_search.as_dict(),
                'n_representative_points': int(len(self.result.representative_xy))}


def _smooth_once(path_xy: np.ndarray, road_class: RoadClass, grid: Grid,
                 iterations: int, multiplier: int, budget_m: float,
                 per_element: bool,
                 selection_radius_m: Optional[float] = None) -> Algorithm1Result:
    """One pass of steps 0 to 3 with the requested budget and acceptance test.

    `selection_radius_m` overrides the radius steps 0 to 2 remove points at, WITHOUT
    changing the class the result is judged against. A smaller one keeps more of the rough
    axis, and with it more development length. It is E1's lever.
    """
    import dataclasses

    selection_class = (road_class if selection_radius_m is None else
                       dataclasses.replace(road_class, r_min_m=float(selection_radius_m)))
    current = np.asarray(path_xy, dtype=float)[:, :2]
    representative = current
    search = SValueSearch(0.0, 0.0, budget_m, 0, False)
    for _ in range(int(iterations)):
        representative = select_representative_points(current, selection_class)
        search = (search_s_value_per_element(representative, road_class, multiplier)
                  if per_element else
                  search_s_value(representative, road_class, max_deviation_m=budget_m,
                                 multiplier=multiplier))
        current = _interpolate(representative, multiplier, search.s_value)
    return Algorithm1Result(representative_xy=representative, smooth_xy=current,
                            smooth_xyz_m=grid.path_xyz_m(current), s_search=search,
                            iterations=int(iterations), road_class_name=road_class.name)


def algorithm_1_enhanced(grid: Grid,
                         rough_path_xy: np.ndarray,
                         road_class: RoadClass,
                         enhancements: Enhancements = Enhancements(),
                         iterations: int = 1,
                         multiplier: int = SPLINE_MULTIPLIER,
                         max_deviation_m: Optional[float] = None) -> EnhancedResult:
    """Algorithm 1 with the requested enhancements. With none, it is `algorithm_1` itself.

    E1 and E4 both move the deviation budget, in opposite directions, and are applied in
    that order: E1 shrinks it until the gradient stops regressing, E4 grows it until the
    radius complies. An arm that enables both would fight itself, which is why each is its
    own arm in `ARMS`.
    """
    base_budget = (DEVIATION_FRACTION_OF_R_MIN * road_class.r_min_m
                   if max_deviation_m is None else float(max_deviation_m))
    rough_xy = np.asarray(rough_path_xy, dtype=float)[:, :2]
    before_i = metrics_mod.i_max_percent(grid.path_xyz_m(rough_xy))
    # what E1 aims at: meet the class, and where the rough axis already failed it, at
    # least do not make the longitudinal profile worse than the ground already was.
    gradient_target = max(road_class.i_max_percent, before_i)

    budget = base_budget
    selection_radius = road_class.r_min_m
    adjustments, hit_cap = 0, False
    result = _smooth_once(rough_xy, road_class, grid, iterations, multiplier, budget,
                          enhancements.per_element_budget)

    if enhancements.gradient_aware:
        while metrics_mod.i_max_percent(result.smooth_xyz_m) > gradient_target:
            if adjustments >= enhancements.max_adjustments:
                hit_cap = True
                break
            selection_radius *= BACKOFF_FACTOR
            adjustments += 1
            result = _smooth_once(rough_xy, road_class, grid, iterations, multiplier,
                                  budget, enhancements.per_element_budget,
                                  selection_radius_m=selection_radius)
    gradient_recovered = (
        bool(metrics_mod.i_max_percent(result.smooth_xyz_m) <= gradient_target)
        if enhancements.gradient_aware else None)

    if enhancements.radius_iterate:
        while metrics_mod.r_min_m(result.smooth_xyz_m) < road_class.r_min_m:
            if adjustments >= enhancements.max_adjustments:
                hit_cap = True
                break
            budget *= GROWTH_FACTOR
            adjustments += 1
            result = _smooth_once(rough_xy, road_class, grid, iterations, multiplier,
                                  budget, enhancements.per_element_budget,
                                  selection_radius_m=selection_radius)
    radius_compliant = (
        bool(metrics_mod.r_min_m(result.smooth_xyz_m) >= road_class.r_min_m)
        if enhancements.radius_iterate else None)

    short_before = short_after = None
    if enhancements.enforce_min_tangent:
        short_before = len(short_interior_tangents(result.smooth_xy, road_class))
        spans = short_interior_tangents(result.smooth_xy, road_class)
        if spans:
            representative = _drop_points_in_spans(result.representative_xy, spans,
                                                   road_class)
            if len(representative) != len(result.representative_xy):
                search = search_s_value(representative, road_class,
                                        max_deviation_m=budget, multiplier=multiplier)
                merged = _interpolate(representative, multiplier, search.s_value)
                result = Algorithm1Result(
                    representative_xy=representative, smooth_xy=merged,
                    smooth_xyz_m=grid.path_xyz_m(merged), s_search=search,
                    iterations=result.iterations, road_class_name=road_class.name)
                adjustments += 1
        short_after = len(short_interior_tangents(result.smooth_xy, road_class))

    xyz = result.smooth_xyz_m
    if enhancements.smooth_z:
        xyz = grade_line_xyz_m(xyz, road_class)

    return EnhancedResult(
        result=result, smooth_xyz_m=xyz, arm=enhancements.name, budget_m=budget,
        selection_radius_m=selection_radius,
        n_adjustments=adjustments, hit_adjustment_cap=hit_cap,
        gradient_recovered=gradient_recovered, radius_compliant=radius_compliant,
        n_short_tangents_before=short_before, n_short_tangents_after=short_after,
        z_from_grade_line=bool(enhancements.smooth_z
                               and road_class.l_design_step_min_m is not None))


def before_after_enhanced(grid: Grid,
                          rough_path_xy: np.ndarray,
                          road_class: RoadClass,
                          enhancements: Enhancements = Enhancements(),
                          iterations: int = 1,
                          metric_step_m: float = metrics_mod.DEFAULT_RESAMPLE_STEP_M,
                          metric_chord_m: float = metrics_mod.DEFAULT_CHORD_M) -> dict:
    """The before/after report of one enhancement arm, with its per-station STAS checks.

    Same shape as `before_after`, plus an `enhancements` block describing what the arm
    did and a `stations` block holding the per-station verdict of every STAS parameter.
    With no enhancement enabled this reproduces `before_after_guarded`'s measurements.
    """
    from core import checks as checks_mod

    if not math.isclose(grid.cell_size_m, road_class.cell_size_m, rel_tol=1e-9):
        raise ValueError(
            f"grid cell {grid.cell_size_m} m != road class cell {road_class.cell_size_m} m; "
            f"call road_class.with_cell_size({grid.cell_size_m}) first")

    rough_xy = np.asarray(rough_path_xy, dtype=float)[:, :2]
    before_xyz = grid.path_xyz_m(rough_xy)
    enhanced = algorithm_1_enhanced(grid, rough_xy, road_class, enhancements,
                                    iterations=iterations)
    after_xyz = enhanced.smooth_xyz_m

    before = metrics_mod.measure(before_xyz, step_m=metric_step_m, chord_m=metric_chord_m)
    after = metrics_mod.measure(after_xyz, step_m=metric_step_m, chord_m=metric_chord_m)
    dev = metrics_mod.deviation(after_xyz[:, :2], before_xyz[:, :2], step_m=metric_step_m)

    return {
        'road_class': {'name': road_class.name, 'source': road_class.source,
                       'status': road_class.status, 'standing': road_class.standing,
                       'r_min_m': road_class.r_min_m,
                       'i_max_percent': road_class.i_max_percent,
                       'cell_size_m': road_class.cell_size_m},
        'arm': enhanced.arm,
        'enhancements': enhanced.as_dict(),
        'iterations': enhanced.result.iterations,
        's_search': enhanced.result.s_search.as_dict(),
        'n_representative_points': int(len(enhanced.result.representative_xy)),
        'before': before.as_dict(),
        'after': after.as_dict(),
        'deviation': dev,
        'delta': {
            'r_min_m': after.r_min_m - before.r_min_m,
            'i_max_percent': after.i_max_percent - before.i_max_percent,
            'length_m': after.length_m - before.length_m,
            'sinuosity': after.sinuosity - before.sinuosity,
        },
        'r_min_meets_class': bool(after.r_min_m >= road_class.r_min_m),
        'i_max_meets_class': bool(after.i_max_percent <= road_class.i_max_percent),
        'stations_before': checks_mod.check_path_stations(
            before_xyz, road_class, step_m=metric_step_m, chord_m=metric_chord_m).as_dict(),
        'stations_after': checks_mod.check_path_stations(
            after_xyz, road_class, step_m=metric_step_m, chord_m=metric_chord_m).as_dict(),
        'warnings': _warnings(before, after, road_class),
    }
