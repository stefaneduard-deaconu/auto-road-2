"""Research step 2: does the generated alignment satisfy a predefined road class?

The rule that matters: a class that cannot be met is reported as **infeasible under this
class**. Nothing here relaxes a limit, substitutes a looser class or picks the class that
happens to pass. `RoadClass` objects are frozen, and the only thing that changes between
runs is which class is asked about.

Two of the measures are pass/fail against the norm:

    i_max   <= road_class.i_max_percent      "declivitate longitudinala maxima"
    R_min   >= road_class.r_min_m            "raza minima a curbei in plan"

The others (length, sinuosity, elevation difference) are reported, not judged: the norm
tables read in `data/configs/road_classes.py` set no limit on them.

Caveats that travel with every report:

* `status` is `TO CONFIRM` until the road-design co-author signs the class off, and it is
  printed on every table so no reader mistakes an unconfirmed constraint for a norm.
* `R_min` is the planimetric radius of the axis polyline measured by `core.metrics`, not
  a designed circular arc with transition curves. It is a lower bound on what a designed
  alignment through the same corridor could offer.
* When the class gives a serpentine radius, an `R_min` below `r_min_m` but above
  `r_min_serpentine_m` still FAILS the plain check and is flagged as "only admissible if
  designed as a serpentine", which is a design decision, not something this code makes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from core import alignment as alignment_mod
from core import metrics as metrics_mod
from data.configs.road_classes import ROAD_CLASSES, RoadClass, get

#: relative slack on a comparison, so a value that is exactly at the limit is not failed
#: by floating-point noise (a constant 6.5% gradient comes out as 6.500000000000001).
#: It is far below any meaningful engineering difference.
CHECK_TOLERANCE = 1e-9


def _at_most(value: float, limit: float) -> bool:
    return bool(value <= limit * (1.0 + CHECK_TOLERANCE))


def _at_least(value: float, limit: float) -> bool:
    return bool(value >= limit * (1.0 - CHECK_TOLERANCE))


@dataclass(frozen=True)
class Check:
    """One pass/fail engineering check."""

    name: str
    value: float
    limit: float
    comparison: str  # '<=' or '>='
    passed: bool
    units: str
    source: str
    note: str = ''

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        flag = 'PASS' if self.passed else 'FAIL'
        return (f"{flag} {self.name}: {self.value:.2f} {self.comparison} "
                f"{self.limit:.2f} {self.units}")


@dataclass(frozen=True)
class ChecksReport:
    road_class_name: str
    road_class_status: str
    checks: tuple[Check, ...]
    reported: dict
    step_m: float
    chord_m: float

    @property
    def feasible(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def verdict(self) -> str:
        if self.feasible:
            return f"meets {self.road_class_name}"
        failed = ', '.join(c.name for c in self.checks if not c.passed)
        return f"infeasible under this class ({self.road_class_name}): {failed}"

    def as_dict(self) -> dict:
        return {
            'road_class': self.road_class_name,
            'road_class_status': self.road_class_status,
            'feasible': self.feasible,
            'verdict': self.verdict,
            'checks': {c.name: c.as_dict() for c in self.checks},
            'reported': self.reported,
            'step_m': self.step_m,
            'chord_m': self.chord_m,
        }


def check_path(path_xyz_m: np.ndarray,
               road_class: RoadClass,
               step_m: float = metrics_mod.DEFAULT_RESAMPLE_STEP_M,
               chord_m: float = metrics_mod.DEFAULT_CHORD_M) -> ChecksReport:
    """Measure the axis and compare it against one road class.

    `path_xyz_m` is (n, 3) in meters. A 2-column path can be checked for R_min only.
    """
    arr = np.asarray(path_xyz_m, dtype=float)
    measured = metrics_mod.measure(arr, step_m=step_m, chord_m=chord_m)

    note = ''
    if (road_class.r_min_serpentine_m is not None
            and road_class.r_min_serpentine_m <= measured.r_min_m < road_class.r_min_m):
        note = (f"above the {road_class.r_min_serpentine_m:.0f} m serpentine minimum but "
                f"below the {road_class.r_min_m:.0f} m plain minimum; only admissible if "
                f"the curve is designed as a serpentine, which is a designer's decision")

    checks = [Check(
        name='R_min',
        value=measured.r_min_m,
        limit=road_class.r_min_m,
        comparison='>=',
        passed=_at_least(measured.r_min_m, road_class.r_min_m),
        units='m',
        source=road_class.source,
        note=note,
    )]
    if measured.i_max_percent is not None:
        checks.append(Check(
            name='i_max',
            value=measured.i_max_percent,
            limit=road_class.i_max_percent,
            comparison='<=',
            passed=_at_most(measured.i_max_percent, road_class.i_max_percent),
            units='%',
            source=road_class.source,
        ))

    reported = {
        'length_m': measured.length_m,
        'length_3d_m': measured.length_3d_m,
        'sinuosity': measured.sinuosity,
        'elevation_net_m': measured.elevation_net_m,
        'elevation_ascent_m': measured.elevation_ascent_m,
        'elevation_descent_m': measured.elevation_descent_m,
        'elevation_span_m': measured.elevation_span_m,
        'design_speed_kmh': road_class.design_speed_kmh,
        'terrain_category': road_class.terrain_category,
    }
    return ChecksReport(road_class_name=road_class.name,
                        road_class_status=road_class.status,
                        checks=tuple(checks), reported=reported,
                        step_m=step_m, chord_m=chord_m)


def check_against_classes(path_xyz_m: np.ndarray,
                          road_class_names: Optional[Sequence[str]] = None,
                          step_m: float = metrics_mod.DEFAULT_RESAMPLE_STEP_M,
                          chord_m: float = metrics_mod.DEFAULT_CHORD_M) -> dict[str, ChecksReport]:
    """One report per class. The default is the whole registry."""
    names = list(ROAD_CLASSES) if road_class_names is None else list(road_class_names)
    return {name: check_path(path_xyz_m, get(name), step_m=step_m, chord_m=chord_m)
            for name in names}


def flags_table(reports: dict[str, ChecksReport]) -> str:
    """The table of pass flags per class that T6 asks for."""
    header = (f"{'road class':<28}{'status':<12}{'V km/h':>7}{'R_min m':>9}{'req':>7}"
              f"{'R ok':>6}{'i_max %':>9}{'lim':>6}{'i ok':>6}  verdict")
    lines = [header, '-' * len(header)]
    for name, report in reports.items():
        by_name = {c.name: c for c in report.checks}
        r, i = by_name.get('R_min'), by_name.get('i_max')
        lines.append(
            f"{name:<28}{report.road_class_status:<12}"
            f"{report.reported['design_speed_kmh']:>7.0f}"
            f"{_fmt(r.value if r else None):>9}{_fmt(r.limit if r else None):>7}"
            f"{('yes' if r.passed else 'NO') if r else '-':>6}"
            f"{_fmt(i.value if i else None):>9}{_fmt(i.limit if i else None):>6}"
            f"{('yes' if i.passed else 'NO') if i else '-':>6}"
            f"  {'OK' if report.feasible else report.verdict}")
    notes = [f"  note ({name}): {c.note}"
             for name, report in reports.items() for c in report.checks if c.note]
    if notes:
        lines += ['', 'Notes:'] + notes
    lines += ['',
              'Every class above is TO CONFIRM until the road-design co-author signs it '
              'off; sources are in data/configs/road_classes.py.',
              'A failing class is reported as infeasible under that class. No limit is '
              'relaxed and no class is substituted.']
    return '\n'.join(lines)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return '-'
    if value == metrics_mod.INF:
        return 'inf'
    return f"{value:.1f}"


def feasible_classes(reports: dict[str, ChecksReport]) -> list[str]:
    return [name for name, report in reports.items() if report.feasible]


def infeasible_classes(reports: dict[str, ChecksReport]) -> list[str]:
    return [name for name, report in reports.items() if not report.feasible]


# ==========================================================================================
# Per-station checks (deliverable (c) of the article)
#
# `check_path` above judges an alignment by its worst value: one R_min, one i_max. That
# answers "does this pass", not "where and for how long does it fail", and the second
# article needs the second question answered for every STAS parameter. The checks below
# therefore run over the station grid of `core.alignment` and report, per parameter, how
# many stations violate it and how many metres of road that is.
#
# WHERE EACH PARAMETER APPLIES. A limit is only checked where it means something:
#
#   R_H           on curve stations. A tangent has no radius to be too small.
#   i             everywhere.
#   L_alignment   on tangents that lie BETWEEN two curves, which is what the standard's
#                 "minimum alignment length" governs. The first and last tangent of an
#                 alignment run off to the terminals and are reported, not judged.
#   R_V           on stations inside a vertical curve, against the concave or the convex
#                 minimum depending on the sign of the grade break. A station on a grade
#                 tangent has no vertical curve to measure.
#
# A class that does not define a limit yields `applicable=False`, never a silent pass.
# ==========================================================================================


@dataclass(frozen=True)
class StationCheck:
    """One STAS parameter, judged at every station it applies to."""

    name: str
    limit: Optional[float]
    comparison: str  # '<=' or '>='
    units: str
    applicable: bool
    n_applicable: int
    n_violations: int
    violation_length_m: float
    worst_value: Optional[float]
    worst_station_m: Optional[float]
    source: str = ''
    note: str = ''

    @property
    def passed(self) -> bool:
        """True when nothing violates it. An inapplicable check passes vacuously."""
        return self.n_violations == 0

    @property
    def status(self) -> str:
        if not self.applicable:
            return 'not applicable'
        return 'pass' if self.passed else 'fail'

    def as_dict(self) -> dict:
        return {**asdict(self), 'status': self.status, 'passed': self.passed}

    def __str__(self) -> str:
        if not self.applicable:
            return f"n/a  {self.name}: {self.note or 'the class does not define this limit'}"
        if self.passed:
            return (f"pass {self.name}: worst {_fmt(self.worst_value)} {self.units} "
                    f"{self.comparison} {_fmt(self.limit)} over {self.n_applicable} stations")
        return (f"FAIL {self.name}: {self.n_violations}/{self.n_applicable} stations, "
                f"{self.violation_length_m:.0f} m, worst {_fmt(self.worst_value)} "
                f"{self.units} at km {self.worst_station_m:.0f}")


@dataclass(frozen=True)
class StationReport:
    """Every STAS parameter of one alignment, judged station by station."""

    road_class_name: str
    road_class_status: str
    road_class_standing: str
    checks: tuple[StationCheck, ...]
    geometry: 'alignment_mod.AlignmentGeometry'

    @property
    def applicable_checks(self) -> tuple[StationCheck, ...]:
        return tuple(c for c in self.checks if c.applicable)

    @property
    def feasible(self) -> bool:
        return all(c.passed for c in self.applicable_checks)

    @property
    def verdict(self) -> str:
        failed = [c.name for c in self.applicable_checks if not c.passed]
        if not failed:
            skipped = [c.name for c in self.checks if not c.applicable]
            tail = f" ({', '.join(skipped)} not defined by the class)" if skipped else ''
            return f"meets {self.road_class_name} at every station{tail}"
        return (f"infeasible under this class ({self.road_class_name}): "
                f"{', '.join(failed)}")

    @property
    def total_violation_length_m(self) -> float:
        return float(sum(c.violation_length_m for c in self.applicable_checks))

    def by_name(self) -> dict[str, StationCheck]:
        return {c.name: c for c in self.checks}

    def as_dict(self) -> dict:
        return {
            'road_class': self.road_class_name,
            'road_class_status': self.road_class_status,
            'road_class_standing': self.road_class_standing,
            'feasible': self.feasible,
            'verdict': self.verdict,
            'length_m': self.geometry.length_m,
            'n_stations': int(len(self.geometry.chainage_m)),
            'step_m': self.geometry.step_m,
            'chord_m': self.geometry.chord_m,
            'straight_radius_m': self.geometry.straight_radius_m,
            'total_violation_length_m': self.total_violation_length_m,
            'checks': {c.name: c.as_dict() for c in self.checks},
        }

    def __str__(self) -> str:
        head = (f"{self.road_class_name} [{self.road_class_status}, "
                f"{self.road_class_standing}] over {self.geometry.length_m:.0f} m")
        return '\n'.join([head] + [f"  {c}" for c in self.checks] + [f"  -> {self.verdict}"])


def _station_weights_m(geometry: 'alignment_mod.AlignmentGeometry') -> np.ndarray:
    """Length of road each station stands for. Sums to the alignment length exactly.

    The end stations carry half a spacing, the interior ones a full one, which is the
    trapezoidal weighting; multiplying a station count by the spacing instead would
    overstate a fully violating alignment by one spacing.
    """
    chainage = geometry.chainage_m
    if len(chainage) < 2:
        return np.zeros(len(chainage))
    weights = np.empty(len(chainage))
    weights[1:-1] = (chainage[2:] - chainage[:-2]) / 2.0
    weights[0] = (chainage[1] - chainage[0]) / 2.0
    weights[-1] = (chainage[-1] - chainage[-2]) / 2.0
    return weights


def _judge(name: str,
           values: np.ndarray,
           applies: np.ndarray,
           limit: Optional[float],
           comparison: str,
           units: str,
           geometry: 'alignment_mod.AlignmentGeometry',
           source: str = '',
           note: str = '') -> StationCheck:
    """Compare one series against one limit, over the stations the limit applies to."""
    if limit is None:
        return StationCheck(name=name, limit=None, comparison=comparison, units=units,
                            applicable=False, n_applicable=0, n_violations=0,
                            violation_length_m=0.0, worst_value=None, worst_station_m=None,
                            source=source,
                            note='the road class does not define this limit')

    applies = applies & np.isfinite(values)
    where = np.flatnonzero(applies)
    if where.size == 0:
        return StationCheck(name=name, limit=limit, comparison=comparison, units=units,
                            applicable=False, n_applicable=0, n_violations=0,
                            violation_length_m=0.0, worst_value=None, worst_station_m=None,
                            source=source,
                            note='no station on this alignment is governed by it')

    subject = values[where]
    if comparison == '<=':
        ok = np.array([_at_most(v, limit) for v in subject])
        worst_local = int(np.argmax(subject))
    else:
        ok = np.array([_at_least(v, limit) for v in subject])
        worst_local = int(np.argmin(subject))

    violating = where[~ok]
    return StationCheck(
        name=name, limit=limit, comparison=comparison, units=units,
        applicable=True,
        n_applicable=int(where.size),
        n_violations=int(violating.size),
        violation_length_m=float(_station_weights_m(geometry)[violating].sum()),
        worst_value=float(subject[worst_local]),
        worst_station_m=float(geometry.chainage_m[where[worst_local]]),
        source=source, note=note,
    )


def _interior_tangent_mask(geometry: 'alignment_mod.AlignmentGeometry') -> np.ndarray:
    """Stations on a tangent that has a curve on both sides."""
    mask = np.zeros(len(geometry.chainage_m), dtype=bool)
    elements = geometry.elements
    for k, element in enumerate(elements):
        if element.kind != 'tangent':
            continue
        if k == 0 or k == len(elements) - 1:
            continue
        mask[element.i_start:element.i_end + 1] = True
    return mask


def check_path_stations(path_xyz_m: np.ndarray,
                        road_class: RoadClass,
                        step_m: float = metrics_mod.DEFAULT_RESAMPLE_STEP_M,
                        chord_m: float = metrics_mod.DEFAULT_CHORD_M,
                        geometry: Optional['alignment_mod.AlignmentGeometry'] = None
                        ) -> StationReport:
    """Judge every STAS parameter of `path_xyz_m` at every station, against one class.

    Pass `geometry` to reuse a measurement already made; otherwise it is computed here
    with the same one curvature method the aggregate checks use.
    """
    if geometry is None:
        geometry = alignment_mod.measure_alignment(path_xyz_m, road_class,
                                                   step_m=step_m, chord_m=chord_m)

    is_concave = np.array([k == 'concave' for k in geometry.vertical_kind])
    is_convex = np.array([k == 'convex' for k in geometry.vertical_kind])

    serpentine_note = ''
    if road_class.r_min_serpentine_m is not None:
        serpentine_note = (
            f"a radius below {road_class.r_min_m:.0f} m but above "
            f"{road_class.r_min_serpentine_m:.0f} m is only admissible if the curve is "
            f"designed as a serpentine, which is a designer's decision, not this code's")

    checks = (
        _judge('R_H', geometry.r_horizontal_m, geometry.is_curve, road_class.r_min_m,
               '>=', 'm', geometry, road_class.source, serpentine_note),
        _judge('i', np.abs(geometry.gradient_percent),
               np.ones(len(geometry.chainage_m), dtype=bool), road_class.i_max_percent,
               '<=', '%', geometry, road_class.source,
               f"measured on the alignment resampled at {geometry.step_m:g} m; the "
               f"aggregate i_max of core.checks.check_path uses the raw segments and is "
               f"the conservative one of the two"),
        _judge('L_alignment', geometry.tangent_length_m, _interior_tangent_mask(geometry),
               road_class.l_alignment_min_m, '>=', 'm', geometry, road_class.source,
               'judged only on tangents between two curves; the two end tangents run to '
               'the terminals and are reported, not judged'),
        _judge('R_V_concave', geometry.r_vertical_m, is_concave,
               road_class.r_vert_concave_min_m, '>=', 'm', geometry, road_class.source,
               'on the designed grade line, not on the terrain profile'),
        _judge('R_V_convex', geometry.r_vertical_m, is_convex,
               road_class.r_vert_convex_min_m, '>=', 'm', geometry, road_class.source,
               'on the designed grade line, not on the terrain profile'),
    )
    return StationReport(road_class_name=road_class.name,
                         road_class_status=road_class.status,
                         road_class_standing=road_class.standing,
                         checks=checks, geometry=geometry)


#: the order STAS parameters are reported in, everywhere.
STATION_CHECK_NAMES: tuple[str, ...] = ('R_H', 'i', 'L_alignment', 'R_V_concave', 'R_V_convex')
