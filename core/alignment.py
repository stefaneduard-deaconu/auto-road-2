"""Alignment geometry: tangent/curve decomposition in plan, and a designed grade line.

All inputs and outputs are in METERS; gradients are in percent unless a name says `ratio`.

This module is what turns a polyline into the quantities STAS 863-85 is written in terms
of. `core.metrics` answers "what is the smallest radius of this path"; this module answers
"where does each curve start and end, how long is the tangent between them, and what
vertical curve does the grade line need at this grade break". The station check of the
article (every STAS parameter at every point) is built on it.

FOUR SERIES, ONE STATION GRID
-----------------------------
The alignment is resampled once at a constant arc-length step and every parameter becomes
a value per station:

    r_horizontal_m      the planimetric curve radius
    gradient_percent    the longitudinal gradient
    tangent_length_m    the length of the tangent the station belongs to (NaN in a curve)
    r_vertical_m        the radius of the vertical curve the station lies in (NaN on a
                        grade tangent), with `vertical_kind` giving concave/convex

THE CURVATURE METHOD IS NOT RE-DEFINED HERE. `core.metrics.curve_radii_m` is called, with
its `step_m` and `chord_m`, and its result is placed on the station grid. There is exactly
one three-point circumradius method in this repository and this module must not add a
second one.

STRAIGHT VERSUS CURVED
----------------------------------------
Three almost-collinear points give a huge but finite circumradius, so a radius alone never
says "this is a tangent". The threshold is stated as a **sagitta**, the middle point's
offset from the chord, because that is the quantity the measurement can actually resolve:
for a chord `c` on a circle of radius `R` the sagitta is about `c^2 / (8R)`. A station
whose sagitta is under `straight_sagitta_m` is a tangent. At the defaults (chord 20 m,
sagitta 5 cm) that is a radius of 1000 m, and both numbers are reported so a result is
reproducible.

Runs shorter than `min_element_length_m` are absorbed into their neighbour: an element
shorter than the measurement chord is not resolvable, and without this a single noisy
station would split one curve into three elements and invent a zero-length tangent
between them.

HOW SHORT A TANGENT THIS CAN SEE, AND IN WHICH DIRECTION IT ERRS
----------------------------------------------------------------
A station's radius comes from a triple spanning `2 * chord_m` of arc centred on it, so a
station within one chord of a curve still measures that curve's radius. The consequences
are systematic, not random, and both are conservative:

* a measured tangent is shorter than the true one by about `chord_m + 2 * step_m`
  (30 m at the defaults), because the stations at each end are still seeing the curve;
* a tangent shorter than that is not detected at all, and the two curves around it are
  reported as one curve.

So `L_alignment` is under-reported, which fails a marginal tangent rather than passing it,
and a **tangent too short to be seen cannot produce a false pass** - it produces no
tangent to check. What it does mean is that "no L_alignment violation" never proves "no
short tangents": below the floor the question is not answered, and the result row records
`step_m` and `chord_m` so a reader can compute the floor. `tests/test_alignment.py` pins
both the bias and the floor.

THE DESIGNED GRADE LINE
-----------------------
A vertical-curve radius cannot be read off the terrain-following profile, because that
profile is the ground and not a design. It is derived, in the order a preliminary
("first step") design works in:

    1. design points every `l_design_step_min_m` of chainage, elevation taken from the
       profile. The standard gives a MINIMUM step, so the spacing used is
       `total / floor(total / step)`, which is the smallest spacing at or above it that
       divides the alignment evenly and lands on both ends.
    2. the grade line, piecewise linear through those points, with one gradient `i_k` per
       interval.
    3. at each interior design point, the grade break `di = i_k+1 - i_k` (as a ratio).
       `di > 0` is concave (a valley), `di < 0` is convex (a crest).
    4. the vertical curve is centred on the break and may use at most half of each
       adjacent grade tangent, so its length is `L_v = min(len_prev, len_next)` and the
       parabolic radius is `R_V = L_v / |di|`.

This is a preliminary vertical design, which is the design stage the article is about. It
is not an earthworks design: nothing here computes cut or fill.

TWO GRADIENT NUMBERS, DELIBERATELY
----------------------------------
`gradient_percent` here is a per-station central difference over the resampled polyline,
so it is smoothed by the resampling. The authoritative aggregate for a compliance check
stays `core.metrics.i_max_percent`, measured on the raw path segments, which is the
conservative (larger) of the two. Both are reported; they are not interchangeable and
must never be put in the same column.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from core import metrics
from data.configs.road_classes import RoadClass

#: middle-point offset from the chord below which a station counts as a tangent, in meters.
DEFAULT_STRAIGHT_SAGITTA_M = 0.05
#: elements shorter than this are absorbed into their neighbour, in meters.
#: `None` means "use the curvature chord", which is the shortest resolvable element.
DEFAULT_MIN_ELEMENT_LENGTH_M: Optional[float] = None


def tangent_detection_floor_m(step_m: float = metrics.DEFAULT_RESAMPLE_STEP_M,
                              chord_m: float = metrics.DEFAULT_CHORD_M) -> float:
    """Shortest tangent this measurement can resolve, and its under-estimate of a longer one.

    A tangent below this length is reported as part of the surrounding curve; a longer one
    is reported about this much shorter than it is. See the module docstring.
    """
    return chord_m + 2.0 * step_m

ElementKind = Literal['tangent', 'curve']
VerticalKind = Literal['concave', 'convex']

INF = float('inf')
NAN = float('nan')


def straight_radius_m(chord_m: float, sagitta_m: float = DEFAULT_STRAIGHT_SAGITTA_M) -> float:
    """The radius at which a `chord_m` chord bulges by `sagitta_m`: `c^2 / (8 s)`."""
    if chord_m <= 0 or sagitta_m <= 0:
        raise ValueError('chord_m and sagitta_m must be positive')
    return (chord_m * chord_m) / (8.0 * sagitta_m)


# -- the station grid ---------------------------------------------------------------------

def resample_xyz(path_xyz_m: np.ndarray,
                 step_m: float = metrics.DEFAULT_RESAMPLE_STEP_M) -> tuple[np.ndarray, np.ndarray]:
    """Stations along the polyline at constant arc-length spacing, and their chainage.

    The x and y columns are identical to `core.metrics.resample_by_arclength`, so the
    stations here and the radii measured there sit on the same grid; z is interpolated at
    the same chainages.
    """
    xyz = np.asarray(path_xyz_m, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f'path must be (n, 3) in meters, got shape {xyz.shape}')
    # drop points that repeat in plan, exactly as metrics does, keeping z aligned
    keep = np.ones(len(xyz), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1) > 1e-12
    xyz = xyz[keep]
    if len(xyz) < 2:
        raise ValueError('path must have at least two distinct points in plan')
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))])
    total = float(s[-1])
    if total < step_m:
        targets = np.array([0.0, total])
    else:
        n_steps = max(int(np.floor(total / step_m)), 1)
        targets = np.linspace(0.0, total, n_steps + 1)
    stations = np.column_stack([np.interp(targets, s, xyz[:, c]) for c in range(3)])
    return stations, targets


def _radii_on_stations(stations_xy: np.ndarray, step_m: float, chord_m: float) -> np.ndarray:
    """`metrics.menger_radii` placed on the station grid, at the middle point of each triple.

    The first and last `stride` stations are not the middle of any triple; they take the
    nearest measured value, which is what a designer reads off the end of a drawing.
    """
    n = len(stations_xy)
    out = np.full(n, NAN)
    stride = max(int(round(chord_m / step_m)), 1)
    while stride > 1 and n < 2 * stride + 1:
        stride -= 1
    radii = metrics.menger_radii(stations_xy, stride=stride)
    if radii.size == 0:
        out[:] = INF
        return out
    out[stride:stride + radii.size] = radii
    out[:stride] = radii[0]
    out[stride + radii.size:] = radii[-1]
    return out


# -- horizontal decomposition ---------------------------------------------------------------

@dataclass(frozen=True)
class Element:
    """One tangent or one curve of the alignment in plan."""

    kind: ElementKind
    i_start: int
    i_end: int  # inclusive station index
    s_start_m: float
    s_end_m: float
    length_m: float
    r_min_m: float
    r_mean_m: float


def _runs(labels: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of every maximal run of equal labels."""
    if labels.size == 0:
        return []
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks - 1, [labels.size - 1]])
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _absorb_short_runs(is_curve: np.ndarray,
                       chainage_m: np.ndarray,
                       min_length_m: float) -> np.ndarray:
    """Flip runs shorter than `min_length_m` until every run is resolvable.

    The shortest offending run is flipped first, which merges it into whichever
    neighbours it now matches; the pass repeats because merging can itself create a run
    that is long enough. A path that is one single short run is left alone.
    """
    labels = is_curve.copy()
    while True:
        runs = _runs(labels)
        if len(runs) < 2:
            return labels
        lengths = [chainage_m[b] - chainage_m[a] for a, b in runs]
        order = sorted(range(len(runs)), key=lambda k: lengths[k])
        shortest = order[0]
        if lengths[shortest] >= min_length_m:
            return labels
        a, b = runs[shortest]
        labels[a:b + 1] = ~labels[a]


def classify_stations(radii_m: np.ndarray,
                      chainage_m: np.ndarray,
                      chord_m: float,
                      sagitta_m: float = DEFAULT_STRAIGHT_SAGITTA_M,
                      min_element_length_m: Optional[float] = None) -> np.ndarray:
    """Boolean per station: True where the alignment is curving, False on a tangent."""
    threshold = straight_radius_m(chord_m, sagitta_m)
    is_curve = np.isfinite(radii_m) & (radii_m <= threshold)
    min_length = chord_m if min_element_length_m is None else min_element_length_m
    return _absorb_short_runs(is_curve, chainage_m, min_length)


def horizontal_elements(is_curve: np.ndarray,
                        chainage_m: np.ndarray,
                        radii_m: np.ndarray) -> list[Element]:
    """The alternating tangents and curves of the alignment, in order along the chainage."""
    elements: list[Element] = []
    for a, b in _runs(is_curve):
        r = radii_m[a:b + 1]
        finite = r[np.isfinite(r)]
        elements.append(Element(
            kind='curve' if is_curve[a] else 'tangent',
            i_start=a,
            i_end=b,
            s_start_m=float(chainage_m[a]),
            s_end_m=float(chainage_m[b]),
            length_m=float(chainage_m[b] - chainage_m[a]),
            r_min_m=float(finite.min()) if finite.size else INF,
            r_mean_m=float(finite.mean()) if finite.size else INF,
        ))
    return elements


def tangent_length_series(elements: list[Element], n_stations: int) -> np.ndarray:
    """Length of the tangent each station belongs to; NaN for stations inside a curve."""
    out = np.full(n_stations, NAN)
    for element in elements:
        if element.kind == 'tangent':
            out[element.i_start:element.i_end + 1] = element.length_m
    return out


# -- the designed grade line ------------------------------------------------------------------

@dataclass(frozen=True)
class GradeLine:
    """A piecewise-linear designed longitudinal profile and its vertical curves."""

    s_design_m: np.ndarray          # chainage of every design point
    z_design_m: np.ndarray          # its elevation, taken from the profile
    step_used_m: float              # the spacing actually used (>= the norm's minimum)
    gradients_ratio: np.ndarray     # one per interval between design points
    break_s_m: np.ndarray           # chainage of every interior grade break
    break_delta_ratio: np.ndarray   # i[k+1] - i[k] at that break
    break_length_m: np.ndarray      # vertical-curve length available there
    break_radius_m: np.ndarray      # L_v / |di|; inf where the break vanishes
    break_kind: list[Optional[VerticalKind]]

    @property
    def gradients_percent(self) -> np.ndarray:
        return 100.0 * self.gradients_ratio


def design_points(chainage_m: np.ndarray,
                  z_m: np.ndarray,
                  l_step_min_m: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Chainages and elevations of the longitudinal profile's design points.

    `l_step_min_m` is a minimum, so the spacing used is the smallest one at or above it
    that divides the alignment evenly and lands on both ends. An alignment shorter than
    one step gets its two ends and nothing else.
    """
    if l_step_min_m <= 0:
        raise ValueError('l_step_min_m must be positive')
    total = float(chainage_m[-1])
    n_intervals = max(int(np.floor(total / l_step_min_m)), 1)
    s_design = np.linspace(0.0, total, n_intervals + 1)
    z_design = np.interp(s_design, chainage_m, z_m)
    return s_design, z_design, float(total / n_intervals)


def grade_line(chainage_m: np.ndarray, z_m: np.ndarray, l_step_min_m: float) -> GradeLine:
    """The designed grade line over the profile, with one vertical curve per grade break."""
    s_design, z_design, step_used = design_points(chainage_m, z_m, l_step_min_m)
    spans = np.diff(s_design)
    gradients = np.diff(z_design) / spans
    delta = np.diff(gradients)
    lengths = np.minimum(spans[:-1], spans[1:])
    with np.errstate(divide='ignore', invalid='ignore'):
        radii = np.where(np.abs(delta) > 0, lengths / np.abs(delta), INF)
    kinds: list[Optional[VerticalKind]] = [
        'concave' if d > 0 else 'convex' if d < 0 else None for d in delta]
    return GradeLine(
        s_design_m=s_design,
        z_design_m=z_design,
        step_used_m=step_used,
        gradients_ratio=gradients,
        break_s_m=s_design[1:-1],
        break_delta_ratio=delta,
        break_length_m=lengths,
        break_radius_m=radii,
        break_kind=kinds,
    )


def vertical_series(line: GradeLine,
                    chainage_m: np.ndarray) -> tuple[np.ndarray, list[Optional[VerticalKind]]]:
    """Per station: the radius of the vertical curve it lies in, and that curve's kind.

    A station on a grade tangent, between two vertical curves, gets NaN and None: there is
    no vertical curve there to have a radius.
    """
    radii = np.full(len(chainage_m), NAN)
    kinds: list[Optional[VerticalKind]] = [None] * len(chainage_m)
    for s_break, half, radius, kind in zip(line.break_s_m,
                                           line.break_length_m / 2.0,
                                           line.break_radius_m,
                                           line.break_kind):
        if kind is None:
            continue
        inside = np.flatnonzero(np.abs(chainage_m - s_break) <= half)
        radii[inside] = radius
        for i in inside:
            kinds[int(i)] = kind
    return radii, kinds


def grade_series(line: GradeLine, chainage_m: np.ndarray) -> np.ndarray:
    """Per station: the designed grade (%) of the grade-line interval it lies in."""
    k = np.clip(np.searchsorted(line.s_design_m, chainage_m, side='right') - 1,
                0, len(line.gradients_ratio) - 1)
    return line.gradients_percent[k]


# -- the bundle -------------------------------------------------------------------------------

@dataclass(frozen=True)
class AlignmentGeometry:
    """Every STAS parameter of one alignment, as a series over its chainage."""

    chainage_m: np.ndarray
    stations_xyz_m: np.ndarray
    r_horizontal_m: np.ndarray
    gradient_percent: np.ndarray
    tangent_length_m: np.ndarray
    r_vertical_m: np.ndarray
    vertical_kind: list[Optional[VerticalKind]]
    is_curve: np.ndarray
    elements: list[Element]
    line: Optional[GradeLine]
    road_class_name: str
    step_m: float
    chord_m: float
    straight_sagitta_m: float
    straight_radius_m: float
    min_element_length_m: float
    #: designed grade (%) at every station; NaN when the class has no design step
    grade_percent: Optional[np.ndarray] = None

    @property
    def length_m(self) -> float:
        return float(self.chainage_m[-1])

    @property
    def has_vertical(self) -> bool:
        return self.line is not None

    def tangents(self) -> list[Element]:
        return [e for e in self.elements if e.kind == 'tangent']

    def curves(self) -> list[Element]:
        return [e for e in self.elements if e.kind == 'curve']


def measure_alignment(path_xyz_m: np.ndarray,
                      road_class: RoadClass,
                      step_m: float = metrics.DEFAULT_RESAMPLE_STEP_M,
                      chord_m: float = metrics.DEFAULT_CHORD_M,
                      straight_sagitta_m: float = DEFAULT_STRAIGHT_SAGITTA_M,
                      min_element_length_m: Optional[float] = None) -> AlignmentGeometry:
    """Every STAS parameter of `path_xyz_m`, at every station along it.

    The vertical parameters need `road_class.l_design_step_min_m`; a class that does not
    define it gets `line=None` and an all-NaN `r_vertical_m`, which the checks report as
    not applicable rather than as a pass.
    """
    stations, chainage = resample_xyz(path_xyz_m, step_m)
    radii = _radii_on_stations(stations[:, :2], step_m, chord_m)
    is_curve = classify_stations(radii, chainage, chord_m, straight_sagitta_m,
                                 min_element_length_m)
    elements = horizontal_elements(is_curve, chainage, radii)
    gradient = 100.0 * np.gradient(stations[:, 2], chainage) if len(chainage) > 1 \
        else np.zeros(len(chainage))

    if road_class.l_design_step_min_m is None:
        line, r_vertical = None, np.full(len(chainage), NAN)
        vertical_kind: list[Optional[VerticalKind]] = [None] * len(chainage)
        grade = np.full(len(chainage), NAN)
    else:
        line = grade_line(chainage, stations[:, 2], road_class.l_design_step_min_m)
        r_vertical, vertical_kind = vertical_series(line, chainage)
        grade = grade_series(line, chainage)

    return AlignmentGeometry(
        chainage_m=chainage,
        stations_xyz_m=stations,
        r_horizontal_m=radii,
        gradient_percent=gradient,
        tangent_length_m=tangent_length_series(elements, len(chainage)),
        r_vertical_m=r_vertical,
        vertical_kind=vertical_kind,
        is_curve=is_curve,
        elements=elements,
        line=line,
        road_class_name=road_class.name,
        step_m=step_m,
        chord_m=chord_m,
        straight_sagitta_m=straight_sagitta_m,
        straight_radius_m=straight_radius_m(chord_m, straight_sagitta_m),
        min_element_length_m=chord_m if min_element_length_m is None else min_element_length_m,
        grade_percent=grade,
    )
