"""S2: the alignment geometry, checked against shapes whose geometry is known exactly."""
import numpy as np
import pytest

from core import alignment, metrics
from data.configs.road_classes import get

V40 = get('STAS863_V40')
V25 = get('STAS863_V25')
NO_VERTICAL = get('RO_CLASS_IV_DEAL')


def straight_line(length_m=1000.0, n=201, slope=0.0):
    x = np.linspace(0.0, length_m, n)
    return np.column_stack([x, np.zeros(n), slope * x])


def circular_arc(radius_m=200.0, sweep_rad=np.pi / 2, n=401, z=0.0):
    t = np.linspace(0.0, sweep_rad, n)
    return np.column_stack([radius_m * np.sin(t),
                            radius_m * (1.0 - np.cos(t)),
                            np.full(n, z)])


def crest(length_m=600.0, n=601, rise=0.04, fall=-0.06):
    """A straight line in plan whose profile rises then falls: one convex grade break."""
    x = np.linspace(0.0, length_m, n)
    half = length_m / 2.0
    z = np.where(x <= half, rise * x, rise * half + fall * (x - half))
    return np.column_stack([x, np.zeros(n), z])


def from_curvature(segments, step_m=1.0, z=0.0):
    """Build a path from a list of (length_m, radius_m or None) by integrating the heading.

    Tangency between consecutive elements is exact by construction, which a fixture built
    by translating separate arcs is not.
    """
    xy = [np.zeros(2)]
    heading = 0.0
    for length_m, radius_m in segments:
        n = max(int(round(length_m / step_m)), 1)
        ds = length_m / n
        turn = 0.0 if radius_m is None else ds / radius_m
        for _ in range(n):
            heading += turn
            xy.append(xy[-1] + ds * np.array([np.cos(heading), np.sin(heading)]))
    xy = np.array(xy)
    return np.column_stack([xy, np.full(len(xy), z)])


# -- the straightness threshold --------------------------------------------------------

def test_straight_radius_is_the_sagitta_formula():
    assert alignment.straight_radius_m(20.0, 0.05) == pytest.approx(1000.0)
    assert alignment.straight_radius_m(20.0, 0.10) == pytest.approx(500.0)


@pytest.mark.parametrize('bad', [(0.0, 0.05), (20.0, 0.0), (-20.0, 0.05)])
def test_straight_radius_rejects_non_positive_inputs(bad):
    with pytest.raises(ValueError):
        alignment.straight_radius_m(*bad)


# -- a straight line -------------------------------------------------------------------

def test_a_straight_line_is_one_tangent_of_its_full_length():
    geometry = alignment.measure_alignment(straight_line(), V40)
    assert [e.kind for e in geometry.elements] == ['tangent']
    assert geometry.elements[0].length_m == pytest.approx(1000.0)
    assert not geometry.is_curve.any()
    assert np.all(geometry.tangent_length_m == pytest.approx(1000.0))


def test_a_straight_line_has_no_measurable_curvature():
    geometry = alignment.measure_alignment(straight_line(), V40)
    finite = geometry.r_horizontal_m[np.isfinite(geometry.r_horizontal_m)]
    assert np.all(finite > geometry.straight_radius_m)


def test_a_constant_slope_gives_that_gradient_at_every_station():
    geometry = alignment.measure_alignment(straight_line(slope=0.05), V40)
    assert np.allclose(geometry.gradient_percent, 5.0)


# -- a circular arc --------------------------------------------------------------------

def test_an_arc_is_one_curve_and_recovers_its_radius():
    geometry = alignment.measure_alignment(circular_arc(radius_m=200.0), V40)
    assert [e.kind for e in geometry.elements] == ['curve']
    measured = geometry.elements[0].r_min_m
    # the method measures slightly below the true radius (see core.metrics); under 1%
    assert 0.99 * 200.0 <= measured <= 200.0 * 1.001


def test_the_arc_radius_agrees_with_the_one_curvature_method():
    path = circular_arc(radius_m=150.0)
    geometry = alignment.measure_alignment(path, V40)
    finite = geometry.r_horizontal_m[np.isfinite(geometry.r_horizontal_m)]
    assert finite.min() == pytest.approx(metrics.r_min_m(path), rel=1e-9)


def test_an_arc_has_no_tangent_length_anywhere():
    geometry = alignment.measure_alignment(circular_arc(), V40)
    assert np.all(np.isnan(geometry.tangent_length_m))
    assert geometry.tangents() == []


# -- tangent between two curves --------------------------------------------------------

def test_a_line_then_an_arc_splits_into_a_tangent_and_a_curve():
    line = straight_line(length_m=400.0, n=81)
    arc = circular_arc(radius_m=100.0, sweep_rad=np.pi / 2, n=201)
    arc = arc + np.array([line[-1, 0], 0.0, 0.0])
    path = np.vstack([line, arc[1:]])
    geometry = alignment.measure_alignment(path, V40)
    kinds = [e.kind for e in geometry.elements]
    assert kinds == ['tangent', 'curve']
    tangent = geometry.tangents()[0]
    assert tangent.length_m == pytest.approx(400.0, abs=3 * geometry.step_m)


def test_a_single_noisy_station_does_not_split_a_tangent():
    path = straight_line(length_m=1000.0, n=201)
    path[100, 1] += 0.5
    geometry = alignment.measure_alignment(path, V40)
    # without the short-run filter this one nudge produces a spurious curve element
    assert len(geometry.curves()) <= 1
    for element in geometry.elements:
        assert element.length_m >= geometry.min_element_length_m or len(geometry.elements) == 1


# -- how short a tangent the measurement can see ---------------------------------------

def two_curves(tangent_m, radius_m=80.0):
    return from_curvature([(120.0, radius_m), (tangent_m, None), (120.0, -radius_m)])


@pytest.mark.parametrize('true_m', [50.0, 60.0, 80.0, 120.0, 200.0])
def test_a_measured_tangent_is_short_by_the_detection_floor(true_m):
    geometry = alignment.measure_alignment(two_curves(true_m), V40)
    tangents = geometry.tangents()
    assert len(tangents) == 1
    floor = alignment.tangent_detection_floor_m(geometry.step_m, geometry.chord_m)
    assert tangents[0].length_m == pytest.approx(true_m - floor, abs=geometry.step_m)


@pytest.mark.parametrize('true_m', [20.0, 30.0, 40.0, 45.0])
def test_a_tangent_shorter_than_the_floor_is_not_detected(true_m):
    geometry = alignment.measure_alignment(two_curves(true_m), V40)
    assert geometry.tangents() == []
    assert [e.kind for e in geometry.elements] == ['curve']


def test_the_bias_under_reports_the_tangent_which_is_the_safe_direction():
    # under-reporting fails a marginal tangent rather than passing it
    geometry = alignment.measure_alignment(two_curves(80.0), V40)
    assert geometry.tangents()[0].length_m < 80.0


def test_the_detection_floor_is_the_chord_plus_two_steps():
    assert alignment.tangent_detection_floor_m(5.0, 20.0) == pytest.approx(30.0)


# -- the designed grade line -----------------------------------------------------------

def test_the_design_step_is_the_smallest_spacing_at_or_above_the_minimum():
    chainage = np.linspace(0.0, 620.0, 125)
    z = np.zeros_like(chainage)
    s_design, _, step_used = alignment.design_points(chainage, z, 50.0)
    assert step_used >= 50.0
    assert len(s_design) == 13          # floor(620 / 50) = 12 intervals
    assert step_used == pytest.approx(620.0 / 12)
    assert s_design[0] == 0.0 and s_design[-1] == pytest.approx(620.0)


def test_design_points_rejects_a_non_positive_step():
    with pytest.raises(ValueError):
        alignment.design_points(np.array([0.0, 10.0]), np.zeros(2), 0.0)


def test_a_crest_gives_one_convex_break_of_the_expected_radius():
    geometry = alignment.measure_alignment(crest(length_m=600.0, rise=0.04, fall=-0.06), V40)
    line = geometry.line
    assert line is not None
    breaks = [k for k in line.break_kind if k is not None]
    assert breaks == ['convex']
    k = line.break_kind.index('convex')
    assert line.break_delta_ratio[k] == pytest.approx(-0.10, abs=1e-9)
    # R_V = L_v / |di| with L_v the shorter of the two adjacent grade tangents
    expected = line.break_length_m[k] / 0.10
    assert line.break_radius_m[k] == pytest.approx(expected)


def test_a_sag_is_reported_as_concave():
    path = crest(length_m=600.0, rise=-0.05, fall=0.05)
    geometry = alignment.measure_alignment(path, V40)
    assert 'concave' in geometry.line.break_kind
    assert 'convex' not in geometry.line.break_kind


def test_a_constant_grade_has_no_vertical_curve_at_all():
    geometry = alignment.measure_alignment(straight_line(slope=0.03), V40)
    assert all(k is None for k in geometry.line.break_kind)
    assert np.all(np.isnan(geometry.r_vertical_m))


def test_stations_on_a_grade_tangent_have_no_vertical_radius():
    geometry = alignment.measure_alignment(crest(), V40)
    assert np.isnan(geometry.r_vertical_m).any()
    assert np.isfinite(geometry.r_vertical_m).any()


def test_the_two_stas_classes_use_the_same_design_step():
    path = crest()
    a = alignment.measure_alignment(path, V40)
    b = alignment.measure_alignment(path, V25)
    assert a.line.step_used_m == pytest.approx(b.line.step_used_m)


# -- a class without vertical geometry --------------------------------------------------

def test_a_class_without_a_design_step_reports_no_grade_line():
    geometry = alignment.measure_alignment(crest(), NO_VERTICAL)
    assert not geometry.has_vertical
    assert geometry.line is None
    assert np.all(np.isnan(geometry.r_vertical_m))
    assert all(k is None for k in geometry.vertical_kind)
    # the horizontal parameters are still measured
    assert geometry.elements


# -- the station grid -------------------------------------------------------------------

def test_stations_share_the_grid_of_the_one_curvature_method():
    path = circular_arc(radius_m=300.0)
    stations, chainage = alignment.resample_xyz(path)
    assert np.allclose(stations[:, :2], metrics.resample_by_arclength(path))
    assert len(chainage) == len(stations)
    assert np.allclose(np.diff(chainage), chainage[1] - chainage[0])


def test_every_series_has_one_value_per_station():
    geometry = alignment.measure_alignment(crest(), V40)
    n = len(geometry.chainage_m)
    for series in (geometry.r_horizontal_m, geometry.gradient_percent,
                   geometry.tangent_length_m, geometry.r_vertical_m, geometry.is_curve):
        assert len(series) == n
    assert len(geometry.vertical_kind) == n


def test_repeated_points_in_plan_are_dropped():
    path = np.vstack([straight_line(n=51), straight_line(n=51)[-1]])
    geometry = alignment.measure_alignment(path, V40)
    assert geometry.length_m == pytest.approx(1000.0)


def test_a_path_with_no_horizontal_extent_is_rejected():
    with pytest.raises(ValueError):
        alignment.resample_xyz(np.zeros((5, 3)))


def test_a_two_column_path_is_rejected():
    with pytest.raises(ValueError):
        alignment.resample_xyz(np.zeros((5, 2)))
