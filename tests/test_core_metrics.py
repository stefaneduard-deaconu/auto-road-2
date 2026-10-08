"""T4 acceptance: every metric is checked against an analytic value on circles and lines."""
import numpy as np
import pytest

from core.metrics import (DEFAULT_CHORD_M, DEFAULT_RESAMPLE_STEP_M, INF, PathMetrics, curve_radii_m,
                          deviation, elevation_difference_m, gradients_percent,
                          hausdorff_m, i_max_percent, measure, menger_radii,
                          path_length_3d_m, path_length_m, r_min_m,
                          resample_by_arclength, segment_lengths_m, sinuosity,
                          straight_line_distance_m)


def circle(radius: float, n: int = 400, arc: float = 2 * np.pi, centre=(0.0, 0.0)):
    t = np.linspace(0, arc, n)
    return np.column_stack([centre[0] + radius * np.cos(t), centre[1] + radius * np.sin(t)])


def straight(length: float = 500.0, n: int = 50):
    return np.column_stack([np.linspace(0, length, n), np.zeros(n)])


# -- length and sinuosity ---------------------------------------------------------------

def test_length_of_a_straight_line_is_exact():
    assert path_length_m(straight(500.0)) == pytest.approx(500.0)
    assert straight_line_distance_m(straight(500.0)) == pytest.approx(500.0)
    assert sinuosity(straight(500.0)) == pytest.approx(1.0)


def test_length_of_a_full_circle_is_2_pi_r():
    r = 120.0
    assert path_length_m(circle(r, n=5000)) == pytest.approx(2 * np.pi * r, rel=1e-5)


def test_sinuosity_of_a_half_circle_is_pi_over_two():
    # a semicircle of radius r: length pi*r, chord 2r -> sinuosity pi/2
    assert sinuosity(circle(50.0, n=5000, arc=np.pi)) == pytest.approx(np.pi / 2, rel=1e-5)


def test_sinuosity_of_a_closed_path_is_infinite():
    # the ends of a full circle coincide only up to floating point, hence the tolerance
    assert sinuosity(circle(10.0, n=100)) == INF
    assert sinuosity(np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 0.0]])) == INF


def test_segment_lengths_of_a_unit_staircase():
    path = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 10.0]])
    np.testing.assert_allclose(segment_lengths_m(path), [5.0, 6.0])
    assert path_length_m(path) == pytest.approx(11.0)


def test_3d_length_uses_the_height():
    path = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 40.0]])
    assert path_length_m(path) == pytest.approx(30.0)
    assert path_length_3d_m(path) == pytest.approx(50.0)


def test_2d_path_has_no_3d_length():
    with pytest.raises(ValueError):
        path_length_3d_m(straight())


# -- the ONE curvature method ------------------------------------------------------------

@pytest.mark.parametrize("radius", [25.0, 95.0, 125.0, 240.0, 650.0])
def test_curvature_method_recovers_a_circle(radius):
    path = circle(radius, n=4000)
    radii = curve_radii_m(path, step_m=5.0, chord_m=20.0)
    assert radii.size > 0
    np.testing.assert_allclose(radii, radius, rtol=5e-3)
    assert r_min_m(path, step_m=5.0, chord_m=20.0) == pytest.approx(radius, rel=5e-3)


def test_exact_circumradius_needs_no_dense_sampling():
    # three points that really lie on a circle give its radius exactly, at any spacing
    for radius in (25.0, 650.0):
        pts = np.array([[radius * np.cos(t), radius * np.sin(t)] for t in (0.0, 0.9, 2.2)])
        assert menger_radii(pts)[0] == pytest.approx(radius)


def test_a_short_chord_is_noisy_on_a_large_radius_which_is_why_the_chord_is_fixed():
    # the failure mode the method is designed around: at 5 m spacing the sagitta on a
    # 650 m curve is about 5 mm, so polyline discretisation moves the estimate percent-wise
    path = circle(650.0, n=2000)
    points = resample_by_arclength(path, step_m=5.0)
    noisy = menger_radii(points, stride=1)
    stable = menger_radii(points, stride=4)  # chord 20 m
    assert noisy.std() > 10 * stable.std()


def test_three_points_on_a_circle_give_its_radius_exactly():
    r = 7.0
    pts = np.array([[r * np.cos(t), r * np.sin(t)] for t in (0.3, 1.1, 2.4)])
    assert menger_radii(pts)[0] == pytest.approx(r)


def test_a_straight_line_has_infinite_radius():
    assert (menger_radii(straight()) == INF).all()
    assert r_min_m(straight()) == INF


def test_radius_of_a_right_angle_is_half_the_hypotenuse():
    # two segments of equal length meeting at 90 degrees: the circumcircle of the
    # right-angled isosceles triangle has the hypotenuse as diameter
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    assert menger_radii(pts)[0] == pytest.approx(np.hypot(10.0, 10.0) / 2)


def test_resampling_keeps_the_ends_and_uses_a_constant_step():
    path = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0]])
    out = resample_by_arclength(path, step_m=10.0)
    np.testing.assert_allclose(out[0], path[0])
    np.testing.assert_allclose(out[-1], path[-1])
    steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    np.testing.assert_allclose(steps, steps[0], rtol=1e-9)


def test_resampling_drops_duplicate_points():
    path = np.array([[0.0, 0.0], [0.0, 0.0], [20.0, 0.0]])
    out = resample_by_arclength(path, step_m=5.0)
    assert len(out) == 5


def test_a_path_shorter_than_one_step_is_just_its_ends():
    assert len(resample_by_arclength(np.array([[0.0, 0.0], [1.0, 0.0]]), step_m=5.0)) == 2


def test_the_measurement_parameters_are_reported():
    path = circle(100.0, n=4000)
    m = measure(path, step_m=12.5, chord_m=25.0)
    assert (m.step_m, m.chord_m) == (12.5, 25.0)
    assert m.r_min_m == pytest.approx(100.0, rel=5e-3)


def test_a_tight_bend_is_found_even_next_to_straight_sections():
    # straight 200 m along +x, a tangent quarter circle of radius 30 m turning left,
    # then straight 200 m along +y. Tangent-continuous, so the only curvature is the bend.
    r = 30.0
    theta = np.linspace(0.0, np.pi / 2, 600)
    before = np.column_stack([np.linspace(-200.0, 0.0, 200), np.zeros(200)])
    bend = np.column_stack([r * np.sin(theta), r - r * np.cos(theta)])
    after = np.column_stack([np.full(200, r), np.linspace(r, r + 200.0, 200)])
    path = np.vstack([before, bend, after])
    assert r_min_m(path, step_m=2.0, chord_m=8.0) == pytest.approx(r, rel=5e-2)
    # a straight-only prefix still reports infinity
    assert r_min_m(before, step_m=2.0, chord_m=8.0) == INF


# -- gradients --------------------------------------------------------------------------

def test_gradient_of_a_constant_slope():
    path = np.column_stack([np.linspace(0, 100, 11), np.zeros(11), np.linspace(0, 6, 11)])
    np.testing.assert_allclose(gradients_percent(path), 6.0)
    assert i_max_percent(path) == pytest.approx(6.0)


def test_gradient_sign_and_maximum():
    path = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 8.0], [200.0, 0.0, 3.0]])
    np.testing.assert_allclose(gradients_percent(path), [8.0, -5.0])
    assert i_max_percent(path) == pytest.approx(8.0)


def test_a_flat_road_has_zero_gradient():
    path = np.column_stack([straight(), np.zeros(50)])
    assert i_max_percent(path) == pytest.approx(0.0)


def test_gradients_need_a_z_column():
    with pytest.raises(ValueError):
        gradients_percent(straight())


def test_elevation_difference():
    path = np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 15.0], [20.0, 0.0, 12.0]])
    assert elevation_difference_m(path) == {
        'net_m': 2.0, 'ascent_m': 5.0, 'descent_m': 3.0, 'span_m': 5.0}


# -- deviation --------------------------------------------------------------------------

def test_deviation_of_a_path_from_itself_is_zero():
    path = circle(100.0, n=200, arc=np.pi)
    d = deviation(path, path)
    assert d['hausdorff_m'] == pytest.approx(0.0)
    assert d['mean_deviation_m'] == pytest.approx(0.0)


def test_deviation_of_two_parallel_lines_is_their_offset():
    a = straight(500.0)
    b = a + np.array([0.0, 7.0])
    d = deviation(a, b, step_m=5.0)
    assert d['hausdorff_m'] == pytest.approx(7.0, abs=1e-6)
    assert d['mean_deviation_m'] == pytest.approx(7.0, abs=1e-6)
    assert d['max_deviation_m'] == pytest.approx(7.0, abs=1e-6)


def test_hausdorff_is_symmetric():
    a, b = straight(100.0), straight(100.0) + np.array([3.0, 4.0])
    assert hausdorff_m(a, b) == pytest.approx(hausdorff_m(b, a))


def test_chord_deviation_of_a_circle_from_its_chord():
    # the semicircle's furthest point from its diameter is r
    r = 60.0
    arc = circle(r, n=2000, arc=np.pi)
    chord = np.array([arc[0], arc[-1]])
    assert deviation(arc, chord, step_m=1.0)['max_deviation_m'] == pytest.approx(r, rel=1e-3)


# -- the bundle --------------------------------------------------------------------------

def test_measure_bundles_everything_for_a_3d_path():
    path = np.column_stack([np.linspace(0, 100, 11), np.zeros(11), np.linspace(0, 6, 11)])
    m = measure(path)
    assert isinstance(m, PathMetrics)
    assert m.length_m == pytest.approx(100.0)
    assert m.r_min_m == INF
    assert m.i_max_percent == pytest.approx(6.0)
    assert m.elevation_net_m == pytest.approx(6.0)
    assert (m.step_m, m.chord_m) == (DEFAULT_RESAMPLE_STEP_M, DEFAULT_CHORD_M)
    assert set(m.as_dict()) == set(PathMetrics.__dataclass_fields__)


def test_measure_of_a_2d_path_leaves_the_z_measures_empty():
    m = measure(straight())
    assert m.length_3d_m is None and m.i_max_percent is None
    assert m.elevation_net_m is None
