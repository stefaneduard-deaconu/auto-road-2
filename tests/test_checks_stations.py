"""S3: the per-station STAS checks, on shapes whose compliance is known in advance."""
import numpy as np
import pytest

from core import alignment, checks, metrics
from data.configs.road_classes import get
from tests.test_alignment import circular_arc, crest, from_curvature, straight_line

V40 = get('STAS863_V40')
V25 = get('STAS863_V25')
NO_VERTICAL = get('RO_CLASS_IV_DEAL')


def test_the_check_names_and_their_order_are_fixed():
    report = checks.check_path_stations(crest(), V40)
    assert [c.name for c in report.checks] == list(checks.STATION_CHECK_NAMES)


# -- R_H --------------------------------------------------------------------------------

def test_a_wide_arc_passes_the_radius_check_at_every_station():
    report = checks.check_path_stations(circular_arc(radius_m=400.0), V40)
    r = report.by_name()['R_H']
    assert r.applicable and r.passed
    assert r.n_violations == 0
    assert r.worst_value == pytest.approx(400.0, rel=0.01)


def test_a_tight_arc_fails_at_every_curve_station_and_reports_the_length():
    radius = 30.0                       # below V40's 60 m
    report = checks.check_path_stations(circular_arc(radius_m=radius), V40)
    r = report.by_name()['R_H']
    assert not r.passed
    assert r.n_violations == r.n_applicable
    assert r.violation_length_m == pytest.approx(report.geometry.length_m, rel=0.05)
    assert r.worst_value < V40.r_min_m


def test_the_same_tight_arc_passes_the_looser_class():
    report = checks.check_path_stations(circular_arc(radius_m=30.0), V25)
    assert report.by_name()['R_H'].passed


def test_a_straight_line_has_no_station_the_radius_limit_governs():
    report = checks.check_path_stations(straight_line(), V40)
    r = report.by_name()['R_H']
    assert not r.applicable
    assert r.n_applicable == 0
    assert 'no station' in r.note


def test_the_worst_radius_matches_the_one_curvature_method():
    path = circular_arc(radius_m=45.0)
    report = checks.check_path_stations(path, V40)
    assert report.by_name()['R_H'].worst_value == pytest.approx(metrics.r_min_m(path), rel=1e-9)


# -- i ----------------------------------------------------------------------------------

def test_a_gentle_constant_grade_passes_everywhere():
    report = checks.check_path_stations(straight_line(slope=0.04), V40)
    i = report.by_name()['i']
    assert i.applicable and i.passed
    assert i.worst_value == pytest.approx(4.0)


def test_a_steep_constant_grade_fails_at_every_station():
    report = checks.check_path_stations(straight_line(slope=0.12), V40)
    i = report.by_name()['i']
    assert not i.passed
    assert i.n_violations == i.n_applicable
    assert i.worst_value == pytest.approx(12.0)


def test_the_gradient_check_uses_the_absolute_value():
    down = checks.check_path_stations(straight_line(slope=-0.12), V40).by_name()['i']
    assert not down.passed and down.worst_value == pytest.approx(12.0)


def test_a_gradient_exactly_at_the_limit_is_not_failed_by_rounding():
    report = checks.check_path_stations(straight_line(slope=V40.i_max_percent / 100.0), V40)
    assert report.by_name()['i'].passed


@pytest.mark.parametrize('road_class', [V40, V25])
def test_the_minimum_alignment_length_is_one_point_four_v(road_class):
    # the article's eq. (1): L_alignment,min = 1.4 V
    assert road_class.l_alignment_min_m == pytest.approx(1.4 * road_class.design_speed_kmh)


# -- L_alignment ------------------------------------------------------------------------

def _two_curves_with_a_tangent(tangent_m):
    """Arc, straight tangent of the given length, arc: exactly one interior tangent.

    The measured tangent comes out about `alignment.tangent_detection_floor_m()` shorter
    than `tangent_m`, so the lengths chosen below allow for that.
    """
    return from_curvature([(120.0, 80.0), (tangent_m, None), (120.0, -80.0)])


def test_a_long_enough_tangent_between_two_curves_passes():
    report = checks.check_path_stations(_two_curves_with_a_tangent(200.0), V40)
    check = report.by_name()['L_alignment']
    assert check.applicable
    assert check.passed, check


def test_a_short_tangent_between_two_curves_fails():
    # 50 m true, about 20 m measured, against V40's 56 m minimum
    report = checks.check_path_stations(_two_curves_with_a_tangent(50.0), V40)
    check = report.by_name()['L_alignment']
    assert check.applicable
    assert not check.passed
    assert check.worst_value < V40.l_alignment_min_m


def test_the_same_tangent_passes_the_looser_class():
    # 80 m true, about 50 m measured: under V40's 56 m, over V25's 35 m
    path = _two_curves_with_a_tangent(80.0)
    assert not checks.check_path_stations(path, V40).by_name()['L_alignment'].passed
    assert checks.check_path_stations(path, V25).by_name()['L_alignment'].passed


def test_a_tangent_too_short_to_measure_yields_no_check_rather_than_a_pass():
    report = checks.check_path_stations(_two_curves_with_a_tangent(30.0), V40)
    check = report.by_name()['L_alignment']
    assert not check.applicable
    assert check.n_violations == 0


def test_end_tangents_are_not_judged():
    line = straight_line(length_m=30.0, n=13)
    arc = circular_arc(radius_m=80.0, n=161) + np.array([line[-1, 0], 0.0, 0.0])
    report = checks.check_path_stations(np.vstack([line, arc[1:]]), V40)
    check = report.by_name()['L_alignment']
    # the only tangent is the leading one, which runs to a terminal
    assert not check.applicable
    assert 'no station' in check.note


# -- R_V --------------------------------------------------------------------------------

def test_a_sharp_crest_fails_the_convex_radius_and_leaves_concave_unjudged():
    report = checks.check_path_stations(crest(length_m=400.0, rise=0.05, fall=-0.05), V40)
    convex, concave = report.by_name()['R_V_convex'], report.by_name()['R_V_concave']
    assert convex.applicable and not convex.passed
    assert not concave.applicable


def test_a_gentle_sag_is_judged_against_the_concave_limit():
    report = checks.check_path_stations(crest(length_m=2000.0, rise=-0.005, fall=0.005), V40)
    concave = report.by_name()['R_V_concave']
    assert concave.applicable
    assert concave.limit == V40.r_vert_concave_min_m


def test_the_two_stas_classes_use_different_vertical_limits():
    path = crest(length_m=600.0, rise=0.04, fall=-0.06)
    strict = checks.check_path_stations(path, V40).by_name()['R_V_convex']
    loose = checks.check_path_stations(path, V25).by_name()['R_V_convex']
    assert strict.limit == 1000.0 and loose.limit == 500.0
    assert strict.worst_value == pytest.approx(loose.worst_value)


def test_a_constant_grade_has_no_vertical_curve_to_judge():
    report = checks.check_path_stations(straight_line(slope=0.03), V40)
    assert not report.by_name()['R_V_convex'].applicable
    assert not report.by_name()['R_V_concave'].applicable


# -- a class that defines no vertical geometry --------------------------------------------

def test_an_unofficial_class_reports_its_undefined_limits_as_such():
    report = checks.check_path_stations(_two_curves_with_a_tangent(40.0), NO_VERTICAL)
    for name in ('L_alignment', 'R_V_concave', 'R_V_convex'):
        check = report.by_name()[name]
        assert not check.applicable, name
        assert check.limit is None, name
        assert check.note == 'the road class does not define this limit', name


def test_an_undefined_limit_never_counts_as_a_failure():
    report = checks.check_path_stations(crest(), NO_VERTICAL)
    assert report.feasible == all(c.passed for c in report.applicable_checks)
    assert {c.name for c in report.applicable_checks} == {'R_H', 'i'} or \
           {c.name for c in report.applicable_checks} == {'i'}


# -- the report -----------------------------------------------------------------------------

def test_the_verdict_names_every_failing_parameter():
    report = checks.check_path_stations(circular_arc(radius_m=20.0), V40)
    assert not report.feasible
    assert 'R_H' in report.verdict
    assert 'infeasible under this class' in report.verdict


def test_a_compliant_alignment_says_so_and_lists_what_was_skipped():
    report = checks.check_path_stations(straight_line(slope=0.02), V40)
    assert report.feasible
    assert 'meets STAS863_V40 at every station' in report.verdict


def test_the_report_carries_the_class_standing_and_status():
    report = checks.check_path_stations(crest(), V40)
    assert report.road_class_standing == 'official'
    assert report.road_class_status == 'CONFIRMED'
    assert checks.check_path_stations(crest(), NO_VERTICAL).road_class_standing == 'unofficial'


def test_as_dict_is_serialisable_and_carries_the_measurement_parameters():
    data = checks.check_path_stations(crest(), V40).as_dict()
    assert set(data['checks']) == set(checks.STATION_CHECK_NAMES)
    assert data['step_m'] == metrics.DEFAULT_RESAMPLE_STEP_M
    assert data['chord_m'] == metrics.DEFAULT_CHORD_M
    assert data['straight_radius_m'] == pytest.approx(1000.0)
    assert data['road_class_standing'] == 'official'
    for check in data['checks'].values():
        assert check['status'] in ('pass', 'fail', 'not applicable')


def test_total_violation_length_is_the_sum_over_applicable_checks():
    report = checks.check_path_stations(circular_arc(radius_m=20.0), V40)
    assert report.total_violation_length_m == pytest.approx(
        sum(c.violation_length_m for c in report.applicable_checks))


def test_a_geometry_can_be_reused_instead_of_remeasured():
    path = crest()
    geometry = alignment.measure_alignment(path, V40)
    a = checks.check_path_stations(path, V40)
    b = checks.check_path_stations(path, V40, geometry=geometry)
    assert a.as_dict()['checks'] == b.as_dict()['checks']
    assert b.geometry is geometry


# -- the aggregate checks are untouched ------------------------------------------------------

def test_the_aggregate_check_still_reports_exactly_two_checks():
    report = checks.check_path(crest(), V40)
    assert [c.name for c in report.checks] == ['R_min', 'i_max']


def test_the_aggregate_and_station_radius_agree_on_a_pure_arc():
    path = circular_arc(radius_m=45.0)
    assert checks.check_path(path, V40).checks[0].value == pytest.approx(
        checks.check_path_stations(path, V40).by_name()['R_H'].worst_value)
