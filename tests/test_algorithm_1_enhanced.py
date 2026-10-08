"""S4: the Algorithm 1 enhancements. Every one is opt-in; the default must not move."""
import numpy as np
import pytest

from core import metrics
from core.checks import STATION_CHECK_NAMES
from core.algorithm_1 import (ARM_ORDER, ARMS, Enhancements, algorithm_1,
                              algorithm_1_enhanced, before_after, before_after_enhanced,
                              before_after_guarded, element_deviations_m,
                              grade_line_xyz_m, search_s_value, search_s_value_per_element,
                              select_representative_points, short_interior_tangents)
from data.configs.road_classes import get
from tests.test_core_algorithm_1 import rough_path

V40 = get('STAS863_V40')
V25 = get('STAS863_V25')
NO_VERTICAL = get('RO_CLASS_IV_DEAL')


@pytest.fixture(scope='module')
def case():
    return rough_path()


@pytest.fixture(scope='module')
def rough_case():
    # the period must divide the grid size, which is what perlin_numpy requires
    return rough_path(seed=3, periods=(5, 5))


# -- the defaults must not move ----------------------------------------------------------

def test_no_enhancement_reproduces_plain_algorithm_1(case):
    grid, rough = case
    plain = algorithm_1(rough, V40, grid=grid)
    enhanced = algorithm_1_enhanced(grid, rough, V40, Enhancements())
    np.testing.assert_array_equal(plain.smooth_xy, enhanced.result.smooth_xy)
    np.testing.assert_array_equal(plain.smooth_xyz_m, enhanced.smooth_xyz_m)
    assert plain.s_search.s_value == enhanced.result.s_search.s_value


def test_the_baseline_arm_measures_what_the_guarded_report_measures(case):
    grid, rough = case
    guarded = before_after_guarded(grid, rough, V40)
    arm = before_after_enhanced(grid, rough, V40, ARMS['baseline'])
    assert arm['after']['r_min_m'] == pytest.approx(guarded['after']['r_min_m'])
    assert arm['after']['i_max_percent'] == pytest.approx(guarded['after']['i_max_percent'])
    assert arm['before'] == guarded['before']


def test_plain_before_after_is_untouched(case):
    grid, rough = case
    report = before_after(grid, rough, V40)
    assert set(report) == {'road_class', 'iterations', 's_search', 'n_representative_points',
                           'before', 'after', 'deviation', 'delta', 'r_min_meets_class',
                           'i_max_meets_class', 'warnings'}


def test_an_empty_enhancement_set_reports_nothing_enabled():
    assert not Enhancements().any_enabled
    assert ARMS['gradient_aware'].any_enabled


def test_every_arm_is_named_after_itself_and_listed_in_order():
    for name, arm in ARMS.items():
        assert arm.name == name
    assert set(ARMS) | {'none'} == set(ARM_ORDER)


# -- E1 gradient-aware smoothing -----------------------------------------------------------

def test_gradient_aware_never_leaves_a_compliant_gradient_worse_than_the_baseline(rough_case):
    grid, rough = rough_case
    rc = V25.with_cell_size(grid.cell_size_m)
    base = algorithm_1_enhanced(grid, rough, rc, ARMS['baseline'])
    aware = algorithm_1_enhanced(grid, rough, rc, ARMS['gradient_aware'])
    before_i = metrics.i_max_percent(grid.path_xyz_m(np.asarray(rough)[:, :2]))
    if before_i <= rc.i_max_percent:
        assert metrics.i_max_percent(aware.smooth_xyz_m) <= \
               metrics.i_max_percent(base.smooth_xyz_m) + 1e-9


def test_gradient_aware_reports_whether_it_recovered(rough_case):
    grid, rough = rough_case
    rc = V25.with_cell_size(grid.cell_size_m)
    aware = algorithm_1_enhanced(grid, rough, rc, ARMS['gradient_aware'])
    before_i = metrics.i_max_percent(grid.path_xyz_m(np.asarray(rough)[:, :2]))
    if before_i <= rc.i_max_percent:
        assert aware.gradient_recovered is not None
    assert algorithm_1_enhanced(grid, rough, rc, ARMS['baseline']).gradient_recovered is None


def test_gradient_aware_only_ever_loosens_the_point_selection(rough_case):
    grid, rough = rough_case
    rc = V25.with_cell_size(grid.cell_size_m)
    aware = algorithm_1_enhanced(grid, rough, rc, ARMS['gradient_aware'])
    # E1's lever is the selection radius, not the deviation budget
    assert aware.selection_radius_m <= rc.r_min_m + 1e-9
    assert aware.budget_m == pytest.approx(0.1 * rc.r_min_m)


def test_the_adjustment_loop_is_capped(rough_case):
    grid, rough = rough_case
    rc = V40.with_cell_size(grid.cell_size_m)
    arm = Enhancements(name='e1', gradient_aware=True, max_adjustments=2)
    assert algorithm_1_enhanced(grid, rough, rc, arm).n_adjustments <= 2


# -- E2 per-element deviation budget --------------------------------------------------------

def test_element_deviations_are_zero_against_the_reference_itself(case):
    grid, rough = case
    representative = select_representative_points(rough, V40)
    from core.algorithm_1 import _interpolate
    tight = _interpolate(representative, 8, 0.0)
    for _, deviation in element_deviations_m(tight, tight, V40):
        assert deviation == pytest.approx(0.0, abs=1e-9)


def test_element_deviations_label_each_element(case):
    grid, rough = case
    representative = select_representative_points(rough, V40)
    from core.algorithm_1 import _interpolate
    tight = _interpolate(representative, 8, 0.0)
    loose = _interpolate(representative, 8, 50.0)
    per_element = element_deviations_m(tight, loose, V40)
    assert per_element
    assert {kind for kind, _ in per_element} <= {'tangent', 'curve'}
    assert all(d >= 0 for _, d in per_element)


def test_the_per_element_search_returns_a_usable_s_value(case):
    grid, rough = case
    representative = select_representative_points(rough, V40)
    search = search_s_value_per_element(representative, V40)
    assert search.s_value >= 0.0
    assert search.n_evaluations > 0
    assert not search.degenerate


def test_the_per_element_arm_produces_a_different_axis_than_the_global_one(case):
    grid, rough = case
    glob = algorithm_1_enhanced(grid, rough, V40, ARMS['baseline'])
    per = algorithm_1_enhanced(grid, rough, V40, ARMS['per_element_budget'])
    assert per.result.s_search.s_value != glob.result.s_search.s_value


def test_a_degenerate_path_skips_the_per_element_search():
    representative = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    assert search_s_value_per_element(representative, V40).degenerate


# -- E3 minimum tangent length ----------------------------------------------------------------

def test_a_class_without_a_minimum_alignment_length_reports_no_short_tangents(case):
    grid, rough = case
    result = algorithm_1(rough, NO_VERTICAL, grid=grid)
    assert short_interior_tangents(result.smooth_xy, NO_VERTICAL) == []


def test_the_min_tangent_arm_reports_a_before_and_after_count(case):
    grid, rough = case
    enhanced = algorithm_1_enhanced(grid, rough, V40, ARMS['min_tangent'])
    assert enhanced.n_short_tangents_before is not None
    assert enhanced.n_short_tangents_after is not None
    assert enhanced.n_short_tangents_after <= enhanced.n_short_tangents_before


def test_other_arms_do_not_report_tangent_counts(case):
    grid, rough = case
    enhanced = algorithm_1_enhanced(grid, rough, V40, ARMS['baseline'])
    assert enhanced.n_short_tangents_before is None
    assert enhanced.n_short_tangents_after is None


# -- E4 radius compliance iteration -------------------------------------------------------------

def test_radius_iterate_reports_whether_the_class_was_met(case):
    grid, rough = case
    enhanced = algorithm_1_enhanced(grid, rough, V40, ARMS['radius_iterate'])
    assert enhanced.radius_compliant is not None
    assert enhanced.radius_compliant == (
        metrics.r_min_m(enhanced.smooth_xyz_m) >= V40.r_min_m)


def test_radius_iterate_never_ends_below_the_class_unless_it_hit_the_cap(case):
    grid, rough = case
    enhanced = algorithm_1_enhanced(grid, rough, V40, ARMS['radius_iterate'])
    assert enhanced.radius_compliant or enhanced.hit_adjustment_cap


def test_radius_iterate_only_ever_grows_the_budget(case):
    grid, rough = case
    enhanced = algorithm_1_enhanced(grid, rough, V40, ARMS['radius_iterate'])
    assert enhanced.budget_m >= 0.1 * V40.r_min_m - 1e-9


def test_a_class_that_is_already_met_costs_no_adjustment(case):
    grid, rough = case
    loose = get('SYNTHETIC_LOOSE')
    enhanced = algorithm_1_enhanced(grid, rough, loose, ARMS['radius_iterate'])
    if metrics.r_min_m(algorithm_1(rough, loose, grid=grid).smooth_xyz_m) >= loose.r_min_m:
        assert enhanced.n_adjustments == 0


# -- E5 vertical smoothing ------------------------------------------------------------------------

def test_the_grade_line_replaces_z_and_leaves_the_plan_alone(case):
    grid, rough = case
    result = algorithm_1(rough, V40, grid=grid)
    graded = grade_line_xyz_m(result.smooth_xyz_m, V40)
    assert graded.shape[1] == 3
    # the plan is re-stated on the station grid, so its length differs only by the
    # chord shortening of the resampling, not by any sideways movement
    assert metrics.path_length_m(graded) == pytest.approx(
        metrics.path_length_m(result.smooth_xyz_m), rel=1e-3)
    assert not np.allclose(graded[:, 2],
                           np.interp(np.linspace(0, 1, len(graded)),
                                     np.linspace(0, 1, len(result.smooth_xyz_m)),
                                     result.smooth_xyz_m[:, 2]))


def test_a_class_without_a_design_step_leaves_the_profile_alone(case):
    grid, rough = case
    result = algorithm_1(rough, NO_VERTICAL, grid=grid)
    np.testing.assert_array_equal(
        grade_line_xyz_m(result.smooth_xyz_m, NO_VERTICAL), result.smooth_xyz_m)


def test_the_smooth_z_arm_says_where_z_came_from(case):
    grid, rough = case
    assert algorithm_1_enhanced(grid, rough, V40, ARMS['smooth_z']).z_from_grade_line
    assert not algorithm_1_enhanced(grid, rough, V40, ARMS['baseline']).z_from_grade_line
    assert not algorithm_1_enhanced(grid, rough, NO_VERTICAL, ARMS['smooth_z']).z_from_grade_line


def test_smoothing_z_does_not_raise_the_gradient_above_the_terrain_profile(case):
    grid, rough = case
    plain = algorithm_1_enhanced(grid, rough, V40, ARMS['baseline'])
    graded = algorithm_1_enhanced(grid, rough, V40, ARMS['smooth_z'])
    # the grade line is a chord-wise average of the ground, so it cannot be steeper
    assert metrics.i_max_percent(graded.smooth_xyz_m) <= \
           metrics.i_max_percent(plain.smooth_xyz_m) + 1e-9


# -- the enhanced report ----------------------------------------------------------------------------

def test_the_enhanced_report_carries_the_station_checks_before_and_after(case):
    grid, rough = case
    report = before_after_enhanced(grid, rough, V40, ARMS['baseline'])
    for key in ('stations_before', 'stations_after'):
        assert set(report[key]['checks']) == set(STATION_CHECK_NAMES)


def test_the_enhanced_report_names_its_arm(case):
    grid, rough = case
    for name in ARMS:
        report = before_after_enhanced(grid, rough, V40, ARMS[name])
        assert report['arm'] == name
        assert report['enhancements']['arm'] == name


def test_the_enhanced_report_carries_the_class_standing(case):
    grid, rough = case
    report = before_after_enhanced(grid, rough, V40, ARMS['baseline'])
    assert report['road_class']['standing'] == 'official'


def test_a_cell_size_mismatch_is_refused(case):
    grid, rough = case
    with pytest.raises(ValueError, match='cell'):
        before_after_enhanced(grid, rough, V40.with_cell_size(grid.cell_size_m + 1.0),
                              ARMS['baseline'])


def test_every_arm_runs_end_to_end_on_a_real_path(case):
    grid, rough = case
    for name in ARMS:
        report = before_after_enhanced(grid, rough, V40, ARMS[name])
        assert np.isfinite(report['after']['length_m'])
        assert report['deviation']['hausdorff_m'] >= 0.0
