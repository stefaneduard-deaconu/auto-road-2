"""T5 acceptance: one `algorithm_1(path, road_class)`, a real s-value search, Z
re-sampled from the terrain, and switching the class moves the result as section 3b says.
"""
import math

import numpy as np
import pytest

from core.algorithm_1 import (DEVIATION_FRACTION_OF_R_MIN, algorithm_1, before_after,
                              format_before_after, insert_midpoints,
                              search_s_value, select_representative_points)
from core.grid import Grid
from core.hag import build_hag, select_areas
from core.metrics import INF, measure
from core.search import build_graph, dijkstra
from core.terrain import TerrainSpec, generate_terrain
from data.configs.road_classes import get

STRICT = get('RO_CLASS_III_DEAL')   # R >= 240 m, i <= 6%
MIDDLE = get('RO_CLASS_IV_DEAL')    # R >= 125 m, i <= 6.5%
LOOSE = get('SYNTHETIC_LOOSE')      # R >= 25 m, i <= 10%


def rough_path(seed: int = 0, size: int = 100, periods=(4, 4)):
    """A real Dijkstra output on the HAG-restricted grid, which is what T5 smooths."""
    grid = generate_terrain(TerrainSpec(seed=seed, grid_size=(size, size), periods=periods))
    start, target = (5, 5), (size - 6, size - 6)
    hag = build_hag(grid, height_delta=3.0)
    mask = hag.mask(select_areas(hag, start, target, 'yellow'))
    mask[start] = mask[target] = True
    graph = build_graph(grid, mask=mask)
    result = dijkstra(graph, grid.index(start), grid.index(target))
    assert result.reached
    return grid, result.coords(grid).astype(float)


@pytest.fixture(scope="module")
def case():
    return rough_path()


# -- the mid-point rule reads the class -------------------------------------------

def test_midpoint_rule_uses_the_class_radius():
    # segments of 60 m at a 10 m cell size
    rep = np.array([[0., 0.], [6., 0.], [6., 6.], [12., 6.]])
    # r_min 25 m -> threshold pi*25 = 78.5 m, so a 60 m segment gets no mid-point
    assert len(insert_midpoints(rep, LOOSE)) == len(rep)
    # the same class on a 50 m DEM: the 6-cell segments are 300 m, well past pi*25 m
    assert len(insert_midpoints(rep, LOOSE.with_cell_size(50.0))) == len(rep) + 2




def test_short_paths_are_returned_unchanged():
    rep = np.array([[0., 0.], [1., 1.]])
    np.testing.assert_array_equal(insert_midpoints(rep, LOOSE), rep)


# -- the s-value search really searches ---------------------------------------------

def test_the_s_value_search_actually_searches(case):
    grid, rough = case
    rep = select_representative_points(rough, MIDDLE)
    search = search_s_value(rep, MIDDLE)
    # a real search evaluates several s-values
    assert search.n_evaluations > 2
    assert search.s_value != 1.0
    assert search.s_value > 0.0
    assert search.deviation_m <= search.budget_m
    assert not search.hit_cap


def test_the_s_value_found_is_the_largest_within_budget(case):
    grid, rough = case
    rep = select_representative_points(rough, MIDDLE)
    search = search_s_value(rep, MIDDLE)
    from core.algorithm_1 import _interpolate
    from core.metrics import hausdorff_m
    tight = _interpolate(rep, 8, 0.0) * MIDDLE.cell_size_m
    just_above = _interpolate(rep, 8, search.s_value * 1.5) * MIDDLE.cell_size_m
    assert hausdorff_m(tight, just_above) > search.budget_m


def test_the_deviation_budget_is_in_meters(case):
    grid, rough = case
    rep = select_representative_points(rough, MIDDLE)
    assert search_s_value(rep, MIDDLE).budget_m == pytest.approx(
        DEVIATION_FRACTION_OF_R_MIN * MIDDLE.r_min_m)
    # the budget must follow the cell size, not silently change meaning with it
    assert search_s_value(rep, MIDDLE.with_cell_size(30.0)).budget_m == pytest.approx(
        DEVIATION_FRACTION_OF_R_MIN * MIDDLE.r_min_m)


def test_a_larger_budget_allows_a_looser_spline(case):
    grid, rough = case
    rep = select_representative_points(rough, MIDDLE)
    tight = search_s_value(rep, MIDDLE, max_deviation_m=1.0)
    loose = search_s_value(rep, MIDDLE, max_deviation_m=50.0)
    assert loose.s_value > tight.s_value
    assert loose.deviation_m >= tight.deviation_m


def test_the_search_is_deterministic(case):
    grid, rough = case
    rep = select_representative_points(rough, MIDDLE)
    assert search_s_value(rep, MIDDLE).s_value == search_s_value(rep, MIDDLE).s_value


# -- XY only, then Z from the terrain (research step 3) -------------------------------

def test_smoothing_is_xy_only_and_z_comes_from_the_terrain(case):
    grid, rough = case
    result = algorithm_1(rough, MIDDLE, grid=grid)
    assert result.smooth_xy.shape[1] == 2
    assert result.smooth_xyz_m.shape[1] == 3
    # every Z is the bilinear terrain height at the smoothed XY, in meters
    np.testing.assert_allclose(result.smooth_xyz_m[:, 2],
                               grid.interpolate_height(result.smooth_xy))
    # and the XY columns are in meters
    np.testing.assert_allclose(result.smooth_xyz_m[:, :2],
                               result.smooth_xy * grid.cell_size_m)


def test_z_is_resampled_not_carried_over(case):
    grid, rough = case
    result = algorithm_1(rough, MIDDLE, grid=grid)
    rough_z = grid.path_xyz_m(rough)[:, 2]
    smooth_z = result.smooth_xyz_m[:, 2]
    # the axis moved, so the elevation profile is a different one
    assert not np.isclose(smooth_z.max(), rough_z.max(), atol=1e-9) or \
        not np.isclose(smooth_z.min(), rough_z.min(), atol=1e-9)


def test_without_a_grid_only_xy_is_produced(case):
    _, rough = case
    assert algorithm_1(rough, MIDDLE).smooth_xyz_m is None


def test_iterations_must_be_at_least_one(case):
    _, rough = case
    with pytest.raises(ValueError):
        algorithm_1(rough, MIDDLE, iterations=0)


def test_repeating_the_steps_smooths_further(case):
    grid, rough = case
    once = algorithm_1(rough, MIDDLE, grid=grid, iterations=1)
    twice = algorithm_1(rough, MIDDLE, grid=grid, iterations=2)
    assert twice.iterations == 2
    m1 = measure(once.smooth_xy * grid.cell_size_m)
    m2 = measure(twice.smooth_xy * grid.cell_size_m)
    assert m2.length_m <= m1.length_m * 1.05


# -- no prints, no literals ----------------------------------------------------------------

def test_algorithm_1_prints_nothing(case, capsys):
    grid, rough = case
    algorithm_1(rough, MIDDLE, grid=grid)
    assert capsys.readouterr().out == ''


def test_the_module_holds_no_radius_or_cell_size_literals():
    import inspect

    from core import algorithm_1 as mod
    code = [line.split('#')[0] for line in inspect.getsource(mod).splitlines()]
    code = chr(10).join(l for l in code if not l.strip().startswith(('*', '"', "'")))
    for literal in ('3.1458', 'MIDPOINT_RULE_RADIUS', 'MIN_RADIUS', 'SCALE ='):
        assert literal not in code, literal
    # the only radius and cell size in the signature come from the RoadClass
    params = inspect.signature(mod.algorithm_1).parameters
    assert 'road_class' in params
    assert not {'MINIMAL_RADIUS', 'GRID_RATIO_TO_METERS', 'minimal_radius'} & set(params)


# -- the class drives the result (section 3b acceptance test) --------------------------------

def test_a_stricter_radius_gives_a_larger_r_min_and_a_larger_deviation(case):
    grid, rough = case
    loose = before_after(grid, rough, LOOSE)
    strict = before_after(grid, rough, STRICT)
    assert strict['after']['r_min_m'] > loose['after']['r_min_m']
    assert strict['deviation']['hausdorff_m'] > loose['deviation']['hausdorff_m']


def test_the_before_after_report_has_both_sides(case):
    grid, rough = case
    report = before_after(grid, rough, MIDDLE)
    assert set(report) >= {'before', 'after', 'delta', 'deviation', 'road_class',
                           's_search', 'warnings'}
    assert report['road_class']['status'] == 'TO CONFIRM'
    assert report['before']['length_m'] > 0 and report['after']['length_m'] > 0
    assert report['before']['i_max_percent'] is not None
    assert report['after']['i_max_percent'] is not None
    assert report['delta']['r_min_m'] == report['after']['r_min_m'] - report['before']['r_min_m']


def test_a_radius_below_the_class_is_warned_about_and_never_relaxed(case):
    grid, rough = case
    report = before_after(grid, rough, STRICT)
    if report['after']['r_min_m'] < STRICT.r_min_m:
        assert not report['r_min_meets_class']
        assert any('infeasible under this class' in w for w in report['warnings'])
    # the class itself is untouched whatever happened
    assert STRICT.r_min_m == 240.0


def test_smoothing_that_worsens_the_gradient_is_reported(case):
    grid, rough = case
    report = before_after(grid, rough, STRICT)
    worse = report['after']['i_max_percent'] > report['before']['i_max_percent']
    assert worse == any('made the longitudinal geometry worse' in w
                        for w in report['warnings'])


def test_the_report_formats_as_a_table(case):
    grid, rough = case
    text = format_before_after(before_after(grid, rough, MIDDLE))
    for label in ('length (m)', 'R_min (m)', 'i_max (%)', 'before', 'after',
                  'TO CONFIRM', 's_value'):
        assert label in text




def test_endpoints_are_kept_close(case):
    grid, rough = case
    result = algorithm_1(rough, MIDDLE, grid=grid)
    assert np.linalg.norm(result.representative_xy[0] - rough[0]) < 1e-9
    assert np.linalg.norm(result.representative_xy[-1] - rough[-1]) < 1e-9
    assert np.linalg.norm(result.smooth_xy[0] - rough[0]) * grid.cell_size_m < MIDDLE.r_min_m
