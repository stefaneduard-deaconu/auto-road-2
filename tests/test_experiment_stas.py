"""S6/S7: the STAS runner and its figures. Fast paths only; the full matrix is a CLI run."""
import csv

import numpy as np
import pytest

from core import experiment_stas as stas
from core import figures_stas
from core.checks import STATION_CHECK_NAMES, check_path_stations
from core.experiment_matrix import HEIGHT_DELTA_M, QUICK_TERRAINS
from core.hag import build_hag
from core.od_sampler import sample_od_pairs
from core.terrain import generate_terrain
from data.configs.road_classes import STAS_MATRIX_CLASSES, get
from tests.test_alignment import crest

V40 = get('STAS863_V40')


@pytest.fixture(scope='module')
def scenario():
    case = QUICK_TERRAINS[0]
    grid = generate_terrain(case.spec)
    hag = build_hag(grid, height_delta=HEIGHT_DELTA_M)
    sample = sample_od_pairs(grid, hag, terrain_id=case.terrain_id,
                             terrain_seed=case.spec.seed, n_pairs=1)
    return grid, hag, sample.pairs[0]


# -- the cost sweep -------------------------------------------------------------------

def test_every_cost_arm_resolves():
    for arm in stas.COST_ARMS:
        cost, cap = stas.resolve_cost(arm, V40)
        assert cost is not None
        assert cap is None or cap == V40.i_max_percent


def test_only_gradcut_restricts_the_graph():
    for arm in stas.COST_ARMS:
        _, cap = stas.resolve_cost(arm, V40)
        assert (cap is not None) == (arm == 'climb_gradient_cut')


def test_an_unknown_cost_arm_is_refused():
    with pytest.raises(ValueError, match='unknown grid edge cost'):
        stas.resolve_cost('curvature_lookahead', V40)


def test_the_gradient_penalty_is_tied_to_the_class():
    from core import costs
    strict, _ = stas.resolve_cost('climb_gradient_penalty', V40)
    loose, _ = stas.resolve_cost('climb_gradient_penalty', get('STAS863_V25'))
    assert costs.name_of(strict) != costs.name_of(loose)


# -- one scenario ---------------------------------------------------------------------

@pytest.fixture(scope='module')
def rows(scenario, tmp_path_factory):
    grid, hag, od = scenario
    return stas.run_stas_scenario(
        grid, hag, od, 'STAS863_V40',
        cost_arms=('climb_tiebreak',), series_dir=tmp_path_factory.mktemp('series'),
        series='all')


def test_one_row_per_search_arm_times_smoothing_arm(rows):
    smoothed = len(stas.SMOOTHING_ARMS) - 1
    assert len(rows) == len(stas.SEARCH_ARMS) * (1 + smoothed * len(stas.ITERATIONS))


def test_iterations_one_and_two_are_both_recorded(rows):
    smoothed = {row['algorithm_1_iterations'] for row in rows if row['smoothing_arm'] != 'none'}
    assert smoothed == set(stas.ITERATIONS)
    assert {row['algorithm_1_iterations'] for row in rows
            if row['smoothing_arm'] == 'none'} == {0}


def test_every_row_declares_its_arms_and_class(rows):
    for row in rows:
        assert row['search_space'] in {a[0] for a in stas.SEARCH_ARMS}
        assert row['hag_edge_cost'] == stas.HAG_EDGE_COST
        assert row['smoothing_arm'] in stas.SMOOTHING_ARMS
        assert row['road_class'] == 'STAS863_V40'
        assert row['road_class_standing'] == 'official'
        assert row['road_class_status'] == 'CONFIRMED'


def test_scenario_ids_are_unique(rows):
    ids = [row['scenario_id'] for row in rows]
    assert len(set(ids)) == len(ids)


def test_the_full_grid_arm_searches_the_whole_grid(rows):
    full = [r for r in rows if r['search_space'] == 'full_grid']
    assert all(r['search_space_percent_of_full'] == pytest.approx(100.0) for r in full)


def test_the_hag_arm_searches_less_than_the_full_grid(rows):
    hag = [r for r in rows if r['search_space'] == 'hag_cta']
    assert all(r['search_space_percent_of_full'] <= 100.0 for r in hag)
    assert all(r['search_nodes_expanded'] > 0 for r in hag)


def test_the_unsmoothed_arm_has_no_deviation(rows):
    for row in (r for r in rows if r['smoothing_arm'] == 'none'):
        assert row['deviation_hausdorff_m'] == 0.0
        assert row['length_m'] == row['rough_length_m']


def test_every_stas_parameter_gets_its_column_group(rows):
    for row in rows:
        for name in STATION_CHECK_NAMES:
            assert f'stas_{name}_status' in row
            assert row[f'stas_{name}_status'] in ('pass', 'fail', 'not applicable')
            assert row[f'stas_{name}_n_violations'] >= 0


def test_a_not_applicable_parameter_reports_no_violations(rows):
    for row in rows:
        for name in STATION_CHECK_NAMES:
            if row[f'stas_{name}_status'] == 'not applicable':
                assert row[f'stas_{name}_n_violations'] == 0
                assert row[f'stas_{name}_violation_length_m'] == 0.0


def test_feasible_means_no_applicable_parameter_failed(rows):
    for row in rows:
        failed = any(row[f'stas_{n}_status'] == 'fail' for n in STATION_CHECK_NAMES)
        assert row['stas_feasible'] == (not failed)


def test_every_column_of_the_contract_is_produced(rows):
    known = set(stas.COLUMNS)
    for row in rows:
        assert set(row) <= known, set(row) - known


def test_the_eq1_height_cost_is_the_sum_of_absolute_height_differences(rows):
    for row in rows:
        assert row['height_cost'] >= 0.0


# -- the per-station series file ---------------------------------------------------------

def test_a_series_file_has_one_row_per_station(tmp_path):
    report = check_path_stations(crest(), V40)
    path = stas.write_series(tmp_path / 'one.csv', report)
    with open(path, newline='', encoding='utf-8') as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == len(report.geometry.chainage_m)


def test_a_series_file_carries_every_parameter_and_its_limit(tmp_path):
    report = check_path_stations(crest(), V40)
    path = stas.write_series(tmp_path / 'one.csv', report)
    series = figures_stas.read_series(path)
    for column in ('chainage_m', 'r_horizontal_m', 'gradient_percent',
                   'tangent_length_m', 'r_vertical_m', 'vertical_kind',
                   'limit_R_H', 'limit_i', 'limit_L_alignment', 'limit_R_V'):
        assert column in series


def test_the_series_round_trips_the_measured_values(tmp_path):
    report = check_path_stations(crest(), V40)
    series = figures_stas.read_series(stas.write_series(tmp_path / 'one.csv', report))
    np.testing.assert_allclose(series['chainage_m'], report.geometry.chainage_m, rtol=1e-5)
    np.testing.assert_allclose(series['gradient_percent'],
                               report.geometry.gradient_percent, rtol=1e-4)


def test_the_vertical_limit_follows_the_sign_of_the_grade_break(tmp_path):
    report = check_path_stations(crest(), V40)
    series = figures_stas.read_series(stas.write_series(tmp_path / 'one.csv', report))
    for kind, limit in zip(series['vertical_kind'], series['limit_R_V']):
        if kind == 'concave':
            assert limit == V40.r_vert_concave_min_m
        elif kind == 'convex':
            assert limit == V40.r_vert_convex_min_m
        else:
            assert np.isnan(limit)


def test_a_class_without_vertical_geometry_writes_empty_limits(tmp_path):
    report = check_path_stations(crest(), get('RO_CLASS_IV_DEAL'))
    series = figures_stas.read_series(stas.write_series(tmp_path / 'one.csv', report))
    assert np.isnan(series['limit_R_V']).all()
    assert np.isnan(series['limit_L_alignment']).all()


def test_only_the_reference_scenarios_get_a_series_by_default():
    reference = {'search_space': 'hag_cta', 'grid_edge_cost': 'climb_tiebreak'}
    other = {'search_space': 'full_grid', 'grid_edge_cost': 'length_3d'}
    assert stas._wants_series('reference', reference)
    assert not stas._wants_series('reference', other)
    assert stas._wants_series('all', other)
    assert not stas._wants_series('none', reference)


# -- the figures ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def series(tmp_path_factory):
    path = tmp_path_factory.mktemp('series') / 's.csv'
    return figures_stas.read_series(
        stas.write_series(path, check_path_stations(crest(length_m=900.0), V40)))


def test_each_panel_draws(series):
    import matplotlib.pyplot as plt
    for column, limit_column, label, units, side in figures_stas.PANELS:
        fig = figures_stas.fig_parameter(series, column, limit_column, label, units, side)
        assert fig.axes
        plt.close(fig)


def test_the_stacked_figure_has_one_axis_per_parameter(series):
    import matplotlib.pyplot as plt
    fig = figures_stas.fig_all_parameters(series, title='t')
    assert len(fig.axes) == len(figures_stas.PANELS)
    plt.close(fig)


def test_the_compliance_strip_has_one_row_per_parameter(series):
    import matplotlib.pyplot as plt
    fig = figures_stas.fig_compliance_strip(series)
    assert len(fig.axes[0].get_yticks()) == len(figures_stas.PANELS)
    plt.close(fig)


def test_a_caption_never_claims_compliance_for_a_limit_that_governs_nothing(series):
    import matplotlib.pyplot as plt
    # a straight profile in plan: no curve, so R_H governs no station
    straight = dict(series)
    straight['r_horizontal_m'] = np.full(len(series['chainage_m']), np.inf)
    fig, ax = plt.subplots()
    figures_stas.fig_parameter(straight, 'r_horizontal_m', 'limit_R_H', 'R', 'm',
                               'above', ax=ax)
    assert 'no station on this alignment is governed by it' in ax.get_title(loc='left')
    plt.close(fig)


def test_a_caption_counts_the_stations_a_limit_actually_governs(series):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    figures_stas.fig_parameter(series, 'gradient_percent', 'limit_i', 'i', '%',
                               'below', ax=ax)
    title = ax.get_title(loc='left')
    assert 'stations' in title
    assert 'no station on this alignment is governed by it' not in title
    plt.close(fig)


def test_violations_are_detected_on_the_correct_side_of_the_limit():
    values = np.array([10.0, 30.0, 50.0])
    limit = np.full(3, 25.0)
    np.testing.assert_array_equal(
        figures_stas._violating(values, limit, 'above'), [True, False, False])
    np.testing.assert_array_equal(
        figures_stas._violating(values, limit, 'below'), [False, True, True])


def test_a_nan_station_never_counts_as_a_violation():
    values = np.array([np.nan, 10.0])
    limit = np.full(2, 25.0)
    np.testing.assert_array_equal(
        figures_stas._violating(values, limit, 'above'), [False, True])


def test_the_arm_boxplot_draws_from_summary_rows():
    import matplotlib.pyplot as plt
    rows = [{'smoothing_arm': 'baseline', 'stas_R_H_violation_length_m': '12.0'},
            {'smoothing_arm': 'smooth_z', 'stas_R_H_violation_length_m': '4.0'}]
    fig = figures_stas.fig_arm_violations(rows, 'R_H')
    assert fig.axes
    plt.close(fig)


def test_the_column_contract_matches_the_classes_and_arms():
    assert set(STAS_MATRIX_CLASSES) == {'STAS863_V40', 'STAS863_V25'}
    assert 'none' in stas.SMOOTHING_ARMS
    assert len(stas.COLUMNS) == len(set(stas.COLUMNS))
