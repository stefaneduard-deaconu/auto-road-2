"""The result-file contract of the study protocol sections 2 and 4."""
import math

import pytest

from core.results_io import (MATRIX_COLUMNS, fmt_value, flatten_scenario, parse_value,
                             path_csv_name, read_csv, read_jsonl, write_csv,
                             write_path_csv)

#: a hard-coded copy of the field list of the study protocol This test IS
#: the spec check: if the plan changes, this list changes, and the code must follow.
SPEC_FIELDS = {
    # 2.1 identity and provenance
    'scenario_id', 'terrain_id', 'terrain_seed', 'grid_size', 'periods',
    'height_interval', 'height_delta', 'od_id', 'od_seed', 'start', 'target',
    'road_class', 'road_class_status', 'road_class_source', 'arm', 'rule', 'k_hops',
    'cost_model', 'connectivity', 'algorithm_1_iterations', 'metric_step_m',
    'metric_chord_m', 'timestamp_utc', 'git_sha', 'git_dirty', 'git_branch', 'python',
    'python_free_threaded', 'platform', 'numpy', 'scipy',
    # 2.2 step 1, cost
    'n_areas_total', 'n_areas_selected', 'n_cells_selected', 'n_cells_total',
    'search_space_percent_of_full', 'search_space_reduction_percent', 'n_graph_edges',
    'search_nodes_expanded', 'search_nodes_pushed', 'search_nodes_reached',
    'wall_time_s_median', 'wall_time_s_min', 'wall_time_s_max', 'peak_memory_bytes',
    'hag_build_time_s', 'time_reduction_percent',
    # 2.3 step 1, quality
    'path_found', 'infeasible_reason', 'n_path_cells', 'path_length_m',
    'objective_cost_sum_abs_dh_m', 'objective_cost_ratio_to_baseline',
    'length_ratio_to_baseline', 'vs_baseline_hausdorff_m',
    'vs_baseline_mean_deviation_m', 'vs_baseline_median_deviation_m',
    'vs_baseline_max_deviation_m',
    # 2.4 step 3 (the before_/after_ families are checked separately below)
    'n_representative_points', 's_value', 's_deviation_m', 's_budget_m',
    's_n_evaluations', 's_hit_cap', 's_degenerate', 'deviation_hausdorff_m',
    'deviation_mean_m', 'deviation_median_m', 'deviation_max_m', 'delta_r_min_m',
    'delta_i_max_percent', 'delta_length_m', 'delta_sinuosity',
    # 2.5 step 2
    'check_r_min_value', 'check_r_min_limit', 'check_r_min_passed', 'check_r_min_note',
    'check_i_max_value', 'check_i_max_limit', 'check_i_max_passed', 'feasible',
    'verdict',
}

BEFORE_AFTER = ('length_m', 'length_3d_m', 'sinuosity', 'r_min_m', 'i_max_percent',
                'i_min_signed_percent', 'i_max_signed_percent', 'elevation_net_m',
                'elevation_ascent_m', 'elevation_descent_m', 'elevation_span_m')


def test_every_field_the_plan_lists_is_a_column():
    missing = SPEC_FIELDS - set(MATRIX_COLUMNS)
    assert not missing, f'the study protocol lists these, the CSV does not: {missing}'


def test_the_before_and_after_families_are_complete():
    for name in BEFORE_AFTER:
        assert f'before_{name}' in MATRIX_COLUMNS
        assert f'after_{name}' in MATRIX_COLUMNS


def test_the_columns_are_unique():
    assert len(MATRIX_COLUMNS) == len(set(MATRIX_COLUMNS))


@pytest.mark.parametrize('value,text', [
    (None, ''), (math.inf, 'inf'), (-math.inf, '-inf'), (1.5, '1.5'),
    (True, 'True'), (3, '3'), ([100, 100], '100x100'),
])
def test_fmt_value(value, text):
    assert fmt_value(value) == text


def test_nan_is_written_as_a_word():
    assert fmt_value(math.nan) == 'nan'


@pytest.mark.parametrize('text,value', [
    ('', None), ('inf', math.inf), ('-inf', -math.inf), ('1.5', 1.5),
    ('True', True), ('False', False), ('3', 3), ('a b', 'a b'),
])
def test_parse_value(text, value):
    assert parse_value(text) == value


def test_non_finite_values_survive_a_csv_round_trip(tmp_path):
    """`r_min_m` is genuinely infinite for a straight alignment; it must not become 0."""
    rows = [{'a': math.inf, 'b': -math.inf, 'c': None, 'd': 2.5}]
    path = write_csv(tmp_path / 'x.csv', rows, ('a', 'b', 'c', 'd'))
    back = read_csv(path)[0]
    assert back['a'] == math.inf
    assert back['b'] == -math.inf
    assert back['c'] is None
    assert back['d'] == 2.5


def _scenario(path_found=True):
    metrics = {name: 1.0 for name in BEFORE_AFTER}
    report = {
        'road_class': {'name': 'C', 'status': 'TO CONFIRM', 'source': 'MT-2017'},
        'n_representative_points': 7,
        's_search': {'s_value': 2.0, 'deviation_m': 1.0, 'budget_m': 12.0,
                     'n_evaluations': 4, 'hit_cap': False, 'degenerate': False},
        'before': dict(metrics, i_max_percent=5.0),
        'after': dict(metrics, i_max_percent=9.0),
        'deviation': {'hausdorff_m': 3.0, 'mean_deviation_m': 1.0,
                      'median_deviation_m': 0.5, 'max_deviation_m': 3.0},
        'delta': {'r_min_m': 10.0, 'i_max_percent': 4.0, 'length_m': -5.0,
                  'sinuosity': -0.1},
    }
    checks = {'feasible': False, 'verdict': 'infeasible under this class (C): i_max',
              'checks': {'R_min': {'value': 30.0, 'limit': 25.0, 'passed': True,
                                   'note': ''},
                         'i_max': {'value': 9.0, 'limit': 6.0, 'passed': False}}}
    arm = {'arm': 'hag_yellow', 'rule': 'yellow', 'k_hops': 0,
           'path_found': path_found, 'search_space_percent_of_full': 24.80}
    geometry = ({'hag_yellow': {'C': {'1': {'before_after': report, 'checks': checks},
                                      '2': {'before_after': report, 'checks': checks}}}}
                if path_found else {})
    return {'scenario_id': 'T__od1', 'terrain_id': 'T', 'terrain_seed': 0,
            'od_id': 'od1', 'od_seed': 1000, 'od_protocol': True,
            'hag_build_time_s': 0.1,
            'config': {'terrain': {'grid_size': [100, 100], 'periods': [4, 4],
                                   'height_interval': [100, 120]},
                       'height_delta': 3.0, 'start': [5, 5], 'target': [94, 94],
                       'cost': 'height_tiebreak', 'connectivity': 8,
                       'metric_step_m': 5.0, 'metric_chord_m': 20.0},
            'provenance': {'git_sha': 'abc', 'python': '3.14.7',
                           'python_free_threaded': True},
            'arms': {'hag_yellow': arm}, 'geometry': geometry}


def test_one_row_per_arm_class_and_iteration_setting():
    rows = flatten_scenario(_scenario())
    assert len(rows) == 2          # 1 arm x 1 class x 2 iteration settings
    assert {r['algorithm_1_iterations'] for r in rows} == {1, 2}
    assert all(r['road_class'] == 'C' for r in rows)
    assert all(r['road_class_status'] == 'TO CONFIRM' for r in rows)


def test_the_flattened_row_carries_the_checks_and_the_verdict():
    row = flatten_scenario(_scenario())[0]
    assert row['check_i_max_passed'] is False
    assert row['verdict'].startswith('infeasible under this class')
    assert row['feasible'] is False


def test_smoothing_that_worsens_the_gradient_is_flagged():
    row = flatten_scenario(_scenario())[0]
    assert row['smoothing_made_i_max_worse'] is True


def test_an_arm_with_no_path_still_produces_a_row():
    rows = flatten_scenario(_scenario(path_found=False))
    assert len(rows) == 1
    assert rows[0]['path_found'] is False
    # the geometry columns are simply absent; write_csv turns a missing key into an
    # empty cell, which is what "excluded from every quality statistic" looks like
    assert rows[0].get('before_r_min_m') is None


def test_path_csv_names_separate_the_rough_and_smoothed_axes():
    assert path_csv_name('T', 'od1', 'hag_yellow') == 'T__od1__hag_yellow__rough.csv'
    assert (path_csv_name('T', 'od1', 'hag_yellow', 'C', 2)
            == 'T__od1__hag_yellow__C__it2.csv')


def test_write_path_csv(tmp_path):
    path = write_path_csv(tmp_path / 'p.csv', [[0.0, 0.0, 1.0], [10.0, 0.0, 2.0]])
    lines = path.read_text(encoding='utf-8').splitlines()
    assert lines[0] == 'x_m,y_m,z_m'
    assert len(lines) == 3


def test_read_jsonl_skips_a_truncated_last_line(tmp_path):
    path = tmp_path / 'x.jsonl'
    path.write_text('{"a": 1}\n{"b": 2}\n{"c":', encoding='utf-8')
    assert read_jsonl(path) == [{'a': 1}, {'b': 2}]
