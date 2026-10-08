"""Summary statistics and the provenance guard."""
import math

import pytest

from core.summary import (ProvenanceMismatch, assert_one_build, describe,
                          population_geometry, population_search, render_markdown)

BUILD = {'git_sha': 'abc', 'python': '3.14.7', 'python_free_threaded': True,
         'git_dirty': False, 'git_branch': 'b', 'platform': 'p', 'numpy': '2.5.3',
         'scipy': '1.18.1', 'timestamp_utc': 't'}


def _row(**kwargs):
    row = dict(BUILD)
    row.update({'od_protocol': True, 'road_class': 'RO_CLASS_IV_DEAL',
                'algorithm_1_iterations': 1, 'terrain_id': 'T', 'od_id': 'od1',
                'arm': 'hag_yellow'})
    row.update(kwargs)
    return row


def test_describe_matches_the_obvious_statistics():
    stat = describe([1.0, 2.0, 3.0, 4.0, 5.0], 'x')
    assert stat.n == 5
    assert stat.mean == pytest.approx(3.0)
    assert stat.median == pytest.approx(3.0)
    assert stat.minimum == 1.0 and stat.maximum == 5.0
    assert stat.sd == pytest.approx(1.5811, rel=1e-3)


def test_infinite_radii_are_counted_and_excluded_not_averaged():
    """`r_min_m` is infinite for a straight alignment; averaging it in is meaningless."""
    stat = describe([10.0, 20.0, math.inf, 30.0, math.inf], 'r_min_m')
    assert stat.n_inf == 2
    assert stat.n == 3
    assert stat.mean == pytest.approx(20.0)


def test_missing_values_are_counted_separately():
    stat = describe([1.0, None, 3.0], 'x')
    assert stat.n_missing == 1
    assert stat.n == 2


def test_a_thin_statistic_says_so():
    assert 'n<5' in describe([1.0, 2.0], 'x').note
    assert describe([1.0] * 6, 'x').note == ''


def test_a_column_with_no_finite_values_does_not_crash():
    stat = describe([None, math.inf], 'x')
    assert stat.n == 0
    assert 'no finite values' in stat.note


# -- B25: a finite but effectively-straight r_min must not dominate the mean ------------

def test_near_collinear_r_min_is_censored_like_a_genuine_straight_line():
    """A FINITE 2.476e+15 m is the floating-point artefact the study protocol's matrix recorded;
    it must be excluded exactly as a literal inf straight alignment already is."""
    from core.summary import _censor_straight
    assert _censor_straight(2.476e15) == math.inf
    assert _censor_straight(math.inf) == math.inf              # a genuine straight line
    assert _censor_straight(240.0) == 240.0                    # a real radius, untouched
    assert _censor_straight(None) is None


def test_the_rendered_geometry_table_excludes_the_straight_line_artefact():
    rows = [_row(before_r_min_m=20.0, after_r_min_m=20.0 if k else 2.476e15,
                delta_r_min_m=0.0 if k else 2.476e15 - 20.0,
                check_r_min_passed=True, check_i_max_passed=True, feasible=True)
            for k in range(6)]
    text = render_markdown(rows)
    assert '2.476e' not in text and '4.3e+' not in text
    assert 'after_r_min_m' in text and 'B25' in text


# -- B22: the objective-cost RATIO is noise on near-flat ground -------------------------

def test_a_flat_baseline_scenario_is_excluded_from_the_ratio_but_not_the_abs_diff():
    from core.summary import _is_flat_baseline, _objective_abs_diff
    flat_baseline = {('T', 'od1'): 2.59}
    steep_baseline = {('T', 'od1'): 100.0}
    row = _row(objective_cost_sum_abs_dh_m=21.9)
    assert _is_flat_baseline(row, flat_baseline) is True
    assert _is_flat_baseline(row, steep_baseline) is False
    assert _objective_abs_diff(row, flat_baseline) == pytest.approx(21.9 - 2.59)


def test_the_rendered_quality_table_reports_the_abs_diff_and_names_the_exclusion():
    rows = []
    for k in range(6):
        od_id = f'od{k}'
        rows.append(_row(arm='full_grid', od_id=od_id, objective_cost_sum_abs_dh_m=2.59))
        rows.append(_row(arm='hag_yellow', od_id=od_id,
                         objective_cost_ratio_to_baseline=8.47,
                         objective_cost_sum_abs_dh_m=21.9, length_ratio_to_baseline=1.1,
                         vs_baseline_hausdorff_m=5.0, vs_baseline_mean_deviation_m=2.0))
    text = render_markdown(rows)
    assert 'objective_cost_abs_diff_m' in text
    assert 'B22' in text
    assert '8.47' not in text            # the noisy ratio must not appear in the table


def test_rows_from_different_builds_are_refused():
    rows = [_row(), _row(python='3.13.0')]
    with pytest.raises(ProvenanceMismatch, match='python'):
        assert_one_build(rows)


def test_rows_from_different_commits_are_refused():
    with pytest.raises(ProvenanceMismatch, match='git_sha'):
        assert_one_build([_row(), _row(git_sha='def')])


def test_rows_from_different_operating_systems_are_refused():
    """The same code does not give the same floats on both."""
    rows = [_row(platform='Windows-11-10.0.26200-SP0'),
            _row(platform='Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39')]
    with pytest.raises(ProvenanceMismatch, match='os_family'):
        assert_one_build(rows)


def test_two_builds_of_the_same_os_still_aggregate():
    """Only the OS family is compared: a build-number bump must not refuse a valid matrix."""
    block = assert_one_build([_row(platform='Windows-11-10.0.26200-SP0'),
                              _row(platform='Windows-11-10.0.26300-SP0')])
    assert block['os_family'] == 'Windows'


def test_a_dirty_tree_warns_but_does_not_refuse():
    block = assert_one_build([_row(git_dirty=True)])
    assert block['git_dirty_warning'] is True


def test_the_search_population_does_not_triple_count_the_classes():
    """The search does not depend on the road class, so it must be counted once."""
    rows = [_row(road_class=name, search_space_percent_of_full=24.80)
            for name in ('RO_CLASS_III_DEAL', 'RO_CLASS_IV_DEAL', 'RO_CLASS_V_DEAL_REDUS')]
    assert len(population_search(rows)) == 1
    assert len(population_geometry(rows)) == 3


def test_the_reference_pair_and_synthetic_classes_are_excluded():
    rows = [_row(), _row(od_protocol=False), _row(road_class='SYNTHETIC_LOOSE')]
    assert len(population_search(rows)) == 1
    assert len(population_geometry(rows)) == 1


def test_the_geometry_population_takes_one_iteration_setting():
    rows = [_row(algorithm_1_iterations=1), _row(algorithm_1_iterations=2)]
    assert len(population_geometry(rows, iterations=1)) == 1


def test_the_rendered_summary_carries_the_provenance_block_and_the_caveat():
    text = render_markdown([_row(search_space_percent_of_full=24.80,
                                 check_r_min_passed=True, feasible=False)])
    for line in ('Produced:', 'Commit:', 'Python:', 'Populations:',
                 'Excluded from the statistics:'):
        assert line in text
    assert 'TO CONFIRM' in text
    assert 'evaluated' in text
    # the wording the paper commits to: never "validated for automated road design"
    assert 'validated for' not in text
    assert 'Nothing here is a validated result' in text


def test_the_summary_names_both_populations_with_their_sizes():
    text = render_markdown([_row()])
    assert 'P_search n=1' in text
    assert 'P_geom n=1' in text
