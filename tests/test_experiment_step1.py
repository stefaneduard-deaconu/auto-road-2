"""T3 acceptance: one JSON row per run, and the numbers are reproducible."""
import json

import pytest

from core.experiment_step1 import DEFAULT_ARMS, Step1Config, run_step1, summarise
from core.provenance import provenance, write_jsonl
from core.terrain import TerrainSpec

SMALL = Step1Config(terrain=TerrainSpec(seed=3, grid_size=(40, 40), periods=(2, 2)),
                    start=(3, 3), target=(36, 36), height_delta=3.0)


@pytest.fixture(scope="module")
def row():
    return run_step1(SMALL)


def test_every_arm_is_reported(row):
    assert set(row['arms']) == {label for label, _, _ in DEFAULT_ARMS}
    assert row['experiment'] == 'step1_hag_vs_full_grid'
    assert row['seed'] == 3


def test_the_row_carries_seed_git_sha_and_config(row):
    assert row['config']['terrain']['seed'] == 3
    assert row['config']['start'] == [3, 3]
    assert set(row['provenance']) >= {'git_sha', 'git_dirty', 'timestamp_utc', 'numpy',
                                      'scipy', 'python', 'python_free_threaded'}


def test_full_grid_arm_searches_the_whole_grid(row):
    full = row['arms']['full_grid']
    assert full['search_space_fraction_of_full'] == 1.0
    assert full['n_cells_selected'] == full['n_cells_total'] == 40 * 40
    assert full['path_found']


def test_hag_reduces_the_search_space_and_the_work(row):
    full, yellow = row['arms']['full_grid'], row['arms']['hag_yellow']
    assert yellow['search_space_fraction_of_full'] < 1.0
    assert yellow['n_cells_selected'] < full['n_cells_selected']
    assert yellow['search_nodes_expanded'] <= full['search_nodes_expanded']
    assert yellow['n_graph_edges'] < full['n_graph_edges']


def test_dilation_grows_the_search_space(row):
    assert (row['arms']['hag_yellow_k1']['n_cells_selected']
            >= row['arms']['hag_yellow']['n_cells_selected'])


def test_both_ways_of_stating_the_reduction_are_recorded(row):
    # bad point 10: "to X%" and "by X%" are different claims
    for arm in row['arms'].values():
        assert arm['search_space_percent_of_full'] + arm['search_space_reduction_percent'] \
            == pytest.approx(100.0)


def test_quality_side_is_recorded_for_every_reachable_arm(row):
    for label, arm in row['arms'].items():
        if not arm['path_found']:
            assert 'infeasible_reason' in arm
            continue
        assert arm['path_length_m'] > 0
        assert arm['objective_cost_sum_abs_dh_m'] >= 0
        assert arm['metric_r_min_m'] > 0
        assert arm['metric_i_max_percent'] >= 0
        assert arm['metric_sinuosity'] >= 1.0
        if label != row['config']['baseline_arm']:
            assert arm['vs_baseline_hausdorff_m'] >= 0
            assert arm['objective_cost_ratio_to_baseline'] >= 1.0 - 1e-9


def test_memory_and_time_are_measured_separately(row):
    for arm in row['arms'].values():
        assert arm['wall_time_s'] > 0
        assert arm['peak_memory_bytes'] > 0


def test_the_run_is_reproducible():
    a, b = run_step1(SMALL), run_step1(SMALL)
    for label in a['arms']:
        for key in ('n_cells_selected', 'search_nodes_expanded', 'path_length_m',
                    'objective_cost_sum_abs_dh_m', 'metric_r_min_m'):
            assert a['arms'][label].get(key) == b['arms'][label].get(key), (label, key)


def test_one_json_row_per_run(tmp_path, row):
    path = write_jsonl(tmp_path / 'step1.jsonl', row)
    write_jsonl(path, row)
    lines = path.read_text(encoding='utf-8').strip().split('\n')
    assert len(lines) == 2
    back = json.loads(lines[0])
    assert back['arms']['full_grid']['path_found'] is True
    assert '_path_xyz_m' not in back['arms']['full_grid']  # the path itself is not a result field


def test_summary_table_mentions_every_arm(row):
    text = summarise(row)
    for label in row['arms']:
        assert label in text


def test_out_of_range_objective_is_rejected():
    with pytest.raises(ValueError):
        run_step1(Step1Config(terrain=TerrainSpec(seed=1, grid_size=(20, 20), periods=(2, 2)),
                              start=(0, 0), target=(50, 50)))


def test_provenance_reports_the_repo_state():
    p = provenance()
    assert p['git_sha'] is None or len(p['git_sha']) == 40
    assert p['numpy'] and p['scipy']
    # timings are only comparable within one interpreter, so the build is recorded
    assert p['python_free_threaded'] in (True, False, None)
