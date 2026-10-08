"""The T7 runner. The `--quick` subset is the smoke test."""
import csv

import pytest

from core.experiment_matrix import (MatrixOptions, QUICK_TERRAINS, REFERENCE_PAIR,
                                    _class_names, main, scenario_id)
from core.results_io import MATRIX_COLUMNS, read_jsonl


@pytest.fixture(scope='module')
def quick_run(tmp_path_factory):
    out = tmp_path_factory.mktemp('matrix')
    assert main(['--quick', '--out', str(out)]) == 0
    return out


def test_the_quick_run_writes_every_artefact(quick_run):
    for name in ('matrix.jsonl', 'matrix.csv', 'od_pairs.csv', 'summary.md'):
        assert (quick_run / name).exists(), name
    assert list((quick_run / 'paths').glob('*.csv'))


def test_the_csv_header_is_the_column_contract(quick_run):
    with open(quick_run / 'matrix.csv', encoding='utf-8', newline='') as handle:
        assert next(csv.reader(handle)) == list(MATRIX_COLUMNS)


def test_the_row_count_is_the_enumerated_product(quick_run):
    scenarios = read_jsonl(quick_run / 'matrix.jsonl')
    n_arms = len(scenarios[0]['arms'])
    expected = len(scenarios) * n_arms * len(_class_names(quick=True)) * 1
    with open(quick_run / 'matrix.csv', encoding='utf-8', newline='') as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == expected


def test_every_class_is_marked_to_confirm(quick_run):
    with open(quick_run / 'matrix.csv', encoding='utf-8', newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r['road_class']]
    assert rows and all(r['road_class_status'] == 'TO CONFIRM' for r in rows)


def test_the_log_is_strict_json(quick_run):
    """Bare `Infinity` is accepted by Python but is not valid JSON."""
    import json

    def boom(constant):
        raise AssertionError(f'bare JSON constant written: {constant}')

    for line in (quick_run / 'matrix.jsonl').read_text(encoding='utf-8').splitlines():
        json.loads(line, parse_constant=boom)


def test_rerunning_does_not_duplicate_scenarios(quick_run):
    before = len(read_jsonl(quick_run / 'matrix.jsonl'))
    assert main(['--quick', '--out', str(quick_run)]) == 0
    assert len(read_jsonl(quick_run / 'matrix.jsonl')) == before


def test_force_redoes_without_doubling(quick_run):
    before = len(read_jsonl(quick_run / 'matrix.jsonl'))
    assert main(['--quick', '--out', str(quick_run), '--force']) == 0
    assert len(read_jsonl(quick_run / 'matrix.jsonl')) == before


def test_summary_only_does_not_search(quick_run, monkeypatch):
    import core.experiment_matrix as matrix

    def refuse(*args, **kwargs):
        raise AssertionError('--summary-only must not run a search')

    monkeypatch.setattr(matrix, 'run_step1_full', refuse)
    assert main(['--quick', '--out', str(quick_run), '--summary-only']) == 0
    assert (quick_run / 'summary.md').exists()


def test_the_summary_carries_the_provenance_block(quick_run):
    text = (quick_run / 'summary.md').read_text(encoding='utf-8')
    for line in ('Produced:', 'Commit:', 'Python:', 'Classes:', 'Populations:'):
        assert line in text, line


def test_scenario_ids_are_stable():
    assert scenario_id('T_article', 'od1') == 'T_article__od1'


def test_the_reference_pair_is_the_published_one():
    """The fixed reference pair is (5, 5) -> (94, 94); it is a reference, not a result."""
    assert REFERENCE_PAIR == ((5, 5), (94, 94))


def test_the_quick_terrains_are_small():
    assert all(case.spec.grid_size == (40, 40) for case in QUICK_TERRAINS)


# -- write_paths='all': smoothed axes, so a figure can be drawn without re-running -------

def test_write_paths_all_also_commits_the_smoothed_axis(tmp_path):
    from core.results_io import path_csv_name

    out = tmp_path / 'all_paths'
    assert main(['--quick', '--out', str(out), '--write-paths', 'all']) == 0
    rough = list((out / 'paths').glob('*__rough.csv'))
    smoothed = [p for p in (out / 'paths').glob('*.csv') if p not in rough]
    assert rough and smoothed
    #: one smoothed file per (scenario, arm, class) at the quick run's one iteration setting
    expected = path_csv_name('Q_a', 'od1', 'hag_yellow', road_class='RO_CLASS_III_DEAL',
                             iterations=1)
    assert (out / 'paths' / expected).exists()


def test_write_paths_primary_is_unchanged_by_the_all_option_existing(tmp_path):
    """The default must still write ONLY the rough axes -- 'all' is opt-in."""
    out = tmp_path / 'primary_only'
    assert main(['--quick', '--out', str(out)]) == 0
    names = {p.name for p in (out / 'paths').glob('*.csv')}
    assert names and all(n.endswith('__rough.csv') for n in names)
