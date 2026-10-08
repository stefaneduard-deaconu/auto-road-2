"""The parallel runner (core/parallel.py): same rows as the serial runner except the timing and
worker columns, resume, retry, lanes, the monitor and the calibration sample. Tests run in a
temp directory (conftest.py), so `results/` here is throwaway."""
import csv
import json
import re
import sys
from pathlib import Path

import pytest

from core import native, parallel, run
from core.calibrate import stratified_sample
from core.monitor import latest_run, render, snapshot
from core.programme import EXPERIMENTS, Experiment, RunOptions, Unit
from core.summary import ProvenanceMismatch, assert_one_build

needs_native = pytest.mark.skipif(not native.available(), reason='no C++ compiler for core.native')
FREE_THREADED = getattr(sys, '_is_gil_enabled', lambda: True)() is False
TIMING_OR_RUN = re.compile(r'time|wall|memory|peak|break_even|timestamp|run_id|worker|executor|'
                           r'timing_mode|lane|saving|rss')


def _csv(path: Path) -> dict:
    with open(path, encoding='utf-8') as handle:
        return {r['scenario_id']: r for r in csv.DictReader(handle)}


def _non_timing_differences(a: dict, b: dict) -> set:
    assert set(a) == set(b)
    return {k for s in a for k in a[s] if not TIMING_OR_RUN.search(k) and a[s][k] != b[s].get(k)}


@needs_native
@pytest.mark.parametrize('executor', ['process', 'thread'])
def test_parallel_rows_equal_the_serial_rows(tmp_path, monkeypatch, executor):
    if executor == 'thread' and not FREE_THREADED:
        pytest.skip('threads only on the free-threaded interpreter')
    if executor == 'thread':
        monkeypatch.setenv('HAG_ALLOW_THREADS', '1')  # small quick-tier grids only
    csv_rel = Path('results/x3_reduction_synthetic/x3_reduction_synthetic.csv')
    serial_dir, par_dir = tmp_path / 'serial', tmp_path / 'parallel'
    serial_dir.mkdir()
    par_dir.mkdir()
    monkeypatch.chdir(serial_dir)
    assert run.main(['X3', '--tier', 'quick']) == 0
    serial = _csv(serial_dir / csv_rel)
    monkeypatch.chdir(par_dir)
    assert run.main(['X3', '--tier', 'quick', '--workers', '2', '--executor', executor]) == 0
    par = _csv(par_dir / csv_rel)
    assert not _non_timing_differences(serial, par)
    assert {r['timing_mode'] for r in par.values()} == {'parallel'}
    assert {r['timing_mode'] for r in serial.values()} == {'serial'}
    # one folder per worker, each with its own logs
    run_dir = latest_run(par_dir / 'results' / '_runs')
    workers = sorted(p.name for p in (run_dir / 'workers').iterdir())
    assert workers and all((run_dir / 'workers' / w / 'heartbeat.json').exists() for w in workers)
    assert json.loads((run_dir / 'manifest.json').read_text())['finished_utc']


@needs_native
def test_a_second_parallel_run_resumes_and_runs_nothing():
    assert run.main(['X3', '--tier', 'quick', '--workers', '2']) == 0
    lines = Path('results/x3_reduction_synthetic/units.jsonl').read_text().splitlines()
    assert run.main(['X3', '--tier', 'quick', '--workers', '2']) == 0
    assert Path('results/x3_reduction_synthetic/units.jsonl').read_text().splitlines() == lines


@pytest.mark.skipif(not FREE_THREADED, reason='threads only on the free-threaded interpreter')
def test_a_failed_unit_is_retried_once(monkeypatch):
    monkeypatch.setenv('HAG_ALLOW_THREADS', '1')  # pure-Python units: safe on threads
    calls = []

    def flaky(unit, options):
        calls.append(unit.unit_id)
        if calls.count(unit.unit_id) == 1:
            raise RuntimeError('first attempt fails')
        return [{'value': 1}]

    exp = Experiment('XT', 'xt_test', 'test', 'test', False, False,
                     lambda tier: [Unit('u1', {}), Unit('u2', {})], flaky)
    monkeypatch.setitem(EXPERIMENTS, 'XT', exp)
    result = parallel.run_lanes(parallel.single_lane(['XT'], 'quick'), RunOptions(),
                                workers=2, executor='thread')
    assert result.counts == {'done': 2}
    assert sorted(calls) == ['u1', 'u1', 'u2', 'u2']


@pytest.mark.skipif(not FREE_THREADED, reason='threads only on the free-threaded interpreter')
def test_a_unit_that_always_fails_is_logged_not_retried_forever(monkeypatch):
    monkeypatch.setenv('HAG_ALLOW_THREADS', '1')  # pure-Python units: safe on threads
    exp = Experiment('XT', 'xt_test', 'test', 'test', False, False,
                     lambda tier: [Unit('bad', {})],
                     lambda unit, options: (_ for _ in ()).throw(ValueError('broken')))
    monkeypatch.setitem(EXPERIMENTS, 'XT', exp)
    result = parallel.run_lanes(parallel.single_lane(['XT'], 'quick'), RunOptions(),
                                workers=1, executor='thread')
    assert result.counts == {'failed': 1}
    snap = snapshot(result.run_dir)
    assert snap['failures'] and 'broken' in snap['failures'][0]['reason']
    assert 'failed units' in render(snap)


def test_lanes_take_units_by_parameter_and_priority():
    fine = parallel.Lane('fine', ['X7'], ['standard'], priority=1, where={'cell': {'max': 3}})
    coarse = parallel.Lane('coarse', ['X7'], ['standard'])
    tasks = parallel.collect_tasks([coarse, fine], lambda k, t: {}, lambda k, t: {})
    by_lane = {}
    for task in tasks:
        by_lane.setdefault(task.lane, []).append(task.unit.params['cell'])
    assert by_lane['fine'] and max(by_lane['fine']) <= 3
    assert min(by_lane['coarse']) > 3
    assert len(tasks) == len(EXPERIMENTS['X7'].units('standard'))


def test_plan_files_load(tmp_path):
    root = Path(__file__).resolve().parents[1] / 'runs'
    for path in sorted(root.glob('*.toml')):
        plan = parallel.load_plan(path)
        assert plan['lanes'], path


def test_pick_follows_the_lane_shares():
    lanes = [parallel.Lane('a', [], [], share=0.75), parallel.Lane('b', [], [], share=0.25)]
    pending = {'a': [f'a{k}' for k in range(10)], 'b': [f'b{k}' for k in range(10)]}
    running = {'a': 0, 'b': 0}
    from collections import Counter
    running = Counter()
    picked = []
    for _ in range(4):
        task = parallel._pick(pending, running, lanes, 4)
        lane = task[0]
        running[lane] += 1
        picked.append(lane)
    assert picked.count('a') == 3 and picked.count('b') == 1


def test_auto_workers_is_at_least_one():
    assert parallel.auto_workers({'logical_cpus': 1}) == 1
    assert parallel.auto_workers({'physical_cores': 16, 'logical_cpus': 32}) == 15
    assert parallel.auto_workers({'performance_cores': 12, 'physical_cores': 16}) == 11


def test_the_stratified_sample_covers_every_stratum():
    units = [Unit(f's{s}_n{n}', {'size': n, 'seed': s}) for n in (120, 240, 480) for s in range(10)]
    picked = stratified_sample(units, 6, seed=1)
    assert sorted(u.params['size'] for u in picked) == [120, 120, 240, 240, 480, 480]
    assert stratified_sample(units, 6, seed=1) == picked


def test_parallel_and_serial_timings_never_pool():
    base = {'git_sha': 'a', 'python': '3.14', 'python_free_threaded': True, 'numpy': '2',
            'scipy': '1', 'platform': 'Windows-11', 'engine': 'native'}
    serial = {**base, 'timing_mode': 'serial', 'n_workers': 1}
    par = {**base, 'timing_mode': 'parallel', 'n_workers': 8}
    assert_one_build([serial, par])  # counts may pool
    with pytest.raises(ProvenanceMismatch):
        assert_one_build([serial, par], timing=True)
    assert_one_build([serial, {k: v for k, v in serial.items() if k != 'timing_mode'}],
                     timing=True)  # rows from before the parallel runner were serial




@pytest.mark.skipif(not FREE_THREADED, reason='threads only on the free-threaded interpreter')
def test_the_memory_gate_never_blocks_an_idle_pool(monkeypatch):
    monkeypatch.setenv('HAG_ALLOW_THREADS', '1')  # pure-Python units: safe on threads
    monkeypatch.setattr(parallel, 'available_memory_bytes', lambda: 1)  # always below reserve
    exp = Experiment('XT', 'xt_test', 'test', 'test', False, False,
                     lambda tier: [Unit('u1', {}), Unit('u2', {})], lambda unit, options: [{'v': 1}])
    monkeypatch.setitem(EXPERIMENTS, 'XT', exp)
    result = parallel.run_lanes(parallel.single_lane(['XT'], 'quick'), RunOptions(),
                                workers=2, executor='thread', reserve_gib=8)
    assert result.counts == {'done': 2}


def test_threads_are_opt_in(monkeypatch, capsys):
    monkeypatch.delenv('HAG_ALLOW_THREADS', raising=False)
    exp = Experiment('XT', 'xt_test', 'test', 'test', False, False,
                     lambda tier: [Unit('u', {})], lambda unit, options: [{'v': 1}])
    monkeypatch.setitem(EXPERIMENTS, 'XT', exp)
    result = parallel.run_lanes(parallel.single_lane(['XT'], 'quick'), RunOptions(),
                                workers=1, executor='thread')
    assert result.executor == 'process' and 'experimental' in capsys.readouterr().out
