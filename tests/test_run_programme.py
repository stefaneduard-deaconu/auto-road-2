"""The experiment runner: dry-run, resume, not_possible logging, the brief (tests run in a
temp directory, see conftest.py, so `results/` and the log here are throwaway)."""
import json
from pathlib import Path

import pytest

from core import native, run
from core.experiment_hag import Pair
from core.programme import EXPERIMENTS, TIERS, map_pair

pytestmark = pytest.mark.skipif(not native.available(), reason='no C++ compiler for core.native')


def test_unit_ids_are_unique_per_tier():
    for exp in EXPERIMENTS.values():
        for tier in TIERS[:2]:
            ids = [u.unit_id for u in exp.units(tier)]
            assert len(ids) == len(set(ids)), (exp.key, tier)


def test_dry_run_computes_nothing(capsys):
    assert run.main(['X3', '--tier', 'quick', '--dry-run']) == 0
    assert 'would run syn_s0_n120_p4_a20_d3' in capsys.readouterr().out
    assert not Path('results/x3_reduction_synthetic/rows.jsonl').exists()


def test_run_resume_and_shared_border_rows():
    assert run.main(['X3', '--only', 'syn_s0_n120_p4_a20_d3']) == 0
    csv = Path('results/x3_reduction_synthetic/x3_reduction_synthetic.csv')
    from core.results_io import read_csv
    rows = read_csv(csv)
    cta = [r for r in rows if r['search_space'] == 'hag_cta']
    assert cta and {r['hag_edge_cost'] for r in cta} == {'shared_border'}
    assert all(0 < r['search_space_percent_of_full'] < 100 for r in cta)
    assert {r['engine'] for r in rows} == {'native'}
    before = Path('results/x3_reduction_synthetic/units.jsonl').read_text().count('\n')
    run.main(['X3', '--only', 'syn_s0_n120_p4_a20_d3'])
    after = Path('results/x3_reduction_synthetic/units.jsonl').read_text().count('\n')
    assert after == before
    log = Path('results/experiments-log.md').read_text(encoding='utf-8')
    assert log.count('- X3 (quick)') == 2


def test_a_second_concurrent_run_is_locked_out():
    """A concurrent second run must never proceed, even started right after the first —
    an incident where several `core.run X3` invocations were started close together found
    a plain `exists()`-then-write check racy: both passed the check before either had
    written the file, and both ran, corrupting `rows.jsonl` with interleaved writes."""
    lock = run._lock('X3')
    try:
        with pytest.raises(run.Locked):
            run._lock('X3')
    finally:
        lock.unlink()


def test_missing_raster_is_not_possible_not_failed():
    assert run.main(['X6', '--tif', 'no/such/raster.tif']) == 0
    entries = [json.loads(line) for line in
               Path('results/x6_dem_whole/units.jsonl').read_text().splitlines()]
    assert {e['status'] for e in entries} == {'not_possible'}
    assert 'raster not found' in entries[0]['reason']


def test_report_writes_fact_keys():
    run.main(['X3', '--only', 'syn_s0_n120_p4_a20_d3'])
    path = run.write_brief(Path('brief.md'))
    text = path.read_text(encoding='utf-8')
    assert '`X3.hag_cta.shared_border.3.search_space_percent_of_full`' in text
    assert 'X6 - The whole Idrija DEM' in text


def test_map_pair_keeps_the_physical_point():
    pair = Pair('p', (3, 7), (10, 0), 1.0)
    assert map_pair(pair, 20.0, 20.0).start == (3, 7)
    assert map_pair(pair, 20.0, 1.0).start == (70, 150)
    assert map_pair(pair, 20.0, 3.0).target == (70, 3)


def test_the_memory_gate_depends_on_total_memory_not_on_whoever_is_running(monkeypatch):
    from core import programme
    gib = 2 ** 30
    monkeypatch.setattr(programme, 'total_memory_bytes', lambda: 16 * gib)
    cells_for = lambda g: int(g * gib / programme.BYTES_PER_CELL)
    # needs more than 80% of the physical memory: never possible here, however idle it is
    monkeypatch.setattr(programme, 'available_memory_bytes', lambda: 16 * gib)
    with pytest.raises(programme.NotPossible, match='80%'):
        programme.require_memory(cells_for(14), 'big')
    # fits the machine but not what is free right now: waits, then is logged not_possible
    monkeypatch.setattr(programme, 'available_memory_bytes', lambda: 1 * gib)
    with pytest.raises(programme.NotPossible, match='stayed available'):
        programme.require_memory(cells_for(3), 'busy', wait_s=0.0)
    # fits and is free: runs
    monkeypatch.setattr(programme, 'available_memory_bytes', lambda: 8 * gib)
    programme.require_memory(cells_for(3), 'fine', wait_s=0.0)


def test_the_memory_gate_reads_macos_sysctl_and_vm_stat(monkeypatch):
    import subprocess
    from core import programme
    vm_stat = ('Mach Virtual Memory Statistics: (page size of 16384 bytes)\n'
               'Pages free:                                     1000.\n'
               'Pages active:                                 700000.\n'
               'Pages inactive:                                 2000.\n'
               'Pages speculative:                               300.\n'
               'Pages wired down:                             200000.\n'
               'Pages purgeable:                                  40.\n'
               '"Translation faults":                      529700017.\n')
    outputs = {'sysctl': '38654705664\n', 'vm_stat': vm_stat}

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=outputs[args[0]], stderr='')

    monkeypatch.setattr(programme.subprocess, 'run', fake_run)
    assert programme._darwin_memory_status() == (38654705664, 16384 * (1000 + 2000 + 300 + 40))
    outputs['vm_stat'] = 'not vm_stat output\n'
    assert programme._darwin_memory_status() is None
