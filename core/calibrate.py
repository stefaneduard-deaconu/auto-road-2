"""Serial timing calibration for a parallel run: `python -m core.run calibrate X3 --sample 60`.

Counts, paths and quality measures do not depend on how many workers shared the machine;
timings do. A parallel run therefore tags its rows `timing_mode = 'parallel'`, and this
command re-runs a STRATIFIED sample of the same units one at a time, on an idle machine, into
`results/<dir>/calibration/` with `timing_mode = 'serial_calibration'`. The article takes its
timing columns (time ratios, break-even, memory) from these rows whenever the main rows are
parallel (`article_2/build_article.py`), and states both sample sizes.

The sample takes the same number of units from every stratum (grid size × Perlin period ×
relief × height interval for the synthetic experiments, cell size for the DEM ones), chosen
with a fixed seed, so it can be reproduced and it covers every terrain class.
"""
from __future__ import annotations

import random
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from core.programme import EXPERIMENTS, NotPossible, RunOptions, Unit
from core.provenance import provenance, write_jsonl
from core.results_io import read_jsonl, write_csv

STRATUM_KEYS = ('size', 'period', 'amplitude_m', 'delta', 'cell', 'kind')


def stratified_sample(units: list[Unit], n: int, seed: int) -> list[Unit]:
    strata: dict[tuple, list[Unit]] = defaultdict(list)
    for unit in units:
        strata[tuple(unit.params.get(k) for k in STRATUM_KEYS)].append(unit)
    rng = random.Random(seed)
    for members in strata.values():
        rng.shuffle(members)
    picked, round_ = [], 0
    keys = sorted(strata, key=str)
    while len(picked) < n and any(len(strata[k]) > round_ for k in keys):
        for k in keys:
            if round_ < len(strata[k]) and len(picked) < n:
                picked.append(strata[k][round_])
        round_ += 1
    return picked


def calibration_dir(key: str, tier: str) -> Path:
    from core import run as run_mod
    return run_mod.out_dir(key, tier) / 'calibration'


def calibrate(key: str, tier: str, options: RunOptions, *, sample: int = 60, seed: int = 2026,
              note: str = '', dry_run: bool = False) -> int:
    from core import run as run_mod
    if key not in EXPERIMENTS:
        print(f'unknown experiment {key}; known: {list(EXPERIMENTS)}')
        return 2
    exp = EXPERIMENTS[key]
    latest = run_mod.latest_status_for(key, tier)
    done = [u for u in exp.units(tier) if latest.get(u.unit_id, {}).get('status') == 'done']
    out = calibration_dir(key, tier)
    already = {e['unit_id'] for e in (read_jsonl(out / 'units.jsonl')
                                      if (out / 'units.jsonl').exists() else [])
               if e.get('status') == 'done'}
    chosen = [u for u in stratified_sample(done, sample, seed) if u.unit_id not in already]
    print(f'{key} calibration, tier {tier}: {len(done)} units done in the main run, sample of '
          f'{sample} (seed {seed}), {len(chosen)} still to re-time serially')
    if dry_run or not chosen:
        for u in chosen[:20]:
            print(f'  would re-time {u.unit_id}')
        return 0
    print('  TIMING CALIBRATION: run it alone on an idle machine.')
    out.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prov = provenance()
    failed = 0
    for k, unit in enumerate(chosen, 1):
        t0 = time.perf_counter()
        entry = {'unit_id': unit.unit_id, 'run_id': run_id, 'tier': tier,
                 'git_sha': prov['git_sha']}
        try:
            rows = exp.run(unit, options)
            for row in rows:
                row.update({'experiment': key, 'unit_id': unit.unit_id, 'run_id': run_id,
                            **prov, 'n_workers': 1, 'executor': 'serial',
                            'timing_mode': 'serial_calibration'})
                row['scenario_id'] = run_mod._scenario_id(row)
                write_jsonl(out / 'rows.jsonl', row)
            entry.update(status='done', n_rows=len(rows))
        except NotPossible as exc:
            entry.update(status='not_possible', reason=str(exc))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            entry.update(status='failed', reason=f'{type(exc).__name__}: {exc}',
                         traceback=traceback.format_exc(limit=6))
        entry['wall_s'] = time.perf_counter() - t0
        write_jsonl(out / 'units.jsonl', entry)
        print(f'  [{k}/{len(chosen)}] {unit.unit_id}: {entry["status"]} '
              f'({entry["wall_s"]:.1f} s)', flush=True)
    rebuild(key, tier)
    run_mod.append_log({'experiment': key, 'tier': tier, 'run_id': run_id,
                        'counts': {'calibrated': len(chosen) - failed, 'failed': failed},
                        'rows': None, 'units_to_run': len(chosen), 'units_already_done':
                        len(already), 'wall_s': 0.0, 'git_sha': prov['git_sha'],
                        'git_dirty': prov['git_dirty'], 'engine': options.engine,
                        'note': f'serial timing calibration, sample {sample} seed {seed}. '
                                f'{note}'.strip()})
    return 1 if failed else 0


def rebuild(key: str, tier: str):
    from core import run as run_mod
    out = calibration_dir(key, tier)
    rows_path, units_path = out / 'rows.jsonl', out / 'units.jsonl'
    if not rows_path.exists():
        return None
    done = {e['unit_id']: e['run_id'] for e in read_jsonl(units_path) if e.get('status') == 'done'}
    rows = [r for r in read_jsonl(rows_path) if done.get(r.get('unit_id')) == r.get('run_id')]
    if not rows:
        return None
    rows.sort(key=lambda r: (str(r.get('unit_id')), str(r.get('scenario_id'))))
    columns = list(run_mod.IDENTITY)
    for row in rows:
        columns += [c for c in row if c not in columns]
    return write_csv(out / 'calibration.csv', rows, columns)
