"""Watch a parallel run: `python -m core.run monitor [--follow] [--interval 10]`.

Reads only the files a run writes as it goes (`results/_runs/<run_id>/manifest.json`,
`progress.json` and every worker's `heartbeat.json` and `units.jsonl`), so it is safe to run
at any time, from another terminal, while the run is working. It shows, per lane, how many
units are finished, running and queued with an ETA; per worker, the unit it is on, for how
long, its memory and whether it has stopped sending heartbeats (stalled); and the failed
units with their reasons.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from core.parallel import HEARTBEAT_S, RUNS

STALLED_AFTER_S = 120


def latest_run(runs: Path = RUNS) -> Optional[Path]:
    if not runs.exists():
        return None
    candidates = [p for p in runs.iterdir() if (p / 'manifest.json').exists()]
    return max(candidates, key=lambda p: p.name) if candidates else None


def _read(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _fmt_s(seconds: Optional[float]) -> str:
    if seconds is None:
        return '-'
    seconds = int(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f'{h}h{m:02d}m' if h else f'{m}m{s:02d}s'


def snapshot(run_dir: Path) -> dict:
    """Everything the monitor shows, as data (used by the tests and by `render`)."""
    manifest = _read(run_dir / 'manifest.json') or {}
    progress = _read(run_dir / 'progress.json') or {}
    now = time.time()
    workers, failures = [], []
    for folder in sorted((run_dir / 'workers').glob('*')):
        beat = _read(folder / 'heartbeat.json') or {}
        current = beat.get('current') or {}
        age = now - beat['time'] if 'time' in beat else None
        workers.append({'worker_id': folder.name, 'units_done': beat.get('units_done', 0),
                        'unit': current.get('unit'), 'lane': current.get('lane'),
                        'running_s': now - current['started'] if current.get('started') else None,
                        'estimate_s': current.get('estimate_s'),
                        'rss_gib': (beat.get('rss_bytes') or 0) / 2**30,
                        'heartbeat_age_s': age,
                        'stalled': age is not None and age > max(STALLED_AFTER_S, 4 * HEARTBEAT_S)
                        and not manifest.get('finished_utc')})
        units = folder / 'units.jsonl'
        if units.exists():
            for line in units.read_text(encoding='utf-8').splitlines():
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if entry.get('status') == 'failed':
                    failures.append({'worker_id': folder.name, 'unit': entry.get('unit_id'),
                                     'experiment': entry.get('experiment'),
                                     'reason': (entry.get('reason') or '')[:200]})
    return {'run_id': manifest.get('run_id'), 'run_dir': str(run_dir),
            'started_utc': manifest.get('started_utc'),
            'finished_utc': manifest.get('finished_utc'), 'workers_planned': manifest.get('workers'),
            'executor': manifest.get('executor'), 'machine': manifest.get('machine', {}),
            'lanes': progress.get('lanes', {}), 'experiments': progress.get('experiments', {}),
            'counts': progress.get('counts', {}), 'eta_s': progress.get('eta_s_estimate'),
            'elapsed_s': progress.get('elapsed_s'), 'workers': workers, 'failures': failures}


def render(snap: dict) -> str:
    m = snap['machine']
    lines = [f"run {snap['run_id']}  ({snap['run_dir']})",
             f"  {snap['workers_planned']} {snap['executor']} workers on {m.get('os')} "
             f"({m.get('physical_cores')} physical / {m.get('logical_cpus')} logical cores, "
             f"free-threaded {m.get('free_threaded')})",
             f"  started {snap['started_utc']}, elapsed {_fmt_s(snap['elapsed_s'])}, "
             + (f"FINISHED {snap['finished_utc']}" if snap['finished_utc']
                else f"ETA about {_fmt_s(snap['eta_s'])} (from unit estimates)"),
             f"  units: {snap['counts'] or 'none finished yet'}", '', '  lanes:']
    for name, lane in snap['lanes'].items():
        total = lane.get('total') or 0
        share = 100.0 * lane.get('finished', 0) / total if total else 100.0
        lines.append(f"    {name:10s} {lane.get('finished', 0):>6}/{total:<6} finished "
                     f"({share:5.1f}%), {lane.get('running', 0)} running, "
                     f"{lane.get('queued', 0)} queued")
    if snap['experiments']:
        lines.append('  per experiment:')
        for key, c in sorted(snap['experiments'].items()):
            lines.append(f'    {key:14s} {c}')
    lines += ['', '  workers:']
    for w in snap['workers']:
        state = 'STALLED' if w['stalled'] else ('idle' if not w['unit'] else 'working')
        lines.append(f"    {w['worker_id']:5s} {state:8s} done {w['units_done']:>5}  "
                     f"{(w['unit'] or '-')[:52]:52s} {_fmt_s(w['running_s']):>7} "
                     f"(est {_fmt_s(w['estimate_s'])})  {w['rss_gib']:.2f} GiB")
    if snap['failures']:
        lines += ['', f"  failed units ({len(snap['failures'])}):"]
        for f in snap['failures'][-10:]:
            lines.append(f"    {f['worker_id']} {f['experiment']} {f['unit']}: {f['reason']}")
    return '\n'.join(lines)


def monitor(follow: bool = False, interval: float = 10.0, run_dir: Optional[Path] = None) -> int:
    run_dir = run_dir or latest_run()
    if run_dir is None:
        print(f'no parallel run found under {RUNS}/ (start one with --workers or --plan)')
        return 0
    while True:
        snap = snapshot(run_dir)
        if follow:
            print('\033[2J\033[H', end='')
        print(render(snap), flush=True)
        if not follow or snap['finished_utc']:
            return 0
        time.sleep(interval)
