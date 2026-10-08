"""Run the units of the HAG programme on many workers at once.

    python -m core.run X3 --tier standard --workers auto            # one experiment, all cores
    python -m core.run all --tier standard --workers 8 --executor thread
    python -m core.run --plan runs/progressive_dem.toml              # lanes, coarse first
    python -m core.run monitor --follow                              # watch a running plan

The SAME `Experiment.run(unit, options)` functions as the serial runner are called; only the
scheduling differs. What a parallel run adds:

* **Workers** - processes (default; `spawn`, identical on Windows and macOS) or threads (only
  on the free-threaded interpreter, where they really run at once). `--workers auto` is the
  number of physical performance cores, see `machine_info`.
* **Order** - longest unit first (LPT), from the measured `wall_s` of earlier runs of the same
  unit when there is one, else from the unit's size. The longest units start first, so the
  tail of the run is short.
* **Lanes** - a plan file names groups of units (e.g. `coarse` = 20/10/5 m cells, `fine` =
  3/1 m) with a share of the workers and a priority. All lanes start together, the coarse lane
  finishes first, and its CSVs are complete and readable while the fine lane keeps working;
  an idle worker takes work from any lane.
* **Memory gate** - a unit is only started while the free memory stays above `reserve_gib`.
* **Retry** - a unit that fails (not `not_possible`) is retried once; if a worker process
  dies, the pool is rebuilt and the units it held are re-queued.

Files of one run (`<run_id>` is the start time, UTC):

    results/_runs/<run_id>/manifest.json      plan, lanes, workers, executor, machine, provenance
    results/_runs/<run_id>/progress.json      per lane and experiment: done/failed/to run, ETA
    results/_runs/<run_id>/workers/<w>/       units.jsonl, rows.jsonl, worker.log, heartbeat.json
    results/<dir>/rows.jsonl, units.jsonl, <dir>.csv     merged, exactly as the serial runner

The coordinator is the only writer of the merged files, so every reader (`core.run report`,
`article_2/build_article.py`) works unchanged. Every row records `n_workers`,
`executor`, `worker_id`, `lane` and `timing_mode` (`serial` or `parallel`): timings measured
while other workers share the machine are tagged and never pooled with serial ones
(`core.summary.assert_one_build(..., timing=True)`); `core.run calibrate` re-times a sample
serially for the article's timing columns.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import (FIRST_COMPLETED, Future, ProcessPoolExecutor,
                                ThreadPoolExecutor, wait)
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from core.programme import EXPERIMENTS, NotPossible, RunOptions, Unit, available_memory_bytes
from core.provenance import provenance, sanitise, write_jsonl

RUNS = Path('results') / '_runs'
HEARTBEAT_S = 5.0
DEFAULT_RESERVE_GIB = 2.0


# -- the machine ---------------------------------------------------------------------------

def _sysctl(name: str) -> Optional[str]:
    try:
        out = subprocess.run(['sysctl', '-n', name], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _physical_cores() -> Optional[int]:
    try:
        import psutil  # optional
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:  # noqa: BLE001 - psutil is optional
        pass
    if sys.platform == 'darwin':
        value = _sysctl('hw.physicalcpu')
        return int(value) if value and value.isdigit() else None
    if sys.platform.startswith('linux'):
        try:
            cores = set()
            physical = core = None
            for line in Path('/proc/cpuinfo').read_text().splitlines():
                if line.startswith('physical id'):
                    physical = line.split(':')[1].strip()
                elif line.startswith('core id'):
                    core = line.split(':')[1].strip()
                    cores.add((physical, core))
            return len(cores) or None
        except OSError:
            return None
    if sys.platform == 'win32':
        try:
            out = subprocess.run(['powershell', '-NoProfile', '-Command',
                                  '(Get-CimInstance Win32_Processor | Measure-Object '
                                  '-Property NumberOfCores -Sum).Sum'],
                                 capture_output=True, text=True, timeout=20)
            value = out.stdout.strip()
            return int(value) if value.isdigit() else None
        except (OSError, subprocess.SubprocessError):
            return None
    return None


def machine_info() -> dict:
    """What a parallel run needs to know about the machine, recorded in every manifest."""
    logical = os.cpu_count() or 1
    physical = _physical_cores()
    performance = None
    if sys.platform == 'darwin':  # Apple silicon: performance and efficiency cores
        value = _sysctl('hw.perflevel0.physicalcpu')
        performance = int(value) if value and value.isdigit() else None
    total = None
    try:
        import psutil
        total = int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001
        if sys.platform == 'darwin':
            value = _sysctl('hw.memsize')
            total = int(value) if value and value.isdigit() else None
    gil = getattr(sys, '_is_gil_enabled', None)
    return {'os': sys.platform, 'platform': platform.platform(),
            'cpu': platform.processor() or _sysctl('machdep.cpu.brand_string') or '',
            'logical_cpus': logical, 'physical_cores': physical,
            'performance_cores': performance, 'memory_total_bytes': total,
            'memory_available_bytes': available_memory_bytes(),
            'python': platform.python_version(),
            'free_threaded': gil is not None and not gil()}


def auto_workers(info: Optional[dict] = None) -> int:
    """Physical performance cores, minus one for the coordinator when there are many."""
    info = info or machine_info()
    cores = (info.get('performance_cores') or info.get('physical_cores')
             or max(1, (info.get('logical_cpus') or 2) // 2))
    return max(1, cores - 1 if cores >= 8 else cores)


def resolve_workers(value, info: Optional[dict] = None) -> int:
    if value in (None, '', 'auto'):
        return auto_workers(info)
    return max(1, int(value))


def _rss_bytes() -> Optional[int]:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001
        pass
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t),
                        ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t),
                        ('PeakPagefileUsage', ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        try:
            psapi = ctypes.WinDLL('psapi')
            kernel = ctypes.WinDLL('kernel32')
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p,
                                                   wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters),
                                          counters.cb):
                return int(counters.WorkingSetSize)
        except Exception:  # noqa: BLE001 - a memory probe must never fail a unit
            return None
        return None
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == 'darwin' else peak * 1024)  # peak, not current
    except Exception:  # noqa: BLE001
        return None


# -- tasks ---------------------------------------------------------------------------------

@dataclass
class Task:
    key: str
    tier: str
    unit: Unit
    lane: str
    estimate_s: float
    attempt: int = 1

    @property
    def label(self) -> str:
        return f'{self.key}:{self.tier}:{self.unit.unit_id}'


@dataclass
class Lane:
    name: str
    experiments: list
    tiers: list
    share: float = 1.0
    priority: int = 0
    only: Optional[str] = None
    where: dict = field(default_factory=dict)

    def accepts(self, key: str, tier: str, unit: Unit) -> bool:
        if key not in self.experiments or tier not in self.tiers:
            return False
        if self.only and self.only not in unit.unit_id:
            return False
        for param, bounds in self.where.items():
            value = unit.params.get(param)
            if value is None:
                return False
            if isinstance(bounds, dict):
                if 'min' in bounds and value < bounds['min']:
                    return False
                if 'max' in bounds and value > bounds['max']:
                    return False
                if 'in' in bounds and value not in bounds['in']:
                    return False
            elif value != bounds:
                return False
        return True


def load_plan(path: Path) -> dict:
    """A TOML plan: `[run]` (workers, executor, reserve_gib, note) and `[[lane]]` tables."""
    import tomllib
    with open(path, 'rb') as handle:
        raw = tomllib.load(handle)
    run = raw.get('run', {})
    lanes = [Lane(name=l['name'], experiments=[e.upper() for e in l['experiments']],
                  tiers=list(l.get('tiers', [run.get('tier', 'standard')])),
                  share=float(l.get('share', 1.0)), priority=int(l.get('priority', 0)),
                  only=l.get('only'), where=dict(l.get('where', {})))
             for l in raw.get('lane', [])]
    if not lanes:
        raise ValueError(f'{path}: a plan needs at least one [[lane]]')
    unknown = {e for lane in lanes for e in lane.experiments} - set(EXPERIMENTS)
    if unknown:
        raise ValueError(f'{path}: unknown experiments {sorted(unknown)}')
    return {'path': str(path), 'run': run, 'lanes': lanes}


def single_lane(keys: Iterable[str], tier: str, only: Optional[str] = None) -> list[Lane]:
    return [Lane(name='main', experiments=list(keys), tiers=[tier], only=only)]


def unit_cells(unit: Unit) -> float:
    """A size measure of a unit: the number of grid cells it searches (approximate)."""
    p = unit.params
    if 'cell' in p:
        extent = float(p.get('size_m') or p.get('size') or 4000)
        return (extent / float(p['cell'])) ** 2
    if 'size' in p:
        return float(p['size']) ** 2
    return 1.0e5


def seconds_per_cell(units: list[Unit], history: dict[str, float]) -> Optional[float]:
    """Median measured wall time per cell over the units of this experiment already run."""
    rates = sorted(history[u.unit_id] / unit_cells(u) for u in units if u.unit_id in history)
    return rates[len(rates) // 2] if rates else None


def estimate_seconds(key: str, unit: Unit, history: dict[str, float],
                     rate: Optional[float] = None) -> float:
    """Measured wall time of the same unit in an earlier run; else cells x the experiment's
    measured seconds per cell; else a size-based guess."""
    if unit.unit_id in history:
        return history[unit.unit_id]
    return unit_cells(unit) * (rate if rate is not None else 1.0e-5) + 0.2


def collect_tasks(lanes: list[Lane], status_of, history_of) -> list[Task]:
    """Every unit not yet done, assigned to the first lane that accepts it."""
    tasks, seen = [], set()
    for lane in sorted(lanes, key=lambda l: -l.priority):
        for key in lane.experiments:
            exp = EXPERIMENTS[key]
            for tier in lane.tiers:
                status, history = status_of(key, tier), history_of(key, tier)
                units = exp.units(tier)
                if tier != 'standard':  # the standard tier's measurements cover most sizes
                    history = {**history_of(key, 'standard'), **history}                         if tier != 'hilly' else history
                rate = seconds_per_cell(units + exp.units('standard'), history)
                for unit in units:
                    label = (key, tier, unit.unit_id)
                    if label in seen or not lane.accepts(key, tier, unit):
                        continue
                    seen.add(label)
                    if status.get(unit.unit_id, {}).get('status') == 'done':
                        continue
                    tasks.append(Task(key, tier, unit, lane.name,
                                      estimate_seconds(key, unit, history, rate)))
    return tasks


# -- the worker side -------------------------------------------------------------------------

_WORKER = threading.local()
_COUNTER = None  # a multiprocessing.Value in process workers


def _init_process_worker(counter, run_dir: str) -> None:
    global _COUNTER
    _COUNTER = counter
    with counter.get_lock():
        counter.value += 1
        number = counter.value
    _setup_worker(f'p{number:02d}', Path(run_dir))


def _setup_worker(worker_id: str, run_dir: Path) -> None:
    _WORKER.id = worker_id
    _WORKER.dir = run_dir / 'workers' / worker_id
    _WORKER.dir.mkdir(parents=True, exist_ok=True)
    _WORKER.current = None
    _WORKER.units_done = 0
    _WORKER.started = time.time()
    beat = threading.Thread(target=_heartbeat_loop, args=(_WORKER.dir, _WORKER),
                            daemon=True, name=f'heartbeat-{worker_id}')
    beat.start()


_THREAD_IDS: dict[int, str] = {}
_THREAD_LOCK = threading.Lock()


def _ensure_thread_worker(run_dir: Path) -> None:
    if getattr(_WORKER, 'id', None):
        return
    with _THREAD_LOCK:
        number = len(_THREAD_IDS) + 1
        _THREAD_IDS[threading.get_ident()] = f't{number:02d}'
    _setup_worker(_THREAD_IDS[threading.get_ident()], run_dir)


def _write_heartbeat(folder: Path, state) -> None:
    """Best effort: observing a worker must never fail the unit it runs."""
    try:
        beat = {'worker_id': state.id, 'pid': os.getpid(), 'thread': threading.get_ident(),
                'time_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'time': time.time(), 'units_done': state.units_done,
                'rss_bytes': _rss_bytes(), 'current': state.current}
        tmp = folder / f'heartbeat.{threading.get_ident()}.tmp'
        tmp.write_text(json.dumps(beat), encoding='utf-8')
        os.replace(tmp, folder / 'heartbeat.json')
    except Exception:  # noqa: BLE001
        pass


def _heartbeat_loop(folder: Path, state) -> None:
    while True:
        _write_heartbeat(folder, state)
        time.sleep(HEARTBEAT_S)


def _log(line: str) -> None:
    with open(_WORKER.dir / 'worker.log', 'a', encoding='utf-8') as handle:
        handle.write(f'{datetime.now(timezone.utc).isoformat(timespec="seconds")} {line}\n')


def execute(task: Task, options: RunOptions, run_id: str, run_dir: str, extra: dict,
            executor: str) -> dict:
    """Run one unit in a worker; returns `{'entry': ..., 'rows': [...]}`."""
    if executor == 'thread':
        _ensure_thread_worker(Path(run_dir))
    exp = EXPERIMENTS[task.key]
    prov = provenance()
    _WORKER.current = {'unit': task.label, 'lane': task.lane, 'started': time.time(),
                       'estimate_s': task.estimate_s, 'attempt': task.attempt}
    _write_heartbeat(_WORKER.dir, _WORKER)
    _log(f'start {task.label} (lane {task.lane}, attempt {task.attempt})')
    t0 = time.perf_counter()
    entry = {'unit_id': task.unit.unit_id, 'run_id': run_id, 'tier': task.tier,
             'git_sha': prov['git_sha'], 'worker_id': _WORKER.id, 'lane': task.lane,
             'attempt': task.attempt}
    rows: list[dict] = []
    try:
        rows = exp.run(task.unit, options)
        for row in rows:
            row.update({'experiment': task.key, 'unit_id': task.unit.unit_id, 'run_id': run_id,
                        **prov, 'worker_id': _WORKER.id, 'lane': task.lane, **extra})
        entry.update(status='done', n_rows=len(rows))
    except NotPossible as exc:
        entry.update(status='not_possible', reason=str(exc))
    except Exception as exc:  # noqa: BLE001 - logged and returned, the run moves on
        entry.update(status='failed', reason=f'{type(exc).__name__}: {exc}',
                     traceback=traceback.format_exc(limit=6))
    entry['wall_s'] = time.perf_counter() - t0
    entry['rss_bytes'] = _rss_bytes()
    for row in rows:
        write_jsonl(_WORKER.dir / 'rows.jsonl', {'experiment': task.key, 'tier': task.tier, **row})
    write_jsonl(_WORKER.dir / 'units.jsonl', {'experiment': task.key, **entry})
    _WORKER.units_done += 1
    _WORKER.current = None
    _write_heartbeat(_WORKER.dir, _WORKER)
    _log(f'{entry["status"]} {task.label} in {entry["wall_s"]:.1f} s'
         + (f' - {entry.get("reason")}' if entry['status'] != 'done' else ''))
    return {'entry': entry, 'rows': [sanitise(r) for r in rows]}


# -- the coordinator -------------------------------------------------------------------------

@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    counts: dict
    by_experiment: dict
    wall_s: float
    workers: int
    executor: str


def _pick(pending: dict[str, list[Task]], running: Counter, lanes: list[Lane],
          workers: int) -> Optional[Task]:
    """The next task: the lane furthest below its share of the workers, longest unit first."""
    candidates = [l for l in lanes if pending.get(l.name)]
    if not candidates:
        return None
    total_share = sum(l.share for l in candidates) or 1.0

    def deficit(lane: Lane) -> tuple:
        target = workers * lane.share / total_share
        return (running[lane.name] - target, -lane.priority)
    lane = min(candidates, key=deficit)
    return pending[lane.name].pop(0)


def run_lanes(lanes: list[Lane], options: RunOptions, *, workers, executor: str = 'process',
              reserve_gib: float = DEFAULT_RESERVE_GIB, note: str = '',
              plan_path: Optional[str] = None, max_retries: int = 1,
              dry_run: bool = False) -> RunResult:
    from core import run as run_mod  # the merged-file helpers live in core.run

    info = machine_info()
    n_workers = resolve_workers(workers, info)
    if executor == 'thread' and not info['free_threaded']:
        print('  threads need the free-threaded interpreter; using processes instead')
        executor = 'process'
    if executor == 'thread' and os.environ.get('HAG_ALLOW_THREADS') != '1':
        # 2026-10-02 benchmark: X3 standard 240-cell units on 2-15 threads crashed the
        # free-threaded interpreter (Windows 0xC0000005 / 0xC00000FD) inside a native
        # extension; process workers ran every unit. Threads stay opt-in until that is found.
        print('  thread workers are experimental (native crash seen on 3.14t); using '
              'processes. Set HAG_ALLOW_THREADS=1 to force threads.')
        executor = 'process'
    tasks = collect_tasks(lanes, run_mod.latest_status_for, run_mod.wall_history_for)
    by_lane: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        by_lane[task.lane].append(task)
    for name in by_lane:
        by_lane[name].sort(key=lambda t: -t.estimate_s)
    print(f'parallel run: {len(tasks)} units on {n_workers} {executor} workers '
          f'({info["physical_cores"]} physical cores, {info["logical_cpus"]} logical)')
    for lane in lanes:
        est = sum(t.estimate_s for t in by_lane[lane.name])
        print(f'  lane {lane.name}: {len(by_lane[lane.name])} units, about '
              f'{est / 3600:.2f} worker-hours')
    if tasks:
        total = sum(t.estimate_s for t in tasks)
        longest = max(t.estimate_s for t in tasks)
        print(f'  estimated wall time on {n_workers} workers: about '
              f'{max(total / n_workers, longest) / 3600:.2f} h (the longest unit alone: '
              f'{longest / 60:.1f} min). Estimates come from measured units of earlier runs; '
              f'parallel workers share memory bandwidth, so expect it to be longer.')
    if dry_run or not tasks:
        return RunResult('', Path(), {}, {}, 0.0, n_workers, executor)

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = RUNS / run_id
    (run_dir / 'workers').mkdir(parents=True, exist_ok=True)
    timing_mode = 'parallel' if n_workers > 1 else 'serial'
    extra = {'n_workers': n_workers, 'executor': executor, 'timing_mode': timing_mode}
    touched = sorted({(t.key, t.tier) for t in tasks})
    locks = [run_mod.lock_for(key, tier) for key, tier in touched]
    manifest = {'run_id': run_id, 'started_utc': datetime.now(timezone.utc).isoformat(),
                'plan': plan_path, 'workers': n_workers, 'executor': executor,
                'reserve_gib': reserve_gib, 'timing_mode': timing_mode, 'note': note,
                'machine': info, 'provenance': provenance(),
                'lanes': [vars(l) for l in lanes],
                'units': {l.name: len(by_lane[l.name]) for l in lanes},
                'estimate_worker_hours': sum(t.estimate_s for t in tasks) / 3600,
                'pid': os.getpid()}
    _write_json(run_dir / 'manifest.json', manifest)

    counts: Counter = Counter()
    per_exp: dict = defaultdict(Counter)
    running: Counter = Counter()
    in_flight: dict[Future, Task] = {}
    last_rebuild: dict = {}
    started = time.perf_counter()
    totals = {l.name: len(by_lane[l.name]) for l in lanes}
    done_by_lane: Counter = Counter()

    def make_pool():
        if executor == 'thread':
            return ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix='hagw')
        import multiprocessing
        ctx = multiprocessing.get_context('spawn')
        counter = ctx.Value('i', 0)
        return ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                   initializer=_init_process_worker,
                                   initargs=(counter, str(run_dir)))

    def memory_ok() -> bool:
        free = available_memory_bytes()
        return free is None or free > reserve_gib * 2**30

    def progress() -> None:
        elapsed = time.perf_counter() - started
        remaining = sum(t.estimate_s for q in by_lane.values() for t in q)
        remaining += sum(t.estimate_s for t in in_flight.values())
        _write_json(run_dir / 'progress.json', {
            'run_id': run_id, 'updated_utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_s': elapsed, 'workers': n_workers,
            'lanes': {l.name: {'total': totals[l.name], 'finished': done_by_lane[l.name],
                               'running': running[l.name], 'queued': len(by_lane[l.name])}
                      for l in lanes},
            'experiments': {k: dict(v) for k, v in per_exp.items()},
            'counts': dict(counts),
            'eta_s_estimate': remaining / max(1, n_workers)})

    def finish(task: Task, payload: dict) -> None:
        entry, rows = payload['entry'], payload['rows']
        out = run_mod.out_dir(task.key, task.tier)
        for row in rows:
            row.pop('tier', None)
            row['scenario_id'] = run_mod._scenario_id(row)
            write_jsonl(out / 'rows.jsonl', row)
        write_jsonl(out / 'units.jsonl', entry)
        counts[entry['status']] += 1
        per_exp[f'{task.key}:{task.tier}'][entry['status']] += 1
        done_by_lane[task.lane] += 1
        now = time.monotonic()
        if now - last_rebuild.get((task.key, task.tier), 0) > 30:
            run_mod.rebuild_csv(task.key, task.tier)
            last_rebuild[(task.key, task.tier)] = now
        print(f'  [{sum(done_by_lane.values())}/{len(tasks)}] {entry.get("worker_id")} '
              f'{task.label}: {entry["status"]} ({entry["wall_s"]:.1f} s)'
              + (f' - {entry.get("reason")}' if entry['status'] != 'done' else ''), flush=True)

    warned_memory = False
    pool = make_pool()
    try:
        while any(by_lane.values()) or in_flight:
            # with nothing running the gate cannot free memory by waiting: start one unit anyway
            while len(in_flight) < n_workers and (memory_ok() or not in_flight):
                if not in_flight and not memory_ok() and not warned_memory:
                    print(f'  memory gate: free memory below reserve_gib={reserve_gib}; '
                          'running one unit at a time until it frees up', flush=True)
                    warned_memory = True
                task = _pick(by_lane, running, lanes, n_workers)
                if task is None:
                    break
                future = pool.submit(execute, task, options, run_id, str(run_dir), extra,
                                     executor)
                in_flight[future] = task
                running[task.lane] += 1
            if not in_flight:
                time.sleep(5)  # memory gate: wait for memory to free up
                continue
            finished, _ = wait(list(in_flight), timeout=30, return_when=FIRST_COMPLETED)
            broken = False
            for future in finished:
                task = in_flight.pop(future)
                running[task.lane] -= 1
                try:
                    payload = future.result()
                except BrokenProcessPool:
                    broken = True
                    _requeue(task, by_lane, max_retries, 'worker process died', finish)
                    continue
                except Exception as exc:  # noqa: BLE001
                    payload = {'entry': {'unit_id': task.unit.unit_id, 'run_id': run_id,
                                         'tier': task.tier, 'status': 'failed',
                                         'reason': f'{type(exc).__name__}: {exc}',
                                         'wall_s': 0.0, 'lane': task.lane}, 'rows': []}
                if payload['entry']['status'] == 'failed' and task.attempt <= max_retries:
                    task.attempt += 1
                    by_lane[task.lane].insert(0, task)
                    print(f'  retry {task.label}: {payload["entry"].get("reason")}')
                    continue
                finish(task, payload)
            if broken:
                for future, task in list(in_flight.items()):
                    running[task.lane] -= 1
                    _requeue(task, by_lane, max_retries, 'pool rebuilt after a crash', finish)
                in_flight.clear()
                pool.shutdown(wait=False, cancel_futures=True)
                pool = make_pool()
            progress()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        for key, tier in touched:
            run_mod.rebuild_csv(key, tier)
        for lock in locks:
            lock.unlink(missing_ok=True)
        progress()
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['counts'] = dict(counts)
        manifest['wall_s'] = time.perf_counter() - started
        _write_json(run_dir / 'manifest.json', manifest)
    result = RunResult(run_id, run_dir, dict(counts), {k: dict(v) for k, v in per_exp.items()},
                       time.perf_counter() - started, n_workers, executor)
    for key_tier, c in result.by_experiment.items():
        key, tier = key_tier.split(':')
        run_mod.append_log({'experiment': key, 'tier': tier, 'run_id': run_id,
                            'counts': c, 'rows': None, 'units_to_run': sum(c.values()),
                            'units_already_done': None, 'wall_s': result.wall_s,
                            'git_sha': manifest['provenance']['git_sha'],
                            'git_dirty': manifest['provenance']['git_dirty'],
                            'engine': options.engine,
                            'note': f'parallel: {n_workers} {executor} workers, run '
                                    f'results/_runs/{run_id}/. {note}'.strip()})
    return result


def _requeue(task: Task, by_lane, max_retries: int, reason: str, finish) -> None:
    if task.attempt <= max_retries:
        task.attempt += 1
        by_lane[task.lane].insert(0, task)
        print(f'  re-queued {task.label}: {reason}')
    else:
        finish(task, {'entry': {'unit_id': task.unit.unit_id, 'tier': task.tier,
                                'status': 'failed', 'reason': reason, 'wall_s': 0.0,
                                'lane': task.lane, 'run_id': ''}, 'rows': []})


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Windows refuses the replace while a reader (the monitor, an editor, an antivirus
            # scan) has the target open; a progress file is not worth a crashed run.
            time.sleep(0.2 * (attempt + 1))
    print(f'  could not update {path.name} (in use); it will be rewritten on the next tick',
          flush=True)
