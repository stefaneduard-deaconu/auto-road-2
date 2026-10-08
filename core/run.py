"""One command for every experiment of the HAG programme.

    python -m core.run list                         # the experiments and their unit counts
    python -m core.run status                       # what is done, per experiment
    python -m core.run X3 --tier quick --dry-run    # what would run, nothing is computed
    python -m core.run X3 --tier quick              # run (resumes: finished units are skipped)
    python -m core.run all --tier quick             # X1..X7 in order
    python -m core.run report                       # write the study notes

Parallel (see core/parallel.py):

    python -m core.run X3 --tier standard --workers auto        # all physical cores
    python -m core.run all --tier standard --workers 8 --executor thread
    python -m core.run plan runs/progressive_dem.toml           # what a plan would run, and ETA
    python -m core.run --plan runs/progressive_dem.toml         # run a plan (lanes)
    python -m core.run monitor [--follow]                       # watch the latest parallel run
    python -m core.run calibrate X3 --sample 60                 # serial timings for the article
    python -m core.run machine                                  # cores, memory, interpreter
    python -m core.run list --markdown                          # the experiment table (README)

Per experiment it writes `results/<dir>/rows.jsonl` and `units.jsonl` (raw logs, git-ignored)
and rebuilds `results/<dir>/<dir>.csv` (committed) after every unit, so a run can be stopped
and inspected at any time. Each finished run appends one entry to
the study notes. A unit that cannot run here (no raster, not enough
memory) is logged `not_possible` with the reason; one that crashes is logged `failed` with
the error, and the run moves on. Exit code: 0 all done, 1 some failed, 2 bad usage/lock.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from core.programme import EXPERIMENTS, TIERS, NotPossible, RunOptions, Unit
from core.provenance import provenance, sanitise, write_jsonl
from core.results_io import read_jsonl, write_csv
from core.summary import ProvenanceMismatch, assert_one_build, describe

RESULTS = Path('results')
LOG = Path('results/experiments-log.md')
BRIEF = Path('results/results-brief.md')
BRIEF_HILLY = Path('results/results-brief-hilly.md')
IDENTITY = ('experiment', 'unit_id', 'scenario_id', 'case_id', 'family', 'search_space',
            'hag_edge_cost', 'grid_edge_cost', 'engine', 'od_id', 'od_band')


#: `_hilly` while the `hilly` tier is addressed: its results live in `results/<dir>_hilly/` so
#: they are never pooled with the other tiers' statistics.
SUFFIX = ''


def use_tier(tier: str) -> None:
    global SUFFIX
    SUFFIX = '_hilly' if tier == 'hilly' else ''


def _suffix(tier: Optional[str]) -> str:
    if tier is None:
        return SUFFIX
    return '_hilly' if tier == 'hilly' else ''


def _directory(key: str, tier: Optional[str] = None) -> str:
    return EXPERIMENTS[key].directory + _suffix(tier)


def _out(key: str, tier: Optional[str] = None) -> Path:
    return RESULTS / _directory(key, tier)


def out_dir(key: str, tier: str) -> Path:
    """The merged-results folder of one experiment and tier (created if missing)."""
    out = _out(key, tier)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _units_log(key: str, tier: Optional[str] = None) -> list[dict]:
    path = _out(key, tier) / 'units.jsonl'
    return read_jsonl(path) if path.exists() else []


def _latest_status(key: str, tier: Optional[str] = None) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for entry in _units_log(key, tier):
        latest[entry['unit_id']] = entry
    return latest


def latest_status_for(key: str, tier: str) -> dict[str, dict]:
    return _latest_status(key, tier)


def wall_history_for(key: str, tier: str) -> dict[str, float]:
    """Measured `wall_s` per unit: this folder first, then archived `results_*` copies
    (the longest-first order of the parallel runner is built from it)."""
    history: dict[str, float] = {}
    name = _directory(key, tier)
    for path in sorted(Path('.').glob(f'results_*/{name}/units.jsonl')) +             [_out(key, tier) / 'units.jsonl']:
        if not path.exists():
            continue
        for entry in read_jsonl(path):
            if entry.get('status') == 'done' and entry.get('wall_s') is not None:
                history[entry['unit_id']] = float(entry['wall_s'])
    return history


def _scenario_id(row: dict) -> str:
    parts = [row.get('unit_id'), row.get('od_id'), row.get('search_space'),
             row.get('hag_edge_cost'), row.get('grid_edge_cost'), row.get('engine')]
    return '__'.join(str(p) for p in parts if p not in (None, ''))


def rebuild_csv(key: str, tier: Optional[str] = None) -> Optional[Path]:
    rows_path = _out(key, tier) / 'rows.jsonl'
    if not rows_path.exists():
        return None
    rows = read_jsonl(rows_path)
    latest = _latest_status(key, tier)
    done_runs = {u: e.get('run_id') for u, e in latest.items() if e.get('status') == 'done'}
    rows = [r for r in rows if done_runs.get(r.get('unit_id')) == r.get('run_id')]
    if not rows:
        return None
    rows.sort(key=lambda r: (str(r.get('unit_id')), str(r.get('scenario_id'))))
    columns = list(IDENTITY)
    for row in rows:
        columns += [k for k in row if k not in columns]
    return write_csv(_out(key, tier) / f'{_directory(key, tier)}.csv', rows, columns)


class Locked(RuntimeError):
    pass


def lock_for(key: str, tier: str) -> Path:
    return _lock(key, tier)


def _lock(key: str, tier: Optional[str] = None) -> Path:
    """Exclusive-create the lock file: atomic on both POSIX and Windows, unlike a separate
    `exists()` check followed by a write, which lets two processes started close together
    both pass the check before either has written the file."""
    lock = _out(key, tier) / '.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(lock, 'x', encoding='utf-8') as handle:
            handle.write(str(os.getpid()))
    except FileExistsError:
        raise Locked(f'{lock} exists: another run of {key} is active, or one crashed '
                     '(delete the file if no python process is running it)') from None
    return lock


def run_experiment(key: str, tier: str, options: RunOptions, *, dry_run: bool = False,
                   fresh: bool = False, only: Optional[str] = None,
                   max_units: Optional[int] = None, note: str = '') -> dict:
    exp = EXPERIMENTS[key]
    units = [u for u in exp.units(tier) if not only or only in u.unit_id]
    status = {} if fresh else _latest_status(key)
    todo = [u for u in units if status.get(u.unit_id, {}).get('status') != 'done']
    if max_units is not None:
        todo = todo[:max_units]
    summary = {'experiment': key, 'tier': tier, 'units_total': len(units),
               'units_already_done': len(units) - len([u for u in units if status.get(
                   u.unit_id, {}).get('status') != 'done']), 'units_to_run': len(todo)}
    print(f'{key} ({exp.title}), tier {tier}: {summary["units_total"]} units, '
          f'{summary["units_already_done"]} done, {len(todo)} to run')
    if dry_run:
        for unit in todo[:20]:
            print(f'  would run {unit.unit_id}')
        if len(todo) > 20:
            print(f'  ... and {len(todo) - 20} more')
        return summary
    if exp.timing_sensitive:
        print('  TIMING-SENSITIVE: run it alone on an idle machine and say in --note what '
              'else was open (an editor, a browser, a container runtime).')
    if exp.requires_raster and not options.tif.exists():
        print(f'  raster {options.tif} not found: every unit will be logged not_possible')
    out = _out(key)
    lock = _lock(key)
    if fresh:
        for name in ('rows.jsonl', 'units.jsonl'):
            (out / name).unlink(missing_ok=True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prov = provenance()
    counts: Counter = Counter()
    n_rows, started = 0, time.perf_counter()
    try:
        for k, unit in enumerate(todo, 1):
            t0 = time.perf_counter()
            entry = {'unit_id': unit.unit_id, 'run_id': run_id, 'tier': tier,
                     'git_sha': prov['git_sha']}
            try:
                rows = exp.run(unit, options)
                for row in rows:
                    row.update({'experiment': key, 'unit_id': unit.unit_id, 'run_id': run_id,
                                **prov, 'n_workers': 1, 'executor': 'serial',
                                'timing_mode': 'serial'})
                    row['scenario_id'] = _scenario_id(row)
                    write_jsonl(out / 'rows.jsonl', row)
                entry.update(status='done', n_rows=len(rows))
                n_rows += len(rows)
            except NotPossible as exc:
                entry.update(status='not_possible', reason=str(exc))
            except Exception as exc:  # noqa: BLE001 - logged, and the run moves on
                entry.update(status='failed', reason=f'{type(exc).__name__}: {exc}',
                             traceback=traceback.format_exc(limit=6))
            entry['wall_s'] = time.perf_counter() - t0
            write_jsonl(out / 'units.jsonl', entry)
            counts[entry['status']] += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / k * (len(todo) - k)
            print(f'  [{k}/{len(todo)}] {unit.unit_id}: {entry["status"]} '
                  f'({entry["wall_s"]:.1f} s; ETA {eta / 60:.1f} min)'
                  + (f' - {entry.get("reason")}' if entry['status'] != 'done' else ''),
                  flush=True)
            rebuild_csv(key)
    finally:
        lock.unlink(missing_ok=True)
    summary.update({'run_id': run_id, 'counts': dict(counts), 'rows': n_rows,
                    'wall_s': time.perf_counter() - started, 'git_sha': prov['git_sha'],
                    'git_dirty': prov['git_dirty'], 'engine': options.engine, 'note': note})
    append_log(summary)
    return summary


def append_log(summary: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        LOG.write_text('# Experiments log\n\nAppended by `python -m core.run` after every run; '
                       'add observations under **Notes** by hand. Newest last.\n',
                       encoding='utf-8')
    counts = ', '.join(f'{k} {v}' for k, v in sorted(summary['counts'].items())) or 'nothing'
    lines = [
        '', f'## {summary["run_id"]} - {summary["experiment"]} ({summary["tier"]})', '',
        f'- command: `python -m core.run {summary["experiment"]} --tier {summary["tier"]}`',
        f'- git: `{summary["git_sha"]}` (dirty: {summary["git_dirty"]}); engine: '
        f'{summary["engine"]}; machine: {sys.platform}, Python {sys.version.split()[0]}',
        f'- units: {counts} (of {summary["units_to_run"]} to run, '
        f'{summary["units_already_done"]} already done); rows: {summary["rows"]}; '
        f'wall time: {summary["wall_s"] / 60:.1f} min',
        f'- other load while running: {summary["note"] or "not stated"}',
        '- Notes: ',
    ]
    with open(LOG, 'a', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def print_status() -> None:
    for key, exp in EXPERIMENTS.items():
        latest = _latest_status(key)
        counts = Counter(e.get('status') for e in latest.values())
        csv = _out(key) / f'{_directory(key)}.csv'
        shas = {e.get('git_sha') for e in latest.values()}
        print(f'{key} {_directory(key)}: {dict(counts) or "never run"}; csv '
              f'{"present" if csv.exists() else "missing"}; git {sorted(map(str, shas))}')
        for tier in (('hilly',) if SUFFIX else TIERS[:3]):
            ids = {u.unit_id for u in exp.units(tier)}
            done = sum(1 for u in ids if latest.get(u, {}).get('status') == 'done')
            print(f'    {tier:8s} {done}/{len(ids)} units done')


def print_list() -> None:
    for key, exp in EXPERIMENTS.items():
        sizes = ', '.join(f'{t} {len(exp.units(t))}' for t in (('hilly',) if SUFFIX else TIERS[:3]))
        flags = ', '.join(f for f, on in (('timing-sensitive', exp.timing_sensitive),
                                          ('needs raster', exp.requires_raster),
                                          ('runs alone, never with --workers',
                                           exp.exclusive)) if on)
        print(f'{key}  {exp.title}  [{sizes} units]{"  (" + flags + ")" if flags else ""}')
        print(f'    {exp.question}')
        print(f'    -> results/{_directory(key)}/')


def experiments_markdown() -> str:
    """The table of experiments for README.md, generated from `core.programme.EXPERIMENTS`
    so it cannot drift (tests/test_parallel.py compares it with the README)."""
    lines = ['| ID | Name | Question it answers | Units: quick / standard / full / hilly '
             '| Results folder | Notes |', '|---|---|---|---|---|---|',
             '| X0 | The compiled engine equals the Python reference | Do `core/native` (the '
             'ctypes library and the standalone executable) return the same distances, paths '
             'and node counts as `core.search`? | tests only | - | run first: `pytest '
             'tests/test_native_engine.py tests/test_native_exe.py` |']
    x8 = ('| X8 | STAS 863-85 at every point of the alignment | Does each alignment meet '
          'every STAS 863-85 parameter at every station, before and after Algorithm 1? '
          '| one run | `results/stas/` | `python -m core.experiment_stas`, not part of '
          '`core.run` |')
    for key, exp in EXPERIMENTS.items():
        if key == 'X9':
            lines.append(x8)
        sizes = ' / '.join(str(len(exp.units(t))) for t in ('quick', 'standard', 'full', 'hilly'))
        flags = [f for f, on in (('timing-sensitive', exp.timing_sensitive),
                                 ('needs the Idrija raster', exp.requires_raster),
                                 ('runs alone, never with --workers', exp.exclusive)) if on]
        lines.append(f'| {key} | {exp.title} | {exp.question} | {sizes} | '
                     f'`results/{exp.directory}/` | {", ".join(flags) or "-"} |')
    if x8 not in lines:
        lines.append(x8)
    return '\n'.join(lines) + '\n'


# -- the results brief ---------------------------------------------------------------------

#: per experiment: the grouping columns and the quantities summarised per group
BRIEF_SPEC = {
    'X1': (('family', 'cell_size_m', 'height_delta_m'),
           ('n_areas', 'edges_per_area_mean', 'degree_max', 'planarity_ratio',
            'area_m2_median', 'area_size_gini', 'hop_diameter_lb', 'hag_build_time_s',
            'terrain_slope_gon_p50', 'share_slope_20_25_gon')),
    'X2': (('grid_edge_cost', 'search_space', 'hag_edge_cost'),
           ('search_space_percent_of_full', 'objective_ratio_to_full', 'hausdorff_to_full_m',
            'cta_hops', 'path_found')),
    'X3': (('search_space', 'hag_edge_cost', 'height_delta_m'),
           ('search_space_percent_of_full', 'objective_ratio_to_full', 'length_ratio_to_full',
            'hausdorff_to_full_m', 'nodes_expanded_ratio_to_full', 'search_time_ratio_to_full',
            'memory_ratio_to_full', 'break_even_queries', 'path_found')),
    'X4': (('cell_size_m', 'search_space'),
           ('search_space_percent_of_full', 'objective_ratio_to_full', 'hausdorff_to_full_m',
            'hag_n_areas', 'search_time_ratio_to_full')),
    'X5': (('engine', 'grid_rows', 'search_space'),
           ('search_time_s_median', 'search_nodes_expanded', 'peak_memory_bytes',
            'hag_build_time_s', 'break_even_queries')),
    'X6': (('cell_size_m', 'height_delta_m', 'search_space', 'hag_edge_cost', 'od_band'),
           ('search_space_percent_of_full', 'objective_ratio_to_full', 'hausdorff_to_full_m',
            'search_time_ratio_to_full', 'break_even_queries', 'path_found')),
    'X9': (('engine_x9', 'grid_rows', 'workers'),
           ('throughput_qps', 'speedup_vs_1', 'efficiency', 'search_s_median',
            'load_or_build_ms', 'costs_match_reference')),
    'X7': (('cell_size_m', 'search_space', 'hag_edge_cost'),
           ('search_space_percent_of_full', 'objective_ratio_to_full', 'hausdorff_to_full_m',
            'search_time_ratio_to_full', 'path_found')),
}


def _num(value):
    if isinstance(value, bool):
        return float(value)
    return value


def _reported(row: dict) -> bool:
    """A row the brief reports: a search space and HAG edge cost the study still uses. Rows
    of an older run with an option since removed stay in the CSV but are not summarised."""
    from core.hag import HAG_EDGE_COSTS
    from core.naming import SEARCH_SPACES
    space, edge = row.get('search_space'), row.get('hag_edge_cost')
    return ((space is None or space in SEARCH_SPACES)
            and (edge in (None, '') or edge in HAG_EDGE_COSTS))


def write_brief(path: Path = BRIEF) -> Path:
    lines = ['# Results brief', '',
             '<!-- GENERATED by `python -m core.run report` from results/x*/ ; never edit. '
             'Every number is a statistic of the named CSV column over the named group. -->',
             '', 'Fact keys are `X<n>.<group values>.<column>.<stat>`; quote numbers only '
                 'from here or from `article_2/build_article.py` facts.', '']
    for key, exp in EXPERIMENTS.items():
        csv = _out(key) / f'{_directory(key)}.csv'
        latest = _latest_status(key)
        counts = Counter(e.get('status') for e in latest.values())
        lines += [f'## {key} - {exp.title}', '', f'*{exp.question}*', '',
                  f'- source: `{csv.as_posix()}`; units: {dict(counts) or "never run"}']
        for status in ('not_possible', 'failed'):
            reasons = Counter(e.get('reason', '')[:160] for e in latest.values()
                              if e.get('status') == status)
            for reason, n in reasons.most_common(5):
                lines.append(f'- {status} ({n}): {reason}')
        if not csv.exists():
            lines += ['- no CSV yet: nothing to report', '']
            continue
        from core.results_io import read_csv
        rows = [r for r in read_csv(csv) if _reported(r)]
        group_cols, quantities = BRIEF_SPEC[key]
        # an engine-split table may hold both engines; the build check then ignores it
        checked = ([{k: v for k, v in r.items() if k != 'engine'} for r in rows]
                   if 'engine' in group_cols else rows)
        try:
            block = assert_one_build(checked)
            lines.append(f'- build: git `{block["git_sha"]}`, Python {block["python"]} '
                         f'(free-threaded {block["python_free_threaded"]}), '
                         f'{block["os_family"]}, engine {block.get("engine", "per row")}; '
                         f'dirty rows: {block["git_dirty_values"]}')
        except ProvenanceMismatch as exc:
            lines.append(f'- **mixed builds, not summarised**: {exc}')
            lines.append('')
            continue
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            groups[tuple(row.get(c) for c in group_cols)].append(row)
        lines += ['', '| group (' + ', '.join(group_cols) + ') | column | n | mean | SD | '
                  'min | median | max | fact key |', '|---|---|---|---|---|---|---|---|---|']
        for group in sorted(groups, key=lambda g: tuple(str(v) for v in g)):
            label = ', '.join(str(v) for v in group)
            for q in quantities:
                values = [_num(r.get(q)) for r in groups[group] if r.get(q) not in (None, '')]
                if not values:
                    continue
                s = describe(values, q)
                fact = f'{key}.' + '.'.join(str(v) for v in group) + f'.{q}'

                def f(x):
                    return '-' if x is None or (isinstance(x, float) and math.isnan(x)) \
                        else f'{x:.4g}'
                lines.append(f'| {label} | {q} | {s.n} | {f(s.mean)} | {f(s.sd)} | '
                             f'{f(s.minimum)} | {f(s.median)} | {f(s.maximum)} | `{fact}` |')
        lines.append('')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m core.run', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', nargs='?', default=None,
                        help="an experiment id (X1..X9), 'all', 'list', 'status', 'report', "
                             "'machine', 'monitor', 'plan PLAN.toml' or 'calibrate X3'")
    parser.add_argument('target', nargs='?', default=None,
                        help="the plan file of 'plan', the experiment of 'calibrate'")
    parser.add_argument('--tier', choices=TIERS, default='quick')
    parser.add_argument('--dry-run', action='store_true', help='list the units, run nothing')
    parser.add_argument('--fresh', action='store_true',
                        help='discard this experiment\'s logs and start over (no resume)')
    parser.add_argument('--engine', choices=('auto', 'native', 'python'), default='auto',
                        help='Dijkstra engine; auto = native when available (X5 runs both)')
    parser.add_argument('--repeats', type=int, help='timing repeats (default per experiment)')
    parser.add_argument('--only', help='run only units whose id contains this text')
    parser.add_argument('--max-units', type=int, help='stop after this many units')
    parser.add_argument('--tif', type=Path, default=RunOptions().tif, help='the Idrija raster')
    parser.add_argument('--note', default='', help='what else was running, for the log')
    parser.add_argument('--workers', default=None,
                        help="parallel workers: a number or 'auto' (physical performance "
                             "cores); omitted = the serial runner, exactly as before")
    parser.add_argument('--executor', choices=('process', 'thread'), default='process',
                        help='parallel workers as processes (default) or threads '
                             '(free-threaded interpreter only)')
    parser.add_argument('--plan', type=Path, help='run a plan file (lanes); see runs/')
    parser.add_argument('--reserve-gib', type=float, default=None,
                        help='keep this much memory free: no unit starts below it (default 2)')
    parser.add_argument('--sample', type=int, default=60, help="'calibrate': units to re-time")
    parser.add_argument('--seed', type=int, default=2026, help="'calibrate': sample seed")
    parser.add_argument('--follow', action='store_true', help="'monitor': refresh until done")
    parser.add_argument('--interval', type=float, default=10.0, help="'monitor': seconds")
    parser.add_argument('--markdown', action='store_true', help="'list': a Markdown table")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command is None and args.plan is None:
        parser.error('give a command, an experiment id, or --plan PLAN.toml')
    command = args.command or 'plan-run'
    command = command.upper() if command.lower().startswith('x') else command
    use_tier(args.tier)
    options = RunOptions(engine=args.engine, repeats=args.repeats, tif=args.tif,
                         tier_hint=args.tier)
    if command == 'list':
        if args.markdown:
            print(experiments_markdown(), end='')
        else:
            print_list()
        return 0
    if command == 'status':
        print_status()
        return 0
    if command == 'report':
        print(write_brief(BRIEF_HILLY if SUFFIX else BRIEF))
        return 0
    if command == 'machine':
        from core.parallel import auto_workers, machine_info
        info = machine_info()
        print(json.dumps({**info, 'auto_workers': auto_workers(info)}, indent=2))
        return 0
    if command == 'monitor':
        from core.monitor import monitor
        return monitor(follow=args.follow, interval=args.interval)
    if command == 'calibrate':
        if not args.target:
            parser.error("'calibrate' needs an experiment, e.g. calibrate X3")
        from core.calibrate import calibrate
        return calibrate(args.target.upper(), args.tier, options, sample=args.sample,
                         seed=args.seed, note=args.note, dry_run=args.dry_run)
    if command in ('plan', 'plan-run'):
        plan_path = Path(args.target) if command == 'plan' else args.plan
        if plan_path is None:
            parser.error("'plan' needs a plan file, e.g. plan runs/progressive_dem.toml")
        from core.parallel import DEFAULT_RESERVE_GIB, load_plan, run_lanes
        plan = load_plan(plan_path)
        run = plan['run']
        try:
            result = run_lanes(plan['lanes'], options,
                               workers=args.workers or run.get('workers', 'auto'),
                               executor=args.executor if args.workers
                               else run.get('executor', args.executor),
                               reserve_gib=args.reserve_gib or run.get('reserve_gib',
                                                                       DEFAULT_RESERVE_GIB),
                               note=args.note or run.get('note', ''),
                               plan_path=str(plan_path),
                               dry_run=command == 'plan' or args.dry_run)
        except Locked as exc:
            print(exc, file=sys.stderr)
            return 2
        return 1 if result.counts.get('failed') else 0
    keys = list(EXPERIMENTS) if command == 'all' else [command]
    unknown = [k for k in keys if k not in EXPERIMENTS]
    if unknown:
        parser.error(f'unknown experiment {unknown}; known: {list(EXPERIMENTS)}')
    if command == 'all':
        keys = [k for k in keys if not EXPERIMENTS[k].exclusive]
    if args.workers is not None and not args.dry_run:
        exclusive = [k for k in keys if EXPERIMENTS[k].exclusive]
        if exclusive:
            parser.error(f'{exclusive} measure the machine itself and must run alone: '
                         'run them without --workers')
        from core.parallel import DEFAULT_RESERVE_GIB, run_lanes, single_lane
        try:
            result = run_lanes(single_lane(keys, args.tier, args.only), options,
                               workers=args.workers, executor=args.executor,
                               reserve_gib=args.reserve_gib or DEFAULT_RESERVE_GIB,
                               note=args.note)
        except Locked as exc:
            print(exc, file=sys.stderr)
            return 2
        return 1 if result.counts.get('failed') else 0
    failed = False
    for key in keys:
        try:
            summary = run_experiment(key, args.tier, options, dry_run=args.dry_run,
                                     fresh=args.fresh, only=args.only,
                                     max_units=args.max_units, note=args.note)
        except Locked as exc:
            print(exc, file=sys.stderr)
            return 2
        failed |= bool(summary.get('counts', {}).get('failed'))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
