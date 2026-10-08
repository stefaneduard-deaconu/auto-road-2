"""T7, the robustness matrix: the study protocol, research step 4.

    python -m core.experiment_matrix --out results

3 terrains x 3 O-D pairs x 3 road classes. The four search arms are computed once per
(terrain, O-D) and reused across the classes, because the objective is Equation 1 and
does not depend on the road class; only Algorithm 1 and the engineering checks are
repeated per class and per iteration setting.

This module defines no geometry literal of its own. The terrains come from
`TerrainSpec`, the classes from `data.configs.road_classes.PROPOSED_MATRIX_CLASSES`,
the O-D pairs from `core.od_sampler`, and the only coordinates written here are the
fixed reference pair (5, 5)-(94, 94), which is cited where it appears.
"""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

from core.algorithm_1 import before_after
from core.checks import check_path
from core.experiment_step1 import DEFAULT_ARMS, Step1Config, run_step1_full
from core.hag import build_hag
from core.od_sampler import ODPair, od_pairs_rows, reference_pair, sample_od_pairs
from core.provenance import write_jsonl
from core.results_io import (MATRIX_COLUMNS, OD_PAIRS_COLUMNS, matrix_rows,
                             path_csv_name, read_jsonl, write_csv, write_path_csv)
from core.summary import summary_from_csv
from core.terrain import TerrainSpec, generate_terrain
from data.configs.road_classes import (PROPOSED_MATRIX_CLASSES,
                                       PROPOSED_MATRIX_SYNTHETIC, get)


@dataclass(frozen=True)
class TerrainCase:
    terrain_id: str
    spec: TerrainSpec
    rationale: str


TERRAINS: tuple[TerrainCase, ...] = (
    TerrainCase('T_article', TerrainSpec(seed=0),
                'the reference terrain (seed 0), so the new numbers can be '
                'compared with earlier ones'),
    TerrainCase('T_smooth', TerrainSpec(seed=7, periods=(2, 2)),
                'longer wavelengths: fewer, larger height areas, so the HAG chain is '
                'short and wide'),
    TerrainCase('T_rough', TerrainSpec(seed=13, periods=(5, 5),
                                       height_interval=(100.0, 130.0)),
                'shorter wavelengths and a larger height range: many small areas, so '
                'the chain is long and narrow'),
)

QUICK_TERRAINS: tuple[TerrainCase, ...] = (
    TerrainCase('Q_a', TerrainSpec(seed=0, grid_size=(40, 40), periods=(2, 2)), 'plumbing'),
    TerrainCase('Q_b', TerrainSpec(seed=7, grid_size=(40, 40), periods=(2, 2)), 'plumbing'),
)

HEIGHT_DELTA_M = 3.0
ITERATIONS: tuple[int, ...] = (1, 2)
TIMING_REPEATS = 5
#: the study protocol: the fixed reference pair, reported as a
#: reference row and excluded from every statistic
REFERENCE_PAIR = ((5, 5), (94, 94))


@dataclass(frozen=True)
class MatrixOptions:
    out_dir: Path
    quick: bool = False
    force: bool = False
    summary_only: bool = False
    write_paths: str = 'primary'          # 'none' | 'primary' | 'all'
    timing_repeats: int = TIMING_REPEATS
    iterations: tuple[int, ...] = ITERATIONS


def scenario_id(terrain_id: str, od_id: str) -> str:
    return f'{terrain_id}__{od_id}'


def _class_names(quick: bool) -> tuple[str, ...]:
    return PROPOSED_MATRIX_CLASSES[:2] if quick else PROPOSED_MATRIX_CLASSES


def run_scenario(case: TerrainCase, od: ODPair, class_names: Sequence[str],
                 options: MatrixOptions) -> tuple[dict, dict, dict]:
    """One (terrain, O-D): every arm, then Algorithm 1 and the checks per arm x class.

    Returns `(row, rough_paths_xyz, smoothed_paths_xyz)`. The third dict is keyed by
    `(arm, class_name, iterations)` and is only ever consulted when `options.write_paths ==
    'all'` (`_write_paths`); it costs nothing extra to collect, since `_smoothed_xyz` was
    already being computed and thrown away right after `check_path` used it.
    """
    config = Step1Config(terrain=case.spec, start=od.start, target=od.target,
                         height_delta=HEIGHT_DELTA_M, arms=DEFAULT_ARMS,
                         timing_repeats=options.timing_repeats,
                         terrain_label=case.terrain_id)
    run = run_step1_full(config)

    geometry: dict = {}
    smoothed_paths: dict = {}
    for arm, path_xy in run.paths_xy.items():
        per_arm: dict = {}
        for class_name in class_names:
            road_class = get(class_name).with_cell_size(run.grid.cell_size_m)
            per_class: dict = {}
            for iterations in options.iterations:
                report = before_after(run.grid, path_xy, road_class,
                                      iterations=iterations,
                                      metric_step_m=config.metric_step_m,
                                      metric_chord_m=config.metric_chord_m)
                smoothed = _smoothed_xyz(run.grid, path_xy, road_class, iterations)
                smoothed_paths[(arm, class_name, iterations)] = smoothed
                checks = check_path(smoothed, road_class,
                                    step_m=config.metric_step_m,
                                    chord_m=config.metric_chord_m)
                per_class[str(iterations)] = {'before_after': report,
                                              'checks': checks.as_dict()}
            per_arm[class_name] = per_class
        geometry[arm] = per_arm

    row = dict(run.row)
    row.update({
        'scenario_id': scenario_id(case.terrain_id, od.od_id),
        'terrain_id': case.terrain_id,
        'terrain_seed': case.spec.seed,
        'terrain_rationale': case.rationale,
        'od_id': od.od_id,
        'od_seed': None if not od.protocol else None,
        'od_protocol': od.protocol,
        'od_pair': od.as_dict(),
        'geometry': geometry,
    })
    return row, run.paths_xyz_m, smoothed_paths


def _smoothed_xyz(grid, path_xy, road_class, iterations):
    from core.algorithm_1 import algorithm_1
    return algorithm_1(path_xy, road_class, grid=grid, iterations=iterations).smooth_xyz_m


def _write_paths(options: MatrixOptions, case: TerrainCase, od: ODPair,
                 paths_xyz: dict, smoothed_paths: dict) -> None:
    """`'none'` writes nothing; `'primary'` (the default) writes each arm's rough axis, the
    same as always; `'all'` additionally writes the smoothed axis of every (arm, class,
    iterations), which is what lets a figure be drawn from committed files without
    re-running Algorithm 1 (`core.figures` never re-runs a search itself)."""
    if options.write_paths == 'none':
        return
    folder = options.out_dir / 'paths'
    for arm, xyz in paths_xyz.items():
        write_path_csv(folder / path_csv_name(case.terrain_id, od.od_id, arm), xyz)
    if options.write_paths == 'all':
        for (arm, class_name, iterations), xyz in smoothed_paths.items():
            name = path_csv_name(case.terrain_id, od.od_id, arm,
                                 road_class=class_name, iterations=iterations)
            write_path_csv(folder / name, xyz)


class MatrixAlreadyRunning(RuntimeError):
    """Another run holds the lock on this output directory."""


@contextmanager
def _lock(out_dir: Path):
    """One runner per output directory.

    Without this, two concurrent runs both read the log at start, both decide the same
    scenarios are outstanding, and both append them: the log ends up with duplicate
    scenario_ids and the summary silently double-counts them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / '.matrix.lock'
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise MatrixAlreadyRunning(
            f"{lock} exists, so another run is using {out_dir}. If no run is active, "
            f"delete it.") from None
    try:
        os.write(handle, str(os.getpid()).encode())
        os.close(handle)
        yield lock
    finally:
        lock.unlink(missing_ok=True)


def _assert_still_ours(lock: Path) -> None:
    """The lock file must still exist and still name this process.

    Deleting the output directory while a run is in progress removes the lock, after
    which a second run starts happily and both append to the same log. That produced
    duplicate scenario_ids, which the summary would silently double-count. Checking
    before every append turns it into a loud failure.
    """
    try:
        owner = lock.read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        raise MatrixAlreadyRunning(
            f"{lock} disappeared mid-run: the output directory was deleted or another "
            f"run took it over. Stopping rather than appending to a log that may "
            f"already hold duplicates.") from None
    if owner != str(os.getpid()):
        raise MatrixAlreadyRunning(
            f"{lock} now belongs to process {owner}, not {os.getpid()}.")


def run_matrix(options: MatrixOptions) -> dict:
    with _lock(options.out_dir) as lock:
        return _run_matrix_locked(options, lock)


def _run_matrix_locked(options: MatrixOptions, lock: Path) -> dict:
    out = options.out_dir
    jsonl = out / 'matrix.jsonl'
    if options.force and jsonl.exists():
        jsonl.unlink()      # a forced rerun must not leave two rows for one scenario

    done = {row.get('scenario_id') for row in read_jsonl(jsonl)}
    terrains = QUICK_TERRAINS if options.quick else TERRAINS
    class_names = _class_names(options.quick)
    iterations = (1,) if options.quick else options.iterations
    options = replace(options, iterations=iterations)

    samples = []
    for case in terrains:
        grid = generate_terrain(case.spec)
        hag = build_hag(grid, height_delta=HEIGHT_DELTA_M)
        sample = sample_od_pairs(grid, hag, terrain_id=case.terrain_id,
                                 terrain_seed=case.spec.seed,
                                 n_pairs=2 if options.quick else 3)
        samples.append(sample)

        pairs = list(sample.pairs)
        if case.terrain_id == 'T_article':
            pairs.append(reference_pair(grid, hag, *REFERENCE_PAIR))

        for od in pairs:
            sid = scenario_id(case.terrain_id, od.od_id)
            if sid in done:
                print(f'  skip {sid} (already in the log; --force to redo)')
                continue
            print(f'  run  {sid}: {od.start} -> {od.target}')
            _assert_still_ours(lock)
            row, paths, smoothed_paths = run_scenario(case, od, class_names, options)
            row['od_seed'] = sample.od_seed
            write_jsonl(jsonl, row)
            _write_paths(options, case, od, paths, smoothed_paths)

    write_csv(out / 'od_pairs.csv', od_pairs_rows(samples), OD_PAIRS_COLUMNS)
    return _rebuild_tables(options)


def _rebuild_tables(options: MatrixOptions) -> dict:
    """Regenerate the CSV and the summary from the WHOLE log, so they cannot go stale."""
    out = options.out_dir
    scenarios = read_jsonl(out / 'matrix.jsonl')
    if not scenarios:
        raise SystemExit(f'no scenarios in {out / "matrix.jsonl"}')
    csv_path = write_csv(out / 'matrix.csv', matrix_rows(scenarios), MATRIX_COLUMNS)

    classes = ', '.join(_class_names(options.quick))
    header = {
        'Terrains:': ', '.join(f'{c.terrain_id}(seed {c.spec.seed})'
                               for c in (QUICK_TERRAINS if options.quick else TERRAINS))
                     + f'; height_delta {HEIGHT_DELTA_M:g} m',
        'O-D:': 'protocol of the study protocol, od_seed = 1000 + terrain seed',
        'Classes:': f'{classes}  [all TO CONFIRM]',
        'Sources:': 'Ordin MT 1296/2017, Tabelul nr. 1 a) and Tabelul nr. 2 B)',
    }
    summary_path = out / 'summary.md'
    summary_path.write_text(summary_from_csv(csv_path, header_extra=header),
                            encoding='utf-8')
    return {'jsonl': out / 'matrix.jsonl', 'csv': csv_path,
            'od': out / 'od_pairs.csv', 'summary': summary_path}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='results', type=Path)
    parser.add_argument('--quick', action='store_true',
                        help='a small subset on 40x40 terrains, for checking the plumbing')
    parser.add_argument('--force', action='store_true',
                        help='truncate the log and redo every scenario')
    parser.add_argument('--summary-only', action='store_true',
                        help='regenerate the CSV and summary from the existing log')
    parser.add_argument('--write-paths', choices=('none', 'primary', 'all'),
                        default='primary')
    parser.add_argument('--repeats', type=int, default=TIMING_REPEATS)
    args = parser.parse_args(argv)

    out = args.out
    if args.quick and args.out == Path('results'):
        out = args.out / 'quick'      # a quick run must never poison the real log

    options = MatrixOptions(out_dir=out, quick=args.quick, force=args.force,
                            summary_only=args.summary_only,
                            write_paths=args.write_paths, timing_repeats=args.repeats)
    paths = _rebuild_tables(options) if args.summary_only else run_matrix(options)
    for name, path in paths.items():
        print(f'{name:>8}: {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
