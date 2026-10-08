"""Drives the three figure functions `core.figures` implements but never calls: T9.

    python -m core.figures_scenarios --from results --out results/figures

`core.figures` reads `results/matrix.csv` and `results/paths/*.csv` and "never re-runs a
search", so that a figure is always reproducible from committed files. That rule is kept
here too, but it needs one thing `core.figures` alone cannot give it: `fig_terrain_and_areas`
draws the height-area partition, and no committed file holds the raster or the HAG labels.

The distinction this module draws, and states once so it does not have to be re-argued at
every call site: **regenerating the TERRAIN and its HAG from the identity already in
`core.experiment_matrix.TERRAINS` (the same `TerrainSpec`s the committed matrix was built
from) is deterministic and cheap** -- `core.terrain.generate_terrain` and `core.hag.build_hag`
are pure functions of that config, not a timed search, and `tests/test_figures.py` already
regenerates a terrain the same way for its own fixtures. What this module still never does
is run Dijkstra or Algorithm 1: every ALIGNMENT it draws, rough or smoothed, is read from
`results/paths/*.csv`, whose names it constructs with `core.results_io.path_csv_name` --
the same function that wrote them -- rather than parsed back out of a glob, since both the
terrain id and the arm name can themselves contain underscores.

Two figures per terrain, at its `od1` scenario:

    fig_terrain_and_areas     the yellow chain vs. the full HAG, all four arms overlaid
    fig_longitudinal_profile  before/after, one per (arm, road class) found in the paths dir
    fig_curvature             before/after, one per (arm, road class) found in the paths dir

If `results/paths/` holds only the rough axes (`--write-paths primary`, the default), the
longitudinal-profile and curvature figures are skipped and this says so on stderr, rather
than drawing an empty or misleading figure.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from core import figures  # noqa: E402
from core.experiment_matrix import ITERATIONS, TERRAINS  # noqa: E402
from core.figures import ARM_ORDER  # noqa: E402
from core.hag import build_hag, select_areas  # noqa: E402
from core.od_sampler import sample_od_pairs  # noqa: E402
from core.results_io import path_csv_name  # noqa: E402
from core.terrain import generate_terrain  # noqa: E402
from data.configs.road_classes import PROPOSED_MATRIX_CLASSES, get  # noqa: E402


def _load_xyz(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=',', skiprows=1)


def _terrain_and_areas_figure(case, od_id: str, paths_dir: Path, out_dir: Path,
                              grid, hag, selected, start=None, target=None) -> Optional[list]:
    paths = {}
    for arm in ARM_ORDER:
        rough = paths_dir / path_csv_name(case.terrain_id, od_id, arm)
        if rough.exists():
            paths[arm] = _load_xyz(rough)
    if not paths:
        return None
    fig = figures.fig_terrain_and_areas(grid, hag, selected, paths=paths, start=start,
                                        target=target)
    return figures.save(fig, out_dir / f'terrain_areas_{case.terrain_id}_{od_id}')


def _profile_and_curvature_figures(case, od_id: str, paths_dir: Path, out_dir: Path, *,
                                   arms: Sequence[str], class_names: Sequence[str],
                                   iterations_wanted: Sequence[int]) -> list:
    written: list = []
    for arm in arms:
        rough_path = paths_dir / path_csv_name(case.terrain_id, od_id, arm)
        if not rough_path.exists():
            continue
        before_xyz = _load_xyz(rough_path)
        for class_name in class_names:
            for iterations in iterations_wanted:
                smooth_path = paths_dir / path_csv_name(
                    case.terrain_id, od_id, arm, road_class=class_name, iterations=iterations)
                if not smooth_path.exists():
                    continue
                after_xyz = _load_xyz(smooth_path)
                road_class = get(class_name)
                tag = f'{case.terrain_id}_{od_id}_{arm}_{class_name}_it{iterations}'
                written += figures.save(
                    figures.fig_longitudinal_profile(before_xyz, after_xyz, road_class),
                    out_dir / f'profile_{tag}')
                written += figures.save(
                    figures.fig_curvature(before_xyz, after_xyz, road_class),
                    out_dir / f'curvature_{tag}')
    return written


#: the default is deliberately narrow: EVERY (arm x class x iteration) combination is
#: 4 x 3 x 2 = 24 profile/curvature PAIRS per terrain, 96 files per terrain at two formats
#: each -- reproducible on demand (`--all`), but not something to commit to git by default.
#: `full_grid` (the baseline) and `hag_yellow` (the method's own arm, the candidate terrain areas)
#: are the pair every other figure and table in the paper is drawn against.
DEFAULT_ARMS_FOR_FIGURES: tuple[str, ...] = ('full_grid', 'hag_yellow')


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--from', dest='source', default='results', type=Path)
    parser.add_argument('--out', default=None, type=Path)
    parser.add_argument('--all', action='store_true',
                        help='every arm x class x iteration, not just the default subset '
                             '(DEFAULT_ARMS_FOR_FIGURES, iterations=1)')
    args = parser.parse_args(argv)
    out = args.out or (args.source / 'figures')
    paths_dir = args.source / 'paths'
    arms = ARM_ORDER if args.all else DEFAULT_ARMS_FOR_FIGURES
    iterations_wanted = ITERATIONS if args.all else (1,)

    if not paths_dir.exists():
        print(f'no {paths_dir}: nothing to draw', file=sys.stderr)
        return 1

    written: list = []
    for case in TERRAINS:
        grid = generate_terrain(case.spec)                    # deterministic, see docstring
        hag = build_hag(grid, height_delta=3.0)
        sample = sample_od_pairs(grid, hag, terrain_id=case.terrain_id,
                                 terrain_seed=case.spec.seed, n_pairs=1)
        od_id = sample.pairs[0].od_id
        selected = select_areas(hag, sample.pairs[0].start, sample.pairs[0].target,
                                rule='yellow')

        areas_written = _terrain_and_areas_figure(case, od_id, paths_dir, out, grid, hag,
                                                   selected, start=sample.pairs[0].start,
                                                   target=sample.pairs[0].target)
        if areas_written:
            written += areas_written
        written += _profile_and_curvature_figures(case, od_id, paths_dir, out, arms=arms,
                                                   class_names=PROPOSED_MATRIX_CLASSES,
                                                   iterations_wanted=iterations_wanted)

    if not any(str(p.name).startswith('profile_') for p in written):
        print(f'note: {paths_dir} holds no smoothed axes (run `python -m '
              f'core.experiment_matrix --write-paths all` first) -- profile and curvature '
              f'figures were skipped', file=sys.stderr)

    for item in written:
        print(item)
    return 0


if __name__ == '__main__':
    sys.exit(main())
