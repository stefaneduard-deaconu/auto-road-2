"""The fixed-cluster baseline of the HAG (before-release task PR4).

Hierarchical path-finding (HPA*-like) abstracts a grid into FIXED square clusters; the HAG
abstracts it into HEIGHT AREAS. To compare the two abstractions and nothing else, the block
graph built here has the same structure as a HAG: its "areas" are square blocks of cells,
sized so that a block has the mean area size of the HAG of the same terrain (same
granularity), and everything downstream is shared and unchanged: the shared-border edge cost,
the minimum-weight chain, the one-ring dilation, the masked Dijkstra and the measures of
`core.experiment_hag.search_rows`. The only difference between the two arms is therefore how
the terrain is partitioned.

    python -m core.experiment_baseline --out results/baseline [--quick]

Writes `baseline.csv`: one row per (scenario, abstraction, search space), with
`abstraction` = `hag` or `blocks` and the block side in cells.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from core.experiment_hag import search_rows
from core.hag import HAG, _area_sums, build_hag
from core.provenance import provenance, write_jsonl
from core.results_io import read_jsonl, write_csv

#: the synthetic classes of the study (Perlin period, relief in m), full range and hilly
CLASSES = tuple((p, a) for p in (2, 3, 4, 6, 8) for a in (20.0, 60.0, 150.0)) + \
    ((4, 200.0), (6, 120.0), (8, 100.0))
SEEDS = range(5)
SIZES = (120, 240)
HEIGHT_DELTA_M = 3.0
N_PAIRS = 3
SPACES = (('full_grid', ''), ('hag_cta', 'shared_border'), ('hag_cta_ring1', 'shared_border'))


def build_block_hag(hag: HAG) -> tuple[HAG, int]:
    """A HAG-shaped graph over square blocks with the mean area size of `hag`, and the
    block side in cells."""
    grid = hag.grid
    side = max(2, int(round(math.sqrt(hag.n_valid_cells / max(hag.n_areas, 1)))))
    rows, cols = np.indices(grid.shape)
    n_block_cols = math.ceil(grid.shape[1] / side)
    labels = (rows // side) * n_block_cols + (cols // side)
    labels = np.where(hag.labels >= 0, labels, -1)
    used, labels_c = np.unique(labels[labels >= 0], return_inverse=True)
    compact = np.full(grid.shape, -1, dtype=np.int64)
    compact[labels >= 0] = labels_c
    n = int(used.size)
    size, row_sum, col_sum = _area_sums(compact, n)
    surf = np.asarray(grid.surf, dtype=float)
    height = np.bincount(compact[compact >= 0], weights=surf[compact >= 0], minlength=n) / size
    neighbours: list[set[int]] = [set() for _ in range(n)]
    from core.grid import offsets
    for di, dj in offsets(8):
        a = compact[max(di, 0):compact.shape[0] + min(di, 0), max(dj, 0):compact.shape[1] + min(dj, 0)]
        b = compact[max(-di, 0):compact.shape[0] + min(-di, 0), max(-dj, 0):compact.shape[1] + min(-dj, 0)]
        differ = (a != b) & (a >= 0) & (b >= 0)
        for u, v in np.unique(np.stack([a[differ], b[differ]], axis=1), axis=0).tolist():
            neighbours[u].add(v)
            neighbours[v].add(u)
    block = HAG(grid=grid, height_delta=float('nan'), connectivity=8, labels=compact,
                area_height=height, area_size=size,
                centroid=np.column_stack([row_sum, col_sum]) / size[:, None],
                neighbours=neighbours)
    return block, side


def run(out_dir: Path, *, quick: bool = False) -> dict:
    from core import programme
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / 'baseline.jsonl'
    done = {r.get('row_id') for r in read_jsonl(jsonl)}
    classes = CLASSES[:2] if quick else CLASSES
    seeds = range(1) if quick else SEEDS
    sizes = SIZES[:1] if quick else SIZES
    n_new = 0
    for period, relief in classes:
        for seed in seeds:
            for size in sizes:
                case = programme.synthetic_case(seed, size, period, relief)
                hag = build_hag(case.grid, HEIGHT_DELTA_M, native=True)
                blocks, side = build_block_hag(hag)
                pairs = programme._protocol_pairs(case, HEIGHT_DELTA_M, 1000 + seed, N_PAIRS)
                for pair in pairs:
                    for name, graph in (('hag', hag), ('blocks', blocks)):
                        rows = search_rows(case, graph, pair, spaces=SPACES,
                                           grid_edge='climb_tiebreak', engine='native',
                                           repeats=1, memory=False)
                        for row in rows:
                            row_id = (f'p{period}_a{relief:g}_s{seed}_n{size}__{pair.od_id}__'
                                      f'{name}__{row["search_space"]}')
                            if row_id in done:
                                continue
                            row.update({'row_id': row_id, 'abstraction': name,
                                        'periods': period, 'relief_amplitude_m': relief,
                                        'terrain_seed': seed, 'grid_cols': size,
                                        'height_delta_m': HEIGHT_DELTA_M,
                                        'n_areas': graph.n_areas,
                                        'block_side_cells': side if name == 'blocks' else None,
                                        **provenance()})
                            write_jsonl(jsonl, row)
                            n_new += 1
    logged = read_jsonl(jsonl)
    columns = sorted({k for r in logged for k in r})
    csv_path = write_csv(out_dir / 'baseline.csv', logged, columns)
    return {'csv': csv_path, 'n_new_rows': n_new, 'n_total_rows': len(logged)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=Path('results/baseline'))
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args(argv)
    out = args.out / 'quick' if args.quick and args.out == Path('results/baseline') else args.out
    for k, v in run(out, quick=args.quick).items():
        print(f'{k:>12}: {v}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
