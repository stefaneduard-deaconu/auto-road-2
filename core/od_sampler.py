"""The origin-destination selection protocol of the study protocol

O-D pairs are *generated, not hand-picked*, so no reader can suspect the examples were
chosen because they look good. The protocol is deterministic given the terrain seed:
`od_seed = 1000 + terrain_seed`, and one candidate consumes exactly four draws from
`numpy.random.default_rng(od_seed)`, in the order i1, j1, i2, j2. **Changing that order
changes every published pair**, so it is fixed here and asserted by a test.

One thing the protocol deliberately does NOT do: it never requires that a path exists
inside the yellow selection. Filtering on that would hide exactly the failure mode the
summary is asked to count ("number of scenarios where the yellow selection produced no
path at all", the study protocol).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.costs import DEFAULT_COST, CostSpec
from core.grid import Grid
from core.hag import HAG, vdist
from core.search import build_graph, dijkstra

#: no endpoint sits within this many cells of a border, so the corridor is not clipped
MARGIN_CELLS = 5
#: the straight line between the endpoints must be at least this fraction of the diagonal
MIN_DIAGONAL_FRACTION = 0.60
#: the HAG hop distance between the endpoint areas, so the yellow chain has an interior
MIN_HAG_HOPS = 3
N_PAIRS = 3
MAX_CANDIDATES = 10_000
OD_SEED_OFFSET = 1000

#: evaluated in this order; each rejected candidate is charged to exactly one of them
REJECTION_CRITERIA = (
    'too_short',
    'same_height_area',
    'hops_below_min',
    'no_full_grid_path',
    'duplicate_start_area',
    'duplicate_target_area',
)


class ODProtocolFailed(RuntimeError):
    """The protocol could not find `n_pairs` acceptable pairs within `max_candidates`."""


@dataclass(frozen=True)
class ODPair:
    od_id: str
    start: tuple[int, int]
    target: tuple[int, int]
    start_area: int
    target_area: int
    hag_hops: int
    straight_line_m: float
    diagonal_m: float
    diagonal_fraction: float
    start_height_m: float
    target_height_m: float
    height_difference_m: float
    #: False for the fixed reference pair (5, 5)-(94, 94), which is
    #: reported as a reference row and excluded from the summary statistics
    protocol: bool
    candidate_index: int

    def as_dict(self) -> dict:
        return {
            'od_id': self.od_id,
            'start_i': self.start[0], 'start_j': self.start[1],
            'target_i': self.target[0], 'target_j': self.target[1],
            'start_area': self.start_area, 'target_area': self.target_area,
            'hag_hops': self.hag_hops,
            'straight_line_m': self.straight_line_m,
            'diagonal_m': self.diagonal_m,
            'diagonal_fraction': self.diagonal_fraction,
            'start_height_m': self.start_height_m,
            'target_height_m': self.target_height_m,
            'height_difference_m': self.height_difference_m,
            'protocol': self.protocol,
            'candidate_index': self.candidate_index,
        }


@dataclass(frozen=True)
class ODSample:
    terrain_id: str
    terrain_seed: int
    od_seed: int
    pairs: tuple[ODPair, ...]
    n_candidates_drawn: int
    rejections: dict


def od_seed_for(terrain_seed: int) -> int:
    return OD_SEED_OFFSET + int(terrain_seed)


def _pair_from(grid: Grid, hag: HAG, start, target, od_id: str, hops: int,
               protocol: bool, candidate_index: int) -> ODPair:
    diagonal = math.hypot(grid.n_rows, grid.n_cols) * grid.cell_size_m
    straight = math.hypot(start[0] - target[0], start[1] - target[1]) * grid.cell_size_m
    z0 = float(grid.surf[start[0], start[1]])
    z1 = float(grid.surf[target[0], target[1]])
    return ODPair(
        od_id=od_id,
        start=(int(start[0]), int(start[1])),
        target=(int(target[0]), int(target[1])),
        start_area=int(hag.area_of(start)), target_area=int(hag.area_of(target)),
        hag_hops=int(hops), straight_line_m=straight, diagonal_m=diagonal,
        diagonal_fraction=straight / diagonal if diagonal else 0.0,
        start_height_m=z0, target_height_m=z1, height_difference_m=abs(z1 - z0),
        protocol=protocol, candidate_index=candidate_index)


def sample_od_pairs(grid: Grid, hag: HAG, *, terrain_id: str, terrain_seed: int,
                    margin_cells: int = MARGIN_CELLS,
                    min_diagonal_fraction: float = MIN_DIAGONAL_FRACTION,
                    min_hops: int = MIN_HAG_HOPS,
                    n_pairs: int = N_PAIRS,
                    max_candidates: int = MAX_CANDIDATES,
                    cost: CostSpec = DEFAULT_COST,
                    connectivity: int = 8) -> ODSample:
    """Draw `n_pairs` O-D pairs by the protocol. Deterministic for a given terrain seed.

    Criterion 5 (pairwise distinct start areas and target areas) is evaluated greedily
    against the pairs already accepted. the study protocol states it as an extra
    requirement on the accepted set, which is not implementable as written -- "the first
    three that pass" and a set constraint can conflict, with no backtracking rule given
    -- so it is promoted to a per-candidate criterion with its own two counters.
    """
    seed = od_seed_for(terrain_seed)
    rng = np.random.default_rng(seed)
    lo, hi = margin_cells, min(grid.n_rows, grid.n_cols) - margin_cells
    if hi <= lo:
        raise ODProtocolFailed(f"a {grid.shape} terrain has no cells left after a "
                               f"{margin_cells}-cell margin")

    diagonal_cells = math.hypot(grid.n_rows, grid.n_cols)
    full_graph = build_graph(grid, cost=cost, connectivity=connectivity)

    rejections = {name: 0 for name in REJECTION_CRITERIA}
    accepted: list[ODPair] = []
    drawn = 0
    while len(accepted) < n_pairs and drawn < max_candidates:
        drawn += 1
        i1 = int(rng.integers(lo, hi))
        j1 = int(rng.integers(lo, hi))
        i2 = int(rng.integers(lo, hi))
        j2 = int(rng.integers(lo, hi))
        start, target = (i1, j1), (i2, j2)

        if math.hypot(i1 - i2, j1 - j2) < min_diagonal_fraction * diagonal_cells:
            rejections['too_short'] += 1
            continue
        area_a, area_b = int(hag.area_of(start)), int(hag.area_of(target))
        if area_a == area_b:
            rejections['same_height_area'] += 1
            continue
        hops = vdist(hag, area_a)[area_b]
        if not np.isfinite(hops) or hops < min_hops:
            rejections['hops_below_min'] += 1
            continue
        if not dijkstra(full_graph, grid.index(start), grid.index(target)).reached:
            rejections['no_full_grid_path'] += 1
            continue
        if any(p.start_area == area_a for p in accepted):
            rejections['duplicate_start_area'] += 1
            continue
        if any(p.target_area == area_b for p in accepted):
            rejections['duplicate_target_area'] += 1
            continue

        accepted.append(_pair_from(grid, hag, start, target, f'od{len(accepted) + 1}',
                                   hops, protocol=True, candidate_index=drawn))

    if len(accepted) < n_pairs:
        raise ODProtocolFailed(
            f"{terrain_id}: only {len(accepted)} of {n_pairs} pairs after {drawn} "
            f"candidates. Rejections: {rejections}")

    return ODSample(terrain_id=terrain_id, terrain_seed=int(terrain_seed), od_seed=seed,
                    pairs=tuple(accepted), n_candidates_drawn=drawn,
                    rejections=rejections)


def reference_pair(grid: Grid, hag: HAG, start=(5, 5), target=(94, 94)) -> ODPair:
    """The fixed reference pair (5, 5)-(94, 94). Reported, never in the statistics."""
    area_a, area_b = int(hag.area_of(start)), int(hag.area_of(target))
    hops = vdist(hag, area_a)[area_b]
    return _pair_from(grid, hag, start, target, 'reference',
                      int(hops) if np.isfinite(hops) else -1,
                      protocol=False, candidate_index=0)


def od_pairs_rows(samples: Sequence[ODSample]) -> list[dict]:
    """Flat rows for `results/od_pairs.csv`: the audit of the protocol."""
    rows = []
    for sample in samples:
        for pair in sample.pairs:
            row = {'terrain_id': sample.terrain_id, 'terrain_seed': sample.terrain_seed,
                   'od_seed': sample.od_seed,
                   'n_candidates_drawn': sample.n_candidates_drawn}
            row.update(pair.as_dict())
            row.update({f'rejected_{k}': v for k, v in sample.rejections.items()})
            rows.append(row)
    return rows
