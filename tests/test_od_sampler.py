"""The O-D protocol of the study protocol

These tests are the protocol's specification: if one of them has to change, a published
O-D pair changes with it.
"""
import numpy as np
import pytest

from core.grid import Grid
from core.hag import build_hag
from core.od_sampler import (MIN_DIAGONAL_FRACTION, MIN_HAG_HOPS, ODProtocolFailed,
                             od_pairs_rows, od_seed_for, reference_pair,
                             sample_od_pairs)
from core.terrain import TerrainSpec, generate_terrain

TERRAIN_SEEDS = (0, 7, 13)


def _sample(seed, **kwargs):
    grid = generate_terrain(TerrainSpec(seed=seed))
    hag = build_hag(grid, height_delta=3.0)
    return grid, hag, sample_od_pairs(grid, hag, terrain_id=f'T{seed}',
                                      terrain_seed=seed, **kwargs)


@pytest.mark.parametrize('seed', TERRAIN_SEEDS)
def test_od_seed_is_1000_plus_the_terrain_seed(seed):
    assert od_seed_for(seed) == 1000 + seed
    _, _, sample = _sample(seed)
    assert sample.od_seed == 1000 + seed


@pytest.mark.parametrize('seed', TERRAIN_SEEDS)
def test_the_protocol_is_reproducible(seed):
    _, _, a = _sample(seed)
    _, _, b = _sample(seed)
    assert [p.as_dict() for p in a.pairs] == [p.as_dict() for p in b.pairs]
    assert a.n_candidates_drawn == b.n_candidates_drawn


@pytest.mark.parametrize('seed', TERRAIN_SEEDS)
def test_every_accepted_pair_satisfies_every_criterion(seed):
    grid, _, sample = _sample(seed)
    margin = 5
    for pair in sample.pairs:
        for coord in (pair.start, pair.target):
            assert margin <= coord[0] < grid.n_rows - margin
            assert margin <= coord[1] < grid.n_cols - margin
        assert pair.diagonal_fraction >= MIN_DIAGONAL_FRACTION
        assert pair.start_area != pair.target_area
        assert pair.hag_hops >= MIN_HAG_HOPS
        assert pair.protocol is True


@pytest.mark.parametrize('seed', TERRAIN_SEEDS)
def test_the_three_pairs_are_not_three_variations_of_one_corridor(seed):
    _, _, sample = _sample(seed)
    starts = [p.start_area for p in sample.pairs]
    targets = [p.target_area for p in sample.pairs]
    assert len(set(starts)) == len(starts)
    assert len(set(targets)) == len(targets)


@pytest.mark.parametrize('seed', TERRAIN_SEEDS)
def test_every_candidate_is_charged_to_exactly_one_criterion(seed):
    _, _, sample = _sample(seed)
    assert sum(sample.rejections.values()) + len(sample.pairs) == sample.n_candidates_drawn


def test_an_unsatisfiable_terrain_raises_instead_of_looping():
    """A constant plane is one height area, so the hop distance is never >= 3."""
    grid = Grid(surf=np.full((60, 60), 100.0), cell_size_m=10.0)
    hag = build_hag(grid, height_delta=3.0)
    with pytest.raises(ODProtocolFailed):
        sample_od_pairs(grid, hag, terrain_id='flat', terrain_seed=0,
                        max_candidates=200)


def test_the_reference_pair_is_marked_as_not_from_the_protocol():
    grid, hag, _ = _sample(0)
    pair = reference_pair(grid, hag)
    assert pair.protocol is False
    assert pair.od_id == 'reference'
    assert pair.start == (5, 5) and pair.target == (94, 94)


def test_the_audit_rows_carry_the_seed_and_the_rejection_counts():
    _, _, sample = _sample(0)
    rows = od_pairs_rows([sample])
    assert len(rows) == len(sample.pairs)
    for row in rows:
        assert row['od_seed'] == sample.od_seed
        assert row['n_candidates_drawn'] == sample.n_candidates_drawn
        for name, count in sample.rejections.items():
            assert row[f'rejected_{name}'] == count


def test_the_draw_order_is_fixed():
    """One candidate consumes exactly four draws, in the order i1, j1, i2, j2.

    Changing that order silently changes every published pair, so it is pinned by
    replaying the RNG and checking that the draw at an accepted pair's
    `candidate_index` really is that pair.
    """
    grid, _, sample = _sample(0)
    rng = np.random.default_rng(sample.od_seed)
    lo, hi = 5, min(grid.n_rows, grid.n_cols) - 5

    replayed = []
    for _ in range(sample.n_candidates_drawn):
        i1 = int(rng.integers(lo, hi)); j1 = int(rng.integers(lo, hi))
        i2 = int(rng.integers(lo, hi)); j2 = int(rng.integers(lo, hi))
        replayed.append(((i1, j1), (i2, j2)))

    for pair in sample.pairs:
        start, target = replayed[pair.candidate_index - 1]
        assert pair.start == start, f'{pair.od_id} start moved: draw order changed'
        assert pair.target == target, f'{pair.od_id} target moved: draw order changed'
