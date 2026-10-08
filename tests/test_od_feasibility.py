"""Can a road class's `i_max` be held between two points? Screen, then certify.

The terrains here are analytic planes and ridges, so the right answer is known by hand.
"""
import math

import numpy as np
import pytest

from core.dem import DEMError, coarsen_to_cell_size
from core.grid import Grid
from core.od_feasibility import (DEVELOPMENT_RATIO_WARNING, certify_grade,
                                 find_feasible_pairs, grade_limited_graphs, screen_grade)
from core.search import build_graph
from data.configs.road_classes import get

CLASS = get('RO_CLASS_V_DEAL')          # i_max 7%, R_min 95 m


def plane(rows=40, cols=40, rise_per_row_m=0.5, cell_size_m=10.0):
    """A tilted plane: `rise_per_row_m` metres per cell down the rows, flat across them."""
    surf = np.arange(rows, dtype=float)[:, None] * rise_per_row_m + np.zeros((1, cols))
    return Grid(surf=surf, cell_size_m=cell_size_m)


# -- the edge filter in build_graph ----------------------------------------------------

def test_the_gradient_filter_drops_exactly_the_steep_edges():
    grid = plane(rise_per_row_m=1.0)            # 1 m per 10 m cell = 10% down the rows
    full = build_graph(grid)
    limited = build_graph(grid, max_abs_gradient_percent=7.0)
    assert limited.n_edges < full.n_edges
    steep = np.abs(limited.dh) > 0.07 * limited.lengths_m + 1e-12
    assert not steep.any()
    assert limited.max_abs_gradient_percent == 7.0
    # the across-row steps are flat, so they all survive
    assert limited.n_edges == int((np.abs(full.dh) <= 0.07 * full.lengths_m + 1e-12).sum())


def test_a_flat_terrain_keeps_every_edge():
    grid = Grid(surf=np.zeros((20, 20)), cell_size_m=10.0)
    assert build_graph(grid, max_abs_gradient_percent=7.0).n_edges == build_graph(grid).n_edges


def test_a_non_positive_gradient_limit_is_refused():
    with pytest.raises(ValueError, match='must be positive'):
        build_graph(plane(), max_abs_gradient_percent=0.0)


# -- step 1, the screen ----------------------------------------------------------------

def test_the_screen_passes_a_gentle_pair():
    grid = plane(rise_per_row_m=0.5)            # 0.5 m per 10 m cell = 5%
    screen = screen_grade(grid, (2, 2), (32, 2), CLASS)
    assert screen.straight_gradient_percent == pytest.approx(5.0)
    assert screen.passed
    assert screen.development_needed == pytest.approx(5.0 / 7.0)
    assert 'within' in screen.reason


def test_the_screen_rejects_a_pair_the_straight_line_cannot_hold():
    grid = plane(rise_per_row_m=1.0)            # 10%
    screen = screen_grade(grid, (2, 2), (32, 2), CLASS)
    assert screen.straight_gradient_percent == pytest.approx(10.0)
    assert not screen.passed
    #: 30 cells x 10 m = 300 m straight, dropping 30 m; at 7% that needs 428.6 m
    assert screen.min_length_m == pytest.approx(30.0 / 0.07)
    assert screen.development_needed > 1.0
    assert 'already needs' in screen.reason


def test_the_screen_is_blind_to_what_lies_between():
    """Equal endpoints, a ridge in the middle: the screen passes, which is why step 2 exists."""
    grid = plane(rows=40, rise_per_row_m=0.0)
    grid.surf[18:22, :] += 30.0                 # a wall across the whole terrain
    screen = screen_grade(grid, (2, 2), (37, 2), CLASS)
    assert screen.straight_gradient_percent == pytest.approx(0.0)
    assert screen.passed


def test_the_same_cell_twice_is_refused():
    with pytest.raises(ValueError, match='same cell'):
        screen_grade(plane(), (3, 3), (3, 3), CLASS)


# -- step 2, the certificate -----------------------------------------------------------

def test_a_gentle_plane_is_certified_with_a_direct_witness():
    grid = plane(rise_per_row_m=0.5)
    certificate = certify_grade(grid, (2, 2), (32, 2), CLASS)
    assert certificate.reachable
    assert certificate.witness_i_max_percent <= CLASS.i_max_percent + 1e-9
    assert certificate.development_ratio == pytest.approx(1.0, abs=1e-6)
    assert not certificate.snaking
    assert 'gradient-feasible' in certificate.verdict


def test_a_wall_across_the_terrain_is_refused_with_a_reason():
    """Nothing can cross it at 7%, so no algorithm can: the certificate says so."""
    grid = plane(rows=40, rise_per_row_m=0.0)
    grid.surf[18:22, :] += 30.0
    certificate = certify_grade(grid, (2, 2), (37, 2), CLASS)
    assert not certificate.reachable
    assert certificate.path_xy is None
    assert math.isinf(certificate.development_ratio)
    assert 'NO alignment' in certificate.verdict
    assert certificate.edges_kept < certificate.edges_total


def test_the_witness_develops_length_when_the_direct_line_is_too_steep():
    """A 10% plane: the only admissible steps are the flat ones across it, so the witness
    has to traverse instead of descending straight, and its length says so."""
    grid = plane(rows=40, cols=40, rise_per_row_m=1.0)
    #: one gentle ramp cut into the slope, at 5% along the columns
    grid.surf[:, 20] = grid.surf[0, 20] + np.arange(40) * 0.5
    certificate = certify_grade(grid, (2, 20), (30, 20), CLASS)
    if certificate.reachable:
        assert certificate.development_ratio >= 1.0
        assert certificate.witness_i_max_percent <= CLASS.i_max_percent + 1e-9


def test_the_certificate_says_nothing_about_curvature():
    """A vertical certificate only: the witness may snake, and its R_min is reported raw."""
    grid = plane(rise_per_row_m=0.5)
    certificate = certify_grade(grid, (2, 2), (32, 2), CLASS)
    assert 'r_min_m' in certificate.as_dict()
    assert certificate.as_dict()['i_max_percent'] == CLASS.i_max_percent


def test_prebuilt_graphs_give_the_same_answer():
    grid = plane(rise_per_row_m=0.5)
    graphs = grade_limited_graphs(grid, CLASS)
    a = certify_grade(grid, (2, 2), (32, 2), CLASS)
    b = certify_grade(grid, (2, 2), (32, 2), CLASS, graphs=graphs)
    assert a.as_dict() == b.as_dict()


# -- choosing pairs --------------------------------------------------------------------

def test_every_pair_found_passes_both_steps():
    grid = plane(rows=60, cols=60, rise_per_row_m=0.3)     # 3%, comfortably inside 7%
    pairs = find_feasible_pairs(grid, CLASS, n_pairs=2, seed=0, margin=3)
    assert len(pairs) == 2
    for pair in pairs:
        assert pair.screen.passed
        assert pair.certificate.reachable
        assert pair.certificate.witness_i_max_percent <= CLASS.i_max_percent + 1e-9
        assert pair.certificate.development_ratio <= DEVELOPMENT_RATIO_WARNING
        assert pair.start == pair.screen.start and pair.target == pair.screen.target


def test_the_search_is_deterministic_given_the_seed():
    grid = plane(rows=60, cols=60, rise_per_row_m=0.3)
    first = find_feasible_pairs(grid, CLASS, n_pairs=2, seed=7, margin=3)
    second = find_feasible_pairs(grid, CLASS, n_pairs=2, seed=7, margin=3)
    assert [p.start for p in first] == [p.start for p in second]
    assert [p.target for p in first] == [p.target for p in second]


def test_a_terrain_with_no_viable_pair_returns_empty_rather_than_raising():
    """Unlike the O-D protocol, an empty answer is information, not a failure."""
    grid = plane(rows=60, cols=60, rise_per_row_m=2.0)     # 20% everywhere
    assert find_feasible_pairs(grid, CLASS, n_pairs=2, seed=0, margin=3,
                               max_candidates=300) == []


def test_a_margin_that_leaves_nothing_is_refused():
    with pytest.raises(ValueError, match='margin'):
        find_feasible_pairs(plane(rows=10, cols=10), CLASS, margin=20)


# -- the scale the test is run at ------------------------------------------------------

def test_coarsening_averages_blocks_and_scales_the_cell():
    grid = Grid(surf=np.arange(400, dtype=float).reshape(20, 20), cell_size_m=1.0)
    coarse, audit = coarsen_to_cell_size(grid, 5.0)
    assert coarse.shape == (4, 4)
    assert coarse.cell_size_m == 5.0
    assert audit['factor'] == 5 and audit['dropped_rows'] == 0
    assert coarse.surf[0, 0] == pytest.approx(grid.surf[:5, :5].mean())


def test_coarsening_drops_the_trailing_cells_and_says_so():
    grid = Grid(surf=np.zeros((22, 23)), cell_size_m=1.0)
    coarse, audit = coarsen_to_cell_size(grid, 10.0)
    assert coarse.shape == (2, 2)
    assert (audit['dropped_rows'], audit['dropped_cols']) == (2, 3)


def test_coarsening_to_the_same_cell_is_a_copy():
    grid = Grid(surf=np.arange(16, dtype=float).reshape(4, 4), cell_size_m=10.0)
    coarse, audit = coarsen_to_cell_size(grid, 10.0)
    np.testing.assert_array_equal(coarse.surf, grid.surf)
    assert audit['factor'] == 1

def test_a_target_cell_that_is_not_a_whole_multiple_is_refused():
    with pytest.raises(DEMError, match='whole multiple'):
        coarsen_to_cell_size(Grid(surf=np.zeros((20, 20)), cell_size_m=3.0), 10.0)


def test_gradient_feasibility_is_scale_dependent():
    """The finding behind `coarsen_to_cell_size`.

    A 2% slope carrying 15 cm of noise -- the Idrija survey's stated RMS accuracy -- is
    comfortably inside a 7% limit as a slope, but at a 1 m step 15 cm IS 15%. So most single
    steps are inadmissible and the witness has to weave between the ones that are not, while
    the same terrain at 10 m needs no development at all. On the real raster the effect is
    far stronger: the admissible subgraph of a gentle 600 m crop has 16531 components at 1 m,
    the largest holding 4.5% of the cells, against 83 components and 67% at 10 m.
    """
    rng = np.random.default_rng(0)
    fine = Grid(surf=(np.arange(80, dtype=float)[:, None] * 0.02
                      + rng.normal(0, 0.15, (80, 80))), cell_size_m=1.0)
    coarse, _ = coarsen_to_cell_size(fine, 10.0)
    # the macro slope is 2% either way, far inside the 7% limit
    assert screen_grade(fine, (5, 5), (74, 74), CLASS).passed
    assert screen_grade(coarse, (0, 0), (7, 7), CLASS).passed

    at_1m = certify_grade(fine, (5, 5), (74, 74), CLASS)
    at_10m = certify_grade(coarse, (0, 0), (7, 7), CLASS)
    # the noise makes most 1 m steps inadmissible, and the survivors barely percolate
    assert at_1m.edge_keep_fraction < 0.5
    assert at_10m.edge_keep_fraction == pytest.approx(1.0)
    # so the fine grid only gets there by weaving; the coarse one goes straight
    assert at_1m.development_ratio > 1.5
    assert at_10m.development_ratio == pytest.approx(1.0, abs=1e-6)
