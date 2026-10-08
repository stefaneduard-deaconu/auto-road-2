"""T1 acceptance: the new core search matches scipy on random grids, and the full-grid
graph actually exists."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

from core import costs
from core.grid import Grid, edge_arrays, step_lengths_m
from core.search import (build_graph, dijkstra, path_cost, path_height_cost, path_length_m)
from core.terrain import TerrainSpec, generate_terrain

COSTS = ['height', 'height_tiebreak', 'length', '3d']


def random_grid(seed: int, size: int = 12) -> Grid:
    rng = np.random.default_rng(seed)
    return Grid(surf=rng.uniform(100, 120, size=(size, size)), cell_size_m=10.0)


def scipy_distances(graph, start):
    matrix = csr_matrix((graph.weights, graph.indices, graph.indptr),
                        shape=(graph.n_nodes, graph.n_nodes))
    return scipy_dijkstra(matrix, directed=True, indices=start)


# -- the oracle test -----------------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_matches_scipy_on_random_grids(seed):
    grid = random_grid(seed)
    graph = build_graph(grid, cost='height_tiebreak')
    start, target = 0, grid.n_nodes - 1
    got = dijkstra(graph, start, target, stop_at_target=False)
    expected = scipy_distances(graph, start)
    np.testing.assert_allclose(got.distances, expected, rtol=0, atol=1e-9)
    assert got.cost == pytest.approx(expected[target])
    assert path_cost(graph, got.path) == pytest.approx(got.cost)


@pytest.mark.parametrize("cost", COSTS)
def test_every_cost_model_matches_scipy(cost):
    grid = random_grid(101, size=15)
    graph = build_graph(grid, cost=cost)
    start, target = grid.index((1, 1)), grid.index((13, 12))
    got = dijkstra(graph, start, target, stop_at_target=False)
    assert got.cost == pytest.approx(scipy_distances(graph, start)[target])


def test_stopping_at_the_target_gives_the_same_cost_but_less_work():
    grid = random_grid(7, size=25)
    graph = build_graph(grid)
    start, target = grid.index((2, 2)), grid.index((12, 12))
    full = dijkstra(graph, start, target, stop_at_target=False)
    early = dijkstra(graph, start, target, stop_at_target=True)
    assert early.cost == pytest.approx(full.cost)
    assert early.stats.nodes_expanded <= full.stats.nodes_expanded
    assert list(early.path) == list(full.path)


# -- the full-grid graph exists ------------------------------------------------

def test_full_grid_graph_has_the_expected_number_of_edges():
    grid = Grid(surf=np.zeros((5, 5)))
    graph = build_graph(grid)
    # 8-connectivity on a 5x5 grid: 2*(4*5) horizontal+vertical + 2*(2*4*4) diagonals
    assert graph.n_nodes == 25
    assert graph.n_edges == 2 * (4 * 5) * 2 + 2 * (2 * 4 * 4)
    assert graph.n_edges > 0


def test_four_connectivity_has_fewer_edges():
    grid = Grid(surf=np.zeros((5, 5)))
    assert build_graph(grid, connectivity=4).n_edges == 2 * (4 * 5) * 2


def test_graph_is_rebuilt_per_terrain_and_never_cached(tmp_path, monkeypatch):
    """No graph cache: a graph built for one terrain is never reused for another."""
    monkeypatch.chdir(tmp_path)
    g1 = build_graph(random_grid(1, size=5))
    g2 = build_graph(random_grid(2, size=4))
    assert g2.n_nodes == 16
    assert not any(p.suffix == '.pkl' for p in tmp_path.iterdir())
    assert not np.array_equal(g1.grid.surf.shape, g2.grid.surf.shape)


# -- one length function --------------------------------------------------------

def test_step_lengths_use_one_convention():
    assert step_lengths_m(1, 0, 10.0) == pytest.approx(10.0)
    assert step_lengths_m(1, 1, 10.0) == pytest.approx(10.0 * np.sqrt(2))
    assert step_lengths_m(0, -1, 30.0) == pytest.approx(30.0)


def test_masked_and_unmasked_graphs_agree_on_edge_lengths():
    grid = random_grid(3, size=8)
    mask = np.ones(grid.shape, dtype=bool)
    full = build_graph(grid, cost='length')
    masked = build_graph(grid, cost='length', mask=mask)
    np.testing.assert_allclose(np.sort(full.lengths_m), np.sort(masked.lengths_m))


def test_diagonal_edges_are_longer_than_straight_ones():
    grid = Grid(surf=np.zeros((3, 3)), cell_size_m=10.0)
    _, _, lengths, _ = edge_arrays(grid)
    assert set(np.round(np.unique(lengths), 6)) == {10.0, round(10.0 * np.sqrt(2), 6)}


# -- reverse edges ---------------------------------------------------------------

def test_reverse_edge_negates_the_height_difference_not_the_length():
    grid = Grid(surf=np.array([[0.0, 2.0], [0.0, 0.0]]), cell_size_m=10.0)
    u, v, lengths, dh = edge_arrays(grid)
    forward = [k for k in range(len(u)) if (u[k], v[k]) == (0, 1)][0]
    backward = [k for k in range(len(u)) if (u[k], v[k]) == (1, 0)][0]
    assert dh[forward] == pytest.approx(2.0)
    assert dh[backward] == pytest.approx(-2.0)
    assert lengths[forward] == lengths[backward] > 0


# -- masks ----------------------------------------------------------------------------

def test_mask_removes_nodes_from_the_search():
    grid = random_grid(11, size=10)
    mask = np.zeros(grid.shape, dtype=bool)
    mask[:, :3] = True
    graph = build_graph(grid, mask=mask)
    assert graph.n_searchable_nodes == 30
    blocked = dijkstra(graph, grid.index((0, 0)), grid.index((9, 9)))
    assert not blocked.reached
    reachable = dijkstra(graph, grid.index((0, 0)), grid.index((9, 2)))
    assert reachable.reached
    assert all(c[1] < 3 for c in reachable.coords(grid))


# -- path measures --------------------------------------------------------------------

def test_path_measures_on_a_flat_terrain():
    grid = Grid(surf=np.zeros((4, 4)), cell_size_m=10.0)
    graph = build_graph(grid, cost='length')
    result = dijkstra(graph, grid.index((0, 0)), grid.index((0, 3)))
    assert path_length_m(graph, result.path) == pytest.approx(30.0)
    assert path_height_cost(graph, result.path) == pytest.approx(0.0)


def test_height_objective_ignores_horizontal_distance():
    surf = np.zeros((6, 6))
    surf[3, :] = 5.0
    grid = Grid(surf=surf)
    graph = build_graph(grid, cost='height')
    result = dijkstra(graph, grid.index((0, 0)), grid.index((5, 5)), stop_at_target=False)
    assert result.cost == pytest.approx(10.0)  # up 5 and down 5, whatever the detour


def test_tiebreak_prefers_the_shorter_of_two_equal_height_paths():
    grid = Grid(surf=np.zeros((5, 9)), cell_size_m=10.0)
    plain = build_graph(grid, cost='height')
    tied = build_graph(grid, cost='height_tiebreak')
    start, target = grid.index((2, 0)), grid.index((2, 8))
    assert dijkstra(plain, start, target, stop_at_target=False).cost == pytest.approx(0.0)
    result = dijkstra(tied, start, target, stop_at_target=False)
    assert len(result.path) == 9  # the straight row, not a detour
    assert result.cost == pytest.approx(costs.TIE_BREAK_PER_METER * 80.0)


# -- terrain --------------------------------------------------------------------------

def test_terrain_is_reproducible_and_does_not_leak_the_global_seed():
    np.random.seed(1234)
    before = np.random.rand()
    np.random.seed(1234)
    a = generate_terrain(TerrainSpec(seed=0, grid_size=(20, 20), periods=(2, 2)))
    after = np.random.rand()
    b = generate_terrain(TerrainSpec(seed=0, grid_size=(20, 20), periods=(2, 2)))
    np.testing.assert_array_equal(a.surf, b.surf)
    assert before == after  # the global RNG state was restored


def test_terrain_height_range_is_the_documented_one():
    # The generator computes noise*(high-low)+low with noise in [-1, 1], so the real
    # span is [low-(high-low), high]. Documented, not silently corrected: changing it
    # would change every terrain the two articles report.
    spec = TerrainSpec(seed=3, grid_size=(40, 40), periods=(2, 2), height_interval=(100, 120))
    grid = generate_terrain(spec)
    low, high = spec.actual_height_range
    assert (low, high) == (80.0, 120.0)
    assert low <= grid.surf.min() and grid.surf.max() <= high
    assert grid.cell_size_m == 10.0




# -- bilinear height -------------------------------------------------------------

def test_interpolate_height_is_bilinear():
    surf = np.tile(np.arange(6, dtype=float)[:, None], (1, 6))  # height == row index
    grid = Grid(surf=surf)
    assert grid.interpolate_height(np.array([1.1, 1.0])) == pytest.approx(1.1)
    assert grid.interpolate_height(np.array([[2.5, 3.0], [4.25, 1.0]])).tolist() == \
        pytest.approx([2.5, 4.25])


def test_interpolate_height_clamps_outside_the_grid():
    grid = Grid(surf=np.arange(25, dtype=float).reshape(5, 5))
    assert grid.interpolate_height(np.array([-3.0, 7.0])) == grid.surf[0, 4]


def test_path_xyz_m_uses_meters():
    grid = Grid(surf=np.zeros((5, 5)), cell_size_m=20.0)
    xyz = grid.path_xyz_m(np.array([[0.0, 0.0], [1.0, 2.0]]))
    np.testing.assert_allclose(xyz[:, :2], [[0.0, 0.0], [20.0, 40.0]])
