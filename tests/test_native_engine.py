"""X0: the compiled engine is the Python reference, only faster.

Every experiment that runs at scale uses `core.native`; this is the gate that lets its
numbers stand next to the Python ones. Distances, paths and work counters must match
`core.search.dijkstra` on random terrains for every grid edge cost, with and without a
cell mask and a gradient cut; the labeller must reproduce `build_hag` exactly.
"""
import numpy as np
import pytest

from core import native
from core.grid import Grid
from core.hag import build_hag
from core.naming import GRID_EDGE_COST_DESCRIPTIONS, grid_edge_cost
from core.search import build_graph, dijkstra
from core.terrain import TerrainSpec, generate_terrain

pytestmark = pytest.mark.skipif(not native.available(), reason='no C++ compiler for core.native')

GRID_COSTS = sorted(GRID_EDGE_COST_DESCRIPTIONS)


def random_grid(seed: int, shape=(23, 31)) -> Grid:
    rng = np.random.default_rng(seed)
    surf = np.cumsum(rng.normal(0.0, 1.5, shape), axis=0) + rng.normal(0.0, 0.4, shape)
    return Grid(surf=surf, cell_size_m=float(rng.choice([1.0, 3.0, 10.0])))


def both(grid, start, target, cost_name, mask=None, stop_at_target=True):
    cost, cap = grid_edge_cost(cost_name, i_max_percent=12.0)
    graph = build_graph(grid, cost=cost, mask=mask, max_abs_gradient_percent=cap)
    ref = dijkstra(graph, start, target, stop_at_target=stop_at_target)
    got = native.dijkstra(grid, start, target, cost=cost, mask=mask,
                          max_abs_gradient_percent=cap, stop_at_target=stop_at_target,
                          n_edges=native.count_edges(grid, mask, 8, cap))
    return graph, ref, got


@pytest.mark.parametrize('cost_name', GRID_COSTS)
@pytest.mark.parametrize('seed', range(8))
def test_native_equals_python_on_random_terrain(seed, cost_name):
    grid = random_grid(seed)
    start, target = 0, grid.n_nodes - 1
    graph, ref, got = both(grid, start, target, cost_name)
    assert got.stats.n_edges == graph.n_edges
    assert got.reached == ref.reached
    if ref.reached:
        assert got.cost == ref.cost
        np.testing.assert_array_equal(got.path, ref.path)
    assert got.stats.nodes_expanded == ref.stats.nodes_expanded
    assert got.stats.nodes_pushed == ref.stats.nodes_pushed
    assert got.stats.nodes_reached == ref.stats.nodes_reached
    np.testing.assert_array_equal(got.distances, ref.distances)


@pytest.mark.parametrize('cost_name', ['climb_tiebreak', 'climb_gradient_cut'])
@pytest.mark.parametrize('seed', range(4))
def test_native_respects_the_cell_mask(seed, cost_name):
    grid = random_grid(100 + seed)
    rng = np.random.default_rng(seed)
    mask = rng.random(grid.shape) > 0.25
    start, target = 0, grid.n_nodes - 1
    mask.reshape(-1)[[start, target]] = True
    graph, ref, got = both(grid, start, target, cost_name, mask=mask)
    assert got.stats.n_nodes == ref.stats.n_nodes
    assert got.stats.n_edges == graph.n_edges
    assert got.reached == ref.reached
    np.testing.assert_array_equal(got.distances, ref.distances)
    if ref.reached:
        np.testing.assert_array_equal(got.path, ref.path)


def test_native_full_sweep_matches():
    grid = random_grid(7)
    _, ref, got = both(grid, 5, 17, 'climb', stop_at_target=False)
    np.testing.assert_array_equal(got.distances, ref.distances)
    assert got.stats.nodes_expanded == ref.stats.nodes_expanded


def test_python_only_cost_is_refused():
    grid = random_grid(1)
    with pytest.raises(native.NotNativeCost):
        native.dijkstra(grid, 0, 5, cost=lambda dh, length_m: np.abs(dh) + 1.0)


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('height_delta', [1, 3])
def test_native_labeller_reproduces_build_hag(seed, height_delta):
    grid = generate_terrain(TerrainSpec(seed=seed, grid_size=(60, 60), periods=(3, 3)))
    ref = build_hag(grid, height_delta)
    got = build_hag(grid, height_delta, native=True)
    np.testing.assert_array_equal(got.labels, ref.labels)
    np.testing.assert_array_equal(got.area_height, ref.area_height)
    assert got.neighbours == ref.neighbours


def test_native_labeller_with_nodata():
    grid = generate_terrain(TerrainSpec(seed=3, grid_size=(40, 40), periods=(2, 2)))
    valid = np.ones(grid.shape, dtype=bool)
    valid[10:20, 5:30] = False
    ref = build_hag(grid, 2, valid_mask=valid)
    got = build_hag(grid, 2, valid_mask=valid, native=True)
    np.testing.assert_array_equal(got.labels, ref.labels)
    assert got.neighbours == ref.neighbours
