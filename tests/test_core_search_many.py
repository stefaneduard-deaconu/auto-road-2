"""T11 acceptance: one sweep answers many targets, and a cost swap rebuilds nothing.

`dijkstra_many` and `with_cost` exist only to make the portal overlay of `core.network`
affordable, so both are checked against the thing they replace rather than against
themselves: `dijkstra_many` against N separate `dijkstra` calls, and `with_cost` against
`build_graph` from scratch. `tests/test_core_search.py` keeps the scipy oracle underneath
both, so the chain of trust ends at scipy and not at this file.
"""
from __future__ import annotations

import numpy as np
import pytest

from core import costs
from core.grid import Grid
from core.search import build_graph, dijkstra, dijkstra_many, path_cost, with_cost
from core.terrain import TerrainSpec, generate_terrain

COSTS = ['height', 'height_tiebreak', 'length', '3d']


def random_grid(seed: int, size: int = 12) -> Grid:
    rng = np.random.default_rng(seed)
    return Grid(surf=rng.uniform(100, 120, size=(size, size)), cell_size_m=10.0)


def spread_targets(grid: Grid, rng) -> np.ndarray:
    return np.unique(rng.integers(0, grid.n_nodes, size=8))


# -- dijkstra_many against N separate searches -------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_one_sweep_gives_the_same_costs_and_paths_as_separate_searches(seed):
    grid = random_grid(seed)
    graph = build_graph(grid)
    rng = np.random.default_rng(1000 + seed)
    start = int(rng.integers(0, grid.n_nodes))
    targets = spread_targets(grid, rng)

    many = dijkstra_many(graph, start, targets)
    for k, target in enumerate(targets):
        one = dijkstra(graph, start, int(target))
        assert many.costs[k] == pytest.approx(one.cost, rel=0, abs=1e-9)
        #: a tie can make two different paths equally good, so compare the COST of the
        #: path that came back, not the sequence of nodes
        assert path_cost(graph, many.paths[k]) == pytest.approx(one.cost, rel=0, abs=1e-9)
        assert many.paths[k][0] == start
        assert many.paths[k][-1] == target


@pytest.mark.parametrize("cost", COSTS)
def test_every_cost_model_agrees_with_separate_searches(cost):
    grid = random_grid(7)
    graph = build_graph(grid, cost=cost)
    rng = np.random.default_rng(99)
    start = int(rng.integers(0, grid.n_nodes))
    targets = spread_targets(grid, rng)
    many = dijkstra_many(graph, start, targets)
    for k, target in enumerate(targets):
        assert many.costs[k] == pytest.approx(dijkstra(graph, start, int(target)).cost,
                                              rel=0, abs=1e-9)


def test_one_sweep_does_no_more_work_than_the_separate_searches_together():
    """The point of the sweep: |P| targets cost one expansion front, not |P| of them."""
    grid = random_grid(4)
    graph = build_graph(grid)
    rng = np.random.default_rng(4)
    start = int(rng.integers(0, grid.n_nodes))
    targets = spread_targets(grid, rng)
    many = dijkstra_many(graph, start, targets)
    separate = sum(dijkstra(graph, start, int(t)).stats.nodes_expanded for t in targets)
    assert many.stats.nodes_expanded <= separate


def test_the_sweep_stops_once_the_last_target_is_settled():
    grid = random_grid(5, size=20)
    graph = build_graph(grid)
    near = dijkstra_many(graph, 0, [1])
    everything = dijkstra_many(graph, 0, np.arange(grid.n_nodes))
    assert near.stats.nodes_expanded < everything.stats.nodes_expanded


def test_an_unreachable_target_gets_an_infinite_cost_and_no_path():
    """A mask that isolates the start: the sweep exhausts and reports, it does not hang."""
    grid = random_grid(6, size=6)
    mask = np.zeros(grid.shape, dtype=bool)
    mask[0, 0] = True
    mask[5, 5] = True
    graph = build_graph(grid, mask=mask)
    many = dijkstra_many(graph, grid.index((0, 0)), [grid.index((5, 5))])
    assert not np.isfinite(many.costs[0])
    assert many.paths[0] is None


def test_the_start_is_its_own_target_at_zero_cost():
    graph = build_graph(random_grid(8))
    many = dijkstra_many(graph, 17, [17])
    assert many.costs[0] == 0.0
    assert many.paths[0].tolist() == [17]


def test_no_targets_expands_nothing():
    graph = build_graph(random_grid(9))
    many = dijkstra_many(graph, 3, [])
    assert many.stats.nodes_expanded == 0
    assert many.paths == []


def test_a_target_outside_the_graph_is_refused():
    graph = build_graph(random_grid(10))
    with pytest.raises(IndexError):
        dijkstra_many(graph, 0, [graph.n_nodes])


def test_dijkstra_also_hands_back_its_predecessors_now():
    """`previous` is what makes one sweep reusable; `dijkstra` kept it to itself before."""
    graph = build_graph(random_grid(11))
    result = dijkstra(graph, 0, 50, stop_at_target=False)
    assert result.previous is not None
    node = 50
    walked = [node]
    while node != 0:
        node = int(result.previous[node])
        walked.append(node)
    assert walked[::-1] == result.path.tolist()


# -- with_cost against build_graph from scratch ------------------------------------------

@pytest.mark.parametrize("cost", COSTS)
def test_swapping_the_cost_gives_the_graph_build_graph_would_have_built(cost):
    grid = generate_terrain(TerrainSpec(seed=3, grid_size=(20, 20), periods=(2, 2)))
    swapped = with_cost(build_graph(grid, cost='height'), cost)
    fresh = build_graph(grid, cost=cost)
    assert np.array_equal(swapped.indptr, fresh.indptr)
    assert np.array_equal(swapped.indices, fresh.indices)
    assert np.array_equal(swapped.weights, fresh.weights)
    assert np.array_equal(swapped.lengths_m, fresh.lengths_m)
    assert np.array_equal(swapped.dh, fresh.dh)
    assert swapped.cost_name == fresh.cost_name == costs.name_of(cost)


def test_a_swapped_cost_answers_the_same_searches():
    grid = generate_terrain(TerrainSpec(seed=5, grid_size=(20, 20), periods=(2, 2)))
    swapped = with_cost(build_graph(grid, cost='height_tiebreak'), 'length')
    fresh = build_graph(grid, cost='length')
    start, target = grid.index((1, 1)), grid.index((18, 18))
    assert dijkstra(swapped, start, target).cost == pytest.approx(
        dijkstra(fresh, start, target).cost, rel=0, abs=1e-9)


def test_a_gradient_limited_graph_stays_limited_and_says_so():
    """The filter describes edges already dropped; re-costing must not disown it (B21)."""
    grid = generate_terrain(TerrainSpec(seed=5, grid_size=(20, 20), periods=(2, 2)))
    limited = build_graph(grid, cost='height', max_abs_gradient_percent=7.0)
    swapped = with_cost(limited, 'length')
    assert swapped.max_abs_gradient_percent == 7.0
    assert swapped.n_edges == limited.n_edges
    assert np.array_equal(swapped.weights, build_graph(
        grid, cost='length', max_abs_gradient_percent=7.0).weights)


def test_a_masked_graph_keeps_its_mask():
    grid = generate_terrain(TerrainSpec(seed=5, grid_size=(20, 20), periods=(2, 2)))
    mask = np.zeros(grid.shape, dtype=bool)
    mask[:10, :10] = True
    swapped = with_cost(build_graph(grid, mask=mask), 'length')
    assert swapped.mask is mask
    assert swapped.n_searchable_nodes == 100


def test_swapping_in_a_bad_cost_is_refused_exactly_as_building_one_is():
    grid = generate_terrain(TerrainSpec(seed=5, grid_size=(10, 10), periods=(2, 2)))
    graph = build_graph(grid)
    with pytest.raises(ValueError, match='negative'):
        with_cost(graph, lambda dh, length_m: -length_m)


def test_swapping_accepts_an_unregistered_callable_and_records_its_name():
    grid = generate_terrain(TerrainSpec(seed=5, grid_size=(10, 10), periods=(2, 2)))
    model = costs.weighted_sum(height=1.0, length=0.5)
    swapped = with_cost(build_graph(grid), model)
    assert swapped.cost_name == costs.name_of(model)
    assert np.array_equal(swapped.weights, build_graph(grid, cost=model).weights)


# -- subgraph against build_graph from scratch -------------------------------------------

def test_slicing_a_subgraph_gives_what_build_graph_would_have_built():
    from core.search import subgraph
    grid = generate_terrain(TerrainSpec(seed=3, grid_size=(20, 20), periods=(2, 2)))
    mask = np.zeros(grid.shape, dtype=bool)
    mask[3:15, 2:17] = True
    sliced = subgraph(build_graph(grid, cost='height_tiebreak'), mask)
    fresh = build_graph(grid, cost='height_tiebreak', mask=mask)
    assert np.array_equal(sliced.indptr, fresh.indptr)
    assert np.array_equal(sliced.indices, fresh.indices)
    assert np.array_equal(sliced.weights, fresh.weights)
    assert np.array_equal(sliced.lengths_m, fresh.lengths_m)
    assert np.array_equal(sliced.dh, fresh.dh)
    assert sliced.n_searchable_nodes == fresh.n_searchable_nodes


def test_slicing_a_subgraph_matches_on_a_ragged_area_mask():
    """The real case: a height area, not a rectangle."""
    from core.hag import build_hag
    from core.search import subgraph
    grid = generate_terrain(TerrainSpec(seed=7, grid_size=(24, 24), periods=(2, 2)))
    hag = build_hag(grid, 3.0)
    full = build_graph(grid)
    for area in range(min(hag.n_areas, 6)):
        mask = hag.labels == area
        sliced, fresh = subgraph(full, mask), build_graph(grid, mask=mask)
        assert np.array_equal(sliced.indptr, fresh.indptr)
        assert np.array_equal(sliced.indices, fresh.indices)
        assert np.array_equal(sliced.weights, fresh.weights)


def test_slicing_intersects_an_existing_mask_instead_of_dropping_it():
    from core.search import subgraph
    grid = generate_terrain(TerrainSpec(seed=3, grid_size=(16, 16), periods=(2, 2)))
    first, second = np.zeros(grid.shape, dtype=bool), np.zeros(grid.shape, dtype=bool)
    first[:12, :] = True
    second[4:, :] = True
    sliced = subgraph(build_graph(grid, mask=first), second)
    fresh = build_graph(grid, mask=first & second)
    assert np.array_equal(sliced.indices, fresh.indices)
    assert sliced.n_searchable_nodes == fresh.n_searchable_nodes


def test_slicing_keeps_the_gradient_filter():
    from core.search import subgraph
    grid = generate_terrain(TerrainSpec(seed=5, grid_size=(16, 16), periods=(2, 2)))
    mask = np.zeros(grid.shape, dtype=bool)
    mask[2:14, 2:14] = True
    sliced = subgraph(build_graph(grid, max_abs_gradient_percent=7.0), mask)
    assert sliced.max_abs_gradient_percent == 7.0
    assert np.array_equal(sliced.indices,
                          build_graph(grid, mask=mask, max_abs_gradient_percent=7.0).indices)
