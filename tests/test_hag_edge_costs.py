"""HAG edge costs, nodata cells, and the cheapest chain of candidate terrain areas (CTA)."""
import numpy as np
import pytest

from core import costs
from core.grid import Grid
from core.hag import (HAG_EDGE_COSTS, area_graph, area_mean_heights, build_hag,
                      cheapest_area_chain, crossing_arrays, hag_edge_table,
                      search_space_fraction, select_areas, shared_boundaries)
from core.terrain import TerrainSpec, generate_terrain

SURF = np.array([
    [1, 1, 1, 2, 2, 2],
    [1, 1, 1, 2, 2, 2],
    [1, 1, 3, 3, 2, 2],
    [4, 4, 3, 3, 3, 3],
    [4, 4, 4, 3, 3, 3],
    [4, 4, 4, 3, 5, 5],
], dtype=float)


def small_hag():
    return build_hag(Grid(surf=SURF, cell_size_m=10.0), height_delta=1)


def test_crossing_arrays_agree_with_shared_boundaries():
    hag = small_hag()
    low, high, _, _ = crossing_arrays(hag)
    per_pair = shared_boundaries(hag)
    assert low.size == sum(len(v) for v in per_pair.values())
    assert set(zip(low.tolist(), high.tolist())) == set(per_pair)


@pytest.mark.parametrize('edge_cost', HAG_EDGE_COSTS)
def test_every_hag_edge_gets_one_positive_weight(edge_cost):
    hag = small_hag()
    u, v, w = hag_edge_table(hag, edge_cost)
    expected = {(a, b) for a in range(hag.n_areas) for b in hag.neighbours[a] if a < b}
    assert set(zip(u.tolist(), v.tolist())) == expected
    assert (w > 0).all() and np.isfinite(w).all()


def test_area_means_prices_the_mean_height_difference():
    hag = small_hag()
    means = area_mean_heights(hag)
    u, v, w = hag_edge_table(hag, 'area_means', cost='height')
    np.testing.assert_allclose(w, np.maximum(np.abs(means[v] - means[u]), 1e-12))


def test_shared_border_prices_the_step_across_the_contour():
    hag = small_hag()
    surf = SURF.reshape(-1)
    low, high, node_low, node_high = crossing_arrays(hag)
    u, v, w = hag_edge_table(hag, 'shared_border', cost='height')
    for a, b, weight in zip(u, v, w):
        pick = (low == a) & (high == b)
        step = np.abs(surf[node_high[pick]] - surf[node_low[pick]]).mean()
        assert weight == pytest.approx(step)


@pytest.mark.parametrize('edge_cost', ('area_means', 'shared_border'))
def test_terrain_priced_chain_joins_start_and_target(edge_cost):
    grid = generate_terrain(TerrainSpec(seed=0, grid_size=(100, 100), periods=(4, 4)))
    hag = build_hag(grid, height_delta=3)
    graph = area_graph(hag, edge_cost)
    chain = cheapest_area_chain(graph, hag.area_of((5, 5)), hag.area_of((94, 94)))
    assert chain[0] == hag.area_of((5, 5)) and chain[-1] == hag.area_of((94, 94))
    assert all(b in hag.neighbours[a] for a, b in zip(chain, chain[1:]))
    assert set(chain) == select_areas(hag, (5, 5), (94, 94), edge_cost=edge_cost, graph=graph)


def test_nodata_cells_join_no_area_and_are_not_counted():
    valid = np.ones_like(SURF, dtype=bool)
    valid[:, 0] = False
    hag = build_hag(Grid(surf=SURF, cell_size_m=10.0), height_delta=1, valid_mask=valid)
    assert (hag.labels[:, 0] == -1).all()
    assert (hag.labels[valid] >= 0).all()
    assert hag.n_valid_cells == valid.sum()
    assert not hag.mask(range(hag.n_areas))[:, 0].any()
    assert search_space_fraction(hag, range(hag.n_areas)) == 1.0
    low, high, _, _ = crossing_arrays(hag)
    assert (low >= 0).all() and (high >= 0).all()


def test_unknown_edge_cost_is_refused():
    with pytest.raises(ValueError):
        hag_edge_table(small_hag(), 'hops')


def test_gradient_penalty_survives_nested_areas():
    hag = small_hag()
    model = costs.gradient_penalised(i_max_percent=7.0)
    for edge_cost in ('area_means', 'shared_border'):
        _, _, w = hag_edge_table(hag, edge_cost, cost=model)
        assert np.isfinite(w).all()
