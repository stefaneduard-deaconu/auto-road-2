"""The standalone executable (core/native/gridpath_main.cpp) equals the ctypes engine, which
`tests/test_native_engine.py` already proves equal to `core.search`: same cost, same node
counts, the same distance field, bit for bit."""
import numpy as np
import pytest

from core import native
from core.terrain import TerrainSpec, generate_terrain

pytestmark = pytest.mark.skipif(not native.executable_available(),
                                reason='no C++ compiler for the standalone executable')


@pytest.mark.parametrize('seed,size,cost,cap', [
    (1, 30, 'height_tiebreak', None),
    (2, 40, 'height', None),
    (3, 36, '3d', None),
    (4, 30, 'height_tiebreak', 12.0),
])
def test_executable_equals_the_ctypes_engine(tmp_path, seed, size, cost, cap):
    grid = generate_terrain(TerrainSpec(seed=seed, grid_size=(size, size), periods=(2, 2)))
    npy = tmp_path / 'grid.npy'
    np.save(npy, np.ascontiguousarray(grid.surf, dtype=np.float64))
    rng = np.random.default_rng(seed)
    queries = [tuple(int(v) for v in rng.integers(0, grid.n_nodes, 2)) for _ in range(4)]
    out = native.run_executable(npy, queries, cell_size_m=grid.cell_size_m, cost=cost,
                                max_abs_gradient_percent=cap, dist_out=tmp_path / 'dist.bin')
    assert len(out['queries']) == len(queries)
    for (s, t), got in zip(queries, out['queries']):
        ref = native.dijkstra(grid, s, t, cost=cost, max_abs_gradient_percent=cap)
        assert got['cost'] == ref.cost
        assert got['expanded'] == ref.stats.nodes_expanded
        assert got['pushed'] == ref.stats.nodes_pushed
        assert got['reached'] == ref.stats.nodes_reached
    last = native.dijkstra(grid, *queries[-1], cost=cost, max_abs_gradient_percent=cap)
    assert np.array_equal(np.fromfile(tmp_path / 'dist.bin'), last.distances)


def test_executable_honours_a_mask(tmp_path):
    grid = generate_terrain(TerrainSpec(seed=5, grid_size=(32, 32), periods=(2, 2)))
    mask = np.zeros(grid.shape, dtype=np.uint8)
    mask[4:28, 4:28] = 1
    np.save(tmp_path / 'g.npy', np.ascontiguousarray(grid.surf, dtype=np.float64))
    np.save(tmp_path / 'm.npy', mask)
    s, t = 5 * 32 + 5, 26 * 32 + 26
    out = native.run_executable(tmp_path / 'g.npy', [(s, t)], cell_size_m=grid.cell_size_m,
                                mask_npy=tmp_path / 'm.npy')
    ref = native.dijkstra(grid, s, t, mask=mask)
    assert out['queries'][0]['cost'] == ref.cost
    assert out['queries'][0]['expanded'] == ref.stats.nodes_expanded
