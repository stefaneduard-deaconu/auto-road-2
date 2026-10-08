"""The fixed-cluster baseline: square blocks with the HAG's mean area size, same structure."""
import numpy as np

from core.experiment_baseline import build_block_hag
from core.grid import Grid
from core.hag import build_hag


def _hag():
    rng = np.random.default_rng(3)
    surf = np.cumsum(np.cumsum(rng.normal(size=(40, 40)), axis=0), axis=1)
    return build_hag(Grid(surf=surf, cell_size_m=10.0), height_delta=3.0)


def test_blocks_cover_every_valid_cell_once_with_the_hag_granularity():
    hag = _hag()
    blocks, side = build_block_hag(hag)
    assert (blocks.labels >= 0).all()
    assert blocks.area_size.sum() == hag.area_size.sum()
    assert side == max(2, round((hag.n_valid_cells / hag.n_areas) ** 0.5))
    assert abs(blocks.area_size.mean() - hag.area_size.mean()) <= hag.area_size.mean()


def test_block_neighbours_are_symmetric_and_touching():
    blocks, _ = build_block_hag(_hag())
    for a, nbrs in enumerate(blocks.neighbours):
        for b in nbrs:
            assert a in blocks.neighbours[b]
    assert all(len(n) <= 8 for n in blocks.neighbours)
