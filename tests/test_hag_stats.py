"""HAG descriptors on graphs whose answers are known by hand."""
import math

import numpy as np
import pytest

from core.grid import Grid
from core.hag import build_hag
from core.hag_stats import describe_hag, describe_selection, gini, hop_diameter_lower_bound

# three vertical bands 0 | 1 | 2: a path graph of 3 areas
BANDS = np.repeat(np.array([[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]]), 4, axis=0)
# a hill: centre 2, ring 1, outer ring 0 -> nested areas, a path graph too
HILL = np.array([[0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 0],
                 [0, 1, 2, 1, 0],
                 [0, 1, 1, 1, 0],
                 [0, 0, 0, 0, 0]], dtype=float)


def hag_of(surf):
    return build_hag(Grid(surf=surf, cell_size_m=10.0), height_delta=1)


def test_path_graph_descriptors():
    row = describe_hag(hag_of(BANDS))
    assert row['n_areas'] == 3 and row['n_hag_edges'] == 2
    assert row['edges_per_area_mean'] == pytest.approx(4 / 3)
    assert row['degree_max'] == 2 and row['share_degree_1'] == pytest.approx(2 / 3)
    assert row['cyclomatic_number'] == 0 and row['n_components'] == 1
    assert row['hop_diameter_lb'] == 2
    assert row['area_size_gini'] == pytest.approx(0.0)
    assert row['area_m2_median'] == pytest.approx(8 * 100.0)


def test_nested_hill_is_a_path_with_one_enclosed_peak():
    row = describe_hag(hag_of(HILL))
    assert row['n_areas'] == 3 and row['n_hag_edges'] == 2
    assert row['single_cell_area_share'] == pytest.approx(1 / 3)
    assert row['largest_area_share'] == pytest.approx(16 / 25)


def test_nodata_is_not_counted():
    valid = np.ones(BANDS.shape, dtype=bool)
    valid[:, :2] = False
    hag = build_hag(Grid(surf=BANDS, cell_size_m=10.0), 1, valid_mask=valid)
    row = describe_hag(hag)
    assert row['n_areas'] == 2 and row['valid_share'] == pytest.approx(16 / 24)


def test_gini_bounds():
    assert gini(np.ones(5)) == pytest.approx(0.0)
    assert gini(np.array([0, 0, 0, 10.0])) == pytest.approx(0.75)


def test_selection_width_and_hops():
    hag = hag_of(BANDS)
    out = describe_selection(hag, [0, 1, 2], [0, 1, 2])
    assert out['cta_hops'] == 2 and out['n_cells_selected'] == 24
    assert out['cta_narrowest_width_m'] == pytest.approx(2 * 1.0 * 10.0)


def test_diameter_of_a_single_area_is_zero():
    hag = hag_of(np.zeros((3, 3)))
    assert hop_diameter_lower_bound(hag) == 0
    assert math.isnan(describe_hag(hag)['planarity_ratio'])
