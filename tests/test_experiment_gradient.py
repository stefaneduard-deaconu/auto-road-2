"""B21 (gradient-aware search arms) and B23 (the smoothing guard), together.

The `--quick` run is the smoke test for the plumbing, following `tests/test_experiment_matrix.py`'s
pattern. The one slow test reproduces the study protocol's own finding on the real Idrija
DEM -- a scenario already known, by hand, to trigger a compliant-to-non-compliant regression --
so `before_after_guarded`'s decision is checked against a real case, not a fabricated one.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from core.algorithm_1 import before_after_guarded
from core.dem import coarsen_to_cell_size, load_geotiff_grid
from core.experiment_gradient import (COLUMNS, GRADIENT_ARMS, _arm_graph,
                                      run_gradient_scenario, main)
from core.hag import build_hag, select_areas
from core.od_sampler import ODPair
from core.terrain import TerrainSpec, generate_terrain
from data.configs.road_classes import get

REAL_TIF = Path(__file__).resolve().parents[1] / 'raster' / 'Idrija_Fault_LiDAR_DEM.tif'


def _od_pair(grid, hag, start, target, od_id='od1') -> ODPair:
    z0, z1 = float(grid.surf[start]), float(grid.surf[target])
    return ODPair(od_id=od_id, start=start, target=target,
                 start_area=hag.area_of(start), target_area=hag.area_of(target),
                 hag_hops=0, straight_line_m=0.0, diagonal_m=0.0, diagonal_fraction=0.0,
                 start_height_m=z0, target_height_m=z1, height_difference_m=abs(z1 - z0),
                 protocol=True, candidate_index=0)


# -- the three arms share a mask and differ only in the graph they search ---------------

def test_gradpen_never_exceeds_the_unconstrained_arms_edge_count():
    """gradpen keeps every edge (it only re-weights them); gradcut can only remove some."""
    grid = generate_terrain(TerrainSpec(seed=0, grid_size=(40, 40), periods=(2, 2)))
    hag = build_hag(grid, 3.0)
    mask = hag.mask(select_areas(hag, (2, 2), (37, 37), rule='yellow'))
    road_class = get('RO_CLASS_IV_DEAL').with_cell_size(grid.cell_size_m)

    base = _arm_graph(grid, mask, road_class, 'hag_yellow')
    cut = _arm_graph(grid, mask, road_class, 'hag_yellow_gradcut')
    pen = _arm_graph(grid, mask, road_class, 'hag_yellow_gradpen')
    assert cut.n_edges <= base.n_edges
    assert pen.n_edges == base.n_edges
    assert cut.max_abs_gradient_percent == road_class.i_max_percent
    assert pen.max_abs_gradient_percent is None       # gradpen never drops edges


def test_an_unknown_arm_name_is_refused():
    grid = generate_terrain(TerrainSpec(seed=0, grid_size=(20, 20)))
    hag = build_hag(grid, 3.0)
    road_class = get('RO_CLASS_IV_DEAL').with_cell_size(grid.cell_size_m)
    with pytest.raises(ValueError, match='unknown gradient arm'):
        _arm_graph(grid, np.ones(grid.shape, dtype=bool), road_class, 'not_an_arm')


def test_a_steep_direct_gradcut_can_be_infeasible_while_gradpen_still_finds_a_path():
    """Constructed, not hoped-for: a ridge steeper than i_max sits across the only corridor."""
    surf = np.zeros((20, 20))
    surf[:, 10] = 40.0                          # a wall: crossing it in one 10 m step is 400%
    grid = generate_terrain(TerrainSpec(seed=0, grid_size=(20, 20)))
    from core.grid import Grid
    grid = Grid(surf=surf, cell_size_m=10.0)
    hag = build_hag(grid, 3.0)
    road_class = get('RO_CLASS_V_DEAL_REDUS').with_cell_size(grid.cell_size_m)   # i_max 7%
    od = _od_pair(grid, hag, (5, 2), (5, 17))

    rows = {r['arm']: r for r in run_gradient_scenario(grid, hag, od, road_class.name)}
    assert rows['hag_yellow_gradcut']['path_found'] is False
    assert rows['hag_yellow_gradpen']['path_found'] is True
    #: the penalised arm's rough axis must hold the limit far better than the blind one
    assert rows['hag_yellow_gradpen']['rough_i_max_percent'] < rows['hag_yellow']['rough_i_max_percent']


#: filled by run_matrix/main once a row is logged, not by run_gradient_scenario itself
RUNNER_FILLED = ('scenario_id', 'terrain_id', 'terrain_seed', 'od_id', 'timestamp_utc',
                 'git_sha', 'git_dirty', 'git_branch', 'python', 'python_free_threaded',
                 'platform', 'numpy', 'scipy')
#: only written when a path was found; an infeasible row has none of these (by design --
#: there is nothing to measure), which is what the second half of this test checks
FOUND_ONLY = tuple(c for c in COLUMNS if c not in RUNNER_FILLED
                   and c not in ('scenario_id', 'road_class', 'road_class_status', 'arm',
                                 'cost_model', 'connectivity', 'n_cells_selected',
                                 'n_graph_edges_kept', 'n_graph_edges_total',
                                 'edge_keep_fraction', 'search_nodes_expanded', 'wall_time_s',
                                 'path_found', 'infeasible_reason'))


def test_every_row_has_every_column():
    """A found row has every column FOUND_ONLY names; an infeasible one has none of them --
    checked on od2, which the study protocol-style exploration showed mixes both outcomes across arms
    on a small terrain (the gradient cut can disconnect the yellow selection)."""
    grid = generate_terrain(TerrainSpec(seed=0, grid_size=(40, 40), periods=(2, 2)))
    hag = build_hag(grid, 3.0)
    od = _od_pair(grid, hag, (2, 30), (37, 5))
    rows = run_gradient_scenario(grid, hag, od, 'RO_CLASS_IV_DEAL')
    assert any(r['path_found'] for r in rows) and any(not r['path_found'] for r in rows), \
        'the fixture must exercise both outcomes for this test to mean anything'
    for row in rows:
        for column in COLUMNS:
            if column in RUNNER_FILLED:
                continue
            if column in FOUND_ONLY:
                assert (column in row) == row['path_found'], (row['arm'], column)
            elif column == 'infeasible_reason':
                assert (column in row) == (not row['path_found']), (row['arm'], column)
            else:
                assert column in row, (row['arm'], column)


# -- the end-to-end --quick run: the smoke test ------------------------------------------

@pytest.fixture(scope='module')
def quick_run(tmp_path_factory):
    out = tmp_path_factory.mktemp('gradient')
    assert main(['--quick', '--out', str(out)]) == 0
    return out


def test_quick_run_writes_the_column_contract(quick_run):
    with open(quick_run / 'gradient.csv', encoding='utf-8', newline='') as handle:
        assert next(csv.reader(handle)) == list(COLUMNS)


def test_quick_run_covers_every_arm(quick_run):
    with open(quick_run / 'gradient.csv', encoding='utf-8', newline='') as handle:
        arms = {row['arm'] for row in csv.DictReader(handle)}
    assert arms == set(GRADIENT_ARMS)


def test_rerunning_the_quick_run_does_not_duplicate_rows(quick_run):
    from core.results_io import read_jsonl
    before = len(read_jsonl(quick_run / 'gradient.jsonl'))
    assert main(['--quick', '--out', str(quick_run)]) == 0
    assert len(read_jsonl(quick_run / 'gradient.jsonl')) == before


def test_summary_only_rebuilds_the_csv_from_the_log(quick_run):
    import os
    csv_path = quick_run / 'gradient.csv'
    mtime_before = os.path.getmtime(csv_path)
    assert main(['--summary-only', '--out', str(quick_run)]) == 0
    with open(csv_path, encoding='utf-8', newline='') as handle:
        assert next(csv.reader(handle)) == list(COLUMNS)
    assert os.path.getmtime(csv_path) >= mtime_before


# -- the real-DEM oracle: the study protocol's own B23 case ----------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not REAL_TIF.exists(), reason='raster.zip has not been unzipped')
def test_the_guard_fires_on_the_documented_idrija_regression():
    """the study protocol/3.4: crop (1500,4200), pair (53,42)->(19,2), RO_CLASS_V_DEAL, 10 m cells.

    Certified at 6.89% over 986 m; smoothing shortens it to 669 m at 14.70%, over twice the
    limit. The pair was COMPLIANT before (certified <= 7%) and NOT after, which is exactly
    the regression `before_after_guarded` exists to catch.
    """
    fine, _, _ = load_geotiff_grid(REAL_TIF, row0=1500, col0=4200, size=600, sha256=False,
                                   max_void_fraction=0.01)
    grid, _ = coarsen_to_cell_size(fine, target_cell_m=10.0)
    road_class = get('RO_CLASS_V_DEAL').with_cell_size(grid.cell_size_m)

    start, target = (53, 42), (19, 2)
    path_xy = np.array([[float(start[0]), float(start[1])], [float(target[0]), float(target[1])]])
    #: the certified witness IS the rough axis here: a straight
    #: two-point polyline understates it, so certify the real witness instead
    from core.od_feasibility import certify_grade
    certificate = certify_grade(grid, start, target, road_class)
    assert certificate.reachable, 'the crop/pair/class must still certify as the study protocol found'
    assert certificate.witness_i_max_percent <= road_class.i_max_percent + 1e-6

    report = before_after_guarded(grid, certificate.path_xy.astype(float), road_class,
                                  iterations=1)
    assert report['before']['i_max_percent'] <= road_class.i_max_percent + 1e-6
    assert report['after']['i_max_percent'] > road_class.i_max_percent
    assert report['smoothing_rejected'] is True
    assert report['retained_source'] == 'before'
    assert report['retained']['i_max_percent'] == pytest.approx(report['before']['i_max_percent'])
