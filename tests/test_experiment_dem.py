"""core.experiment_dem: census, ladder, swath, tiles -- the real-DEM experiments of research
step 5, none of which the synthetic T7 matrix can answer (see the module docstring).

`_run_dem_scenario` is exercised on a small, directly-constructed `Grid`, so most of this
file needs neither the real raster nor a slow HAG build. Only the CLI-level census/ladder
smoke tests touch `raster/Idrija_Fault_LiDAR_DEM.tif`, guarded exactly like `tests/test_dem.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.dem import DEMError
from core.experiment_dem import (CENSUS_COLUMNS, LADDER_COLUMNS, SWATH_TILE_COLUMNS,
                                 TILES_DROPPED_COLUMNS, _run_dem_scenario, main, run_census,
                                 run_ladder)
from core.grid import Grid
from test_dem import synthetic_geotiff

REAL_TIF = Path(__file__).resolve().parents[1] / 'raster' / 'Idrija_Fault_LiDAR_DEM.tif'


def rolling_hill(size: int = 60, cell_m: float = 10.0, amplitude: float = 40.0) -> Grid:
    """A single smooth ridge: enough relief for a real HAG, small enough to search fast."""
    i, j = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    surf = 100.0 + amplitude * np.sin(i / size * np.pi) * np.cos(j / size * np.pi * 0.7)
    return Grid(surf=surf, cell_size_m=cell_m)


# -- _run_dem_scenario: the glue between core.experiment_step1 and the DEM CSVs ---------

def test_run_dem_scenario_produces_one_row_per_arm_including_the_two_gradient_arms():
    """The four class-blind DEFAULT_ARMS plus hag_yellow_gradcut and hag_yellow_gradpen
    (B21 on real terrain): five rows, and hag_yellow is NOT run twice."""
    from core.experiment_dem import DEFAULT_ARMS
    grid = rolling_hill()
    rows = _run_dem_scenario(grid, scenario_id='t1', source='unit_test', crop={'x': 1},
                             start=(2, 2), target=(57, 57), road_class_name='RO_CLASS_V_DEAL')
    expected = {label for label, _, _ in DEFAULT_ARMS} | {'hag_yellow_gradcut',
                                                          'hag_yellow_gradpen'}
    assert len(rows) == 5
    assert {r['arm'] for r in rows} == expected
    assert all(r['scenario_id'].startswith('t1__') for r in rows)
    assert all(r['source'] == 'unit_test' and r['cell_size_m'] == grid.cell_size_m
              for r in rows)


def test_run_dem_scenario_carries_the_crop_fields_through():
    grid = rolling_hill()
    crop = {'crop_row0': 42, 'crop_col0': 7, 'relief_m': 123.0}
    rows = _run_dem_scenario(grid, scenario_id='t2', source='unit_test', crop=crop,
                             start=(2, 2), target=(57, 57), road_class_name='RO_CLASS_V_DEAL')
    assert all(r['crop_row0'] == 42 and r['crop_col0'] == 7 and r['relief_m'] == 123.0
              for r in rows)


# -- find_pair: the gradient-screened / geometric-fallback split ------------------------

def test_find_pair_prefers_a_gradient_feasible_pair_when_one_exists():
    from core.experiment_dem import find_pair
    from data.configs.road_classes import get

    grid = rolling_hill(size=50, cell_m=10.0, amplitude=5.0)   # gentle: 7% should be easy
    start, target, screened = find_pair(grid, get('RO_CLASS_V_DEAL').with_cell_size(10.0),
                                        seed=0, margin=5)
    assert screened is True
    assert start != target


def test_find_pair_falls_back_to_a_geometric_pair_on_terrain_too_steep_for_the_class():
    """A cliff makes every direct alignment fail the screen; the demonstration must not
    raise for that -- it reports the resulting infeasibility instead (project convention:
    'a class that cannot be met is reported, never relaxed', the study protocol)."""
    from core.experiment_dem import find_pair
    from data.configs.road_classes import get

    surf = np.zeros((30, 30))
    surf[:, 15:] = 500.0                       # a step far steeper than any i_max
    grid = Grid(surf=surf, cell_size_m=10.0)
    start, target, screened = find_pair(grid, get('RO_CLASS_III_DEAL').with_cell_size(10.0),
                                        seed=0, margin=3, max_candidates=200)
    assert screened is False
    assert start != target


def test_find_pair_raises_only_when_no_geometric_candidate_exists_at_all():
    from core.experiment_dem import find_pair
    from data.configs.road_classes import get

    grid = rolling_hill(size=10)
    #: a margin of 5 on a 10-cell grid leaves nothing to draw from; core.od_feasibility
    #: raises ValueError for that precondition, before find_pair's own DEMError fallback
    #: (no candidate satisfied the diagonal-fraction filter) is ever reached
    with pytest.raises((DEMError, ValueError)):
        find_pair(grid, get('RO_CLASS_V_DEAL').with_cell_size(10.0), seed=0, margin=5,
                  max_candidates=50)


def test_a_gradient_infeasible_fallback_scenario_is_reported_not_dropped():
    """The end-to-end version of the fallback: the arm rows exist and say infeasible."""
    from core.experiment_dem import find_pair
    from data.configs.road_classes import get

    surf = np.zeros((30, 30))
    surf[:, 15:] = 500.0
    grid = Grid(surf=surf, cell_size_m=10.0)
    road_class = get('RO_CLASS_III_DEAL')
    start, target, screened = find_pair(grid, road_class.with_cell_size(10.0), seed=0,
                                        margin=3, max_candidates=200)
    assert screened is False
    rows = _run_dem_scenario(grid, scenario_id='fallback', source='unit_test', crop={},
                             start=start, target=target, road_class_name='RO_CLASS_III_DEAL',
                             gradient_screened=screened)
    assert all(r['gradient_screened'] is False for r in rows)
    found = [r for r in rows if r['path_found']]
    assert found and any(r['feasible'] is False for r in found)


def test_paths_out_collects_the_rough_axis_of_every_arm_that_found_a_path():
    grid = rolling_hill()
    collected: dict = {}
    rows = _run_dem_scenario(grid, scenario_id='t5', source='unit_test', crop={},
                             start=(2, 2), target=(57, 57), road_class_name='RO_CLASS_V_DEAL',
                             paths_out=collected)
    found = {r['arm'] for r in rows if r['path_found']}
    assert set(collected) == found and found
    for arm, xyz in collected.items():
        assert xyz.ndim == 2 and xyz.shape[1] == 3
        #: metres, not grid units: a 60-cell grid at 10 m spans up to ~590 m
        assert xyz[:, :2].max() > 100.0, arm
        assert xyz[0, 0] == pytest.approx(2 * grid.cell_size_m)      # starts at the start


def test_a_found_arm_reports_the_full_before_after_and_checks_columns():
    grid = rolling_hill()
    rows = _run_dem_scenario(grid, scenario_id='t3', source='unit_test', crop={},
                             start=(2, 2), target=(57, 57), road_class_name='RO_CLASS_V_DEAL')
    found = [r for r in rows if r['path_found']]
    assert found, 'a full-grid search on a small, connected terrain must find a path'
    for row in found:
        for key in ('path_length_m', 'objective_cost_sum_abs_dh_m', 'before_r_min_m',
                   'before_i_max_percent', 'after_r_min_m', 'after_i_max_percent',
                   'smoothing_rejected', 'retained_source', 'check_r_min_passed',
                   'feasible', 'verdict'):
            assert key in row, (row['arm'], key)
        assert row['retained_source'] in ('before', 'after')


def test_every_row_matches_the_declared_column_contract():
    grid = rolling_hill()
    rows = _run_dem_scenario(grid, scenario_id='t4', source='unit_test', crop={},
                             start=(2, 2), target=(57, 57), road_class_name='RO_CLASS_V_DEAL')
    for row in rows:
        for key in row:
            assert key in SWATH_TILE_COLUMNS, key


# -- ladder: the connectivity-collapse measurement, on a small hand-built grid ----------

def test_ladder_cell_count_matches_the_coarsening_geometry():
    """A property that holds by construction, whatever the terrain: coarsening by a factor
    of `f` on an `n x n` grid leaves `(n // f) ** 2` cells. The qualitative B20 pattern
    itself (fraction stable, components collapse) is NOT asserted on synthetic terrain --
    block-averaging can locally raise or lower a gradient depending on the terrain's own
    curvature, so it is only claimed, and checked, on the real raster below."""
    import core.experiment_dem as dem

    class _FakeTif:
        pixel_scale_m = (1.0, 1.0)

    grid = rolling_hill(size=40, cell_m=1.0, amplitude=15.0)

    def fake_load(path, *, row0, col0, size, sha256, tif, max_void_fraction):
        return grid, np.ones(grid.shape, dtype=bool), {}

    import core.dem as dem_mod

    original = dem_mod.load_geotiff_grid
    original_header = dem_mod.read_geotiff_header
    try:
        dem.load_geotiff_grid = fake_load
        dem.read_geotiff_header = lambda path, sha256=False: _FakeTif()
        rows = dem.run_ladder(Path('unused'), row0=0, col0=0, size=40,
                              cells_m=(1.0, 2.0, 4.0), road_class_name='RO_CLASS_V_DEAL')
    finally:
        dem.load_geotiff_grid = original
        dem.read_geotiff_header = original_header

    by_cell = {r['cell_size_m']: r for r in rows}
    assert by_cell[1.0]['n_cells'] == 40 * 40
    assert by_cell[2.0]['n_cells'] == 20 * 20
    assert by_cell[4.0]['n_cells'] == 10 * 10
    for row in rows:
        assert 0.0 <= row['admissible_edge_fraction'] <= 1.0
        assert 1 <= row['n_components'] <= row['n_cells']
        assert row['largest_component_cells'] <= row['n_cells']


# -- census / ladder CLI, on the real raster ----------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not REAL_TIF.exists(), reason='raster.zip has not been unzipped')
def test_census_reproduces_the_previously_ad_hoc_figure(tmp_path):
    """The 600 m / stride 25 / relief >= 30 m count an earlier, unsaved session reported as
    "102 336 windows" -- now reproducible, with its parameters stated."""
    rows = run_census(REAL_TIF, sizes_m=(600,), stride_m=25, min_relief_m=30.0)
    assert rows[0]['n_candidates'] == 102336
    assert rows[0]['valid_fraction'] == pytest.approx(0.29, abs=0.01)


@pytest.mark.slow
@pytest.mark.skipif(not REAL_TIF.exists(), reason='raster.zip has not been unzipped')
def test_ladder_shows_the_b20_pattern_on_the_real_raster():
    """Fraction stays in a narrow band; components collapse by roughly two orders of
    magnitude from 1 m to 20 m."""
    rows = run_ladder(REAL_TIF, row0=8825, col0=5350, size=600,
                      cells_m=(1.0, 10.0, 20.0), road_class_name='RO_CLASS_V_DEAL')
    by_cell = {r['cell_size_m']: r for r in rows}
    fine, coarse = by_cell[1.0], by_cell[20.0]
    assert 0.3 < fine['admissible_edge_fraction'] < 0.9
    assert 0.3 < coarse['admissible_edge_fraction'] < 0.9
    assert coarse['n_components'] < fine['n_components'] / 10
    assert coarse['largest_component_fraction'] > fine['largest_component_fraction']


@pytest.mark.slow
@pytest.mark.skipif(not REAL_TIF.exists(), reason='raster.zip has not been unzipped')
def test_cli_census_and_ladder_write_the_declared_columns(tmp_path):
    assert main(['--mode', 'census', '--tif', str(REAL_TIF), '--out', str(tmp_path)]) == 0
    import csv
    with open(tmp_path / 'census.csv', encoding='utf-8', newline='') as handle:
        assert next(csv.reader(handle)) == list(CENSUS_COLUMNS)

    assert main(['--mode', 'ladder', '--tif', str(REAL_TIF), '--out', str(tmp_path),
                '--row0', '8825', '--col0', '5350', '--size', '600',
                '--cells', '10,20']) == 0
    with open(tmp_path / 'ladder.csv', encoding='utf-8', newline='') as handle:
        assert next(csv.reader(handle)) == list(LADDER_COLUMNS)


def test_cli_refuses_cleanly_when_the_raster_is_missing(tmp_path):
    assert main(['--mode', 'census', '--tif', str(tmp_path / 'nope.tif'),
                '--out', str(tmp_path)]) == 1


# -- tiles: partial results, resumption and the audit of dropped candidates -----------------

def _tile_raster(tmp_path, *, size=700, amplitude=80.0, hole=None):
    """A fully valid sine-hill raster at 1 m, big enough for 600 m windows, with an optional
    single nodata pixel at `hole` = (row, col)."""
    i, j = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    surf = (100.0 + amplitude * np.sin(i / size * np.pi)
            * np.cos(j / size * np.pi * 0.7)).astype(np.float32)
    if hole is not None:
        surf[hole] = -99999.0
    return synthetic_geotiff(tmp_path, surf, name='tiles.tif', rows_per_strip=50)


def test_run_tiles_reports_each_tile_as_it_finishes(tmp_path):
    from core.experiment_dem import run_tiles

    tif = _tile_raster(tmp_path)
    seen: list[list[dict]] = []
    rows = run_tiles(tif, n=2, cell_m=10.0, seed=1, size_m=600, min_relief_m=10.0,
                     on_tile=seen.append)
    assert len(seen) == 2, 'one callback per tile, not one per run'
    assert all(len(tile_rows) == 5 for tile_rows in seen)       # five arms per tile
    assert [r for tile_rows in seen for r in tile_rows] == rows
    assert len({r['scenario_id'].split('__')[0] for r in rows}) == 2


def test_skipping_a_done_tile_does_not_move_the_pair_of_the_tile_after_it(tmp_path):
    """A tile's pair seed is `seed + its position in the candidate order`. If skipping
    renumbered the tiles that follow, a resumed run would put different origin-destination
    pairs on them than an uninterrupted run, and the two logs could not be merged."""
    from core.experiment_dem import run_tiles

    tif = _tile_raster(tmp_path)
    first = run_tiles(tif, n=2, cell_m=10.0, seed=1, size_m=600, min_relief_m=10.0)
    ids = list(dict.fromkeys(r['scenario_id'].split('__')[0] for r in first))
    assert len(ids) == 2

    resumed = run_tiles(tif, n=2, cell_m=10.0, seed=1, size_m=600, min_relief_m=10.0,
                        skip=frozenset({ids[0]}))
    assert {r['scenario_id'].split('__')[0] for r in resumed} == {ids[1]}

    def pair(rows, tile_id):
        row = next(r for r in rows if r['scenario_id'].startswith(tile_id + '__'))
        return tuple(row[k] for k in ('start_i', 'start_j', 'target_i', 'target_j'))

    assert pair(resumed, ids[1]) == pair(first, ids[1])


def test_skipping_every_tile_runs_nothing(tmp_path):
    from core.experiment_dem import run_tiles

    tif = _tile_raster(tmp_path)
    everything = run_tiles(tif, n=1, cell_m=10.0, seed=1, size_m=600, min_relief_m=10.0)
    done = frozenset(r['scenario_id'].split('__')[0] for r in everything)
    called: list = []
    rows = run_tiles(tif, n=1, cell_m=10.0, seed=1, size_m=600, min_relief_m=10.0,
                     skip=done, on_tile=called.append)
    assert rows == [] and called == []


def test_a_window_the_decimated_scan_over_approved_is_dropped_and_recorded(tmp_path):
    """One nodata pixel between the lattice points of the decimated census is invisible to
    it, so every window that contains the pixel is proposed and then refused by the exact
    read. The article states how many were refused and why; that needs the record."""
    from core.experiment_dem import run_tiles

    tif = _tile_raster(tmp_path, hole=(133, 133))     # 133 is not a multiple of stride 25
    dropped: list[dict] = []
    rows = run_tiles(tif, n=3, cell_m=10.0, seed=1, size_m=600, min_relief_m=10.0,
                     stride_m=25, dropped_out=dropped)
    assert rows == []
    assert len(dropped) == 3
    assert all('nodata' in d['reason'] for d in dropped)
    assert {tuple(sorted(d)) for d in dropped} == {tuple(sorted(TILES_DROPPED_COLUMNS))}
    assert all(d['size_m'] == 600 for d in dropped)


def test_cli_tiles_logs_incrementally_resumes_and_rebuilds_the_csv_without_the_raster(tmp_path):
    import csv
    import json

    tif = _tile_raster(tmp_path)
    out = tmp_path / 'out'
    argv = ['--mode', 'tiles', '--tif', str(tif), '--out', str(out), '--n', '2',
            '--cell', '10', '--seed', '1']
    # the CLI leaves min_relief_m at its default of 30 m; the fixture's relief clears it
    assert main(argv) == 0
    lines = (out / 'tiles.jsonl').read_text(encoding='utf-8').splitlines()
    assert len(lines) == 10 and all(json.loads(line)['source'] == 'idrija_tile'
                                    for line in lines)
    with open(out / 'tiles.csv', encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle)
        assert next(reader) == list(SWATH_TILE_COLUMNS)
        assert sum(1 for _ in reader) == 10
    assert (out / 'tiles_dropped.csv').exists()

    # the same command again computes nothing new
    assert main(argv) == 0
    assert len((out / 'tiles.jsonl').read_text(encoding='utf-8').splitlines()) == 10

    # and the table can be rebuilt from the log alone, with the raster gone
    (out / 'tiles.csv').unlink()
    assert main(['--mode', 'tiles', '--summary-only', '--out', str(out),
                 '--tif', str(tmp_path / 'gone.tif')]) == 0
    with open(out / 'tiles.csv', encoding='utf-8', newline='') as handle:
        assert sum(1 for _ in csv.reader(handle)) == 11         # header + 10


def test_summary_only_is_refused_for_the_modes_that_have_no_log(tmp_path):
    assert main(['--mode', 'census', '--summary-only', '--out', str(tmp_path)]) == 2


# -- b23: the witness of a gradient certificate, smoothed -----------------------------------

def _hill_raster(tmp_path, *, size=60, amplitude=1.0, cliff=False):
    i, j = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    surf = 100.0 + amplitude * np.sin(i / size * np.pi) * np.cos(j / size * np.pi * 0.7)
    if cliff:
        surf = np.where(j >= size // 2, surf + 400.0, surf)
    return synthetic_geotiff(tmp_path, surf.astype(np.float32), name='hill.tif')


def test_run_b23_reports_a_certified_alignment_and_the_guards_verdict(tmp_path):
    from core.experiment_dem import B23_COLUMNS, run_b23

    tif = _hill_raster(tmp_path)
    (row,) = run_b23(tif, row0=0, col0=0, size=60, cell_m=1.0, start=(5, 5),
                     target=(50, 50), road_class_name='RO_CLASS_V_DEAL')
    assert set(row) <= set(B23_COLUMNS)
    assert row['certified'] is True
    assert row['certified_i_max_percent'] <= row['i_max_limit_percent'] + 1e-9
    assert row['smoothing_rejected'] in (True, False)
    assert row['retained_source'] in ('before', 'after')
    assert row['after_length_m'] > 0 and row['development_ratio'] >= 1.0 - 1e-9


def test_run_b23_says_so_when_no_alignment_holds_the_limit(tmp_path):
    """A 400 m step is steeper than any class limit on any 1 m grid: the row exists, says
    `certified` is False, and carries no smoothing numbers to mistake for a result."""
    from core.experiment_dem import run_b23

    tif = _hill_raster(tmp_path, cliff=True)
    (row,) = run_b23(tif, row0=0, col0=0, size=60, cell_m=1.0, start=(5, 5),
                     target=(50, 50), road_class_name='RO_CLASS_V_DEAL')
    assert row['certified'] is False
    assert 'after_i_max_percent' not in row and 'smoothing_rejected' not in row


def test_cli_b23_writes_the_declared_columns(tmp_path):
    import csv

    from core.experiment_dem import B23_COLUMNS

    tif = _hill_raster(tmp_path)
    out = tmp_path / 'out'
    assert main(['--mode', 'b23', '--tif', str(tif), '--out', str(out), '--row0', '0',
                 '--col0', '0', '--size', '60', '--cell', '1']) == 0
    with open(out / 'b23_case.csv', encoding='utf-8', newline='') as handle:
        assert next(csv.reader(handle)) == list(B23_COLUMNS)


@pytest.mark.slow
@pytest.mark.skipif(not REAL_TIF.exists(), reason='raster.zip has not been unzipped')
def test_b23_case_on_the_real_raster_is_certified_and_its_numbers_are_self_consistent():
    """The committed instance of B23. Only invariants are asserted here: what the smoothed
    gradient IS, is a result of `results/dem/b23_case.csv`, not of this test."""
    from core.experiment_dem import run_b23

    (row,) = run_b23(REAL_TIF)
    assert row['certified'] is True
    assert row['certified_i_max_percent'] <= row['i_max_limit_percent'] + 1e-9
    if row['after_i_max_percent'] > row['i_max_limit_percent']:
        assert row['smoothing_rejected'] is True
    else:
        assert row['smoothing_rejected'] is False


def test_every_row_records_the_hag_build_time_and_the_straight_line_distance():
    """A single query pays for building the HAG, so the article cannot state a time saving
    without it; and the pair's straight-line distance is what tells a reader how far apart
    the two points are."""
    grid = rolling_hill()
    rows = _run_dem_scenario(grid, scenario_id='t6', source='unit_test', crop={},
                             start=(2, 2), target=(57, 42), road_class_name='RO_CLASS_V_DEAL')
    builds = {r['hag_build_time_s'] for r in rows}
    assert len(builds) == 1 and builds.pop() > 0.0, 'one HAG per scenario, shared by every arm'
    expected = float(np.hypot(55, 40)) * grid.cell_size_m
    assert all(r['straight_line_m'] == pytest.approx(expected) for r in rows)


def test_timings_are_repeated_and_recorded_with_their_extremes():
    """One invocation of a 4 km crop varied threefold from another on one machine, so a
    single run is not a timing: every timed step is repeated and the median is recorded
    with the minimum and maximum, for the class-blind arms and the gradient arms alike."""
    grid = rolling_hill()
    rows = _run_dem_scenario(grid, scenario_id='t7', source='unit_test', crop={},
                             start=(2, 2), target=(57, 57), road_class_name='RO_CLASS_V_DEAL',
                             timing_repeats=3)
    assert len(rows) == 5
    for row in rows:
        assert row['timing_repeats'] == 3, row['arm']
        assert row['wall_time_s_min'] <= row['wall_time_s'] <= row['wall_time_s_max'], row['arm']
    once = _run_dem_scenario(grid, scenario_id='t8', source='unit_test', crop={},
                             start=(2, 2), target=(57, 57), road_class_name='RO_CLASS_V_DEAL',
                             timing_repeats=1)
    assert {r['timing_repeats'] for r in once} == {1}
    assert all(r['wall_time_s_min'] == r['wall_time_s'] == r['wall_time_s_max'] for r in once)
