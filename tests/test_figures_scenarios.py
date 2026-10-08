"""T9: the driver that gives `fig_terrain_and_areas`, `fig_longitudinal_profile` and
`fig_curvature` real callers, reading only committed matrix/path files (core.figures_scenarios)."""
from __future__ import annotations

import numpy as np
import pytest

import core.experiment_matrix as experiment_matrix
import core.figures_scenarios as figures_scenarios
from core.results_io import path_csv_name




@pytest.fixture(scope='module')
def quick_matrix_with_all_paths(tmp_path_factory):
    """A real (small) matrix run with smoothed axes committed, via --write-paths all."""
    out = tmp_path_factory.mktemp('figs_scenarios')
    assert experiment_matrix.main(['--quick', '--out', str(out), '--write-paths', 'all']) == 0
    return out


def test_main_draws_terrain_areas_for_every_quick_terrain(quick_matrix_with_all_paths,
                                                           monkeypatch):
    monkeypatch.setattr(figures_scenarios, 'TERRAINS', experiment_matrix.QUICK_TERRAINS)
    out_dir = quick_matrix_with_all_paths / 'figures'
    assert figures_scenarios.main(['--from', str(quick_matrix_with_all_paths),
                                   '--out', str(out_dir)]) == 0
    written = {p.name for p in out_dir.iterdir()}
    for case in experiment_matrix.QUICK_TERRAINS:
        assert f'terrain_areas_{case.terrain_id}_od1.svg' in written, written


def test_main_draws_profile_and_curvature_when_smoothed_axes_exist(
        quick_matrix_with_all_paths, monkeypatch):
    monkeypatch.setattr(figures_scenarios, 'TERRAINS', experiment_matrix.QUICK_TERRAINS)
    out_dir = quick_matrix_with_all_paths / 'figures2'
    assert figures_scenarios.main(['--from', str(quick_matrix_with_all_paths),
                                   '--out', str(out_dir)]) == 0
    written = {p.name for p in out_dir.iterdir()}
    assert any(name.startswith('profile_Q_a_od1_hag_yellow_') for name in written), written
    assert any(name.startswith('curvature_Q_a_od1_hag_yellow_') for name in written), written


def test_main_skips_profile_figures_and_warns_when_only_rough_axes_exist(
        tmp_path_factory, monkeypatch, capsys):
    """The default write_paths='primary' matrix has no smoothed axis to draw from."""
    out = tmp_path_factory.mktemp('primary_only')
    assert experiment_matrix.main(['--quick', '--out', str(out)]) == 0     # no --write-paths
    monkeypatch.setattr(figures_scenarios, 'TERRAINS', experiment_matrix.QUICK_TERRAINS)

    assert figures_scenarios.main(['--from', str(out), '--out', str(out / 'figures')]) == 0
    written = {p.name for p in (out / 'figures').iterdir()}
    assert not any(name.startswith('profile_') for name in written)
    assert not any(name.startswith('curvature_') for name in written)
    assert any(name.startswith('terrain_areas_') for name in written)   # this part still works
    assert 'no smoothed axes' in capsys.readouterr().err


def test_main_refuses_cleanly_when_there_is_no_paths_directory(tmp_path):
    (tmp_path / 'matrix.csv').write_text('od_protocol\n', encoding='utf-8')
    assert figures_scenarios.main(['--from', str(tmp_path)]) == 1


# -- the two figure-building helpers, directly -------------------------------------------

def _write_xyz(path, xyz):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, xyz, delimiter=',', header='x_m,y_m,z_m', comments='')


def test_terrain_and_areas_figure_is_none_without_any_committed_rough_path(tmp_path):
    from core.hag import build_hag
    from core.terrain import TerrainSpec, generate_terrain

    grid = generate_terrain(TerrainSpec(seed=0, grid_size=(20, 20)))
    hag = build_hag(grid, 3.0)
    case = experiment_matrix.TerrainCase('T_empty', TerrainSpec(seed=0, grid_size=(20, 20)),
                                         'unit test')
    result = figures_scenarios._terrain_and_areas_figure(
        case, 'od1', tmp_path, tmp_path, grid, hag, {0})
    assert result is None


def test_profile_and_curvature_figures_finds_exactly_the_committed_pairs(tmp_path):
    case = experiment_matrix.TerrainCase(
        'T_x', experiment_matrix.TerrainSpec(seed=0, grid_size=(20, 20)), 'unit test')
    t = np.linspace(0, 1, 20)
    xyz = np.column_stack([t * 100, t * 10, 100 + t * 5])
    _write_xyz(tmp_path / path_csv_name('T_x', 'od1', 'hag_yellow'), xyz)
    _write_xyz(tmp_path / path_csv_name('T_x', 'od1', 'hag_yellow', road_class='RO_CLASS_IV_DEAL',
                                        iterations=1), xyz)

    written = figures_scenarios._profile_and_curvature_figures(
        case, 'od1', tmp_path, tmp_path, arms=figures_scenarios.ARM_ORDER,
        class_names=('RO_CLASS_III_DEAL', 'RO_CLASS_IV_DEAL', 'RO_CLASS_V_DEAL_REDUS'),
        iterations_wanted=(1, 2))
    names = {p.name for p in written}
    assert any('hag_yellow_RO_CLASS_IV_DEAL_it1' in n and n.startswith('profile_')
              for n in names)
    assert any('hag_yellow_RO_CLASS_IV_DEAL_it1' in n and n.startswith('curvature_')
              for n in names)
    #: no OTHER (arm, class) pair was invented from thin air
    assert not any('RO_CLASS_III_DEAL' in n for n in names)
    assert not any('hag_yellow_k1' in n for n in names)
