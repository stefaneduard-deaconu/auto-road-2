"""Task T9. The figures must agree with the tables they sit next to."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from core import figures
from core.hag import build_hag, select_areas
from core.metrics import DEFAULT_CHORD_M, DEFAULT_RESAMPLE_STEP_M
from core.terrain import TerrainSpec, generate_terrain
from data.configs.road_classes import get

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def terrain():
    grid = generate_terrain(TerrainSpec(seed=0, grid_size=(40, 40), periods=(2, 2)))
    return grid, build_hag(grid, height_delta=3.0)


@pytest.fixture
def axis_pair():
    t = np.linspace(0.0, 1.0, 60)
    before = np.column_stack([t * 300.0, np.sin(t * 6.0) * 40.0, 100.0 + t * 25.0])
    after = np.column_stack([t * 300.0, np.sin(t * 6.0) * 20.0, 100.0 + t * 30.0])
    return before, after


def test_the_module_imports_no_matplotlib_gui_backend():
    """`core.figures` selects a headless backend when it is imported."""
    code = ("import sys, core.figures;"
            "import matplotlib; assert matplotlib.get_backend().lower() == 'agg'")
    result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                            text=True, env={'PATH': '', 'SYSTEMROOT': '',
                                            'PYTHONPATH': str(REPO)})
    assert result.returncode == 0, result.stderr


def test_it_selects_a_headless_backend_even_without_mplbackend_set():
    """A standalone script gets no conftest.py, so the module has to do it itself."""
    import os
    env = {k: v for k, v in os.environ.items() if k != 'MPLBACKEND'}
    env['PYTHONPATH'] = str(REPO)
    result = subprocess.run(
        [sys.executable, '-c',
         "import core.figures, matplotlib; print(matplotlib.get_backend())"],
        capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().lower() == 'agg'


def test_terrain_and_areas_draws_the_selected_chain(terrain):
    grid, hag = terrain
    start, target = (5, 5), (grid.n_rows - 6, grid.n_cols - 6)
    selected = select_areas(hag, start, target, rule='yellow')
    fig = figures.fig_terrain_and_areas(grid, hag, selected)
    assert fig.axes
    assert str(len(set(selected))) in fig.axes[0].get_title()


def test_alignments_labels_every_arm(terrain):
    grid, _ = terrain
    paths = {'full_grid': np.array([[0.0, 0.0, 1.0], [100.0, 100.0, 2.0]]),
             'hag_yellow': np.array([[0.0, 0.0, 1.0], [50.0, 120.0, 2.0]])}
    fig = figures.fig_alignments(grid, paths)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ['full_grid', 'hag_yellow']


def test_the_profile_limit_line_comes_from_the_road_class(axis_pair):
    """Typing the limit in would let a figure contradict its own table."""
    before, after = axis_pair
    road_class = get('RO_CLASS_IV_DEAL')
    fig = figures.fig_longitudinal_profile(before, after, road_class)
    gradient_axis = fig.axes[1]
    limits = [line.get_ydata()[0] for line in gradient_axis.lines
              if len(set(line.get_ydata())) == 1]
    assert road_class.i_max_percent in limits


def test_the_curvature_limit_line_comes_from_the_road_class(axis_pair):
    before, after = axis_pair
    road_class = get('RO_CLASS_III_DEAL')
    fig = figures.fig_curvature(before, after, road_class)
    limits = [line.get_ydata()[0] for line in fig.axes[0].lines
              if len(set(line.get_ydata())) == 1]
    assert road_class.r_min_m in limits


def test_the_curvature_title_names_the_step_and_chord(axis_pair):
    """R_min depends on both, so a figure that omits them is not interpretable."""
    before, after = axis_pair
    fig = figures.fig_curvature(before, after, get('RO_CLASS_IV_DEAL'))
    title = fig.axes[0].get_title()
    assert f'{DEFAULT_RESAMPLE_STEP_M:g} m' in title
    assert f'{DEFAULT_CHORD_M:g} m' in title


def test_every_caption_carries_the_to_confirm_status(axis_pair):
    before, after = axis_pair
    road_class = get('RO_CLASS_IV_DEAL')
    assert road_class.status == 'TO CONFIRM'
    for fig in (figures.fig_longitudinal_profile(before, after, road_class),
                figures.fig_curvature(before, after, road_class)):
        titles = ' '.join(ax.get_title() for ax in fig.axes)
        assert 'TO CONFIRM' in titles


def test_distributions_skip_non_finite_values():
    rows = [{'arm': 'hag_yellow', 'search_space_percent_of_full': v}
            for v in (10.0, 20.0, float('inf'), None, 30.0)]
    fig = figures.fig_matrix_distributions(rows,
                                           quantities=('search_space_percent_of_full',))
    assert fig.axes


def test_save_writes_both_formats(tmp_path, axis_pair):
    before, after = axis_pair
    fig = figures.fig_curvature(before, after, get('RO_CLASS_IV_DEAL'))
    written = figures.save(fig, tmp_path / 'curvature')
    assert {p.suffix for p in written} == {'.svg', '.png'}
    assert all(p.stat().st_size > 0 for p in written)


def test_main_draws_an_alignment_figure_for_every_terrain_not_just_the_first_three(tmp_path):
    """Bug B26: sorted() puts T_article's paths before T_rough/T_smooth, so the old
    `[:3]` slice always drew T_article's three pairs and NEVER T_rough or T_smooth --
    `results/figures/` never had an alignments_T_rough_* or alignments_T_smooth_* file."""
    import csv as csv_mod

    src = tmp_path / 'results'
    (src / 'paths').mkdir(parents=True)
    with open(src / 'matrix.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv_mod.writer(handle)
        writer.writerow(['od_protocol', 'arm', 'search_space_percent_of_full',
                         'search_nodes_expanded', 'wall_time_s_median',
                         'objective_cost_ratio_to_baseline'])
        writer.writerow(['True', 'full_grid', '100', '10', '0.01', ''])

    #: four terrains, alphabetically past the old cap of 3
    for terrain in ('T_article', 'T_four', 'T_rough', 'T_smooth'):
        path = src / 'paths' / f'{terrain}__od1__full_grid__rough.csv'
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv_mod.writer(handle)
            writer.writerow(['x_m', 'y_m', 'z_m'])
            writer.writerow(['0.0', '0.0', '100.0'])
            writer.writerow(['10.0', '10.0', '101.0'])

    assert figures.main(['--from', str(src), '--out', str(tmp_path / 'out')]) == 0
    written = {p.name for p in (tmp_path / 'out').iterdir()}
    for terrain in ('T_article', 'T_four', 'T_rough', 'T_smooth'):
        assert f'alignments_{terrain}_od1.svg' in written, (terrain, written)


# -- the figures added for the article -----------------------------------------------

SIX_ARMS = ('full_grid', 'hag_yellow', 'hag_yellow_k1',
            'hag_yellow_gradcut', 'hag_yellow_gradpen')


def _axis(offset):
    t = np.linspace(0.0, 1.0, 30)
    return np.column_stack([t * 300.0, t * 250.0 + offset, 100.0 + 10.0 * t])


def test_every_arm_has_a_colour_and_a_line_style():
    for arm in SIX_ARMS:
        assert arm in figures.ARM_COLOURS and arm in figures.ARM_STYLES
    assert len(set(figures.ARM_COLOURS.values())) == len(figures.ARM_COLOURS)


def test_dem_scenario_draws_all_six_arms_including_the_dash_tuple():
    """`hag_yellow_gradpen`'s style is a dash tuple, which matplotlib refuses as a
    positional format string; drawing all six arms on one figure used to raise."""
    surf = 100.0 + 0.5 * np.add.outer(np.arange(40.0), np.arange(40.0))
    paths = {arm: _axis(20.0 * k) for k, arm in enumerate(SIX_ARMS)}
    fig = figures.fig_dem_scenario(surf, 10.0, paths, start=(2, 2), target=(35, 35),
                                   title='unit test')
    labels = {text.get_text() for text in fig.axes[0].get_legend().get_texts()}
    assert set(SIX_ARMS) <= labels and {'origin', 'destination'} <= labels


def test_dem_scenario_skips_an_arm_that_found_no_path():
    surf = 100.0 + np.zeros((20, 20))
    fig = figures.fig_dem_scenario(surf, 10.0, {'full_grid': _axis(0.0)}, title='x')
    labels = {text.get_text() for text in fig.axes[0].get_legend().get_texts()}
    assert 'full_grid' in labels and 'hag_yellow' not in labels


def test_alignments_accepts_the_gradient_arms(terrain):
    grid, _ = terrain
    paths = {arm: _axis(10.0 * k) for k, arm in enumerate(SIX_ARMS)}
    fig = figures.fig_alignments(grid, paths)
    assert {t.get_text() for t in fig.axes[0].get_legend().get_texts()} == set(SIX_ARMS)


def test_ladder_has_one_point_per_cell_size_and_a_log_component_axis():
    rows = [{'cell_size_m': c, 'admissible_edge_fraction': 0.6, 'n_components': n,
             'largest_component_fraction': f, 'road_class': 'RO_CLASS_V_DEAL',
             'i_max_percent': 7.0}
            for c, n, f in ((20.0, 80, 0.7), (1.0, 16000, 0.05), (5.0, 900, 0.3))]
    fig = figures.fig_ladder(rows)
    assert len(fig.axes) == 3
    for ax in fig.axes:
        xs = ax.lines[0].get_xdata()
        assert list(xs) == [1.0, 5.0, 20.0], 'sorted by cell size whatever the row order'
    assert fig.axes[1].get_yscale() == 'log'
    assert 'RO_CLASS_V_DEAL' in fig._suptitle.get_text()


def _tile_rows(n_tiles=3):
    rows = []
    for tile in range(n_tiles):
        for k, arm in enumerate(SIX_ARMS):
            rows.append({'scenario_id': f'tile{tile}__{arm}', 'arm': arm,
                         'search_space_percent_of_full': 100.0 if k == 0 else 10.0 + tile,
                         'path_length_m': 1000.0 + 50.0 * k, 'before_i_max_percent': 20.0 + k,
                         'path_found': True, 'feasible': (tile + k) % 2 == 0})
    return rows


def test_tiles_distributions_has_four_panels_and_the_class_limit():
    fig = figures.fig_tiles_distributions(_tile_rows(), i_max_limit_percent=7.0)
    assert len(fig.axes) == 4
    assert fig.axes[2].get_yscale() == 'log'
    legend = fig.axes[2].get_legend()
    assert legend is not None and '7%' in legend.get_texts()[0].get_text()
    assert 'of 3' in fig.axes[3].get_title()


def test_tiles_distributions_survives_an_arm_that_never_found_a_path():
    rows = _tile_rows()
    for row in rows:
        if row['arm'] == 'hag_yellow_gradcut':
            row.update(path_found=False, path_length_m=None, before_i_max_percent=None,
                       feasible=None)
    fig = figures.fig_tiles_distributions(rows, i_max_limit_percent=7.0)
    assert len(fig.axes) == 4


def _network_rows():
    rows = []
    for terrain, cell in (('synthetic', 1.0), ('idrija', 20.0)):
        for spacing, be, quality in ((20.0, None, 1.05), (40.0, 900.0, 1.1), (80.0, 200.0, 1.2)):
            rows.append({'terrain_id': terrain, 'cell_size_m': cell,
                         'spacing_m': spacing * cell, 'break_even_n_queries': be,
                         'quality_ratio_mean': quality, 'quality_ratio_max': quality * 1.3})
    return rows


def test_network_break_even_puts_spacing_in_cells_and_skips_a_never_paying_overlay():
    fig = figures.fig_network_break_even(_network_rows())
    ax_be, ax_q = fig.axes
    for line in ax_be.lines:
        assert list(line.get_xdata()) == [40.0, 80.0], (
            'spacing in CELLS, and the point whose break-even is never is not drawn')
    assert ax_be.get_xscale() == 'log' and ax_be.get_yscale() == 'log'
    assert len(ax_be.lines) == 2
    assert len([ln for ln in ax_q.lines if ln.get_linestyle() == '--']) == 2


def test_workflow_names_every_stage_of_the_evaluated_pipeline():
    fig = figures.fig_workflow()
    text = ' '.join(t.get_text().replace('\n', ' ') for t in fig.axes[0].texts)
    for stage in ('Height Area', 'Algorithm 1', 'RoadClass'):
        assert stage in text


def test_saving_the_same_figure_twice_gives_byte_identical_files(tmp_path, axis_pair):
    """No wall-clock date and no random element ids: a re-run of the figure drivers must not
    show up as a diff of every committed SVG."""
    before, after = axis_pair
    road_class = get('RO_CLASS_IV_DEAL')
    first = figures.save(figures.fig_curvature(before, after, road_class), tmp_path / 'a' / 'c')
    second = figures.save(figures.fig_curvature(before, after, road_class), tmp_path / 'b' / 'c')
    for one, other in zip(first, second):
        assert one.read_bytes() == other.read_bytes(), one.suffix


def test_terrain_and_areas_marks_origin_destination_and_the_selected_areas(terrain):
    grid, hag = terrain
    start, target = (5, 5), (grid.n_rows - 6, grid.n_cols - 6)
    selected = select_areas(hag, start, target, rule='yellow')
    paths = {'full_grid': np.array([[50.0, 50.0, 1.0], [300.0, 300.0, 2.0]])}
    fig = figures.fig_terrain_and_areas(grid, hag, selected, paths=paths, start=start,
                                        target=target)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ['selected areas (yellow)', 'full_grid', 'origin', 'destination']


def test_terrain_and_areas_without_paths_or_endpoints_still_explains_the_yellow(terrain):
    grid, hag = terrain
    selected = select_areas(hag, (5, 5), (grid.n_rows - 6, grid.n_cols - 6), rule='yellow')
    fig = figures.fig_terrain_and_areas(grid, hag, selected)
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == [
        'selected areas (yellow)']
