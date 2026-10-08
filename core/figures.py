"""Figures for the article, drawn from committed results. Task T9.

This is the one module in `core/` that imports matplotlib, and none of the others
import it. It reads `results/matrix.csv` and `results/paths/*.csv` and never re-runs a
search: that is why section 4 of the study protocol commits the path CSVs.

Two rules it will not bend:

* every radius in a figure comes from `core.metrics.curve_radii_m` with the same
  `step_m` and `chord_m` the tables report. A figure that sits next to a result table must not
  contradict it.
* the limit lines are read from the `RoadClass`, never typed in, and each caption
  carries the class's `TO CONFIRM` status.
"""
from __future__ import annotations

import os

#: before matplotlib is imported: a standalone script gets no conftest.py.
os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional, Sequence  # noqa: E402

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core import metrics as metrics_mod  # noqa: E402
from core.results_io import read_csv  # noqa: E402

#: the arms, in the order they should appear in a legend
ARM_ORDER = ('full_grid', 'hag_yellow', 'hag_yellow_k1',
             'hag_yellow_gradcut', 'hag_yellow_gradpen')
#: the first three are the T7 arms and keep their original hue; the two gradient-aware arms
#: (core.experiment_gradient) take two more Okabe-Ito colours, and each arm also has its own
#: dash pattern so a figure never leans on colour alone.
ARM_COLOURS = {'full_grid': '#333333', 'hag_yellow': '#d1a000',
               'hag_yellow_k1': '#0072b2',
               'hag_yellow_gradcut': '#cc79a7', 'hag_yellow_gradpen': '#009e73'}
ARM_STYLES = {'full_grid': '-', 'hag_yellow': '-', 'hag_yellow_k1': '--',
              'hag_yellow_gradcut': '-.',
              'hag_yellow_gradpen': (0, (3, 1, 1, 1))}

DISTRIBUTION_QUANTITIES = ('search_space_percent_of_full', 'search_nodes_expanded',
                           'wall_time_s_median', 'objective_cost_ratio_to_baseline')


def _status_caption(road_class) -> str:
    return f'{road_class.name} ({road_class.status})'


def fig_terrain_and_areas(grid, hag, selected_areas, paths: Optional[dict] = None,
                          ax=None, start=None, target=None):
    """The terrain, its height areas, and the yellow chain the search was restricted to.

    The terrain is drawn as a grey hillshade with a faint height tint, so that the yellow
    of the selected areas is the only saturated colour on the figure (on a full 'terrain'
    colour map the yellow bands of the map itself are indistinguishable from the overlay).
    """
    from matplotlib.colors import LightSource
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 6))
    shade = LightSource(azdeg=315, altdeg=45).hillshade(
        np.asarray(grid.surf, dtype=float), vert_exag=2.0, dx=grid.cell_size_m,
        dy=grid.cell_size_m)
    ax.imshow(shade, cmap='Greys_r', origin='lower', vmin=0.0, vmax=1.0)
    ax.imshow(grid.surf, cmap='terrain', origin='lower', alpha=0.28)
    mask = hag.mask(selected_areas)
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask] = (1.0, 0.85, 0.0, 0.55)          # the yellow areas
    ax.imshow(overlay, origin='lower')
    ax.contour(hag.labels, levels=np.arange(hag.n_areas) + 0.5, colors='k',
               linewidths=0.2, alpha=0.4)
    if mask.any() and not mask.all():
        ax.contour(mask.astype(float), levels=[0.5], colors='#7a5c00', linewidths=1.3)
    handles = [Patch(facecolor=(1.0, 0.85, 0.0, 0.55), edgecolor='#7a5c00',
                     label='selected areas (yellow)')]
    for arm, xyz in (paths or {}).items():
        ij = np.asarray(xyz)[:, :2] / grid.cell_size_m
        line, = ax.plot(ij[:, 1], ij[:, 0], linestyle=ARM_STYLES.get(arm, '-'), lw=1.4,
                        color=ARM_COLOURS.get(arm), label=arm)
        handles.append(line)
    for point, marker, label in ((start, 'o', 'origin'), (target, 's', 'destination')):
        if point is not None:
            ax.plot(point[1], point[0], marker, ms=8, mfc='white', mec='black')
            handles.append(Line2D([], [], marker=marker, ls='', ms=7, mfc='white', mec='black',
                                  label=label))
    ax.legend(handles=handles, loc='lower right', fontsize=8, framealpha=0.9)
    ax.set_title(f'Height areas ({hag.n_areas}) and the selected chain '
                 f'({len(set(selected_areas))} areas)')
    ax.set_xlabel('j (cells)')
    ax.set_ylabel('i (cells)')
    return fig


def fig_alignments(grid, paths: dict, ax=None):
    """Every arm's alignment on one terrain, in metres."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 6))
    ax.imshow(grid.surf, cmap='terrain', origin='lower',
              extent=(0, grid.n_cols * grid.cell_size_m,
                      0, grid.n_rows * grid.cell_size_m))
    for arm in ARM_ORDER:
        if arm not in paths:
            continue
        xyz = np.asarray(paths[arm])
        ax.plot(xyz[:, 1], xyz[:, 0], linestyle=ARM_STYLES[arm], lw=1.6,
                color=ARM_COLOURS[arm], label=arm)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title('Alignments per selection rule')
    ax.set_xlabel('y (m)')
    ax.set_ylabel('x (m)')
    return fig


def fig_longitudinal_profile(before_xyz_m, after_xyz_m, road_class, ax=None):
    """The profile before and after smoothing, against the class's i_max.

    This is where the XY_smooth -> Z_new -> i_new effect shows: improving the plan can
    make the profile worse, and the figure must not hide that.
    """
    fig, (ax_z, ax_i) = (ax[0].figure, ax) if ax is not None else plt.subplots(
        2, 1, figsize=(8, 6), sharex=True)

    for xyz, label, colour in ((before_xyz_m, 'before', '#909090'),
                               (after_xyz_m, 'after', '#0072b2')):
        xyz = np.asarray(xyz)
        s = np.concatenate([[0.0], np.cumsum(metrics_mod.segment_lengths_m(xyz[:, :2]))])
        ax_z.plot(s, xyz[:, 2], color=colour, label=label, lw=1.4)
        gradients = metrics_mod.gradients_percent(xyz)
        ax_i.plot(s[1:], np.abs(gradients), color=colour, label=label, lw=1.2)

    limit = road_class.i_max_percent
    ax_i.axhline(limit, color='#c00000', ls='--', lw=1.2,
                 label=f'i_max = {limit:g}%')
    ax_z.set_ylabel('z (m)')
    ax_i.set_ylabel('|i| (%)')
    ax_i.set_xlabel('chainage (m)')
    ax_z.legend(fontsize=8)
    ax_i.legend(fontsize=8)
    ax_z.set_title(f'Longitudinal profile, {_status_caption(road_class)}')
    return fig


def fig_curvature(before_xyz_m, after_xyz_m, road_class,
                  step_m: float = metrics_mod.DEFAULT_RESAMPLE_STEP_M,
                  chord_m: float = metrics_mod.DEFAULT_CHORD_M, ax=None):
    """Curve radius along the axis, against the class's R_min.

    The radii come from `core.metrics.curve_radii_m` at the reported `step_m` and
    `chord_m`, which are printed in the title because the value of R_min depends on them.
    """
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8, 4))
    for xyz, label, colour in ((before_xyz_m, 'before', '#909090'),
                               (after_xyz_m, 'after', '#0072b2')):
        radii = metrics_mod.curve_radii_m(np.asarray(xyz)[:, :2], step_m=step_m,
                                          chord_m=chord_m)
        finite = np.where(np.isfinite(radii), radii, np.nan)
        ax.plot(np.arange(len(finite)) * step_m, finite, color=colour, label=label,
                lw=1.2)
    ax.axhline(road_class.r_min_m, color='#c00000', ls='--', lw=1.2,
               label=f'R_min = {road_class.r_min_m:g} m')
    ax.set_yscale('log')
    ax.set_xlabel('chainage (m)')
    ax.set_ylabel('curve radius (m)')
    ax.legend(fontsize=8)
    ax.set_title(f'Curve radius, {_status_caption(road_class)} '
                 f'(step {step_m:g} m, chord {chord_m:g} m)')
    return fig


def fig_matrix_distributions(rows: Sequence[dict],
                             quantities: Sequence[str] = DISTRIBUTION_QUANTITIES,
                             by: str = 'arm'):
    """Box plots of the matrix, one panel per quantity: robustness, not one example."""
    groups: dict = {}
    for row in rows:
        groups.setdefault(row.get(by), []).append(row)
    labels = [a for a in ARM_ORDER if a in groups] or sorted(groups, key=str)

    fig, axes = plt.subplots(1, len(quantities), figsize=(4 * len(quantities), 4))
    axes = np.atleast_1d(axes)
    for ax, quantity in zip(axes, quantities):
        data = []
        for label in labels:
            values = [r.get(quantity) for r in groups[label]]
            data.append([float(v) for v in values
                         if isinstance(v, (int, float)) and np.isfinite(v)])
        ax.boxplot(data, tick_labels=labels)
        ax.set_title(quantity, fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
    fig.tight_layout()
    return fig


def fig_ladder(rows: Sequence[dict]):
    """B20: how the gradient-admissible subgraph falls apart as the cell shrinks.

    One row per cell size (`core.experiment_dem.run_ladder`). The FRACTION of admissible
    edges barely moves; what collapses is their CONNECTIVITY, which is the whole point of
    stating every `i_max` result together with the cell size it was measured at.
    """
    rows = sorted(rows, key=lambda r: float(r['cell_size_m']))
    cells = [float(r['cell_size_m']) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    series = (('admissible_edge_fraction', 'admissible edges (fraction)', False),
              ('n_components', 'components of the admissible subgraph', True),
              ('largest_component_fraction', 'largest component (fraction of cells)', False))
    for ax, (key, label, log_y) in zip(axes, series):
        ax.plot(cells, [float(r[key]) for r in rows], 'o-', color='#0072b2', lw=1.6)
        ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        else:
            ax.set_ylim(0, 1)
        ax.set_xlabel('cell size (m)')
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticks(cells)
        ax.set_xticklabels([f'{c:g}' for c in cells])
        ax.grid(alpha=0.25)
    road_class = rows[0].get('road_class', '') if rows else ''
    limit = rows[0].get('i_max_percent', '') if rows else ''
    fig.suptitle(f'Gradient feasibility is scale dependent ({road_class}, '
                 f'i_max = {limit}%)', fontsize=10)
    fig.tight_layout()
    return fig


def fig_tiles_distributions(rows: Sequence[dict], i_max_limit_percent: Optional[float] = None):
    """The real-DEM tile population, one box per arm: robustness on real terrain.

    Four panels: the search space, the path length relative to the full-grid path of the SAME
    tile (paired, so terrain difficulty cancels), the rough-axis i_max on a log axis with the
    class limit drawn, and the share of tiles that meet the class.
    """
    by_scenario: dict = {}
    for row in rows:
        by_scenario.setdefault(str(row['scenario_id']).split('__')[0], {})[row['arm']] = row
    arms = [a for a in ARM_ORDER if any(a in group for group in by_scenario.values())]

    def values(arm, getter):
        out = []
        for group in by_scenario.values():
            row = group.get(arm)
            value = getter(group, row) if row else None
            if isinstance(value, (int, float)) and np.isfinite(value):
                out.append(float(value))
        return out

    def length_ratio(group, row):
        base = group.get('full_grid')
        if row is None or base is None:
            return None
        if row.get('path_length_m') in (None, '') or base.get('path_length_m') in (None, ''):
            return None
        return float(row['path_length_m']) / float(base['path_length_m'])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    panels = (
        ('search space (% of the full grid)',
         lambda g, r: r.get('search_space_percent_of_full')),
        ('path length / full-grid path length', length_ratio),
        ('rough-axis i_max (%)', lambda g, r: r.get('before_i_max_percent')),
    )
    for ax, (title, getter) in zip(axes[:3], panels):
        data = [values(a, getter) for a in arms]
        ax.boxplot([d if d else [np.nan] for d in data], tick_labels=arms)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        for label in ax.get_xticklabels():
            label.set_ha('right')
    axes[2].set_yscale('log')
    if i_max_limit_percent is not None:
        axes[2].axhline(i_max_limit_percent, color='#c00000', ls='--', lw=1.2,
                        label=f'class limit {i_max_limit_percent:g}%')
        axes[2].legend(fontsize=8)

    share = []
    for arm in arms:
        found = [g[arm] for g in by_scenario.values() if arm in g and g[arm].get('path_found') is True]
        share.append(100.0 * sum(1 for r in found if r.get('feasible') is True) / max(len(by_scenario), 1))
    axes[3].bar(range(len(arms)), share, color=[ARM_COLOURS.get(a, '#888888') for a in arms])
    axes[3].set_xticks(range(len(arms)))
    axes[3].set_xticklabels(arms, rotation=45, ha='right', fontsize=7)
    axes[3].set_title(f'tiles meeting the class (% of {len(by_scenario)})', fontsize=9)
    fig.tight_layout()
    return fig


def fig_dem_scenario(surf, cell_size_m: float, paths: dict, start=None, target=None,
                     title: str = ''):
    """A hillshaded DEM crop with every arm's alignment on it, in metres, north up.

    `paths` maps an arm to an (n, 3) axis in metres as `core.experiment_dem` commits them
    (`x_m` is the row axis, `y_m` the column axis). An existing road, if one were drawn here,
    would be a REFERENCE only -- never a ground truth (article.md 5.5).
    """
    from matplotlib.colors import LightSource

    surf = np.asarray(surf, dtype=float)
    extent = (0, surf.shape[1] * cell_size_m, surf.shape[0] * cell_size_m, 0)
    light = LightSource(azdeg=315, altdeg=40)
    shade = light.shade(surf, cmap=plt.get_cmap('terrain'), blend_mode='overlay',
                        vert_exag=1.0, dx=cell_size_m, dy=cell_size_m)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.imshow(shade, extent=extent, origin='upper')
    for arm in ARM_ORDER:
        if arm not in paths:
            continue
        xyz = np.asarray(paths[arm])
        ax.plot(xyz[:, 1], xyz[:, 0], linestyle=ARM_STYLES[arm], lw=1.8,
                color=ARM_COLOURS[arm], label=arm)
    for point, marker, label in ((start, 'o', 'origin'), (target, 's', 'destination')):
        if point is not None:
            ax.plot(point[1] * cell_size_m, point[0] * cell_size_m, marker, ms=8, mfc='white',
                    mec='black', label=label)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.85)
    ax.set_xlabel('easting offset (m)')
    ax.set_ylabel('northing offset, southward (m)')
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def fig_network_break_even(rows: Sequence[dict]):
    """The portal overlay: what a query saves, what the build costs, what the route loses.

    Left: the break-even number of queries per `spacing_m` (log axis; below it a single
    full-grid search is cheaper). Right: the route's cost relative to the full-grid optimum,
    mean and worst case, which is the price of a sparser overlay.
    """
    by_terrain: dict = {}
    for row in rows:
        by_terrain.setdefault(str(row.get('terrain_id', '')), []).append(row)
    palette = ('#0072b2', '#d1a000', '#009e73', '#cc79a7', '#d55e00')
    fig, (ax_be, ax_q) = plt.subplots(1, 2, figsize=(10.5, 3.9))

    def column(group, key):
        return [float(r[key]) if r.get(key) not in (None, '') else np.nan for r in group]

    for k, (terrain, group) in enumerate(sorted(by_terrain.items())):
        group = sorted(group, key=lambda r: float(r['spacing_m']))
        #: spacing in CELLS, so terrains with different cell sizes share one axis
        x = [float(r['spacing_m']) / float(r.get('cell_size_m') or 1.0) for r in group]
        colour = palette[k % len(palette)]
        finite = [(xi, float(r['break_even_n_queries'])) for xi, r in zip(x, group)
                  if r.get('break_even_n_queries') not in (None, '')
                  and np.isfinite(float(r['break_even_n_queries']))]
        if finite:
            ax_be.plot([p[0] for p in finite], [p[1] for p in finite], 'o-', color=colour,
                       lw=1.6, label=terrain)
        ax_q.plot(x, column(group, 'quality_ratio_mean'), 'o-', color=colour, lw=1.6,
                  label=f'{terrain}, mean')
        ax_q.plot(x, column(group, 'quality_ratio_max'), 's--', color=colour, lw=1.1, alpha=0.7,
                  label=f'{terrain}, worst query')

    ax_be.set_yscale('log')
    ax_be.set_xscale('log')
    ax_be.set_xlabel('portal spacing (cells)')
    ax_be.set_ylabel('break-even (queries)')
    ax_be.set_title('queries needed before the overlay pays for its build', fontsize=9)
    ax_be.legend(fontsize=7)
    ax_be.grid(alpha=0.25)

    ax_q.axhline(1.0, color='#333333', lw=1.0, ls=':', label='full-grid optimum')
    ax_q.set_xscale('log')
    ax_q.set_xlabel('portal spacing (cells)')
    ax_q.set_ylabel('route cost / full-grid cost')
    ax_q.set_title('what a sparser overlay costs in route quality', fontsize=9)
    ax_q.legend(fontsize=6.5)
    ax_q.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_workflow():
    """The evaluated pipeline as a picture: terrain to engineering checks, in one row."""
    from matplotlib.patches import FancyBboxPatch

    steps = ('terrain\n(grid or DEM)', 'Height Area\nGraph', 'selected areas\n(yellow chain)',
             'search on the\nreduced grid', 'Algorithm 1\n(XY smoothing)',
             'Z re-sampled\nfrom terrain', 'checks against a\nRoadClass')
    fig, ax = plt.subplots(figsize=(13, 2.2))
    ax.set_xlim(0, len(steps) * 1.6)
    ax.set_ylim(0, 1)
    ax.axis('off')
    for k, text in enumerate(steps):
        x = 0.1 + k * 1.6
        ax.add_patch(FancyBboxPatch((x, 0.22), 1.3, 0.56, boxstyle='round,pad=0.02',
                                    fc='#e8f0fa', ec='#0072b2', lw=1.3))
        ax.text(x + 0.65, 0.5, text, ha='center', va='center', fontsize=8.5)
        if k < len(steps) - 1:
            ax.annotate('', xy=(x + 1.58, 0.5), xytext=(x + 1.32, 0.5),
                        arrowprops=dict(arrowstyle='->', color='#333333', lw=1.3))
    fig.tight_layout()
    return fig


def save(fig, path, formats: Sequence[str] = ('svg', 'png'), dpi: int = 300) -> list:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    #: no wall-clock date in an SVG and a fixed salt for its element ids, so the same data
    #: gives byte-identical files and a re-run does not appear as a diff in git
    metadata = {'svg': {'Date': None}}
    with matplotlib.rc_context({'svg.hashsalt': 'core.figures'}):
        for suffix in formats:
            target = path.with_suffix(f'.{suffix}')
            fig.savefig(target, dpi=dpi, bbox_inches='tight', metadata=metadata.get(suffix))
            written.append(target)
    plt.close(fig)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Figures from a committed results dir')
    parser.add_argument('--from', dest='source', default='results', type=Path)
    parser.add_argument('--out', default=None, type=Path)
    args = parser.parse_args(argv)
    out = args.out or (args.source / 'figures')

    rows = read_csv(args.source / 'matrix.csv')
    protocol = [r for r in rows if r.get('od_protocol') is True]
    written = save(fig_matrix_distributions(protocol), out / 'matrix_distributions')

    paths_dir = args.source / 'paths'
    if paths_dir.exists():
        import collections
        by_scenario = collections.defaultdict(dict)
        for csv_path in sorted(paths_dir.glob('*__rough.csv')):
            terrain, od, arm, _ = csv_path.stem.split('__')
            xyz = np.loadtxt(csv_path, delimiter=',', skiprows=1)
            by_scenario[(terrain, od)][arm] = xyz
        #: every scenario gets its figure (bug B26): the old `[:3]` slice always drew
        #: T_article's three protocol pairs, because sorted() puts T_article before
        #: T_rough/T_smooth -- so T_rough and T_smooth never had an alignment figure at all.
        for (terrain, od), paths in by_scenario.items():
            fig, ax = plt.subplots(figsize=(7, 6))
            for arm in ARM_ORDER:
                if arm in paths:
                    ax.plot(paths[arm][:, 1], paths[arm][:, 0], linestyle=ARM_STYLES[arm],
                            lw=1.5, color=ARM_COLOURS[arm], label=arm)
            ax.legend(fontsize=8)
            ax.set_title(f'{terrain} {od}: alignment per selection rule')
            ax.set_xlabel('y (m)')
            ax.set_ylabel('x (m)')
            written += save(fig, out / f'alignments_{terrain}_{od}')

    for item in written:
        print(item)
    return 0


if __name__ == '__main__':
    sys.exit(main())
