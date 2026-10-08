"""The figures of the article, in one print style (agreed item C5, amended 2026-10-08).

Like every `core/figures*.py` module, this one only draws: the numbers come from the committed
results through `article_2.build_article`, and the one map that needs a search reads the
example saved by `core.figure_cases`. Run from the repository root:

    python -m core.figures_article --out article_2/figures

Style (the dataviz reference palette): ink #0b0b0b / #52514e, hairline grid #e1e0d9, at most
three categorical hues in fixed order (blue #2a78d6, orange #eb6834, aqua #1baf7a), and the
one-hue blue ramp for terrain height. Every series also differs by marker, so the figures
read in grayscale.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from core.plotting import set_backend

set_backend()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
BLUES = LinearSegmentedColormap.from_list('heights', ['#cde2fb', '#86b6ef', '#3987e5',
                                                      '#1c5cab', '#0d366b'])
FULL_WIDTH_IN = 6.7      # a two-column journal page, in inches
SURFACE = '#ffffff'


def _style() -> None:
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 8.5, 'axes.titlesize': 9,
        'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2,
        'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.dpi': 300,
        'savefig.bbox': 'tight'})


def _data():
    sys.path.insert(0, '.')
    from article_2 import build_article as b
    return b, b.Data.load(b.DEFAULT_RESULTS)


# -- Figure 1: the evaluated workflow ---------------------------------------------------------------

WORKFLOW = ('Terrain grid\nor LiDAR DEM', 'Height areas\n(height interval)',
            'HAG: shared-\nborder edge costs', 'Cheapest HAG\npath: CTA (+ 1 ring)',
            'Dijkstra on the\nselected cells', 'Algorithm 1\n(1–2 iterations)',
            'Elevations\nreassigned', 'Stations every\n2 m on the axis',
            'STAS 863-85 check\nat every station')


def fig_workflow():
    """Nine numbered steps in two rows, read left to right then right to left (a snake)."""
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 2.7))
    ax.set_axis_off()
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.12, 2.05)
    w, h = 0.9, 0.66
    places = [(c, 1.45) for c in range(5)] + [(4 - c, 0.45) for c in range(4)]
    centres = []
    for n, ((col, y), label) in enumerate(zip(places, WORKFLOW), 1):
        x = col + 0.5
        stage = 'search' if n <= 5 else 'design'
        face = '#eef4fc' if stage == 'search' else '#fdf0ea'
        edge = BLUE if stage == 'search' else ORANGE
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle='round,pad=0.02,rounding_size=0.06',
                                    facecolor=face, edgecolor=edge, linewidth=1.0))
        ax.text(x + 0.04, y - 0.04, label, ha='center', va='center', fontsize=6.6, color=INK)
        ax.text(x - w / 2 + 0.08, y + h / 2 - 0.1, str(n), ha='center', va='center',
                fontsize=7, color='white', fontweight='bold',
                bbox=dict(boxstyle='circle,pad=0.18', facecolor=edge, edgecolor='none'))
        centres.append((x, y))
    for (x0, y0), (x1, y1) in zip(centres, centres[1:]):
        if y0 == y1:
            dx = w / 2 + 0.02
            sign = 1 if x1 > x0 else -1
            ax.annotate('', xy=(x1 - sign * dx, y1), xytext=(x0 + sign * dx, y0),
                        arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.9))
        else:
            ax.annotate('', xy=(x1, y1 + h / 2 + 0.02), xytext=(x0, y0 - h / 2 - 0.02),
                        arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.9))
    ax.text(0.05, 2.03, 'Search-domain reduction and path planning (steps 1–5)', fontsize=7.5,
            color=BLUE, va='top')
    ax.text(0.05, -0.1, 'Geometric refinement and engineering verification (steps 6–9)',
            fontsize=7.5, color=ORANGE, va='bottom')
    return fig


# -- D1: the reduction per terrain class -----------------------------------------------------------

def fig_reduction_by_class():
    """Box plot of the CTA share per class, ordered by median slope; one box per class ID."""
    b, d = _data()
    entries = []
    for key, tc in b.TERRAIN_CLASSES.items():
        values = b.finite(b.column(tc.select(d), 'search_space_percent_of_full'))
        entries.append((b.class_slope_gon(d, tc), b.CLASS_IDS[key], key[:3], values))
    entries.sort()
    colours = {'syn': BLUE, 'hil': ORANGE, 'dem': AQUA}
    names = {'syn': 'Synthetic, full range (S)', 'hil': 'Synthetic, hilly (H)',
             'dem': 'Real LiDAR DEM (D)'}
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 3.0))
    pos = np.arange(len(entries))
    for p, (_, _, kind, values) in zip(pos, entries):
        bp = ax.boxplot([values], positions=[p], widths=0.6, whis=(5, 95), showfliers=False,
                        patch_artist=True)
        for part in bp['boxes']:
            part.set(facecolor=colours[kind], alpha=0.35, edgecolor=colours[kind], linewidth=0.9)
        for part in (*bp['whiskers'], *bp['caps']):
            part.set(color=colours[kind], linewidth=0.8)
        for part in bp['medians']:
            part.set(color=INK, linewidth=1.1)
    ax.set_yscale('log')
    ax.set_ylim(0.02, 150)
    ax.set_xticks(pos, [e[1] for e in entries], rotation=90)
    ax.set_ylabel('CTA, % of the grid (log scale)')
    ax.set_xlabel('Terrain class, ordered by median slope')
    ax.axvspan(*_hilly_span(entries, b), color=GRID, alpha=0.6, zorder=0, linewidth=0)
    for kind in ('syn', 'hil', 'dem'):
        ax.scatter([], [], marker='s', s=40, color=colours[kind], alpha=0.5, label=names[kind])
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=7.2, ncol=3)
    ax.text(_hilly_span(entries, b)[0] + 0.2, 120, 'STAS 863-85 hilly class (20–25 gon)',
            fontsize=7, color=INK2, va='top')
    ax.grid(axis='x', visible=False)
    return fig


def _hilly_span(entries, b):
    lo, hi = b.STAS_HILLY_GON
    inside = [i for i, e in enumerate(entries) if lo <= e[0] <= hi]
    return (min(inside) - 0.5, max(inside) + 0.5) if inside else (0, 0)


# -- D2: what the reduction costs ------------------------------------------------------------------

def fig_cost_of_reduction():
    """Per class: mean share searched against mean objective ratio, CTA and CTA + 1 ring."""
    b, d = _data()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN * 0.62, 3.0))
    for key, tc in b.TERRAIN_CLASSES.items():
        rows, ring = tc.select(d), b.class_ring_rows(d, key)
        x0 = statistics.fmean(b.finite(b.column(rows, 'search_space_percent_of_full')))
        y0 = statistics.fmean(b.finite(b.column(rows, 'objective_ratio_to_full')))
        x1 = statistics.fmean(b.finite(b.column(ring, 'search_space_percent_of_full')))
        y1 = statistics.fmean(b.finite(b.column(ring, 'objective_ratio_to_full')))
        ax.plot([x0, x1], [y0, y1], color=AXIS, linewidth=0.8, zorder=1)
        ax.scatter([x0], [y0], s=26, marker='o', color=BLUE, edgecolor='white',
                   linewidth=0.8, zorder=3)
        ax.scatter([x1], [y1], s=26, marker='^', color=ORANGE, edgecolor='white',
                   linewidth=0.8, zorder=3)
        if b.CLASS_IDS[key] in ('S1', 'S15', 'D2', 'D4', 'D9'):
            ax.annotate(b.CLASS_IDS[key], (x0, y0), xytext=(4, 3), textcoords='offset points',
                        fontsize=7, color=INK2)
    ax.scatter([], [], s=26, marker='o', color=BLUE, label='CTA')
    ax.scatter([], [], s=26, marker='^', color=ORANGE, label='CTA + 1 ring')
    ax.axhline(1.0, color=AXIS, linewidth=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('Search domain, % of the grid (mean, log scale)')
    ax.set_ylabel('Objective ratio to the full grid (mean)')
    ax.legend(loc='upper right', fontsize=7.5)
    return fig


# -- D4: where the HAG pays off -------------------------------------------------------------------

def fig_speedup_vs_build():
    """Per synthetic class: speed-up per query against HAG construction time."""
    b, d = _data()
    cal = {False: b._calibration_pairs(d, False), True: b._calibration_pairs(d, True)}
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN * 0.62, 3.0))
    for key, tc in b.TERRAIN_CLASSES.items():
        if key.startswith('dem_'):
            continue
        first = tc.select(d)[0]
        pairs = cal[key.startswith('hil_')][(first['periods'], float(first['relief_amplitude_m']))]
        speed = statistics.fmean(f['search_time_s_median'] / c['search_time_s_median']
                                 for f, c in pairs if c['search_time_s_median'] > 0)
        build = statistics.fmean(1000 * c['hag_build_time_s'] for _, c in pairs)
        hilly = key.startswith('hil_')
        ax.scatter([build], [speed], s=30, marker='^' if hilly else 'o',
                   color=ORANGE if hilly else BLUE, edgecolor='white', linewidth=0.8, zorder=3)
        if b.CLASS_IDS[key] in ('S1', 'S3', 'S6', 'S9', 'S11', 'S12', 'S14', 'S15', 'H1', 'H2',
                                'H3'):
            ax.annotate(b.CLASS_IDS[key], (build, speed), xytext=(4, 2),
                        textcoords='offset points', fontsize=6.8, color=INK2)
    ax.scatter([], [], marker='o', color=BLUE, label='Synthetic, full range (S)')
    ax.scatter([], [], marker='^', color=ORANGE, label='Synthetic, hilly (H)')
    ax.set_xlabel('HAG construction time (ms)')
    ax.set_ylabel('Speed-up per query (×)')
    ax.legend(loc='lower right', fontsize=7.2)
    return fig


# -- D3: the real DEM and one worked section ------------------------------------------------------

def fig_dem_section(npz_path: Path = Path('results/figure_cases/dem_section.npz')):
    """(a) The whole surveyed swath in blue heights with the worked window outlined; (b) the
    window: height areas, CTA, CTA + 1 ring, and the three alignments."""
    if not npz_path.exists():
        return None
    z = np.load(npz_path)
    whole, cw = z['whole'], float(z['whole_cell_m'])
    sec, cs = z['section'], float(z['section_cell_m'])
    vmin, vmax = float(np.nanmin(whole)), float(np.nanmax(whole))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.3), layout='constrained',
                                   gridspec_kw={'width_ratios': [1.25, 1]})
    ext = (0, whole.shape[1] * cw / 1000, whole.shape[0] * cw / 1000, 0)
    im = ax1.imshow(whole, cmap=BLUES, extent=ext, interpolation='nearest', vmin=vmin, vmax=vmax)
    r0, c0, size = (float(v) for v in z['window_rc_m'])
    ax1.add_patch(Rectangle((c0 / 1000, r0 / 1000), size / 1000, size / 1000, fill=False,
                            edgecolor=ORANGE, linewidth=1.5))
    ax1.annotate('(b)', (c0 / 1000 + size / 1000, r0 / 1000), xytext=(3, -2),
                 textcoords='offset points', color=ORANGE, fontsize=8)
    ax1.set_title('(a) Idrija LiDAR swath, 20 m cells', loc='left', color=INK)
    ax1.set_xlabel('km')
    ax1.set_ylabel('km')
    ax1.grid(False)

    ext2 = (0, sec.shape[1] * cs / 1000, sec.shape[0] * cs / 1000, 0)
    ax2.imshow(sec, cmap=BLUES, extent=ext2, interpolation='nearest', vmin=vmin, vmax=vmax)
    gx = (np.arange(sec.shape[1]) + 0.5) * cs / 1000
    gy = (np.arange(sec.shape[0]) + 0.5) * cs / 1000
    ring = z['mask_hag_cta_ring1'].astype(float)
    cta = z['mask_hag_cta'].astype(float)
    ax2.contourf(gx, gy, ring, levels=[0.5, 1.5], colors=['white'], alpha=0.35)
    ax2.contourf(gx, gy, cta, levels=[0.5, 1.5], colors=['white'], alpha=0.55)
    ax2.contour(gx, gy, ring, levels=[0.5], colors=[AQUA], linewidths=0.7, linestyles='--')
    ax2.contour(gx, gy, cta, levels=[0.5], colors=[ORANGE], linewidths=0.8)
    for space, colour, style, label in (('full_grid', INK, '-', 'Full-grid alignment'),
                                        ('hag_cta', ORANGE, '-', 'CTA alignment'),
                                        ('hag_cta_ring1', AQUA, '--', 'CTA + 1 ring alignment')):
        q = z[f'path_{space}']
        if len(q):
            ax2.plot((q[:, 1] + 0.5) * cs / 1000, (q[:, 0] + 0.5) * cs / 1000, style,
                     color=colour, linewidth=1.2, label=label)
    for point, mark in ((z['start'], 'o'), (z['target'], 's')):
        ax2.scatter([(point[1] + 0.5) * cs / 1000], [(point[0] + 0.5) * cs / 1000], marker=mark,
                    s=26, color='white', edgecolor=INK, zorder=5)
    ax2.set_title(f'(b) The window, {cs:g} m cells', loc='left', color=INK)
    ax2.set_xlabel('km')
    ax2.grid(False)
    ax2.legend(loc='upper left', fontsize=6.6, frameon=True, facecolor='white', edgecolor=AXIS)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.05, pad=0.03, shrink=0.75)
    cbar.set_label('Height (m)', color=INK2)
    cbar.outline.set_edgecolor(AXIS)
    return fig


# -- D5: the STAS 863-85 parameters along the chainage ---------------------------------------------

STAS_SERIES = Path('results/stas/series/od1__STAS863_V40__hag_cta__climb_tiebreak__baseline__it2.csv')
STRAIGHT_RADIUS_M = 1.0e4   # a radius above this is drawn as "straight" (the axis has no curve)


def _spans(ax, chainage, bad):
    """Shade every run of violating stations."""
    edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
    for a, z in zip(edges[::2], edges[1::2]):
        ax.axvspan(chainage[a], chainage[min(z, len(chainage) - 1)], color=ORANGE, alpha=0.16,
                   linewidth=0)


def fig_stas_chainage(path: Path = STAS_SERIES):
    """Four panels on one chainage axis: horizontal radius, gradient, tangent length, vertical
    radii, each against its STAS 863-85 limit; violating stations shaded."""
    if not path.exists():
        return None
    from core.figures_stas import read_series
    z = read_series(path)
    ch = z['chainage_m']
    fig, axes = plt.subplots(4, 1, figsize=(FULL_WIDTH_IN, 6.2), sharex=True, layout='constrained')

    ax = axes[0]
    r = np.where(z['is_curve'], np.minimum(z['r_horizontal_m'], STRAIGHT_RADIUS_M), np.nan)
    lim = float(np.nanmax(z['limit_R_H']))
    ax.plot(ch, r, color=BLUE, linewidth=1.0)
    ax.axhline(lim, color=INK2, linestyle='--', linewidth=0.9)
    _spans(ax, ch, z['is_curve'] & (z['r_horizontal_m'] < lim))
    ax.set_yscale('log')
    ax.set_ylabel('R_H on curves (m)')
    ax.text(0.995, 0.92, f'limit {lim:g} m', transform=ax.transAxes, ha='right', va='top',
            fontsize=7, color=INK2)

    ax = axes[1]
    g = np.abs(z['gradient_percent'])
    lim = float(np.nanmax(z['limit_i']))
    ax.plot(ch, g, color=BLUE, linewidth=1.0)
    ax.axhline(lim, color=INK2, linestyle='--', linewidth=0.9)
    _spans(ax, ch, g > lim)
    ax.set_ylabel('Gradient |i| (%)')
    ax.text(0.995, 0.92, f'limit {lim:g}%', transform=ax.transAxes, ha='right', va='top',
            fontsize=7, color=INK2)

    ax = axes[2]
    t = z['tangent_length_m']
    lim = float(np.nanmax(z['limit_L_alignment']))
    ax.plot(ch, t, color=BLUE, linewidth=1.4)
    ax.axhline(lim, color=INK2, linestyle='--', linewidth=0.9)
    _spans(ax, ch, np.isfinite(t) & (t < lim))
    ax.set_ylabel('Tangent length (m)')
    ax.text(0.995, 0.92, f'limit {lim:g} m', transform=ax.transAxes, ha='right', va='top',
            fontsize=7, color=INK2)

    ax = axes[3]
    rv, kind, lim_v = z['r_vertical_m'], z['vertical_kind'], z['limit_R_V']
    for name, colour, mark in (('concave', AQUA, '-'), ('convex', ORANGE, '-')):
        m = kind == name
        ax.plot(np.where(m, ch, np.nan), np.where(m, rv, np.nan), mark, color=colour,
                linewidth=1.3, label=f'{name} curve')
    ax.plot(ch, lim_v, color=INK2, linestyle='--', linewidth=0.9, label='required minimum')
    _spans(ax, ch, np.isfinite(rv) & np.isfinite(lim_v) & (rv < lim_v))
    ax.set_yscale('log')
    ax.set_ylabel('R_V (m)')
    ax.legend(loc='upper right', fontsize=6.8, ncol=3)
    ax.set_xlabel('Chainage (m)')
    for ax in axes:
        ax.set_xlim(float(ch[0]), float(ch[-1]))
    return fig


FIGURES = {'workflow_steps.png': fig_workflow, 'reduction_by_class.png': fig_reduction_by_class,
           'cost_of_reduction.png': fig_cost_of_reduction,
           'speedup_vs_build.png': fig_speedup_vs_build, 'dem_section.png': fig_dem_section,
           'stas_chainage.png': fig_stas_chainage}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=Path('article_2/figures'))
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args(argv)
    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, make in FIGURES.items():
        if args.only and name not in args.only:
            continue
        fig = make()
        if fig is None:
            print(f'skipped {name} (its data are missing)')
            continue
        fig.savefig(args.out / name)
        plt.close(fig)
        print(args.out / name)
    return 0


if __name__ == '__main__':
    sys.exit(main())
