"""One along-chainage panel per STAS 863-85 parameter, drawn from committed files.

    python -m core.figures_stas --from results/stas --out results/stas/figures

Deliverable (c) of the article is "every STAS parameter at every point of the
resulting path, graphed along the chainage". This module draws exactly that, and NOTHING
here re-runs a search: it reads the per-station series written by
`core.experiment_stas.write_series` and the summary rows of `stas.csv`, which is the same
contract `core.figures_scenarios` already follows. A figure that re-ran the pipeline could
disagree with the table beside it.

Four panels, one per parameter, each with its class limit as a horizontal line and its
violating stations shaded, plus a compliance strip that stacks all four so one glance says
where the alignment fails and in what way.
"""
from __future__ import annotations

import os

#: before matplotlib is imported, exactly as core.figures does
os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional, Sequence  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core.figures import save  # noqa: E402

#: the parameters, in reporting order: (series column, limit column, label, units, which
#: side of the limit is compliant; '_abs' judges |value|, since a grade has a sign)
PANELS: tuple[tuple[str, str, str, str, str], ...] = (
    ('r_horizontal_m', 'limit_R_H', 'horizontal radius $R_H$', 'm', 'above'),
    ('gradient_percent', 'limit_i', 'longitudinal gradient $i$', '%', 'below'),
    ('tangent_length_m', 'limit_L_alignment', 'alignment length $L_{alignment}$', 'm', 'above'),
    ('r_vertical_m', 'limit_R_V', 'vertical curve radius $R_V$', 'm', 'above'),
)

VIOLATION_COLOUR = '#d55e00'
LIMIT_COLOUR = '#333333'
SERIES_COLOUR = '#0072b2'
CONCAVE_COLOUR = '#009e73'
CONVEX_COLOUR = '#cc79a7'


def read_series(path: Path) -> dict[str, np.ndarray]:
    """One per-station series file as arrays. Empty cells become NaN, 'inf' stays inf."""
    with open(path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'{path} has no stations')
    out: dict[str, np.ndarray] = {}
    for column in rows[0]:
        values = [row[column] for row in rows]
        if column == 'vertical_kind':
            out[column] = np.array(values, dtype=object)
        elif column == 'is_curve':
            out[column] = np.array([v == 'True' for v in values])
        else:
            out[column] = np.array([np.nan if v == '' else float(v) for v in values])
    return out


def _subject(values: np.ndarray, side: str) -> np.ndarray:
    return np.abs(values) if side in ('below', 'above_abs') else values


def _violating(values: np.ndarray, limit: np.ndarray, side: str) -> np.ndarray:
    """Stations that breach the limit. NaN (the parameter does not apply) never breaches."""
    subject = _subject(values, side)
    with np.errstate(invalid='ignore'):
        bad = subject > limit if side == 'below' else subject < limit
    return bad & np.isfinite(subject) & np.isfinite(limit)


def _shade_violations(ax, chainage: np.ndarray, bad: np.ndarray) -> int:
    """Shade every run of violating stations. Returns how many runs were shaded."""
    if not bad.any():
        return 0
    edges = np.flatnonzero(np.diff(bad.astype(int)) != 0) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(bad)]])
    runs = 0
    for a, b in zip(starts, ends):
        if not bad[a]:
            continue
        ax.axvspan(chainage[a], chainage[min(b, len(chainage) - 1)],
                   color=VIOLATION_COLOUR, alpha=0.18, lw=0)
        runs += 1
    return runs


def fig_parameter(series: dict[str, np.ndarray], column: str, limit_column: str,
                  label: str, units: str, side: str, ax=None, title: str = ''):
    """One STAS parameter along the chainage, with its limit and its violating spans."""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(9, 2.8))
    chainage = series['chainage_m']
    values = series[column]
    limit = series[limit_column]
    plotted = _subject(values, side)

    if column == 'r_vertical_m':
        # concave and convex have different limits and are shown apart
        kind = series['vertical_kind']
        for name, colour in (('concave', CONCAVE_COLOUR), ('convex', CONVEX_COLOUR)):
            mask = kind == name
            if mask.any():
                ax.plot(np.where(mask, chainage, np.nan),
                        np.where(mask, plotted, np.nan), color=colour, lw=1.6,
                        label=name)
        ax.plot(chainage, limit, color=LIMIT_COLOUR, ls='--', lw=1.0,
                label='required minimum')
    else:
        ax.plot(chainage, plotted, color=SERIES_COLOUR, lw=1.4, label='measured')
        if np.isfinite(limit).any():
            ax.axhline(float(np.nanmax(limit)), color=LIMIT_COLOUR, ls='--', lw=1.0,
                       label=f'limit {np.nanmax(limit):g} {units}')

    bad = _violating(values, limit, side)
    n_runs = _shade_violations(ax, chainage, bad)

    if column in ('r_horizontal_m', 'r_vertical_m') and np.isfinite(plotted).any():
        ax.set_yscale('log')
    ax.set_xlabel('chainage (m)')
    ax.set_ylabel(f'{label} ({units})')
    ax.set_xlim(float(chainage[0]), float(chainage[-1]))

    n_applicable = int((np.isfinite(plotted) & np.isfinite(limit)).sum())
    caption = title or label
    if not np.isfinite(limit).any():
        caption += ' - the road class does not define this limit'
    elif n_applicable == 0:
        # never say "compliant" about a limit that governs nothing here: on an alignment
        # with no tangent between two curves there is no alignment length to be too short
        caption += ' - no station on this alignment is governed by it'
    elif n_runs:
        caption += (f' - {int(bad.sum())} of {n_applicable} stations, in {n_runs} run(s), '
                    f'violate the limit')
    else:
        caption += f' - compliant at all {n_applicable} stations it applies to'
    ax.set_title(caption, fontsize=9, loc='left')
    ax.legend(fontsize=7, loc='best', framealpha=0.9)
    ax.grid(alpha=0.25)
    return ax.figure if created else ax


def fig_all_parameters(series: dict[str, np.ndarray], title: str = ''):
    """The panels stacked on one shared chainage axis."""
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(9, 2.3 * len(PANELS)), sharex=True)
    for ax, (column, limit_column, label, units, side) in zip(axes, PANELS):
        fig_parameter(series, column, limit_column, label, units, side, ax=ax)
        ax.set_xlabel('')
    axes[-1].set_xlabel('chainage (m)')
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def fig_compliance_strip(series: dict[str, np.ndarray], title: str = ''):
    """One row per STAS parameter: green where it holds, orange where it does not.

    Grey means the parameter does not apply at that station - a tangent has no radius to
    be too small, a grade tangent has no vertical curve - which is not the same as a pass.
    """
    fig, ax = plt.subplots(figsize=(9, 2.6))
    chainage = series['chainage_m']
    labels = []
    for k, (column, limit_column, label, _units, side) in enumerate(PANELS):
        values, limit = series[column], series[limit_column]
        subject = _subject(values, side)
        applies = np.isfinite(subject) & np.isfinite(limit)
        bad = _violating(values, limit, side)
        colours = np.where(~applies, '#d9d9d9',
                           np.where(bad, VIOLATION_COLOUR, '#4daf4a'))
        ax.scatter(chainage, np.full(len(chainage), k), c=colours, marker='|', s=180,
                   linewidths=1.4)
        labels.append(label)
    ax.set_yticks(range(len(PANELS)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('chainage (m)')
    ax.set_xlim(float(chainage[0]), float(chainage[-1]))
    ax.set_ylim(-0.6, len(PANELS) - 0.4)
    ax.set_title(title or 'STAS compliance along the alignment', fontsize=9, loc='left')
    fig.text(0.01, -0.04, 'green: holds   orange: violates   grey: does not apply here',
             fontsize=7)
    fig.tight_layout()
    return fig


def fig_arm_violations(rows: Sequence[dict], parameter: str = 'R_H'):
    """Violating length per Algorithm 1 arm, for one STAS parameter, as a boxplot."""
    from core.algorithm_1 import ARM_ORDER

    column = f'stas_{parameter}_violation_length_m'
    data, labels = [], []
    for arm in ARM_ORDER:
        values = [float(r[column]) for r in rows
                  if r.get('smoothing_arm') == arm and r.get(column) not in (None, '')]
        if values:
            data.append(values)
            labels.append(arm)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    if data:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel('violating length (m)')
    ax.set_title(f'{parameter}: how much road violates the limit, per Algorithm 1 arm',
                 fontsize=9, loc='left')
    ax.tick_params(axis='x', labelrotation=20)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    return fig


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--from', dest='src', default='results/stas', type=Path)
    parser.add_argument('--out', default=None, type=Path)
    parser.add_argument('--limit', type=int, default=6,
                        help='how many series to draw the full panel set for')
    args = parser.parse_args(argv)

    out = args.out or (args.src / 'figures')
    out.mkdir(parents=True, exist_ok=True)
    series_dir = args.src / 'series'
    if not series_dir.is_dir():
        raise SystemExit(f'no series in {series_dir}; run core.experiment_stas first')

    written = []
    for path in sorted(series_dir.glob('*.csv'))[:args.limit]:
        series = read_series(path)
        stem = path.stem
        written += save(fig_all_parameters(series, title=stem), out / f'stas_{stem}')
        written += save(fig_compliance_strip(series, title=stem),
                        out / f'compliance_{stem}')
        plt.close('all')

    csv_path = args.src / 'stas.csv'
    if csv_path.exists():
        with open(csv_path, newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        for parameter in ('R_H', 'i', 'L_alignment', 'R_V_convex'):
            written += save(fig_arm_violations(rows, parameter),
                            out / f'arm_violations_{parameter}')
            plt.close('all')

    for item in written:
        print(item)
    return 0


if __name__ == '__main__':
    sys.exit(main())
