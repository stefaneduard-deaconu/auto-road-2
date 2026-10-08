"""Assemble `article_2/article.md` from `article_2/src/*.md` and the committed results.

    python -m article_2.build_article                 # from the repo root
    python -m article_2.build_article --strict        # refuse placeholders that came from missing data

The article is GENERATED. Every table, every figure caption and every number in running text is
substituted from the committed result files, so the text cannot quote a number the data does not
contain, and a re-run of an experiment updates the article instead of silently contradicting it.
Edit `src/*.md`, never `article.md`.

Markers in the sources
----------------------
    {{fact:name}}          a formatted number or phrase computed from the results (FACTS below)
    {{tab:key}}            "Table N", numbered in order of first `<!-- TABLE:key -->`
    {{fig:key}}            "Figure N", numbered in order of first `<!-- FIGURE:key -->`
    {{cite:key,key}}       reference numbers, in order of first citation (references.py)
    <!-- TABLE:key -->     the table itself, caption above (MDPI style)
    <!-- FIGURE:key -->    the figure and its caption
    <!-- REFERENCES -->    the numbered reference list

A fact whose dataset is missing becomes `<<VALUE: name (reason)>>` and is listed in TODO.md; a
fact name that does not exist is an error, so a typo cannot pass as a placeholder. `--strict`
turns every generated placeholder into an error (hand-written `TODO(owner):` markers stay: they
are decisions that belong to the co-authors).

Checks that fail the build
--------------------------
    * an unresolved `{{...}}` marker or an unknown table, figure, fact or reference key
    * a sentence containing "validat" that is not a negation: the paper says *evaluated*
    * either of the two committed conclusion sentences (article.md 6.11) missing
    * a reference whose `verified` status is 'TODO' while `--strict` is given
"""
from __future__ import annotations

import argparse
import inspect
import math
import re
import shutil
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import StatisticsError
from typing import Callable, Optional, Sequence

from core.results_io import read_csv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_RESULTS = ROOT / 'results'

COMMITTED_SENTENCES = (
    'The proposed method was evaluated in terms of computational efficiency, solution quality, '
    'geometric feasibility, and robustness across multiple terrain and design scenarios.',
    'The applicability of the framework to real-world terrain data was additionally assessed '
    'using a DEM-based case study.',
)
NEGATIONS = ('not', 'no ', 'never', 'without', "n't", 'nor ', 'avoid')


class Missing(Exception):
    """A dataset a fact or table needs is not there."""


# ==============================================================================================
# data
# ==============================================================================================

@dataclass
class Data:
    results: Path
    matrix: Optional[list] = None
    od_pairs: Optional[list] = None
    gradient: Optional[list] = None
    census: Optional[list] = None
    ladder: Optional[list] = None
    swath: Optional[list] = None
    tiles: Optional[list] = None
    tiles_dropped: Optional[list] = None
    network: Optional[list] = None
    b23: Optional[list] = None
    x1: Optional[list] = None
    x1h: Optional[list] = None
    x3h: Optional[list] = None
    x2: Optional[list] = None
    x3: Optional[list] = None
    x3cal: Optional[list] = None
    x3hcal: Optional[list] = None
    x4: Optional[list] = None
    x5: Optional[list] = None
    x6: Optional[list] = None
    x7: Optional[list] = None
    stas: Optional[list] = None
    stas_dem: Optional[list] = None
    baseline: Optional[list] = None

    @classmethod
    def load(cls, results: Path) -> 'Data':
        def read(rel):
            path = results / rel
            if not path.exists() and path.with_name(path.name + '.gz').exists():
                path = path.with_name(path.name + '.gz')
            return read_csv(path) if path.exists() else None
        return cls(results=results, matrix=read('matrix.csv'), od_pairs=read('od_pairs.csv'),
                   gradient=read('gradient/gradient.csv'), census=read('dem/census.csv'),
                   ladder=read('dem/ladder.csv'), swath=read('dem/swath.csv'),
                   tiles=read('dem_tiles/tiles.csv'),
                   tiles_dropped=read('dem_tiles/tiles_dropped.csv'),
                   network=read('network/network.csv'),
                   b23=read('dem/b23_case.csv'),
                   x1=read('x1_hag_descriptors/x1_hag_descriptors.csv'),
                   x1h=read('x1_hag_descriptors_hilly/x1_hag_descriptors_hilly.csv'),
                   x3h=read('x3_reduction_synthetic_hilly/x3_reduction_synthetic_hilly.csv'),
                   x2=read('x2_hag_edge_cost/x2_hag_edge_cost.csv'),
                   x3=read('x3_reduction_synthetic/x3_reduction_synthetic.csv'),
                   x3cal=read('x3_reduction_synthetic/calibration/calibration.csv'),
                   x3hcal=read('x3_reduction_synthetic_hilly/calibration/calibration.csv'),
                   x4=read('x4_resolution_synthetic/x4_resolution_synthetic.csv'),
                   x5=read('x5_scaling/x5_scaling.csv'),
                   x6=read('x6_dem_whole/x6_dem_whole.csv'),
                   x7=read('x7_dem_windows/x7_dem_windows.csv'),
                   stas=read('stas/stas.csv'), stas_dem=read('stas_dem/stas.csv'),
                   baseline=read('baseline/baseline.csv'))


def need(rows, name: str):
    if rows is None:
        raise Missing(f'{name} not found')
    return rows


# ==============================================================================================
# formatting
# ==============================================================================================

def finite(values) -> list:
    return [float(v) for v in values
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]


def neg(text: str) -> str:
    """A real minus sign (U+2212) for negative numbers in running text and tables."""
    return text.replace('-', '−')


def span(lo: float, hi: float, d: int) -> str:
    """`lo–hi`; with a negative end the en dash would read as a second minus, so ` to `."""
    sep = '–' if lo >= 0 else ' to '
    return neg(f'{lo:.{d}f}{sep}{hi:.{d}f}')


def ms(values, d: int = 2) -> str:
    """`mean ± SD`, the form used inside running text."""
    v = finite(values)
    if not v:
        raise Missing('no finite values')
    if len(v) == 1:
        return neg(f'{v[0]:.{d}f}')
    return neg(f'{statistics.fmean(v):.{d}f} ± {statistics.stdev(v):.{d}f}')


def rng(values, d: int = 2) -> str:
    v = finite(values)
    if not v:
        raise Missing('no finite values')
    return span(min(v), max(v), d)


def med(values, d: int = 2) -> str:
    v = finite(values)
    if not v:
        raise Missing('no finite values')
    return neg(f'{statistics.median(v):.{d}f}')


def cell(values, d: int = 2) -> str:
    """The table cell: mean ± SD (median) [min–max], the four statistics of Section 5.4.4."""
    v = finite(values)
    if not v:
        return '–'
    text = f'{statistics.fmean(v):.{d}f}'
    if len(v) > 1:
        text += f' ± {statistics.stdev(v):.{d}f}'
    return neg(f'{text} ({statistics.median(v):.{d}f}) [') + span(min(v), max(v), d) + ']'


def thousands(n) -> str:
    return f'{int(round(float(n))):,}'


def pct(x: float, d: int = 1) -> str:
    return f'{100.0 * x:.{d}f}%'


def join_and(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return ''.join(items)
    return ', '.join(items[:-1]) + ' and ' + items[-1]


# ==============================================================================================
# groupings of the synthetic matrix (the same populations core.summary reports)
# ==============================================================================================

def search_rows(d: Data) -> list:
    from core.summary import population_search
    return population_search(need(d.matrix, 'results/matrix.csv'))


def geom_rows(d: Data) -> list:
    from core.summary import population_geometry
    return population_geometry(need(d.matrix, 'results/matrix.csv'))


def by_arm(rows, arm: str) -> list:
    return [r for r in rows if r.get('arm') == arm]


def column(rows, key: str) -> list:
    return [r.get(key) for r in rows]


def paired_reduction(d: Data, arm: str, including_hag: bool) -> list:
    """Per scenario: 100 * (1 - t_arm / t_full), optionally with the HAG build added to t_arm."""
    rows = search_rows(d)
    base = {(r['terrain_id'], r['od_id']): r for r in by_arm(rows, 'full_grid')}
    out = []
    for r in by_arm(rows, arm):
        b = base[(r['terrain_id'], r['od_id'])]
        t = r['wall_time_s_median'] + (r['hag_build_time_s'] if including_hag else 0.0)
        out.append(100.0 * (1.0 - t / b['wall_time_s_median']))
    return out


def paired_ratio(d: Data, arm: str, key: str) -> list:
    rows = search_rows(d)
    base = {(r['terrain_id'], r['od_id']): r for r in by_arm(rows, 'full_grid')}
    return [r[key] / base[(r['terrain_id'], r['od_id'])][key] for r in by_arm(rows, arm)]


def nonflat_ratio(d: Data, arm: str) -> list:
    from core.summary import _baseline_objective_by_scenario, _is_flat_baseline
    rows = search_rows(d)
    base = _baseline_objective_by_scenario(rows)
    return [r.get('objective_cost_ratio_to_baseline') for r in by_arm(rows, arm)
            if not _is_flat_baseline(r, base)]


def abs_diff(d: Data, arm: str) -> list:
    from core.summary import _baseline_objective_by_scenario, _objective_abs_diff
    rows = search_rows(d)
    base = _baseline_objective_by_scenario(rows)
    return [_objective_abs_diff(r, base) for r in by_arm(rows, arm)]


def censored(values) -> list:
    from core.summary import _censor_straight
    return [_censor_straight(v) for v in values]


FACTS: dict = {}


def fact(name: str):
    def register(fn):
        FACTS[name] = fn
        return fn
    return register


# ==============================================================================================
# the X1-X8 programme (2026-09-30): HAG descriptors, reduction distributions, resolution and
# real-DEM robustness, the STAS per-station checks. Distinct from the T7 matrix above: much
# larger samples, and `core.naming`'s `search_space` / `hag_edge_cost` / `grid_edge_cost`.
# ==============================================================================================

def x3_group(d: Data, search_space: str, hag_edge_cost: Optional[str] = None) -> list:
    """X3 rows for one search space, excluding the reference unit (reported separately)."""
    rows = [r for r in need(d.x3, 'results/x3_reduction_synthetic/x3_reduction_synthetic.csv')
            if r.get('reference_row') is not True and r.get('search_space') == search_space]
    if hag_edge_cost is not None:
        rows = [r for r in rows if r.get('hag_edge_cost') == hag_edge_cost]
    return rows


def x3_timing_group(d: Data, search_space: str, hag_edge_cost: Optional[str] = None, *,
                    hilly: bool = False) -> list:
    """The rows a TIMING column (time ratio, break-even, memory) is taken from.

    Serial rows time themselves. Rows of a parallel run (`core/parallel.py`) were timed while
    other workers shared the machine, so their timing columns come from the serial
    calibration sample instead (`python -m core.run calibrate X3`); without one the value is
    missing rather than wrong.
    """
    main = (x3h_group if hilly else x3_group)(d, search_space, hag_edge_cost)
    if not any(r.get('timing_mode') == 'parallel' for r in main):
        return main
    cal = d.x3hcal if hilly else d.x3cal
    where = ('results/x3_reduction_synthetic' + ('_hilly' if hilly else '')
             + '/calibration/calibration.csv')
    rows = [r for r in need(cal, where) if r.get('search_space') == search_space
            and (hag_edge_cost is None or r.get('hag_edge_cost') == hag_edge_cost)]
    if not rows:
        raise Missing(f'{where} has no {search_space} rows: run `python -m core.run calibrate X3`')
    return rows


#: `default` uses whatever `core.naming.DEFAULT_HAG_EDGE_COST` is, resolved at call time so a
#: future change to the default is picked up rather than silently stale.
X3_SPACES = {'default': ('hag_cta', None), 'ring1': ('hag_cta_ring1', None)}


def _resolve_hag_edge_cost(space: str, cost: Optional[str]) -> Optional[str]:
    from core.naming import DEFAULT_HAG_EDGE_COST
    if cost is not None:
        return cost
    return DEFAULT_HAG_EDGE_COST if space == 'hag_cta' else None


def _register_x3_facts() -> None:
    for tag, (space, cost) in X3_SPACES.items():
        def rows(d, space=space, cost=cost):
            return x3_group(d, space, _resolve_hag_edge_cost(space, cost))
        FACTS[f'x3_space_{tag}'] = lambda d, rows=rows: ms(
            column(rows(d), 'search_space_percent_of_full'), 2)
        FACTS[f'x3_space_{tag}_median'] = lambda d, rows=rows: med(
            column(rows(d), 'search_space_percent_of_full'), 2)
        FACTS[f'x3_obj_{tag}'] = lambda d, rows=rows: ms(
            column(rows(d), 'objective_ratio_to_full'), 3)
        FACTS[f'x3_hausdorff_{tag}'] = lambda d, rows=rows: ms(
            column(rows(d), 'hausdorff_to_full_m'), 1)
        FACTS[f'x3_hausdorff_{tag}_median'] = lambda d, rows=rows: med(
            column(rows(d), 'hausdorff_to_full_m'), 1)


_register_x3_facts()


def x3h_group(d: Data, search_space: str, hag_edge_cost: Optional[str] = None) -> list:
    """X3 rows of the `hilly` tier (terrains at 20-25 gon), same filters as `x3_group`."""
    rows = [r for r in need(d.x3h, 'results/x3_reduction_synthetic_hilly/'
                                   'x3_reduction_synthetic_hilly.csv')
            if r.get('search_space') == search_space]
    if hag_edge_cost is not None:
        rows = [r for r in rows if r.get('hag_edge_cost') == hag_edge_cost]
    return rows


def _register_x3h_facts() -> None:
    for tag, (space, cost) in X3_SPACES.items():
        def rows(d, space=space, cost=cost):
            return x3h_group(d, space, _resolve_hag_edge_cost(space, cost))
        FACTS[f'x3h_space_{tag}'] = lambda d, rows=rows: ms(
            column(rows(d), 'search_space_percent_of_full'), 2)
        FACTS[f'x3h_space_{tag}_median'] = lambda d, rows=rows: med(
            column(rows(d), 'search_space_percent_of_full'), 2)
        FACTS[f'x3h_obj_{tag}'] = lambda d, rows=rows: ms(
            column(rows(d), 'objective_ratio_to_full'), 3)
        FACTS[f'x3h_hausdorff_{tag}_median'] = lambda d, rows=rows: med(
            column(rows(d), 'hausdorff_to_full_m'), 1)


_register_x3h_facts()


@fact('x3_all_n_units')
def _(d):
    return str(len(x3_group(d, 'hag_cta', 'shared_border'))
               + len(x3h_group(d, 'hag_cta', 'shared_border')))


@fact('x3_n_classes')
def _(d):
    return str(sum(1 for k in TERRAIN_CLASSES if k.startswith(('syn_', 'hil_'))))


@fact('x3h_n_units')
def _(d):
    return str(len(x3h_group(d, 'hag_cta', 'shared_border')))


@fact('x3h_nodes_ratio_default')
def _(d):
    return ms(column(x3h_group(d, 'hag_cta', 'shared_border'), 'nodes_expanded_ratio_to_full'), 3)


@fact('x3h_time_ratio_default')
def _(d):
    return ms(column(x3_timing_group(d, 'hag_cta', 'shared_border', hilly=True),
                     'search_time_ratio_to_full'), 3)


@fact('x3h_break_even_default_median')
def _(d):
    return med(column(x3_timing_group(d, 'hag_cta', 'shared_border', hilly=True),
                      'break_even_queries'), 1)


@fact('x3h_path_found_default')
def _(d):
    rows = x3h_group(d, 'hag_cta', 'shared_border')
    return f"{sum(1 for r in rows if r.get('path_found') is True)} of {len(rows)}"


@fact('x1h_tree_share')
def _(d):
    rows = need(d.x1h, 'results/x1_hag_descriptors_hilly/x1_hag_descriptors_hilly.csv')
    return pct(sum(1 for r in rows if float(r['cyclomatic_number']) == 0) / len(rows), 1)


@fact('x3_n_workers')
def _(d):
    rows = x3_group(d, 'hag_cta', 'shared_border')
    return str(max(int(r.get('n_workers') or 1) for r in rows))


@fact('x3_calibration_n')
def _(d):
    """Scenarios (full tier) whose timing columns were re-measured serially."""
    rows = [r for r in need(d.x3cal, 'results/x3_reduction_synthetic/calibration/calibration.csv')
            if r.get('search_space') == 'hag_cta' and r.get('hag_edge_cost') == 'shared_border']
    return str(len(rows))


@fact('x3_n_units')
def _(d):
    return str(len(x3_group(d, 'hag_cta', 'shared_border')))


@fact('x3_nodes_ratio_default')
def _(d):
    return ms(column(x3_group(d, 'hag_cta', 'shared_border'), 'nodes_expanded_ratio_to_full'), 3)


@fact('x3_time_ratio_default')
def _(d):
    return ms(column(x3_timing_group(d, 'hag_cta', 'shared_border'),
                     'search_time_ratio_to_full'), 3)


@fact('x3_mem_ratio_default')
def _(d):
    return ms(column(x3_timing_group(d, 'hag_cta', 'shared_border'), 'memory_ratio_to_full'), 3)


@fact('x3_break_even_default_median')
def _(d):
    return med(column(x3_timing_group(d, 'hag_cta', 'shared_border'), 'break_even_queries'), 1)


@fact('x3_length_ratio_default')
def _(d):
    return ms(column(x3_group(d, 'hag_cta', 'shared_border'), 'length_ratio_to_full'), 3)


@fact('x3_path_found_default')
def _(d):
    rows = x3_group(d, 'hag_cta', 'shared_border')
    n = sum(1 for r in rows if r.get('path_found') is True)
    return f'{n} of {len(rows)}'


@fact('x2_default_name')
def _(d):
    from core.naming import DEFAULT_HAG_EDGE_COST
    return DEFAULT_HAG_EDGE_COST


@fact('x1_tree_share')
def _(d):
    rows = [r for r in need(d.x1, 'results/x1_hag_descriptors/x1_hag_descriptors.csv')
            if r.get('family') == 'synthetic']
    trees = sum(1 for r in rows if float(r['cyclomatic_number']) == 0)
    return pct(trees / len(rows), 1)


@fact('x1_dem_edges_per_area')
def _(d):
    rows = [r for r in need(d.x1, 'results/x1_hag_descriptors/x1_hag_descriptors.csv')
            if r.get('family') == 'dem']
    return ms(column(rows, 'edges_per_area_mean'), 2)


def _default_cta(rows: list) -> list:
    """The candidate-terrain-area rows priced with the study's HAG edge cost only; rows of
    any other edge cost must not be pooled with them."""
    from core.naming import DEFAULT_HAG_EDGE_COST
    return [r for r in rows if r.get('search_space') == 'hag_cta'
            and r.get('hag_edge_cost') == DEFAULT_HAG_EDGE_COST]


def x4_at(d: Data, cell_size_m: str) -> list:
    rows = need(d.x4, 'results/x4_resolution_synthetic/x4_resolution_synthetic.csv')
    return [r for r in _default_cta(rows) if str(r.get('cell_size_m')) == cell_size_m]


def x7_at(d: Data, cell_size_m: str) -> list:
    rows = need(d.x7, 'results/x7_dem_windows/x7_dem_windows.csv')
    return [r for r in _default_cta(rows) if str(r.get('cell_size_m')) == cell_size_m]


def _register_resolution_facts() -> None:
    for prefix, at in (('x4', x4_at), ('x7', x7_at)):
        for cell in ('1', '3', '5', '10', '20'):
            FACTS[f'{prefix}_space_{cell}m'] = lambda d, at=at, cell=cell: ms(
                column(at(d, cell), 'search_space_percent_of_full'), 2)
            FACTS[f'{prefix}_obj_{cell}m'] = lambda d, at=at, cell=cell: ms(
                column(at(d, cell), 'objective_ratio_to_full'), 3)


_register_resolution_facts()


def x6_at(d: Data, od_band: Optional[str], cell_size_m: str) -> list:
    rows = need(d.x6, 'results/x6_dem_whole/x6_dem_whole.csv')
    return [r for r in _default_cta(rows) if str(r.get('cell_size_m')) == cell_size_m
            and (od_band is None or r.get('od_band') == od_band)]


def _register_x6_facts() -> None:
    for band in ('1km', '10km'):
        for cell in ('5', '10', '20'):
            FACTS[f'x6_space_{band}_{cell}m'] = lambda d, band=band, cell=cell: ms(
                column(x6_at(d, band, cell), 'search_space_percent_of_full'), 3)
            FACTS[f'x6_obj_{band}_{cell}m'] = lambda d, band=band, cell=cell: ms(
                column(x6_at(d, band, cell), 'objective_ratio_to_full'), 3)


_register_x6_facts()


# -- terrain classes: the reduction per kind of terrain ------------------------------------

#: STAS 863-85's hilly class, in gon (red point D6)
STAS_HILLY_GON = (20.0, 25.0)


def gon_from_percent(slope_percent: float) -> float:
    return math.atan(float(slope_percent) / 100.0) * 200.0 / math.pi


@dataclass(frozen=True)
class TerrainClass:
    """One kind of terrain the reduction is reported for. `name` and `looks_like` are fixed
    words; everything numeric in the description is measured on the rows `select` returns."""
    key: str
    name: str
    looks_like: str
    source: str
    select: Callable[['Data'], list]


def _synthetic_class(period: int, relief_m: float) -> TerrainClass:
    features = {2: 'broad', 3: 'moderately broad', 4: 'medium', 6: 'fine',
                8: 'very fine'}[period]
    relief = {20.0: 'low', 60.0: 'high', 150.0: 'very high'}[relief_m]
    looks = {2: 'a few wide hills and valleys', 3: 'several wide hills and valleys',
             4: 'a moderate number of hills and valleys',
             6: 'many short ridges and valleys',
             8: 'very many short ridges and valleys'}[period]

    def select(d, period=period, relief_m=relief_m):
        return [r for r in x3_group(d, 'hag_cta', 'shared_border')
                if r.get('periods') == period and float(r.get('relief_amplitude_m')) == relief_m]
    return TerrainClass(f'syn_p{period}_a{int(relief_m)}', f'{features} features, {relief} relief',
                        f'synthetic Perlin terrain with {looks} (Perlin period {period}, '
                        f'nominal relief {relief_m:g} m)',
                        'results/x3_reduction_synthetic/x3_reduction_synthetic.csv (X3)', select)


HILLY_NAMES = {(4, 200.0): 'hilly: medium features, 200 m relief',
                (6, 120.0): 'hilly: fine features, 120 m relief',
                (8, 100.0): 'hilly: very fine features, 100 m relief'}


def _hilly_class(period: int, relief_m: float) -> TerrainClass:
    looks = {4: 'a moderate number of hills and valleys', 6: 'many short ridges and valleys',
             8: 'very many short ridges and valleys'}[period]

    def select(d, period=period, relief_m=relief_m):
        return [r for r in x3h_group(d, 'hag_cta', 'shared_border')
                if r.get('periods') == period and float(r.get('relief_amplitude_m')) == relief_m]
    return TerrainClass(f'hil_p{period}_a{int(relief_m)}', HILLY_NAMES[(period, relief_m)],
                        f'synthetic Perlin terrain with {looks} (Perlin period {period}, nominal '
                        f'relief {relief_m:g} m), chosen so that its median slope falls in the '
                        f'STAS 863-85 hilly class',
                        'results/x3_reduction_synthetic_hilly/x3_reduction_synthetic_hilly.csv '
                        '(X3, tier hilly)', select)


def _dem_class(kind: str, cell: str) -> TerrainClass:
    if kind == 'whole':
        return TerrainClass(f'dem_whole_{cell}m', f'Idrija swath, whole, {cell} m cells',
                            f'the whole surveyed LiDAR swath of the Idrija Fault, coarsened to '
                            f'{cell} m cells, O–D pairs 1–20 km apart',
                            'results/x6_dem_whole/x6_dem_whole.csv (X6)',
                            lambda d, cell=cell: x6_at(d, None, cell))
    return TerrainClass(f'dem_window_{cell}m', f'Idrija windows, {cell} m cells',
                        f'960–3840 m windows of the Idrija LiDAR swath that are at least 60% '
                        f'surveyed, at {cell} m cells',
                        'results/x7_dem_windows/x7_dem_windows.csv (X7)',
                        lambda d, cell=cell: x7_at(d, cell))


TERRAIN_CLASSES: dict = {c.key: c for c in (
    *[_synthetic_class(p, a) for p in (2, 3, 4, 6, 8) for a in (20.0, 60.0, 150.0)],
    *[_hilly_class(p, a) for (p, a) in HILLY_NAMES],
    *[_dem_class('whole', c) for c in ('20', '10', '5', '3')],
    *[_dem_class('window', c) for c in ('20', '10', '5', '3', '1')],
)}


def class_slope_gon(d: Data, tc: TerrainClass) -> float:
    rows = tc.select(d)
    values = [gon_from_percent(r['terrain_slope_percent_p50']) for r in rows
              if r.get('terrain_slope_percent_p50') is not None]
    if not values:
        raise Missing(f'no slope for terrain class {tc.key}')
    return statistics.median(values)


def class_description(d: Data, tc: TerrainClass) -> str:
    rows = tc.select(d)
    if not rows:
        raise Missing(f'no rows for terrain class {tc.key}')
    gon = class_slope_gon(d, tc)
    lo, hi = STAS_HILLY_GON
    stas = ('inside' if lo <= gon <= hi else 'below' if gon < lo else 'above')
    return (f'{tc.looks_like[0].upper()}{tc.looks_like[1:]}; median terrain slope '
            f'{gon:.1f} gon, {stas} the STAS 863-85 hilly class ({lo:g}–{hi:g} gon); '
            f'{len(rows)} scenarios, from `{tc.source}`.')


def _register_terrain_class_facts() -> None:
    for key, tc in TERRAIN_CLASSES.items():
        FACTS[f'class_space_{key}'] = lambda d, tc=tc: ms(
            column(tc.select(d), 'search_space_percent_of_full'), 2)
        FACTS[f'class_obj_{key}'] = lambda d, tc=tc: ms(
            column(tc.select(d), 'objective_ratio_to_full'), 3)
        FACTS[f'class_slope_{key}'] = lambda d, tc=tc: f'{class_slope_gon(d, tc):.1f}'


_register_terrain_class_facts()


def _synthetic_class_medians(d: Data) -> list:
    out = []
    for tc in TERRAIN_CLASSES.values():
        if tc.key.startswith(('syn_', 'hil_')):
            out.append((statistics.median(finite(column(tc.select(d),
                                                          'search_space_percent_of_full'))), tc))
    return sorted(out, key=lambda pair: pair[0])


@fact('x3_space_class_max')
def _(d):
    value, tc = _synthetic_class_medians(d)[-1]
    return f'{value:.1f}% (median) on the “{tc.name}” class'


@fact('x3_space_class_min')
def _(d):
    value, tc = _synthetic_class_medians(d)[0]
    return f'{value:.1f}% on the “{tc.name}” class'


@fact('x3_space_class_range')
def _(d):
    meds = _synthetic_class_medians(d)
    return f'{meds[0][0]:.1f}–{meds[-1][0]:.1f}%'


@fact('x3_class_slope_max')
def _(d):
    return f"{max(class_slope_gon(d, tc) for k, tc in TERRAIN_CLASSES.items() if k.startswith('syn_')):.1f}"


@fact('x3_class_slope_min')
def _(d):
    return f"{min(class_slope_gon(d, tc) for k, tc in TERRAIN_CLASSES.items() if k.startswith('syn_')):.1f}"


@fact('x3_n_syn_classes')
def _(d):
    return str(sum(1 for k in TERRAIN_CLASSES if k.startswith('syn_')))


@fact('x3_n_syn_classes_below_hilly')
def _(d):
    lo = STAS_HILLY_GON[0]
    return str(sum(1 for k, tc in TERRAIN_CLASSES.items()
                   if k.startswith('syn_') and class_slope_gon(d, tc) < lo))


# -- STAS (X8) ----------------------------------------------------------------------------------

def stas_rows(d: Data, *, path_found_only: bool = True) -> list:
    rows = need(d.stas, 'results/stas/stas.csv')
    return [r for r in rows if r.get('path_found') is True] if path_found_only else rows


def stas_ok(row: dict) -> bool:
    """Every applicable parameter the study checks (`core.checks.STATION_CHECK_NAMES`)
    passes; recomputed here so a column of an older run that the study no longer checks is
    never counted."""
    from core.checks import STATION_CHECK_NAMES
    return (row.get('path_found') is True and
            all(row.get(f'stas_{name}_status') != 'fail' for name in STATION_CHECK_NAMES))


@fact('stas_n_rows')
def _(d):
    return thousands(len(need(d.stas, 'results/stas/stas.csv')))


@fact('stas_n_no_path')
def _(d):
    rows = need(d.stas, 'results/stas/stas.csv')
    n = sum(1 for r in rows if r.get('path_found') is not True)
    return f'{n} of {len(rows)}'


@fact('stas_feasible_share')
def _(d):
    rows = stas_rows(d)
    n = sum(1 for r in rows if stas_ok(r))
    return f'{n} of {len(rows)} ({pct(n / len(rows), 1)})'


@fact('stas_feasible_share_baseline')
def _(d):
    rows = [r for r in stas_rows(d) if r.get('smoothing_arm') == 'none']
    n = sum(1 for r in rows if stas_ok(r))
    return f'{n} of {len(rows)}'


def _stas_feasible_share_by_iteration(d: Data, iterations: int) -> str:
    rows = [r for r in stas_rows(d) if r.get('smoothing_arm') != 'none'
            and r.get('algorithm_1_iterations') == iterations]
    n = sum(1 for r in rows if stas_ok(r))
    return f'{n} of {len(rows)} ({pct(n / len(rows), 1)})'


@fact('stas_feasible_iteration1')
def _(d):
    return _stas_feasible_share_by_iteration(d, 1)


@fact('stas_feasible_iteration2')
def _(d):
    return _stas_feasible_share_by_iteration(d, 2)


@fact('stas_best_arm')
def _(d):
    rows = [r for r in stas_rows(d) if r.get('smoothing_arm') != 'none']
    from collections import defaultdict
    by_arm: dict = defaultdict(lambda: [0, 0])
    for r in rows:
        by_arm[r['smoothing_arm']][1] += 1
        if stas_ok(r):
            by_arm[r['smoothing_arm']][0] += 1
    best = max(by_arm, key=lambda a: by_arm[a][0] / by_arm[a][1])
    n, total = by_arm[best]
    return f'`{best}` ({n} of {total}, {pct(n / total, 1)})'


# ==============================================================================================
# facts
# ==============================================================================================

ARMS = {'yellow': 'hag_yellow', 'k1': 'hag_yellow_k1', 'full': 'full_grid'}


def _register_arm_facts() -> None:
    for short, arm in ARMS.items():
        FACTS[f'space_{short}'] = lambda d, a=arm: ms(column(by_arm(search_rows(d), a),
                                                            'search_space_percent_of_full'), 2)
        FACTS[f'ratio_{short}'] = lambda d, a=arm: ms(nonflat_ratio(d, a), 3)
        FACTS[f'absdiff_{short}'] = lambda d, a=arm: ms(abs_diff(d, a), 2)
        FACTS[f'length_ratio_{short}'] = lambda d, a=arm: ms(
            column(by_arm(search_rows(d), a), 'length_ratio_to_baseline'), 3)
        FACTS[f'length_ratio_{short}_median'] = lambda d, a=arm: med(
            column(by_arm(search_rows(d), a), 'length_ratio_to_baseline'), 3)
        FACTS[f'hausdorff_{short}'] = lambda d, a=arm: ms(
            column(by_arm(search_rows(d), a), 'vs_baseline_hausdorff_m'), 1)
        FACTS[f'hausdorff_{short}_median'] = lambda d, a=arm: med(
            column(by_arm(search_rows(d), a), 'vs_baseline_hausdorff_m'), 1)
        FACTS[f'time_red_{short}'] = lambda d, a=arm: ms(paired_reduction(d, a, False), 1)
        FACTS[f'time_red_incl_{short}'] = lambda d, a=arm: ms(paired_reduction(d, a, True), 1)
        FACTS[f'mem_ratio_{short}'] = lambda d, a=arm: ms(paired_ratio(d, a, 'peak_memory_bytes'), 2)
        FACTS[f'nodes_ratio_{short}'] = lambda d, a=arm: ms(paired_ratio(d, a, 'search_nodes_expanded'), 2)


_register_arm_facts()


@fact('space_yellow_range')
def _(d):
    return rng(column(by_arm(search_rows(d), 'hag_yellow'), 'search_space_percent_of_full'), 2)


@fact('space_yellow_median')
def _(d):
    return med(column(by_arm(search_rows(d), 'hag_yellow'), 'search_space_percent_of_full'), 2)


@fact('reduction_yellow')
def _(d):
    return ms(column(by_arm(search_rows(d), 'hag_yellow'), 'search_space_reduction_percent'), 2)


@fact('n_od_scenarios')
def _(d):
    return str(len({(r['terrain_id'], r['od_id']) for r in search_rows(d)}))


@fact('n_scenarios')
def _(d):
    pairs = {(r['terrain_id'], r['od_id']) for r in search_rows(d)}
    classes = {r['road_class'] for r in geom_rows(d) if r.get('road_class')}
    return str(len(pairs) * len(classes))


@fact('worse_share')
def _(d):
    rows = geom_rows(d)
    k = sum(1 for r in rows if r.get('smoothing_made_i_max_worse') is True)
    return f'{100.0 * k / len(rows):.0f}% ({k} of {len(rows)})'


@fact('degenerate_s')
def _(d):
    return str(sum(1 for r in geom_rows(d) if r.get('s_degenerate') is True))


@fact('degenerate_s_by_class')
def _(d):
    rows = geom_rows(d)
    out = []
    for name in sorted({r['road_class'] for r in rows}):
        k = sum(1 for r in rows if r['road_class'] == name and r.get('s_degenerate') is True)
        out.append(f'{k} for `{name}`' if k else f'none for `{name}`')
    return join_and(out)


@fact('rmin_inf_note')
def _(d):
    rows = geom_rows(d)
    out = []
    for name in sorted({r['road_class'] for r in rows}):
        group = [r for r in rows if r['road_class'] == name]
        k = sum(1 for v in censored(column(group, 'after_r_min_m')) if v == math.inf)
        if k:
            out.append(f'{k} of the {len(group)} smoothed radii of `{name}`')
    if not out:
        return 'no smoothed radius was unbounded'
    return join_and(out) + ' ' + ('is' if len(out) == 1 else 'are') + ' unbounded'


@fact('ratio_excluded')
def _(d):
    from core.summary import _baseline_objective_by_scenario, _is_flat_baseline
    rows = search_rows(d)
    base = _baseline_objective_by_scenario(rows)
    yellow = by_arm(rows, 'hag_yellow')
    return str(sum(1 for r in yellow if _is_flat_baseline(r, base)))


@fact('timing_repeats')
def _(d):
    v = finite(column(need(d.matrix, 'results/matrix.csv'), 'timing_repeats'))
    return str(int(max(v)))


@fact('epsilon')
def _(d):
    from core.costs import TIE_BREAK_PER_METER
    exp = int(round(math.log10(TIE_BREAK_PER_METER)))
    sup = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
    return '10' + str(exp).translate(sup)


@fact('epsilon_tex')
def _(d):
    from core.costs import TIE_BREAK_PER_METER
    return f'10^{{{int(round(math.log10(TIE_BREAK_PER_METER)))}}}'


@fact('step_m')
def _(d):
    from core.metrics import DEFAULT_RESAMPLE_STEP_M
    return f'{DEFAULT_RESAMPLE_STEP_M:g}'


@fact('chord_m')
def _(d):
    from core.metrics import DEFAULT_CHORD_M
    return f'{DEFAULT_CHORD_M:g}'


def _first(d, key):
    rows = need(d.matrix, 'results/matrix.csv')
    return rows[0].get(key)


@fact('python_version')
def _(d):
    return str(_first(d, 'python'))


@fact('numpy_version')
def _(d):
    return str(_first(d, 'numpy'))


@fact('scipy_version')
def _(d):
    return str(_first(d, 'scipy'))


@fact('platform')
def _(d):
    return str(_first(d, 'platform'))


@fact('matrix_git_sha')
def _(d):
    return str(_first(d, 'git_sha'))[:10]


@fact('matrix_dirty')
def _(d):
    dirty = _first(d, 'git_dirty')
    return 'with uncommitted changes in the working tree' if dirty is True else 'from a clean tree'


# -- gradient study ----------------------------------------------------------------------------

@fact('guard_fired')
def _(d):
    return str(sum(1 for r in need(d.gradient, 'results/gradient/gradient.csv')
                   if r.get('smoothing_rejected') is True))


@fact('guard_found')
def _(d):
    return str(sum(1 for r in need(d.gradient, 'results/gradient/gradient.csv')
                   if r.get('path_found') is True))


@fact('guard_rmin_fail')
def _(d):
    rej = [r for r in need(d.gradient, 'results/gradient/gradient.csv')
           if r.get('smoothing_rejected') is True]
    return str(sum(1 for r in rej if r.get('check_r_min_passed') is False))


# -- real DEM ------------------------------------------------------------------------------------

@fact('census_n_sizes')
def _(d):
    return {3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight'}.get(
        len(need(d.census, 'results/dem/census.csv')), str(len(d.census)))


@fact('census_overapproval')
def _(d):
    v = [r['over_approval_percent'] for r in need(d.census, 'results/dem/census.csv')
         if r.get('over_approval_percent') is not None]
    if not v:
        raise Missing('census was run without --verify-sample')
    return f'{min(v):.1f}–{max(v):.1f}%'


@fact('census_max_square_m')
def _(d):
    sizes = [r['size_m'] for r in need(d.census, 'results/dem/census.csv') if r['n_candidates'] > 0]
    return thousands(max(sizes))


@fact('dem_width')
def _(d):
    return thousands(need(d.census, 'results/dem/census.csv')[0]['raster_width'])


@fact('dem_height')
def _(d):
    return thousands(need(d.census, 'results/dem/census.csv')[0]['raster_height'])


@fact('dem_valid_percent')
def _(d):
    return f"{100.0 * float(need(d.census, 'results/dem/census.csv')[0]['valid_fraction']):.1f}"


def _ladder(d, cell_m):
    for r in need(d.ladder, 'results/dem/ladder.csv'):
        if abs(float(r['cell_size_m']) - cell_m) < 1e-9:
            return r
    raise Missing(f'no ladder row for {cell_m} m')


@fact('ladder_components_1m')
def _(d):
    return thousands(_ladder(d, 1.0)['n_components'])


@fact('ladder_components_20m')
def _(d):
    return thousands(_ladder(d, 20.0)['n_components'])


@fact('ladder_largest_1m')
def _(d):
    return pct(float(_ladder(d, 1.0)['largest_component_fraction']), 1)


@fact('ladder_largest_20m')
def _(d):
    return pct(float(_ladder(d, 20.0)['largest_component_fraction']), 1)


@fact('ladder_frac_range')
def _(d):
    v = [float(r['admissible_edge_fraction']) for r in need(d.ladder, 'results/dem/ladder.csv')]
    return f'{100 * min(v):.0f}–{100 * max(v):.0f}%'


def _swath_first(d, scenario: str, key: str):
    for r in need(d.swath, 'results/dem/swath.csv'):
        if str(r['scenario_id']).startswith(scenario + '__') and r['arm'] == 'full_grid':
            return r[key]
    raise Missing(f'no swath row for {scenario}')


FACTS['swath_areas_20'] = lambda d: thousands(_swath_first(d, 'swath_cell20', 'n_areas_total'))
FACTS['swath_areas_10'] = lambda d: thousands(_swath_first(d, 'swath_cell10', 'n_areas_total'))
FACTS['swath_areas_5'] = lambda d: thousands(_swath_first(d, 'swath_cell5', 'n_areas_total'))
FACTS['hd_areas_5'] = lambda d: thousands(_swath_first(d, 'swath_cell10_hd5', 'n_areas_total'))
FACTS['hd_areas_40'] = lambda d: thousands(_swath_first(d, 'swath_cell10_hd40', 'n_areas_total'))


@fact('dem_height_delta')
def _(d):
    return f"{float(_swath_first(d, 'swath_cell10', 'height_delta_m')):g}"


@fact('pair_candidates')
def _(d):
    from core.experiment_dem import find_pair
    n = inspect.signature(find_pair).parameters['max_candidates'].default
    return thousands(n)


def _tile_groups(d) -> dict:
    groups: dict = {}
    for r in need(d.tiles, 'results/dem_tiles/tiles.csv'):
        groups.setdefault(str(r['scenario_id']).split('__')[0], {})[r['arm']] = r
    return groups


@fact('tiles_n')
def _(d):
    return str(len(_tile_groups(d)))


@fact('tiles_screened')
def _(d):
    return str(sum(1 for g in _tile_groups(d).values()
                   if next(iter(g.values())).get('gradient_screened') is True))


@fact('tiles_space_yellow')
def _(d):
    return ms([g['hag_yellow']['search_space_percent_of_full'] for g in _tile_groups(d).values()
               if 'hag_yellow' in g], 1)


@fact('tiles_i_max_median_full')
def _(d):
    return med([g['full_grid'].get('before_i_max_percent') for g in _tile_groups(d).values()
                if 'full_grid' in g], 0)


def _run_config(pattern: str, group: int) -> str:
    script = HERE / 'run_experiments.sh'
    if not script.exists():
        raise Missing('article_2/run_experiments.sh not found')
    match = re.search(pattern, script.read_text(encoding='utf-8'))
    if not match:
        raise Missing(f'pattern {pattern!r} not found in run_experiments.sh')
    return match.group(group)


FACTS['tiles_requested'] = lambda d: _run_config(r'--mode tiles[^\n]*?--n (\d+)', 1)
FACTS['tiles_seed'] = lambda d: _run_config(r'--mode tiles[^\n]*?--seed (\d+)', 1)


def _b23(d) -> dict:
    rows = need(d.b23, 'results/dem/b23_case.csv')
    if not rows:
        raise Missing('empty b23_case.csv')
    return rows[0]


FACTS['b23_cert_gradient'] = lambda d: f"{float(_b23(d)['certified_i_max_percent']):.2f}"
FACTS['b23_cert_length'] = lambda d: thousands(_b23(d)['certified_length_m'])
FACTS['b23_after_length'] = lambda d: thousands(_b23(d)['after_length_m'])
FACTS['b23_after_gradient'] = lambda d: f"{float(_b23(d)['after_i_max_percent']):.2f}"


def _network(d) -> list:
    rows = need(d.network, 'results/network/network.csv')
    if not rows:
        raise Missing('empty network.csv')
    return rows


def _break_even(row) -> Optional[float]:
    value = row.get('break_even_n_queries')
    if value in (None, ''):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _query_speedup(row) -> float:
    return float(row['full_grid_query_s_mean']) / float(row['overlay_query_s_mean'])


@fact('network_configs')
def _(d):
    return str(len(_network(d)))


@fact('network_never_count')
def _(d):
    rows = _network(d)
    return f'{sum(1 for r in rows if _break_even(r) is None)} of {len(rows)}'


@fact('network_never_spacings')
def _(d):
    cells = sorted(float(r['spacing_m']) / float(r['cell_size_m'])
                   for r in _network(d) if _break_even(r) is None)
    if not cells:
        raise Missing('the overlay paid for its build in every configuration')
    return join_and([f'{c:g}' for c in cells]) + ' cells'


@fact('network_break_even_range')
def _(d):
    v = [_break_even(r) for r in _network(d) if _break_even(r) is not None]
    if not v:
        raise Missing('no configuration paid for its build')
    return f'{thousands(min(v))} to {thousands(max(v))}'


@fact('network_quality_range')
def _(d):
    v = finite(column(_network(d), 'quality_ratio_mean'))
    return f'{min(v):.2f} to {max(v):.2f}'


@fact('network_speedup_max')
def _(d):
    return f'{max(_query_speedup(r) for r in _network(d)):.1f}'


@fact('network_build_range')
def _(d):
    v = finite(column(_network(d), 'build_total_s'))
    return f'{min(v):.0f} to {max(v):.0f}'


@fact('network_queries')
def _(d):
    v = sorted({int(float(r['n_queries'])) for r in _network(d)})
    return str(v[0]) if len(v) == 1 else f'{v[0]} to {v[-1]}'


@fact('network_full_query_range')
def _(d):
    v = [1000.0 * x for x in finite(column(_network(d), 'full_grid_query_s_mean'))]
    return f'{min(v):.0f} to {max(v):.0f}'


@fact('network_terrains')
def _(d):
    return join_and([f'`{t}`' for t in dict.fromkeys(r['terrain_id'] for r in _network(d))])


@fact('network_abstract')
def _(d):
    never = sum(1 for r in _network(d) if _break_even(r) is None)
    finite_be = [_break_even(r) for r in _network(d) if _break_even(r) is not None]
    if not finite_be:
        return 'The portal overlay did not pay for its build at any spacing tested.'
    tail = (f", and at spacings of {FACTS['network_never_spacings'](d)} it never did"
            if never else '')
    return (f"The portal overlay answered a query up to {FACTS['network_speedup_max'](d)} times "
            f"faster than a full-grid search but repaid its build only after "
            f"{FACTS['network_break_even_range'](d)} queries, at a mean route cost of "
            f"{FACTS['network_quality_range'](d)} times the full-grid optimum{tail}.")


# -- the tile population's audit ----------------------------------------------------------------

@fact('tiles_dropped_n')
def _(d):
    return str(len(need(d.tiles_dropped, 'results/dem_tiles/tiles_dropped.csv')))


@fact('tiles_sampled')
def _(d):
    dropped = len(need(d.tiles_dropped, 'results/dem_tiles/tiles_dropped.csv'))
    return str(int(FACTS['tiles_n'](d)) + dropped)


@fact('tiles_dropped_reason')
def _(d):
    rows = need(d.tiles_dropped, 'results/dem_tiles/tiles_dropped.csv')
    if not rows:
        return 'none was dropped'
    kinds = []
    for r in rows:
        reason = str(r.get('reason', ''))
        kinds.append('the exact read found nodata that the decimated scan had missed'
                     if 'nodata' in reason else
                     'no origin\u2013destination pair fitted inside the margin')
    counts = {k: kinds.count(k) for k in dict.fromkeys(kinds)}
    if len(counts) == 1:
        return f'because {next(iter(counts))}'
    return join_and([f'{n} because {k}' for k, n in counts.items()])


# -- sentences built from the matrix -----------------------------------------------------------

def _classes(rows) -> list:
    return sorted({r['road_class'] for r in rows if r.get('road_class')})


def _count(rows, key) -> int:
    return sum(1 for r in rows if r.get(key) is True)


@fact('feasible_sentence')
def _(d):
    rows = geom_rows(d)
    parts = []
    for name in _classes(rows):
        g = [r for r in rows if r['road_class'] == name]
        k = _count(g, 'feasible')
        parts.append(f'{k} of {len(g)} for `{name}`' if k else f'none of {len(g)} for `{name}`')
    return join_and(parts)


@fact('rmin_pass_sentence')
def _(d):
    rows = geom_rows(d)
    parts = []
    for name in _classes(rows):
        g = [r for r in rows if r['road_class'] == name]
        parts.append(f"{_count(g, 'check_r_min_passed')} of {len(g)} for `{name}`")
    return join_and(parts)


@fact('iv_tradeoff_sentence')
def _(d):
    g = [r for r in geom_rows(d) if r['road_class'] == 'RO_CLASS_IV_DEAL']
    if not g:
        raise Missing('no RO_CLASS_IV_DEAL rows')
    rp, ip, both = (_count(g, 'check_r_min_passed'), _count(g, 'check_i_max_passed'),
                    _count(g, 'feasible'))
    if both == 0 and rp and ip:
        return (f'the {rp} rows that meet its radius and the {ip} rows that meet its gradient are '
                f'disjoint, so no row meets both — a trade-off failure in every row.')
    return f'{both} rows meet both limits ({rp} meet the radius and {ip} the gradient).'


def _acceptance(d):
    from data.configs.road_classes import get
    rows = geom_rows(d)
    names = sorted(_classes(rows), key=lambda n: -get(n).r_min_m)
    meds = []
    for name in names:
        g = [r for r in rows if r['road_class'] == name]
        meds.append((statistics.median(finite(censored(column(g, 'after_r_min_m')))),
                     statistics.median(finite(column(g, 'deviation_hausdorff_m'))),
                     statistics.median(finite(column(g, 'after_i_max_percent')))))

    def monotone(idx, increasing):
        seq = [m[idx] for m in meds]
        return all((a <= b) if increasing else (a >= b) for a, b in zip(seq, seq[1:]))
    return names, meds, (monotone(0, False), monotone(1, False), monotone(2, True))


@fact('acceptance_sentence')
def _(d):
    _, _, flags = _acceptance(d)
    if all(flags):
        return ('as the class loosens the smoothed radius and the deviation fall and the smoothed '
                'gradient rises, in the direction the constraint asks for.')
    wrong = [n for n, ok in zip(('the smoothed radius', 'the deviation', 'the smoothed gradient'),
                                flags) if not ok]
    return f'**not every column moved as expected: {join_and(wrong)} did not.**'


@fact('space_yellow_by_terrain')
def _(d):
    rows = by_arm(search_rows(d), 'hag_yellow')
    parts = []
    for terrain in sorted({r['terrain_id'] for r in rows}):
        g = [r for r in rows if r['terrain_id'] == terrain]
        parts.append(f"`{terrain}` {statistics.fmean(finite(column(g, 'search_space_percent_of_full'))):.1f}%")
    return join_and(parts)


@fact('no_path_count')
def _(d):
    rows = search_rows(d)
    return f"{sum(1 for r in rows if r.get('path_found') is False)} of {len(rows)}"


@fact('time_incl_slower_yellow')
def _(d):
    v = paired_reduction(d, 'hag_yellow', True)
    return f'{sum(1 for x in v if x < 0)} of {len(v)}'


@fact('full_time_ms')
def _(d):
    v = [1000.0 * t for t in column(by_arm(search_rows(d), 'full_grid'), 'wall_time_s_median')]
    return ms(v, 0)


@fact('grid_cells')
def _(d):
    rows = by_arm(search_rows(d), 'full_grid')
    sizes = {int(r['n_cells_total']) for r in rows if r.get('n_cells_total') is not None}
    if len(sizes) != 1:
        raise Missing('the synthetic terrains do not share one size')
    return thousands(sizes.pop())


# -- sentences built from the gradient study ----------------------------------------------------

def _gradient_pairs(d):
    rows = need(d.gradient, 'results/gradient/gradient.csv')
    idx: dict = {}
    for r in rows:
        idx.setdefault((r['terrain_id'], r['od_id'], r['road_class']), {})[r['arm']] = r
    return rows, idx


@fact('gradcut_not_found')
def _(d):
    rows, _ = _gradient_pairs(d)
    g = [r for r in rows if r['arm'] == 'hag_yellow_gradcut']
    return f"{sum(1 for r in g if r.get('path_found') is not True)} of {len(g)}"


@fact('gradpen_delta')
def _(d):
    _, idx = _gradient_pairs(d)
    diffs = [a['hag_yellow_gradpen']['rough_i_max_percent'] - a['hag_yellow']['rough_i_max_percent']
             for a in idx.values()
             if a.get('hag_yellow', {}).get('path_found') is True
             and a.get('hag_yellow_gradpen', {}).get('path_found') is True]
    if not diffs:
        raise Missing('no paired rows')
    return f'{statistics.fmean(diffs):.2f}'.replace('-', '−')


def _gradient_arm_rows(d, arm: str) -> list:
    rows, _ = _gradient_pairs(d)
    return [r for r in rows if r['arm'] == arm]


@fact('gradient_pairs_n')
def _(d):
    return str(len(_gradient_arm_rows(d, 'hag_yellow')))


@fact('gradient_i_max_pass_yellow')
def _(d):
    return str(_count(_gradient_arm_rows(d, 'hag_yellow'), 'check_i_max_passed'))


@fact('gradient_i_max_pass_gradpen')
def _(d):
    return str(_count(_gradient_arm_rows(d, 'hag_yellow_gradpen'), 'check_i_max_passed'))


@fact('gradpen_flips')
def _(d):
    _, idx = _gradient_pairs(d)
    pairs = [(a['hag_yellow'].get('feasible'), a['hag_yellow_gradpen'].get('feasible'))
             for a in idx.values()
             if a.get('hag_yellow', {}).get('path_found') is True
             and a.get('hag_yellow_gradpen', {}).get('path_found') is True]
    return f"{sum(1 for x, y in pairs if x != y)} of {len(pairs)}"


@fact('guard_rough_rmin_range')
def _(d):
    rej = [r for r in need(d.gradient, 'results/gradient/gradient.csv')
           if r.get('smoothing_rejected') is True]
    return rng(column(rej, 'rough_r_min_m'), 1)


@fact('class_rmin_range')
def _(d):
    from data.configs.road_classes import PROPOSED_MATRIX_CLASSES, get
    v = [get(n).r_min_m for n in PROPOSED_MATRIX_CLASSES]
    return f'{min(v):g}–{max(v):g}'


@fact('census_first_empty_m')
def _(d):
    empty = [r['size_m'] for r in need(d.census, 'results/dem/census.csv') if r['n_candidates'] == 0]
    if not empty:
        raise Missing('every size tested has candidates')
    return thousands(min(empty))


# -- dynamic facts: swath and tiles ---------------------------------------------------------------

SHORT_ARMS = {'full': 'full_grid', 'yellow': 'hag_yellow', 'k1': 'hag_yellow_k1',
              'gradcut': 'hag_yellow_gradcut',
              'gradpen': 'hag_yellow_gradpen'}
SWATH_RE = re.compile(r'^swath(\d+)(?:hd(\d+))?_(full|yellow|k1|gradcut|gradpen)_'
                      r'(space|nodes|length|length_ratio|i_before|i_after|time|build)$')
SWATH_AREAS_RE = re.compile(r'^swath(\d+)(?:hd(\d+))?_areas$')
TILES_RE = re.compile(r'^tiles_(full|yellow|k1|gradcut|gradpen)_'
                      r'(space|length_ratio|i_before_median|i_after_median|feasible|found|time|build)$')


def _swath_scenario(cell: str, hd: Optional[str]) -> str:
    return f'swath_cell{cell}' + (f'_hd{hd}' if hd else '')


def _swath_row(d, cell, hd, arm):
    for r in _swath_rows(d, _swath_scenario(cell, hd)):
        if r['arm'] == arm:
            return r
    raise Missing(f'no swath row for {_swath_scenario(cell, hd)} / {arm}')


def dynamic_fact(name: str, d: Data) -> Optional[str]:
    m = SWATH_AREAS_RE.match(name)
    if m:
        return thousands(_swath_row(d, m.group(1), m.group(2), 'full_grid')['n_areas_total'])
    m = SWATH_RE.match(name)
    if m:
        cell_, hd, short, metric = m.groups()
        row = _swath_row(d, cell_, hd, SHORT_ARMS[short])
        if metric == 'space':
            return f"{float(row['search_space_percent_of_full']):.2f}"
        if metric == 'nodes':
            return thousands(row['search_nodes_expanded'])
        if metric == 'time':
            return f"{float(row['wall_time_s']):.1f}"
        if metric == 'build':
            return f"{float(row['hag_build_time_s']):.1f}"
        if metric in ('length', 'length_ratio'):
            if row.get('path_length_m') is None:
                raise Missing('no path was found')
            if metric == 'length':
                return thousands(row['path_length_m'])
            base = _swath_row(d, cell_, hd, 'full_grid')
            return f"{float(row['path_length_m']) / float(base['path_length_m']):.2f}"
        key = 'before_i_max_percent' if metric == 'i_before' else 'after_i_max_percent'
        if row.get(key) is None:
            raise Missing('no path was found')
        return f'{float(row[key]):.0f}'
    m = TILES_RE.match(name)
    if m:
        short, metric = m.groups()
        arm = SHORT_ARMS[short]
        groups = _tile_groups(d)
        rows = [g[arm] for g in groups.values() if arm in g]
        found = [r for r in rows if r.get('path_found') is True]
        if metric == 'space':
            return ms(column(rows, 'search_space_percent_of_full'), 1)
        if metric == 'length_ratio':
            out = []
            for g in groups.values():
                base, r = g.get('full_grid'), g.get(arm)
                if base and r and r.get('path_length_m') is not None and base.get('path_length_m'):
                    out.append(float(r['path_length_m']) / float(base['path_length_m']))
            return ms(out, 2)
        if metric == 'i_before_median':
            return med(column(found, 'before_i_max_percent'), 0)
        if metric == 'i_after_median':
            return med(column(found, 'after_i_max_percent'), 0)
        if metric == 'time':
            return ms(column(rows, 'wall_time_s'), 3)
        if metric == 'build':
            return ms(column(rows, 'hag_build_time_s'), 3)
        if metric == 'feasible':
            return f"{_count(found, 'feasible')} of {len(rows)}"
        return f'{len(found)} of {len(rows)}'
    return None


@fact('swath_gradcut_found')
def _(d):
    rows = [r for r in need(d.swath, 'results/dem/swath.csv') if r['arm'] == 'hag_yellow_gradcut']
    return str(_count(rows, 'path_found'))


@fact('swath_scenarios')
def _(d):
    return str(len({str(r['scenario_id']).split('__')[0]
                    for r in need(d.swath, 'results/dem/swath.csv')}))


# ==============================================================================================
# tables
# ==============================================================================================

@dataclass
class Table:
    key: str
    caption: str
    header: list
    rows: list
    note: str = ''


TABLES: dict = {}


def table(key: str):
    def register(fn):
        TABLES[key] = fn
        return fn
    return register


STAT_NOTE = 'Cells: mean ± SD (median) [min–max].'


@table('terrains')
def _(d):
    from core.experiment_matrix import HEIGHT_DELTA_M, TERRAINS
    from core.hag import build_hag
    from core.terrain import generate_terrain
    rows = []
    for case in TERRAINS:
        grid = generate_terrain(case.spec)
        hag = build_hag(grid, HEIGHT_DELTA_M)
        rows.append([f'`{case.terrain_id}`', case.spec.seed,
                     f'{case.spec.periods[0]} × {case.spec.periods[1]}',
                     f'{case.spec.height_interval[0]:.0f}–{case.spec.height_interval[1]:.0f}',
                     f'{grid.surf.min():.1f}–{grid.surf.max():.1f}', hag.n_areas, case.rationale])
    return Table('terrains', 'The three synthetic terrains of the robustness matrix (100 × 100 cells '
                 'of 10 m; height bands of 3 m).',
                 ['Terrain', 'Seed', 'Noise periods', 'Height interval requested (m)',
                  'Height range obtained (m)', 'Height areas', 'Why this terrain'], rows,
                 'The range obtained differs from the interval requested because the noise runs '
                 'in [−1, 1] before scaling.')


@table('classes')
def _(d):
    from data.configs.road_classes import PROPOSED_MATRIX_CLASSES, get
    rows = []
    for name in list(PROPOSED_MATRIX_CLASSES) + ['RO_CLASS_V_DEAL']:
        rc = get(name)
        rows.append([f'`{name}`', rc.technical_class, rc.terrain_category,
                     f'{rc.design_speed_kmh:g}', f'{rc.r_min_m:g}', f'{rc.i_max_percent:g}',
                     '–' if rc.r_min_serpentine_m is None else f'{rc.r_min_serpentine_m:g}',
                     rc.status, 'real DEM' if name == 'RO_CLASS_V_DEAL' else 'synthetic matrix'])
    return Table('classes', 'The road classes used. Every value is transcribed from Ordin MT '
                 '1296/2017 and is awaiting sign-off by the road-design co-author.',
                 ['Class', 'Technical class', 'Terrain', 'Design speed (km/h)', 'R_min (m)',
                  'i_max (%)', 'Serpentine R_min (m)', 'Status', 'Used in'], rows,
                 'Sources: Tabelul nr. 1 a) (design speeds), Tabelul nr. 1 b) for the reduced '
                 'speed of `RO_CLASS_V_DEAL_REDUS` (NOTA 2.4.2 makes it conditional on the road '
                 'administrator\'s approval) and Tabelul nr. 2 B) (radii and gradients).')


@table('search_cost')
def _(d):
    rows = search_rows(d)
    out = []
    for arm in ('full_grid', 'hag_yellow', 'hag_yellow_k1'):
        g = by_arm(rows, arm)
        base = arm == 'full_grid'
        out.append([f'`{arm}`', cell(column(g, 'search_space_percent_of_full'), 2),
                    cell(column(g, 'search_nodes_expanded'), 0),
                    cell([v * 1000 for v in finite(column(g, 'wall_time_s_median'))], 1),
                    '–' if base else cell(paired_reduction(d, arm, False), 1),
                    '–' if base else cell(paired_reduction(d, arm, True), 1),
                    cell([v / 2 ** 20 for v in finite(column(g, 'peak_memory_bytes'))], 2)])
    return Table('search_cost', 'Cost of the search, per arm, over the nine terrain × O–D '
                 'scenarios (population P_search).',
                 ['Arm', 'Search space (% of the full grid)', 'Nodes expanded',
                  'Search time (ms)', 'Time reduction, search only (%)',
                  'Time reduction, including the HAG build (%)', 'Peak memory (MB)'], out,
                 STAT_NOTE + ' Search time is the median of repeated graph construction plus '
                 'search; "including the HAG build" adds the one-off cost of building the HAG, '
                 'which is what a single query pays.')


@table('quality')
def _(d):
    rows = search_rows(d)
    out = []
    for arm in ('hag_yellow', 'hag_yellow_k1'):
        g = by_arm(rows, arm)
        out.append([f'`{arm}`', cell(nonflat_ratio(d, arm), 3), cell(abs_diff(d, arm), 2),
                    cell(column(g, 'length_ratio_to_baseline'), 3),
                    cell(column(g, 'vs_baseline_hausdorff_m'), 1),
                    cell(column(g, 'vs_baseline_mean_deviation_m'), 1)])
    return Table('quality', 'What the reduction costs: the alignment of each arm against the '
                 'full-grid alignment of the same scenario (population P_search).',
                 ['Arm', 'Objective ratio (baseline ≥ 25 m)', 'Objective difference (m)',
                  'Length ratio', 'Hausdorff distance (m)', 'Mean deviation (m)'], out,
                 STAT_NOTE + ' The objective ratio is not used where the baseline objective is '
                 'below 25 m (Section 6.10), so it has fewer scenarios than the other columns; '
                 'the absolute difference has all nine.')


@table('engineering')
def _(d):
    rows = geom_rows(d)
    out = []
    for name in sorted({r['road_class'] for r in rows}):
        g = [r for r in rows if r['road_class'] == name]
        n = len(g)

        def count(key):
            k = sum(1 for r in g if r.get(key) is True)
            return f'{k} ({100 * k / n:.0f}%)'
        out.append([f'`{name}`', n, count('check_r_min_passed'), count('check_i_max_passed'),
                    count('feasible'), count('smoothing_made_i_max_worse'),
                    count('s_degenerate')])
    return Table('engineering', 'Engineering checks after smoothing, per road class (population '
                 'P_geom, one iteration of Algorithm 1). A class that cannot be met is reported, '
                 'never relaxed.',
                 ['Class', 'Rows', 'R_min passes', 'i_max passes', 'Feasible (both)',
                  'Smoothing made i_max worse', 'Degenerate s-search'], out,
                 'Each row is one arm × terrain × O–D combination (36 per class). All classes '
                 'are `TO CONFIRM`.')


@table('smoothing')
def _(d):
    rows = geom_rows(d)
    out = []
    for name in sorted({r['road_class'] for r in rows}):
        g = [r for r in rows if r['road_class'] == name]
        after = censored(column(g, 'after_r_min_m'))
        n_unb = sum(1 for v in after if v == math.inf)
        after_text = cell(after, 1) + (f' · {n_unb} unbounded' if n_unb else '')
        out.append([f'`{name}`', cell(censored(column(g, 'before_r_min_m')), 1), after_text,
                    cell(column(g, 'before_i_max_percent'), 2),
                    cell(column(g, 'after_i_max_percent'), 2),
                    cell(column(g, 'deviation_hausdorff_m'), 1),
                    f"{cell(column(g, 'before_sinuosity'), 2)} → {cell(column(g, 'after_sinuosity'), 2)}"])
    return Table('smoothing', 'Algorithm 1, before and after, per road class (population P_geom).',
                 ['Class', 'R_min before (m)', 'R_min after (m)', 'i_max before (%)',
                  'i_max after (%)', 'Deviation, Hausdorff (m)', 'Sinuosity, before → after'], out,
                 STAT_NOTE + ' Radii above 10⁶ m are counted as unbounded and excluded from the '
                 'statistic (Section 6.11). "Before" is the same for every class because the '
                 'search does not depend on the class.')


@table('acceptance')
def _(d):
    from data.configs.road_classes import get
    rows = geom_rows(d)
    names = sorted({r['road_class'] for r in rows}, key=lambda n: -get(n).r_min_m)
    out, meds = [], {}
    for name in names:
        g = [r for r in rows if r['road_class'] == name]
        rc = get(name)
        meds[name] = (statistics.median(finite(censored(column(g, 'after_r_min_m')))),
                      statistics.median(finite(column(g, 'deviation_hausdorff_m'))),
                      statistics.median(finite(column(g, 'after_i_max_percent'))))
        out.append([f'`{name}`', f'{rc.r_min_m:g}', f'{meds[name][0]:.1f}', f'{meds[name][1]:.1f}',
                    f'{rc.i_max_percent:g}', f'{meds[name][2]:.2f}'])

    def monotone(idx, increasing):
        seq = [meds[n][idx] for n in names]
        pairs = list(zip(seq, seq[1:]))
        return 'yes' if all((a <= b) if increasing else (a >= b) for a, b in pairs) else 'no'
    out.append(['**moves as strictness rises?**', '', monotone(0, False), monotone(1, False), '',
                monotone(2, True)])
    return Table('acceptance', 'Acceptance test of the constraint mechanism (Section 5.4.5): '
                 'classes ordered from the strictest minimum radius to the loosest.',
                 ['Class', 'R_min required (m)', 'Median R_min after (m)',
                  'Median deviation, Hausdorff (m)', 'i_max limit (%)',
                  'Median i_max after (%)'], out,
                 'A stricter R_min must give a larger radius after smoothing and a larger deviation '
                 '(both fall as the class loosens); a stricter i_max must give a lower or equal '
                 'maximum gradient (it rises as the limit loosens).')


@table('gradient')
def _(d):
    rows = need(d.gradient, 'results/gradient/gradient.csv')
    out = []
    for name in sorted({r['road_class'] for r in rows}):
        for arm in ('hag_yellow', 'hag_yellow_gradcut', 'hag_yellow_gradpen'):
            g = [r for r in rows if r['road_class'] == name and r['arm'] == arm]
            found = [r for r in g if r.get('path_found') is True]
            out.append([f'`{name}`', f'`{arm}`', f'{len(found)} of {len(g)}',
                        sum(1 for r in found if r.get('check_i_max_passed') is True),
                        sum(1 for r in found if r.get('check_r_min_passed') is True),
                        sum(1 for r in found if r.get('feasible') is True),
                        sum(1 for r in found if r.get('smoothing_rejected') is True),
                        cell(column(found, 'rough_i_max_percent'), 2),
                        cell(column(found, 'retained_i_max_percent'), 2)])
    return Table('gradient', 'Giving the search the road class (bug B21) and guarding the smoothing '
                 '(bug B23), on the nine synthetic scenarios per class.',
                 ['Class', 'Arm', 'Path found', 'i_max passes', 'R_min passes', 'Feasible',
                  'Guard fired', 'Rough-axis i_max (%)', 'Retained-axis i_max (%)'], out,
                 'Checks are on the retained axis (the smoothed one unless the guard fired). '
                 + STAT_NOTE)


@table('census')
def _(d):
    rows = need(d.census, 'results/dem/census.csv')
    out = [[f"{r['size_m']}", f"{r['stride_m']}", thousands(r['n_candidates']),
            '–' if r['relief_min_m'] is None else f"{r['relief_min_m']:.0f}–{r['relief_max_m']:.0f}",
            '–' if r['n_exact_checked'] is None else r['n_exact_checked'],
            '–' if r['over_approval_percent'] is None else f"{r['over_approval_percent']:.1f}%"]
           for r in rows]
    return Table('census', 'Census of the Idrija raster: windows of each size whose decimated '
                 'samples are all valid and whose decimated relief is at least 30 m.',
                 ['Window (m)', 'Stride (m)', 'Candidates', 'Decimated relief (m)',
                  'Read exactly', 'Failed the exact read'], out,
                 'The scan is decimated, so a void between two samples is invisible to it: the '
                 'last two columns say how often that happened on a random sample of the '
                 'candidates. The relief is a lower bound on the true relief.')


@table('ladder')
def _(d):
    rows = sorted(need(d.ladder, 'results/dem/ladder.csv'), key=lambda r: float(r['cell_size_m']))
    out = [[f"{float(r['cell_size_m']):g}", thousands(r['n_cells']),
            pct(float(r['admissible_edge_fraction']), 1), thousands(r['n_components']),
            pct(float(r['largest_component_fraction']), 1)] for r in rows]
    r0 = rows[0]
    return Table('ladder', f"Gradient feasibility against cell size on one {r0['crop_size_m']} m crop "
                 f"({r0['road_class']}, i_max = {r0['i_max_percent']:g}%).",
                 ['Cell (m)', 'Cells', 'Admissible edges', 'Components of the admissible subgraph',
                  'Largest component (% of cells)'], out,
                 'An edge is admissible if its gradient does not exceed the class limit; '
                 'components are those of the 8-connected admissible-edge subgraph over every cell.')


def _swath_rows(d, scenario: str) -> list:
    return [r for r in need(d.swath, 'results/dem/swath.csv')
            if str(r['scenario_id']).startswith(scenario + '__')]


def _swath_pair_note(d) -> str:
    flags = {r.get('gradient_screened') for r in need(d.swath, 'results/dem/swath.csv')}
    if flags == {False}:
        return ('The pair is not gradient-screened (Section 5.5): the alignments are reported '
                'as infeasible under `RO_CLASS_V_DEAL` where they are.')
    if flags == {True}:
        return 'The pair is gradient-screened (Section 5.5).'
    return 'The pairs are gradient-screened in some scenarios and not in others (column of the CSV).'


@table('swath')
def _(d):
    out = []
    for scenario in ('swath_cell20', 'swath_cell10', 'swath_cell5', 'swath_cell1'):
        rows = _swath_rows(d, scenario)
        if not rows:
            continue
        base = next((r for r in rows if r['arm'] == 'full_grid'), None)
        n_cells = (float(rows[0]['crop_n_rows']) / float(rows[0]['cell_size_m'])) * \
                  (float(rows[0]['crop_n_cols']) / float(rows[0]['cell_size_m']))
        for r in rows:
            if r['arm'] not in SHORT_ARMS.values():  # an arm the study no longer reports
                continue
            ratio = '–'
            if base and r.get('path_length_m') is not None and base.get('path_length_m'):
                ratio = f"{float(r['path_length_m']) / float(base['path_length_m']):.2f}"
            build = r.get('hag_build_time_s')
            out.append([f"{float(r['cell_size_m']):g}", thousands(n_cells), thousands(r['n_areas_total']),
                        '–' if build in (None, '') else f'{float(build):.2f}',
                        f"`{r['arm']}`", f"{float(r['search_space_percent_of_full']):.2f}",
                        thousands(r['search_nodes_expanded']), f"{float(r['wall_time_s']):.2f}",
                        '–' if r.get('path_length_m') is None else thousands(r['path_length_m']),
                        ratio,
                        '–' if r.get('before_i_max_percent') is None
                        else f"{float(r['before_i_max_percent']):.0f}",
                        '–' if r.get('after_i_max_percent') is None
                        else f"{float(r['after_i_max_percent']):.0f}",
                        {True: 'yes', False: 'no', None: '–'}[r.get('feasible')]])
    return Table('swath', 'The large real-DEM crop (the fully covered 4 km window of largest relief; a 1 km '
                 'sub-crop at 1 m), all six arms, one origin–destination pair per cell size.',
                 ['Cell (m)', 'Cells', 'Height areas', 'HAG build (s)', 'Arm', 'Search space (%)',
                  'Nodes expanded', 'Search time (s, median of 3)', 'Path length (m)',
                  'Length / full grid', 'i_max before (%)', 'i_max after (%)', 'Meets the class'],
                 out,
                 _swath_pair_note(d) + ' `–` in the length column means no path was found (the '
                 'gradient cut disconnected the yellow selection). The search time excludes the '
                 'HAG build, which is shown once per cell size and is shared by every arm. Times '
                 'are medians of three runs and indicative.')


@table('swath_hd')
def _(d):
    out = []
    for hd, scenario in ((5, 'swath_cell10_hd5'), (10, 'swath_cell10'), (20, 'swath_cell10_hd20'),
                         (40, 'swath_cell10_hd40')):
        rows = _swath_rows(d, scenario)
        if not rows:
            continue
        by = {r['arm']: r for r in rows}
        base = by.get('full_grid')

        def ratio(arm):
            r = by.get(arm)
            if not r or not base or r.get('path_length_m') is None:
                return '–'
            return f"{float(r['path_length_m']) / float(base['path_length_m']):.2f}"

        out.append([hd, thousands(base['n_areas_total']),
                    f"{float(by['hag_yellow']['search_space_percent_of_full']):.2f}", ratio('hag_yellow'),
                    f"{float(by['hag_yellow_k1']['search_space_percent_of_full']):.2f}",
                    ratio('hag_yellow_k1')])
    return Table('swath_hd', 'Sensitivity to the height band Δ on the large crop at 10 m cells.',
                 ['Δ (m)', 'Height areas', 'Yellow: search space (%)', 'Yellow: length / full grid',
                  'k = 1: search space (%)', 'k = 1: length / full grid'], out,
                 'The full-grid path is the same in every row; only the HAG changes.')


@table('tiles')
def _(d):
    groups = _tile_groups(d)
    out = []
    arms = ('full_grid', 'hag_yellow', 'hag_yellow_k1', 'hag_yellow_gradcut',
            'hag_yellow_gradpen')
    for arm in arms:
        rows = [g[arm] for g in groups.values() if arm in g]
        found = [r for r in rows if r.get('path_found') is True]

        def length_ratio(g):
            base = g.get('full_grid')
            r = g.get(arm)
            if not base or not r or r.get('path_length_m') is None or not base.get('path_length_m'):
                return None
            return float(r['path_length_m']) / float(base['path_length_m'])

        out.append([f'`{arm}`', f'{len(found)} of {len(rows)}',
                    cell(column(rows, 'search_space_percent_of_full'), 1),
                    cell(column(rows, 'search_nodes_expanded'), 0),
                    cell([length_ratio(g) for g in groups.values()], 2),
                    cell(column(found, 'before_i_max_percent'), 0),
                    cell(column(found, 'after_i_max_percent'), 0),
                    f"{sum(1 for r in found if r.get('feasible') is True)} of {len(rows)}"])
    return Table('tiles', 'The real-DEM tile population: 600 m tiles at 10 m cells, one pair per '
                 'tile, all six arms.',
                 ['Arm', 'Path found', 'Search space (%)', 'Nodes expanded',
                  'Length / full grid', 'i_max before (%)', 'i_max after (%)', 'Meet the class'], out,
                 STAT_NOTE + ' The length ratio is paired within a tile, so terrain difficulty '
                 'cancels. Only a minority of tiles have a gradient-screened pair (Section 6.15).')


@table('b23')
def _(d):
    r = _b23(d)
    return Table('b23', 'A certified real-DEM alignment before and after smoothing (bug B23).',
                 ['Crop', 'Pair', 'Class', 'Cell (m)', 'Certified length (m)', 'Certified i_max (%)',
                  'After: length (m)', 'After: i_max (%)', 'Guard'],
                 [[f"row {r['crop_row0']}, col {r['crop_col0']}, {r['crop_size_m']} m",
                   f"({r['start_i']}, {r['start_j']}) → ({r['target_i']}, {r['target_j']})",
                   f"`{r['road_class']}`", f"{float(r['cell_size_m']):g}",
                   thousands(r['certified_length_m']), f"{float(r['certified_i_max_percent']):.2f}",
                   thousands(r['after_length_m']), f"{float(r['after_i_max_percent']):.2f}",
                   'fired' if r.get('smoothing_rejected') is True else 'did not fire']],
                 'The alignment is the witness of the gradient certificate: every step holds the '
                 'limit at cell resolution. Smoothing shortens it and the heights are read again '
                 'from the terrain.')


@table('network')
def _(d):
    rows = need(d.network, 'results/network/network.csv')
    out = []
    for r in rows:
        be = r.get('break_even_n_queries')
        be_text = 'never' if be in (None, '') or not math.isfinite(float(be)) else thousands(be)
        out.append([f"`{r['terrain_id']}`", f"{r['grid_size']}", f"{float(r['spacing_m']):g}",
                    thousands(r['n_portals']), f"{float(r['build_total_s']):.1f}",
                    f"{float(r['overlay_query_s_mean']) * 1000:.2f}",
                    f"{float(r['full_grid_query_s_mean']) * 1000:.2f}", be_text,
                    '–' if r.get('quality_ratio_mean') in (None, '') else
                    f"{float(r['quality_ratio_mean']):.3f} ({float(r['quality_ratio_max']):.2f})"])
    return Table('network', 'The portal overlay: build cost, query cost, break-even and route quality.',
                 ['Terrain', 'Grid', 'Spacing (m)', 'Portals', 'Build (s)',
                  'Overlay query (ms)', 'Full-grid query (ms)', 'Break-even (queries)',
                  'Route cost / optimum: mean (worst)'], out,
                 'Break-even = build time ÷ (full-grid query time − overlay query time). Queries '
                 'are random pairs; the route is optimal only through the chosen portals, so the '
                 'ratio is at least 1.')


@table('x3_reduction')
def _(d):
    cols = []
    for tag, (space, cost) in X3_SPACES.items():
        rows = x3_group(d, space, _resolve_hag_edge_cost(space, cost))
        timed = x3_timing_group(d, space, _resolve_hag_edge_cost(space, cost))
        be = finite(column(timed, 'break_even_queries'))
        cols.append([sd(column(rows, 'search_space_percent_of_full'), 1),
                     sd(column(rows, 'objective_ratio_to_full'), 2),
                     sd(column(rows, 'length_ratio_to_full'), 2),
                     f"{med(column(rows, 'hausdorff_to_full_m'), 0)}",
                     sd(column(rows, 'nodes_expanded_ratio_to_full'), 2),
                     sd(column(timed, 'search_time_ratio_to_full'), 2),
                     f'{statistics.median(be):.0f}' if be else 'never'])
    names = ['Search domain (% of grid)', 'Objective ratio', 'Length ratio',
             'Hausdorff distance, median (m)', 'Nodes-expanded ratio', 'Search-time ratio',
             'Break-even, median (queries)']
    out = [[n, a, b] for n, a, b in zip(names, cols[0], cols[1])]
    n_scenarios = len(x3_group(d, 'hag_cta', _resolve_hag_edge_cost('hag_cta', None)))
    return Table('x3_reduction', 'Search-domain reduction and its cost, pooled over the '
                 f'{thousands(n_scenarios)} full-range synthetic scenarios. Ratios are relative '
                 'to the full-grid search of the same scenario.',
                 ['Measure', 'CTA', 'CTA + 1 ring'], out,
                 SD_NOTE + ' Times and break-even come from the serial timing sample. A query '
                 'whose restricted search is not faster than the full-grid search never repays '
                 'the HAG construction and is left out of the break-even median.')


def _stas_step(rows) -> str:
    steps = {float(r['step_m']) for r in rows if r.get('step_m') not in (None, '')}
    if len(steps) != 1:
        raise Missing(f'the STAS rows mix station spacings: {sorted(steps)}')
    return f'{steps.pop():g}'


#: the article's names for the STAS 863-85 station checks of `core.checks`
STAS_PARAMETER_NAMES = {'R_H': 'Horizontal curve radius R_H',
                        'i': 'Longitudinal gradient i',
                        'L_alignment': 'Tangent length L_alignment',
                        'R_V_concave': 'Concave vertical curve radius R_V',
                        'R_V_convex': 'Convex vertical curve radius R_V'}


@table('stas')
def _(d):
    from core.checks import STATION_CHECK_NAMES
    rows = stas_rows(d)
    out = []
    for name in STATION_CHECK_NAMES:
        applicable = [r for r in rows if r.get(f'stas_{name}_status') in ('pass', 'fail')]
        if not applicable:
            out.append([STAS_PARAMETER_NAMES.get(name, name), '0', '–', '–'])
            continue
        n_fail = sum(1 for r in applicable if r[f'stas_{name}_status'] == 'fail')
        out.append([STAS_PARAMETER_NAMES.get(name, name), thousands(len(applicable)),
                    pct(n_fail / len(applicable), 1),
                    sd(column(applicable, f'stas_{name}_violation_length_m'), 0)])
    return Table('stas', 'STAS 863-85 check at every station, over '
                 f'{thousands(len(rows))} alignments that found a path (both design speeds, '
                 'Table 2).',
                 ['Parameter', 'Alignments checked', 'Alignments failing',
                  'Failing length per alignment (m)'], out,
                 SD_NOTE.replace('scenarios of each class', 'alignments checked') + ' A station is a point of the '
                 f'axis; stations are {_stas_step(rows)} m apart along it. An alignment is not checked for a '
                 'parameter that governs none of its stations (for example, no curve between two '
                 'tangents); this is never counted as a pass.')


def sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:]


def _levels(rows, key) -> str:
    values = sorted({r.get(key) for r in rows if r.get(key) is not None})
    return join_and([f'{v:g}' if isinstance(v, (int, float)) else str(v) for v in values])


def _synthetic_pooling_note(d) -> str:
    """What a synthetic class pools, read from the rows (never typed)."""
    parts = []
    for label, keys in (('Each full-range class', 'syn_'), ('Each hilly class', 'hil_')):
        rows = [r for k, tc in TERRAIN_CLASSES.items() if k.startswith(keys) for r in tc.select(d)]
        if not rows:
            continue
        seeds = len({r.get('terrain_seed') for r in rows})
        parts.append(f'{label} pools {seeds} terrain seeds, grids of '
                     f'{_levels(rows, "grid_cols")} cells, height intervals of '
                     f'{_levels(rows, "height_delta_m")} m and three O–D pairs '
                     'per terrain.')
    return ' '.join(parts)


#: short article IDs of the terrain classes (Table 4 defines them): S = full-range synthetic,
#: H = hilly synthetic, D = real DEM. Never the code keys.
def _class_ids() -> dict:
    ids, n = {}, {'S': 0, 'H': 0, 'D': 0}
    for key in TERRAIN_CLASSES:
        letter = 'S' if key.startswith('syn_') else 'H' if key.startswith('hil_') else 'D'
        n[letter] += 1
        ids[key] = f'{letter}{n[letter]}'
    return ids


CLASS_IDS: dict = _class_ids()


def sd(values, d: int = 2) -> str:
    """The narrow table cell: mean ± SD (the median and range are in the CSV files)."""
    v = finite(values)
    if not v:
        return '–'
    text = f'{statistics.fmean(v):.{d}f}'
    if len(v) > 1:
        text += f' ± {statistics.stdev(v):.{d}f}'
    return neg(text)


SD_NOTE = 'Values: mean ± standard deviation over the scenarios of each class.'


def class_ring_rows(d: 'Data', key: str) -> list:
    """The CTA-plus-one-ring rows of the same scenarios as `TERRAIN_CLASSES[key]`."""
    tc = TERRAIN_CLASSES[key]
    rows = tc.select(d)
    if key.startswith(('syn_', 'hil_')):
        group = x3h_group if key.startswith('hil_') else x3_group
        first = rows[0]
        return [r for r in group(d, 'hag_cta_ring1')
                if (r.get('periods'), float(r.get('relief_amplitude_m'))) ==
                (first['periods'], float(first['relief_amplitude_m']))]
    source = need(d.x6, 'results/x6_dem_whole/x6_dem_whole.csv') if 'whole' in key else \
        need(d.x7, 'results/x7_dem_windows/x7_dem_windows.csv')
    cell = str(rows[0].get('cell_size_m'))
    return [r for r in source if r.get('search_space') == 'hag_cta_ring1'
            and str(r.get('cell_size_m')) == cell]


def class_label(key: str) -> str:
    tc = TERRAIN_CLASSES[key]
    if key.startswith('dem_'):
        cell = key.rsplit('_', 1)[1].rstrip('m')
        return ('Idrija, whole swath' if 'whole' in key else 'Idrija, windows') + f', {cell} m cells'
    period = key.split('_')[1][1:]
    relief = key.split('_')[2][1:]
    name = tc.name.split(': ', 1)[-1].split(',')[0]
    return f'{sentence_case(name)}; period {period}, relief {relief} m'


@table('terrain_classes')
def _(d):
    out = []
    for key, tc in TERRAIN_CLASSES.items():
        out.append([CLASS_IDS[key], class_label(key), f'{class_slope_gon(d, tc):.1f}',
                    thousands(len(tc.select(d)))])
    return Table('terrain_classes', 'Terrain classes: synthetic full-range (S), synthetic hilly '
                 '(H) and real LiDAR DEM (D).',
                 ['ID', 'Terrain', 'Median slope (gon)', 'Scenarios'], out,
                 'The STAS 863-85 hilly class is 20–25 gon. For the synthetic classes the period '
                 'is the number of Perlin features across the grid (more features: shorter '
                 'ridges and valleys) and the relief is the height range. ' +
                 _synthetic_pooling_note(d))


def _calibration_pairs(d: 'Data', hilly: bool) -> dict:
    """(periods, relief) -> [(full-grid row, CTA row)] of the serial timing sample."""
    cal = need(d.x3hcal if hilly else d.x3cal, 'calibration sample')
    by: dict = {}
    for r in cal:
        by.setdefault((r['case_id'], r['od_id'], r['height_delta_m']), {})[r['search_space']] = r
    out: dict = {}
    for v in by.values():
        if 'full_grid' in v and 'hag_cta' in v:
            c = v['hag_cta']
            out.setdefault((c['periods'], float(c['relief_amplitude_m'])), []).append(
                (v['full_grid'], c))
    return out


@table('compute_by_terrain')
def _(d):
    out, sizes = [], set()
    cal = {False: _calibration_pairs(d, False), True: _calibration_pairs(d, True)}
    for key, tc in TERRAIN_CLASSES.items():
        if key.startswith('dem_'):
            continue
        first = tc.select(d)[0]
        pairs = cal[key.startswith('hil_')].get(
            (first['periods'], float(first['relief_amplitude_m'])), [])
        if not pairs:
            raise Missing(f'no timing sample for {key}')
        full = [1000.0 * f['search_time_s_median'] for f, _ in pairs]
        cta = [1000.0 * c['search_time_s_median'] for _, c in pairs]
        speed = [f / c for f, c in zip(full, cta) if c > 0]
        build = [1000.0 * c['hag_build_time_s'] for _, c in pairs]
        be = finite(c['break_even_queries'] for _, c in pairs)
        sizes |= {int(c['grid_cols']) for _, c in pairs}
        out.append([CLASS_IDS[key], f'{statistics.fmean(full):.1f}', f'{statistics.fmean(cta):.1f}',
                    f'{statistics.fmean(speed):.1f}', f'{statistics.fmean(build):.0f}',
                    f'{statistics.median(be):.0f}' if be else 'never',
                    f'{len(pairs) - len(be)}/{len(pairs)}'])
    return Table('compute_by_terrain', 'Computation per synthetic terrain class: full-grid '
                 'search, CTA search and HAG construction.',
                 ['ID', 'Full grid (ms)', 'CTA (ms)', 'Speed-up (×)', 'HAG build (ms)',
                  'Break-even (queries)', 'Never repaid'], out,
                 'Times are means per query over a stratified sample of the scenarios of each '
                 f'class (grids of {join_and([str(n) for n in sorted(sizes)])} cells), re-timed one at a time on an otherwise idle '
                 'machine. Speed-up: full-'
                 'grid search time divided by CTA search time. Break-even: median number of '
                 'queries on one terrain after which the time saved repays the HAG '
                 'construction. Never repaid: queries whose CTA search was not faster than the '
                 'full-grid search, out of the sample.')


@table('od_effect')
def _(d):
    import math
    out = []
    key = 'hil_p6_a120'
    rows = TERRAIN_CLASSES[key].select(d)
    for cols in sorted({r['grid_cols'] for r in rows}):
        part = [r for r in rows if r['grid_cols'] == cols]
        dist = statistics.fmean(r['straight_line_m'] for r in part) / 1000.0
        out.append([f'{CLASS_IDS[key]}, grid of {cols} cells', f'{dist:.1f}', str(len(part)),
                    sd(column(part, 'search_space_percent_of_full'), 1),
                    sd(column(part, 'objective_ratio_to_full'), 2),
                    sd(column(part, 'nodes_expanded_ratio_to_full'), 3)])
    for band in ('1km', '2km', '5km', '10km', '20km'):
        part = x6_at(d, band, '10')
        if not part:
            continue
        dist = statistics.fmean(r['straight_line_m'] for r in part) / 1000.0
        out.append([f'{CLASS_IDS["dem_whole_10m"]}, pairs about {band[:-2]} km apart',
                    f'{dist:.1f}', str(len(part)),
                    sd(column(part, 'search_space_percent_of_full'), 2),
                    sd(column(part, 'objective_ratio_to_full'), 2),
                    sd(column(part, 'nodes_expanded_ratio_to_full'), 3)])
    return Table('od_effect', 'Effect of the O–D distance: one hilly synthetic class (H2) and '
                 'the whole Idrija swath at 10 m cells (D2).',
                 ['O–D set', 'Mean O–D distance (km)', 'Scenarios', 'CTA (% of grid)',
                  'Objective ratio', 'Nodes-expanded ratio'], out,
                 SD_NOTE + ' Synthetic pairs are at least 60% of the grid diagonal apart, so '
                 'their distance grows with the grid; on the DEM, three pairs were drawn per '
                 'distance band and height interval. Percentages on the DEM are of the surveyed '
                 'cells.')


@table('height_interval')
def _(d):
    """Per terrain class and HAG height interval: mean CTA share and mean objective ratio
    (agreed FW7: the comparison the authors asked for, from the existing data)."""
    deltas = (1.0, 2.0, 3.0, 5.0)
    out = []
    for key, tc in TERRAIN_CLASSES.items():
        if key.startswith('dem_window'):
            continue
        rows = tc.select(d)
        cells = []
        for delta in deltas:
            part = [r for r in rows if float(r.get('height_delta_m')) == delta]
            if not part:
                cells.append('–')
                continue
            share = statistics.fmean(finite(column(part, 'search_space_percent_of_full')))
            ratio = statistics.fmean(finite(column(part, 'objective_ratio_to_full')))
            cells.append(f'{share:.1f} / {ratio:.2f}' if share >= 1 else
                         f'{share:.2f} / {ratio:.2f}')
        out.append([CLASS_IDS[key], *cells])
    return Table('height_interval', 'Effect of the HAG height interval per terrain class: CTA '
                 'share of the grid (%) / objective ratio (means).',
                 ['ID', '1 m', '2 m', '3 m', '5 m'], out,
                 'Hilly classes were run at 1, 3 and 5 m; the DEM windows (3 m only) are not '
                 'listed. On the DEM, the share is of the surveyed cells.')


@table('stas_dem')
def _(d):
    """The STAS 863-85 station check on real DEM terrain (before-release task PR2)."""
    from core.checks import STATION_CHECK_NAMES
    all_rows = need(d.stas_dem, 'results/stas_dem/stas.csv')
    rows = [r for r in all_rows if r.get('path_found') is True]
    out = []
    for name in STATION_CHECK_NAMES:
        applicable = [r for r in rows if r.get(f'stas_{name}_status') in ('pass', 'fail')]
        if not applicable:
            out.append([STAS_PARAMETER_NAMES.get(name, name), '0', '–', '–'])
            continue
        n_fail = sum(1 for r in applicable if r[f'stas_{name}_status'] == 'fail')
        out.append([STAS_PARAMETER_NAMES.get(name, name), thousands(len(applicable)),
                    pct(n_fail / len(applicable), 1),
                    sd(column(applicable, f'stas_{name}_violation_length_m'), 0)])
    n_ok = sum(1 for r in rows if stas_ok(r))
    out.append(['All parameters pass', thousands(len(rows)), pct(n_ok / len(rows), 1) + ' pass', '–'])
    windows = len({r['terrain_id'] for r in rows})
    return Table('stas_dem', 'STAS 863-85 check at every station on the real Idrija DEM: '
                 f'{windows} window{"s" if windows != 1 else ""} at 10 m cells, {thousands(len(rows))} alignments that found a '
                 'path (both design speeds).',
                 ['Parameter', 'Alignments checked', 'Alignments failing',
                  'Failing length per alignment (m)'], out,
                 SD_NOTE.replace('scenarios of each class', 'alignments checked') +
                 f' Stations {_stas_step(rows)} m apart; edge costs: height change with the '
                 'length tie-break, height change plus length, and the gradient cut.')


@table('baseline')
def _(d):
    """HAG against fixed square blocks of the same mean size (before-release task PR4)."""
    rows = need(d.baseline, 'results/baseline/baseline.csv')
    groups = {'S': lambda r: float(r['relief_amplitude_m']) in (20.0, 60.0, 150.0),
              'H': lambda r: float(r['relief_amplitude_m']) in (200.0, 120.0, 100.0)}
    out = []
    for fam, label in (('S', 'Full-range classes (S1–S15)'), ('H', 'Hilly classes (H1–H3)')):
        for space, sp_label in (('hag_cta', 'CTA'), ('hag_cta_ring1', 'CTA + 1 ring')):
            cells = []
            for abstraction in ('hag', 'blocks'):
                part = [r for r in rows if r['abstraction'] == abstraction
                        and r['search_space'] == space and groups[fam](r)]
                cells += [sd(column(part, 'search_space_percent_of_full'), 1),
                          sd(column(part, 'objective_ratio_to_full'), 2)]
            out.append([f'{label}, {sp_label}', *cells])
    return Table('baseline', 'The HAG against a fixed-cluster abstraction of the same mean '
                 'cluster size (square blocks), on the same synthetic scenarios.',
                 ['Classes, search space', 'HAG: share (%)', 'HAG: objective ratio',
                  'Blocks: share (%)', 'Blocks: objective ratio'], out,
                 SD_NOTE.replace('of each class', 'of each row') + ' Both abstractions use the '
                 'same shared-border edge cost, chain selection, ring dilation and search; only '
                 'the partition of the terrain differs.')


@table('stas_by_cost')
def _(d):
    rows = stas_rows(d)
    labels = {'climb': 'Height change', 'climb_tiebreak': 'Height change + length tie-break',
              'length_3d': 'Length along the terrain', 'climb_plus_length': 'Height change + length',
              'climb_gradient_penalty': 'Height change + gradient penalty',
              'climb_gradient_cut': 'Height change, edges steeper than i_max removed'}
    out = []
    for cost in labels:
        part = [r for r in rows if r.get('grid_edge_cost') == cost]
        if not part:
            continue
        grad = [r for r in part if r.get('stas_i_status') in ('pass', 'fail')]
        n_grad = sum(1 for r in grad if r['stas_i_status'] == 'pass')
        n_all = sum(1 for r in part if stas_ok(r))
        out.append([labels[cost], thousands(len(part)), pct(n_grad / len(grad), 1),
                    pct(n_all / len(part), 1)])
    return Table('stas_by_cost', 'STAS 863-85 compliance by the edge cost of the detailed search.',
                 ['Edge cost of the search', 'Alignments', 'Gradient passes', 'All parameters pass'],
                 out, 'Alignments that found a path, before and after Algorithm 1, both design '
                 'speeds. The gradient penalty adds one metre of cost per metre of edge for every '
                 'percent above i_max; removing steep edges can disconnect the selected areas, '
                 'which is why that row has fewer alignments.')


@table('x3_by_terrain')
def _(d):
    out = []
    for key, tc in TERRAIN_CLASSES.items():
        if not key.startswith(('syn_', 'hil_')):
            continue
        rows, ring1 = tc.select(d), class_ring_rows(d, key)
        out.append([CLASS_IDS[key], sd(column(rows, 'search_space_percent_of_full'), 1),
                    sd(column(rows, 'objective_ratio_to_full'), 2),
                    sd(column(ring1, 'search_space_percent_of_full'), 1),
                    sd(column(ring1, 'objective_ratio_to_full'), 2)])
    return Table('x3_by_terrain', 'Search-domain reduction and its cost per synthetic terrain '
                 'class (classes in Table 4).',
                 ['ID', 'CTA (% of grid)', 'CTA objective ratio', 'CTA + 1 ring (% of grid)',
                  'CTA + 1 ring objective ratio'], out,
                 SD_NOTE + ' Objective ratio: total elevation change of the restricted alignment '
                 'divided by that of the full-grid alignment (1 = no loss).')


@table('dem_by_cell')
def _(d):
    out = []
    for key, tc in TERRAIN_CLASSES.items():
        if not key.startswith('dem_'):
            continue
        rows, ring1 = tc.select(d), class_ring_rows(d, key)
        out.append([CLASS_IDS[key], sd(column(rows, 'search_space_percent_of_full'), 2),
                    sd(column(rows, 'objective_ratio_to_full'), 2),
                    sd(column(ring1, 'search_space_percent_of_full'), 2),
                    sd(column(ring1, 'objective_ratio_to_full'), 2)])
    return Table('dem_by_cell', 'Search-domain reduction and its cost on the real Idrija LiDAR '
                 'DEM, per source and cell size (classes in Table 4).',
                 ['ID', 'CTA (% of surveyed cells)', 'CTA objective ratio',
                  'CTA + 1 ring (% of surveyed cells)', 'CTA + 1 ring objective ratio'], out,
                 SD_NOTE + ' Percentages are of the surveyed cells.')


#: what each table shows, where it comes from and how to read it. Every key of TABLES must
#: have an entry (pinned by tests/test_article_build.py); the catalogue
#: `article_2/index.md` is generated from it.
TABLE_DESCRIPTIONS: dict = {
    'stas_dem': ('The STAS 863-85 station check on real DEM terrain (Idrija windows at 10 m).',
                 'results/stas_dem/stas.csv'),
    'baseline': ('The HAG against fixed square blocks of the same mean size, same pipeline.',
                 'results/baseline/baseline.csv'),
    'height_interval': ('Per terrain class and HAG height interval (1, 2, 3, 5 m): mean CTA share '
                        'and mean objective ratio.', 'results/x3_*, results/x6_dem_whole/'),
    'terrain_classes': ('The terrain classes with their article IDs (S full-range synthetic, H '
                        'hilly synthetic, D real DEM), median slope and number of scenarios.',
                        'TERRAIN_CLASSES in article_2/build_article.py'),
    'compute_by_terrain': ('Per synthetic class: full-grid and CTA search time, speed-up, HAG '
                           'construction time and break-even, from the serial timing sample.',
                           'results/x3_reduction_synthetic*/calibration/calibration.csv'),
    'od_effect': ('The effect of the O–D distance on the reduction and its cost, for one hilly '
                  'class (by grid size) and the whole DEM swath at 10 m (by distance band).',
                  'results/x3_reduction_synthetic_hilly/, results/x6_dem_whole/'),
    'stas_by_cost': ('STAS 863-85 compliance (gradient, all parameters) by the edge cost of the '
                     'detailed search; shows the effect of the gradient cut and penalty.',
                     'results/stas/stas.csv'),
    'terrains': ('The synthetic terrains of the earlier, nine-scenario robustness matrix (T7): '
                 'size, cell size, Perlin settings and height bands. Context for the tables that '
                 'use that matrix.', 'core/experiment_matrix.py (TERRAINS)'),
    'classes': ('The Romanian road classes the T7 matrix is checked against (radius, gradient, '
                'design speed), transcribed from Ordin MT 1296/2017 and awaiting sign-off.',
                'data/configs/road_classes.py'),
    'search_cost': ('T7 matrix: search space, nodes expanded, time and memory per search arm. '
                    'Superseded for the reduction itself by x3_reduction and x3_by_terrain, kept '
                    'for the arms the old matrix compared.', 'results/matrix.csv'),
    'quality': ('T7 matrix: objective, length and Hausdorff distance of each arm against the '
                'full-grid alignment of the same scenario.', 'results/matrix.csv'),
    'engineering': ('T7 matrix: after Algorithm 1, how many arm × scenario combinations meet the '
                    'minimum radius, the maximum gradient and both, per road class. A class that '
                    'cannot be met is reported, not relaxed.', 'results/matrix.csv'),
    'smoothing': ('T7 matrix: Algorithm 1 before and after per road class — radius, gradient, '
                  'length, sinuosity and deviation from the unsmoothed axis.', 'results/matrix.csv'),
    'acceptance': ('Acceptance test: as the class gets stricter, the smoothed radius and the '
                   'deviation must rise and the gradient must not; the last row says whether each '
                   'column moves the right way.', 'results/matrix.csv'),
    'gradient': ('The two gradient-aware search arms (hard cut, penalty) and the smoothing guard '
                 'on the nine T7 scenarios: path found, checks passed, gradients before and '
                 'after.', 'results/gradient/gradient.csv'),
    'census': ('How much of the Idrija raster is usable: per window size, how many fully '
               'surveyed windows exist and how often the decimated scan over-approved one.',
               'results/dem/census.csv'),
    'ladder': ('Gradient feasibility against cell size on one real crop: share of admissible '
               'edges, number of connected components, size of the largest. Shows that a '
               'gradient limit is only meaningful at a stated cell size.', 'results/dem/ladder.csv'),
    'swath': ('The large 4 km real crop (and a 1 km sub-crop at 1 m): every search arm at each '
              'cell size, with search space, length ratio, time and gradients.',
              'results/dem/swath.csv'),
    'swath_hd': ('The same large crop at 10 m cells for several height bands Δ: how the band '
                 'width changes the number of areas, the search space and the path.',
                 'results/dem/swath.csv'),
    'tiles': ('A random population of real 600 m tiles: search space, length ratio, gradients '
              'and class compliance per arm — the typical real case, as opposed to the hardest '
              'crop.', 'results/dem_tiles/tiles.csv'),
    'b23': ('One real alignment certified to hold the gradient limit at cell resolution, smoothed: '
            'the smoothing shortens it and the gradient rises above the limit.',
            'results/dem/b23_case.csv'),
    'network': ('The portal overlay (on hold): build cost, query cost, break-even number of '
                'queries and route quality per portal spacing.', 'results/network/network.csv'),
    'x3_reduction': ('X3, pooled over all synthetic scenarios: search space, objective ratio, '
                     'Hausdorff distance, nodes, time and break-even for each search space '
                     '(candidate terrain areas, and the same plus one ring of dilation).',
                     'results/x3_reduction_synthetic/x3_reduction_synthetic.csv'),
    'x3_by_terrain': ('X3 split by kind of synthetic terrain (Perlin period × relief), from '
                      'gentle to hilly: the reduction is a property of the terrain — read down '
                      'the rows to see the candidate share fall as the terrain gets steeper, and '
                      'the one-ring columns for what dilation buys back.',
                      'results/x3_reduction_synthetic/x3_reduction_synthetic.csv, '
                      'results/x3_reduction_synthetic_hilly/x3_reduction_synthetic_hilly.csv'),
    'dem_by_cell': ('X6/X7 on the real Idrija DEM per source (whole swath, windows) and cell size: '
                    'the reduction and its cost on real terrain, which is steeper than any '
                    'synthetic class.', 'results/x6_dem_whole/x6_dem_whole.csv, '
                    'results/x7_dem_windows/x7_dem_windows.csv'),
    'stas': ('X8: every STAS 863-85 station parameter over all alignments that found a path — how '
             'often each fails, at how many stations and over what length.',
             'results/stas/stas.csv'),
}

#: the same for figures (key -> (description, source)).
FIGURE_DESCRIPTIONS: dict = {
    'workflow': ('The evaluated pipeline as a diagram: terrain → HAG → selected areas → Dijkstra '
                 '→ Algorithm 1 → z re-sampling → checks.', 'core/figures.py (fig_workflow)'),
    'terrain_areas': ('One synthetic terrain with its height areas, the selected chain and every '
                      'arm\'s alignment: what "selecting areas" means on a map.',
                      'results/figures/terrain_areas_T_article_od1.png'),
    'alignments': ('The four class-blind alignments of one T7 scenario in plan, before smoothing.',
                   'results/figures/alignments_T_article_od1.png'),
    'profile': ('Longitudinal profile and gradient of one alignment before and after Algorithm 1: '
                'smoothing in plan can push the gradient over the limit.',
                'results/figures/profile_T_article_od1_hag_yellow_RO_CLASS_IV_DEAL_it1.png'),
    'curvature': ('Curve radius along the same alignment before and after Algorithm 1, against the '
                  'class minimum.',
                  'results/figures/curvature_T_article_od1_hag_yellow_RO_CLASS_IV_DEAL_it1.png'),
    'distributions': ('T7 matrix distributions per arm (search space, nodes, time, objective). '
                      'Superseded by x3_reduction / x3_by_terrain for the reduction.',
                      'results/figures/matrix_distributions.png'),
    'ladder': ('Gradient feasibility against cell size (plot of the ladder table).',
               'results/dem/ladder.csv'),
    'dem_map': ('The large real crop at 10 m with every arm\'s alignment on a hillshade.',
                'results/dem/swath.csv + raster/ (git-ignored)'),
    'tiles': ('Distributions over the real tile population, per arm.', 'results/dem_tiles/tiles.csv'),
    'network': ('Portal overlay (on hold): break-even and route quality against portal spacing.',
                'results/network/network.csv'),
    'stas': ('Every STAS 863-85 parameter along the chainage of one representative alignment, each '
             'against its limit: where along the road a parameter fails, and for how long.',
             'results/stas/series/ via core/figures_stas.py'),
}


def index_markdown(d: 'Data', used_tables=None, used_figures=None) -> str:
    """The catalogue of every table, figure and terrain class with its name and description,
    (`article_2/index.md`). It is NOT appended to an article: an updated article
    ends with `appendix_markdown` instead (agreed item C6)."""
    lines = ['## Tables, figures and terrain classes', '',
             'Every table, figure and terrain class, with its name and what it shows. Generated by '
             '`python -m article_2.build_article`; numbers in the class descriptions are measured.',
             '', '### Tables', '']
    for key in (used_tables or TABLES):
        description, source = TABLE_DESCRIPTIONS[key]
        try:
            name = TABLES[key](d).caption
        except Exception:  # noqa: BLE001 - a table whose data is missing still has a name
            name = key
        lines.append(f'- **`{key}`** — *Name:* {name} *Shows:* {description} '
                     f'*Source:* `{source}`.')
    lines += ['', '### Figures', '']
    for key in (used_figures or FIGURES):
        description, source = FIGURE_DESCRIPTIONS[key]
        lines.append(f'- **`{key}`** — *Name:* {FIGURES[key].caption} *Shows:* {description} '
                     f'*Source:* `{source}`.')
    lines += ['', '### Terrain classes', '']
    for key, tc in TERRAIN_CLASSES.items():
        try:
            text = class_description(d, tc)
        except Missing as error:
            text = f'<<VALUE: {error}>>'
        lines.append(f'- **`{key}`** — *Name:* {tc.name}. *Description:* {text}')
    return '\n'.join(lines) + '\n'


#: Appendix A, table A1: the terms and measures a reader needs to read the Results. Plain
#: definitions; no number may appear here (numbers come from the data).
READING_TERMS: tuple = (
    ('Height area', 'A connected set of grid cells whose heights fall in the same height '
     'interval.'),
    ('Height Area Graph (HAG)', 'The graph whose nodes are the height areas and whose edges join '
     'areas that share a border.'),
    ('Shared-border edge cost', 'The weight of a HAG edge: the mean height step across the common '
     'border of the two areas, priced with the same cost model as the grid search, over the path '
     'centroid → border midpoint → centroid.'),
    ('Candidate terrain areas (CTA)', 'The height areas on the minimum-weight HAG path between the '
     'areas of the origin and of the destination; the detailed search runs on their cells only.'),
    ('CTA and one ring', 'The candidate terrain areas together with every area that borders them.'),
    ('Search space (% of the full grid)', 'Cells of the restricted search as a share of the cells '
     'of the full grid (of the surveyed cells, on the real DEM).'),
    ('Objective ratio', 'Objective (sum of absolute height differences along the alignment) of the '
     'restricted search divided by that of the full-grid search on the same scenario; 1 means no '
     'loss.'),
    ('Hausdorff distance', 'The largest distance from a point of one alignment to the other '
     'alignment; it measures how far the restricted alignment departs from the full-grid one.'),
    ('Nodes-expanded ratio', 'Nodes expanded by the restricted search divided by those of the '
     'full-grid search; independent of the machine.'),
    ('Break-even', 'The number of queries on one terrain after which the time saved per query '
     'repays the construction of the HAG; never reached when the restricted search is not faster.'),
    ('Station', 'A point of the alignment at which every STAS 863-85 parameter is evaluated.'),
    ('Not applicable', 'A parameter that the road class does not define, or that governs no station '
     'of the alignment; it is never counted as a pass.'),
    ('Median slope (gon)', 'The median terrain slope of a class, in gon (400 gon = 360°); the STAS '
     '863-85 hilly class is 20–25 gon.'),
)


def _mean(values, d: int) -> str:
    v = finite(values)
    return neg(f'{statistics.fmean(v):.{d}f}') if v else '–'


def _hilly_word(gon: float) -> str:
    lo, hi = STAS_HILLY_GON
    return 'inside' if lo <= gon <= hi else 'below' if gon < lo else 'above'


def _md_table(header: list, rows: list, widths=None) -> list:
    sep = separator(widths) if widths else '| ' + ' | '.join('---' for _ in header) + ' |'
    return ['| ' + ' | '.join(header) + ' |', sep,
            *['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]]


def appendix_markdown(d: 'Data') -> str:
    """Appendix A of an updated article (agreed items C6, C16): the terms (C1), then the MEAN results
    per synthetic terrain class (C2) and per real-DEM class (C3). Narrow by design: the classes
    are named by the IDs of the terrain-class table of the Results, and every column is a short
    number, so the tables print on a journal page."""
    lines = ['## Appendix A. Reading the Results', '',
             'Table A1 defines the terms used in Section 3. Tables A2 and A3 give one mean value '
             'per terrain class for each main measure; the classes are defined in Table 4 and '
             'the spread of each value is given in Section 3.', '',
             '**Table A1.** Terms and measures used in the Results.', '']
    lines += _md_table(['Term', 'Meaning'], [list(t) for t in READING_TERMS], (1.4, 4))
    cal = {False: _calibration_pairs(d, False), True: _calibration_pairs(d, True)}
    syn = []
    for key, tc in TERRAIN_CLASSES.items():
        if not key.startswith(('syn_', 'hil_')):
            continue
        rows, ring1 = tc.select(d), class_ring_rows(d, key)
        first = rows[0]
        pairs = cal[key.startswith('hil_')].get(
            (first['periods'], float(first['relief_amplitude_m'])), [])
        speed = [f['search_time_s_median'] / c['search_time_s_median'] for f, c in pairs
                 if c['search_time_s_median'] > 0]
        syn.append([CLASS_IDS[key], _mean(column(rows, 'search_space_percent_of_full'), 1),
                    _mean(column(rows, 'objective_ratio_to_full'), 2),
                    _mean(column(rows, 'nodes_expanded_ratio_to_full'), 2),
                    f'{statistics.fmean(speed):.1f}' if speed else '–',
                    _mean(column(ring1, 'search_space_percent_of_full'), 1),
                    _mean(column(ring1, 'objective_ratio_to_full'), 2)])
    lines += ['', '**Table A2.** Mean results per synthetic terrain class.', '']
    lines += _md_table(['ID', 'CTA (%)', 'Objective ratio', 'Nodes ratio', 'Speed-up (×)',
                        'CTA + 1 ring (%)', 'Objective ratio, + 1 ring'], syn,
                        (1, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6))
    lines += ['', '*CTA (%): share of the grid searched. Speed-up: full-grid search time divided '
              'by CTA search time, from the serial timing sample.*']
    dem = []
    for key, tc in TERRAIN_CLASSES.items():
        if not key.startswith('dem_'):
            continue
        rows, ring1 = tc.select(d), class_ring_rows(d, key)
        dem.append([CLASS_IDS[key], _mean(column(rows, 'search_space_percent_of_full'), 2),
                    _mean(column(rows, 'objective_ratio_to_full'), 2),
                    _mean(column(rows, 'nodes_expanded_ratio_to_full'), 3),
                    _mean(column(ring1, 'search_space_percent_of_full'), 2),
                    _mean(column(ring1, 'objective_ratio_to_full'), 2)])
    lines += ['', '**Table A3.** Mean results per real-DEM class.', '']
    lines += _md_table(['ID', 'CTA (%)', 'Objective ratio', 'Nodes ratio', 'CTA + 1 ring (%)',
                        'Objective ratio, + 1 ring'], dem, (1, 1.4, 1.5, 1.4, 1.5, 1.6))
    lines += ['', '*CTA (%): share of the surveyed cells searched.*']
    return '\n'.join(lines) + '\n'


def supplementary_markdown(d: 'Data') -> str:
    """Supplementary Tables S1-S5 (agreed item C10): the full distribution of every main measure,
    as mean ± SD (median) [min–max], per terrain class (IDs of the terrain-class table). Offered
    digitally with the article; the body and Appendix A carry the short forms."""
    from core.checks import STATION_CHECK_NAMES
    lines = ['# Supplementary Materials: full statistics', '',
             'Each cell is mean ± standard deviation (median) [minimum–maximum] over the scenarios '
             'of the row. Terrain classes are those of Table 4 of the article (S: full-range '
             'synthetic, H: hilly synthetic, D: real Idrija LiDAR DEM). The per-scenario records '
             'are published with the open-source implementation.', '']
    measures = (('search_space_percent_of_full', 'Search domain (%)', 2),
                ('objective_ratio_to_full', 'Objective ratio', 3),
                ('length_ratio_to_full', 'Length ratio', 3),
                ('hausdorff_to_full_m', 'Hausdorff distance (m)', 1),
                ('nodes_expanded_ratio_to_full', 'Nodes-expanded ratio', 3))
    for number, (title, prefixes) in enumerate((
            ('synthetic terrain classes', ('syn_', 'hil_')), ('real-DEM classes', ('dem_',))), 1):
        lines += [f'**Table S{number}.** Search-domain reduction and its cost, {title}: candidate '
                  'terrain areas (CTA) and CTA with one ring of neighbouring areas.', '']
        rows = []
        for key, tc in TERRAIN_CLASSES.items():
            if not key.startswith(prefixes):
                continue
            for label, part in (('CTA', tc.select(d)), ('CTA + 1 ring', class_ring_rows(d, key))):
                rows.append([CLASS_IDS[key], label, thousands(len(part)),
                             *[cell(column(part, col), dec) for col, _, dec in measures]])
        lines += _md_table(['ID', 'Search domain', 'Scenarios', *[m[1] for m in measures]], rows)
        lines.append('')
    lines += ['**Table S3.** Computation per synthetic terrain class, serial timing sample: '
              'search time per query (ms) and HAG construction time (ms).', '']
    cal = {False: _calibration_pairs(d, False), True: _calibration_pairs(d, True)}
    rows = []
    for key, tc in TERRAIN_CLASSES.items():
        if key.startswith('dem_'):
            continue
        first = tc.select(d)[0]
        pairs = cal[key.startswith('hil_')].get(
            (first['periods'], float(first['relief_amplitude_m'])), [])
        rows.append([CLASS_IDS[key], str(len(pairs)),
                     cell([1000 * f['search_time_s_median'] for f, _ in pairs], 1),
                     cell([1000 * c['search_time_s_median'] for _, c in pairs], 1),
                     cell([f['search_time_s_median'] / c['search_time_s_median']
                           for f, c in pairs if c['search_time_s_median'] > 0], 1),
                     cell([1000 * c['hag_build_time_s'] for _, c in pairs], 0),
                     cell([c['break_even_queries'] for _, c in pairs], 1)])
    lines += _md_table(['ID', 'Sample', 'Full grid (ms)', 'CTA (ms)', 'Speed-up (×)',
                        'HAG build (ms)', 'Break-even (queries)'], rows)
    lines += ['', '*Break-even excludes the queries that never repay the construction.*', '']
    lines += ['**Table S4.** Search-domain reduction and its cost pooled over the full-range '
              'synthetic scenarios.', '']
    t = TABLES['x3_reduction'](d)
    rows = []
    for tag, (space, cost) in X3_SPACES.items():
        part = x3_group(d, space, _resolve_hag_edge_cost(space, cost))
        timed = x3_timing_group(d, space, _resolve_hag_edge_cost(space, cost))
        rows.append(['CTA' if tag == 'default' else 'CTA + 1 ring',
                     *[cell(column(part, col), dec) for col, _, dec in measures],
                     cell(column(timed, 'search_time_ratio_to_full'), 3),
                     cell(column(timed, 'break_even_queries'), 1)])
    lines += _md_table(['Search domain', *[m[1] for m in measures], 'Search-time ratio',
                        'Break-even (queries)'], rows)
    lines += ['', '**Table S5.** STAS 863-85 station check per parameter: failing stations and '
              'failing length per alignment.', '']
    st_rows = stas_rows(d)
    rows = []
    for name in STATION_CHECK_NAMES:
        applicable = [r for r in st_rows if r.get(f'stas_{name}_status') in ('pass', 'fail')]
        n_fail = sum(1 for r in applicable if r[f'stas_{name}_status'] == 'fail')
        rows.append([STAS_PARAMETER_NAMES.get(name, name), thousands(len(applicable)),
                     f'{n_fail} ({pct(n_fail / len(applicable), 1)})' if applicable else '–',
                     cell(column(applicable, f'stas_{name}_n_violations'), 1),
                     cell(column(applicable, f'stas_{name}_violation_length_m'), 1)])
    lines += _md_table(['Parameter', 'Alignments checked', 'Failing (n, %)',
                        'Failing stations per alignment', 'Failing length per alignment (m)'], rows)
    return '\n'.join(lines) + '\n'


#: relative column widths per table (agreed item B10): pandoc reads them from the number of
#: dashes in the separator line, so a narrow ID column and a wide text column print well.
WIDTHS: dict = {
    'terrain_classes': (1, 6, 1.6, 1.6), 'x3_by_terrain': (1, 2, 2, 2, 2),
    'dem_by_cell': (1, 2, 2, 2, 2), 'x3_reduction': (4, 2, 2),
    'compute_by_terrain': (1, 1.5, 1.5, 1.5, 1.5, 1.6, 1.4),
    'od_effect': (3.2, 1.5, 1.2, 1.8, 1.8, 1.8), 'stas': (3.2, 1.6, 1.6, 2),
    'stas_by_cost': (4.2, 1.5, 1.5, 1.6), 'height_interval': (1, 2, 2, 2, 2),
    'stas_dem': (3.2, 1.6, 1.6, 2), 'baseline': (3.6, 1.6, 1.6, 1.6, 1.6)}


def separator(widths) -> str:
    """A pipe-table separator whose dash counts carry the relative column widths."""
    return '| ' + ' | '.join('-' * max(3, int(round(8 * w))) for w in widths) + ' |'


def render_table(t: Table, number: int) -> str:
    def line(cells):
        return '| ' + ' | '.join(str(c) for c in cells) + ' |'
    widths = WIDTHS.get(t.key)
    sep = separator(widths) if widths and len(widths) == len(t.header) else \
        line(['---'] * len(t.header))
    out = [f'**Table {number}.** {t.caption}', '', line(t.header), sep]
    out += [line(r) for r in t.rows]
    if t.note:
        out += ['', f'*{t.note}*']
    return '\n'.join(out)


# ==============================================================================================
# figures
# ==============================================================================================

@dataclass
class Figure:
    key: str
    caption: str
    filename: str
    make: Callable[['Data', Path, Path], bool]     # (data, figures_dir, results_dir) -> written?


def _copy_from_results(name: str, subdir: str = 'figures') -> Callable:
    def make(d, figures_dir, results):
        source = results / subdir / name
        if not source.exists():
            return False
        figures_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, figures_dir / name)
        return True
    return make


def _draw(fn) -> Callable:
    def make(d, figures_dir, results):
        from core import figures as core_figures
        fig = fn(d)
        if fig is None:
            return False
        figures_dir.mkdir(parents=True, exist_ok=True)
        target = figures_dir / FIGURE_FILES[fn.__name__]
        fig.savefig(target, dpi=200, bbox_inches='tight')
        core_figures.plt.close(fig)
        return True
    return make


FIGURE_FILES: dict = {}


def _make_workflow(d):
    from core import figures
    return figures.fig_workflow()


def _make_ladder(d):
    from core import figures
    return figures.fig_ladder(need(d.ladder, 'results/dem/ladder.csv'))


def _make_tiles(d):
    from core import figures
    from data.configs.road_classes import get
    tiles = need(d.tiles, 'results/dem_tiles/tiles.csv')
    limit = get(tiles[0]['road_class']).i_max_percent
    return figures.fig_tiles_distributions(tiles, i_max_limit_percent=limit)


def _make_network(d):
    from core import figures
    return figures.fig_network_break_even(need(d.network, 'results/network/network.csv'))


def _make_dem_map(d):
    """The large crop at 10 m with every arm's alignment; needs the raster (git-ignored)."""
    import numpy as np
    from core import figures
    from core.dem import coarsen_stream, fill_voids, read_geotiff_header
    tif_path = ROOT / 'raster' / 'Idrija_Fault_LiDAR_DEM.tif'
    rows = _swath_rows(d, 'swath_cell10')
    if not tif_path.exists() or not rows:
        return None
    by = {r['arm']: r for r in rows}
    ref = by['full_grid']
    tif = read_geotiff_header(tif_path, sha256=False)
    factor = int(round(float(ref['cell_size_m']) / tif.pixel_scale_m[0]))
    surf, _ = coarsen_stream(tif, row0=int(ref['crop_row0']), col0=int(ref['crop_col0']),
                             n_rows=int(ref['crop_n_rows']), n_cols=int(ref['crop_n_cols']),
                             factor=factor)
    if np.isnan(surf).any():
        surf, _ = fill_voids(surf)
    paths = {}
    for arm in by:
        p = d.results / 'dem' / 'paths' / f'swath_cell10__{arm}__rough.csv'
        if p.exists():
            paths[arm] = np.loadtxt(p, delimiter=',', skiprows=1)
    return figures.fig_dem_scenario(
        surf, float(ref['cell_size_m']), paths,
        start=(int(ref['start_i']), int(ref['start_j'])),
        target=(int(ref['target_i']), int(ref['target_j'])),
        title='Large real-DEM crop (10 m cells): every arm, one pair; hillshade of the DEM')


FIGURE_FILES.update({'_make_workflow': 'workflow.png', '_make_ladder': 'ladder.png',
                     '_make_tiles': 'tiles.png', '_make_network': 'network.png',
                     '_make_dem_map': 'dem_map.png'})

FIGURES: dict = {
    'workflow': Figure('workflow', 'The pipeline that is evaluated: from terrain to the engineering '
                       'checks. Every stage is run for every scenario.', 'workflow.png',
                       _draw(_make_workflow)),
    'terrain_areas': Figure('terrain_areas', 'The `T_article` terrain (grey hillshade) with its height '
                            'areas (thin outlines), the yellow chain of areas selected between the '
                            'origin and the destination (yellow, heavy outline) and the alignment of '
                            'each arm, pair `od1`.',
                            'terrain_areas_T_article_od1.png',
                            _copy_from_results('terrain_areas_T_article_od1.png')),
    'alignments': Figure('alignments', 'The alignments of the four class-blind arms on `T_article`, '
                         'pair `od1`, in metres.', 'alignments_T_article_od1.png',
                         _copy_from_results('alignments_T_article_od1.png')),
    'profile': Figure('profile', 'Longitudinal profile and gradient before and after Algorithm 1 '
                      '(`T_article`, `od1`, `hag_yellow`, `RO_CLASS_IV_DEAL`, one iteration; class '
                      '`TO CONFIRM`). Smoothing turns a compliant gradient into a non-compliant one: '
                      'the XY_smooth → Z_new → i_new chain at work.',
                      'profile_T_article_od1_hag_yellow_RO_CLASS_IV_DEAL_it1.png',
                      _copy_from_results('profile_T_article_od1_hag_yellow_RO_CLASS_IV_DEAL_it1.png')),
    'curvature': Figure('curvature', 'Curve radius along the same axis before and after Algorithm 1, '
                        'against the class minimum (chord and step as reported).',
                        'curvature_T_article_od1_hag_yellow_RO_CLASS_IV_DEAL_it1.png',
                        _copy_from_results('curvature_T_article_od1_hag_yellow_RO_CLASS_IV_DEAL_it1.png')),
    'distributions': Figure('distributions', 'Distributions over the nine synthetic scenarios, per arm: '
                            'search space, nodes expanded, search time and objective ratio.',
                            'matrix_distributions.png', _copy_from_results('matrix_distributions.png')),
    'ladder': Figure('ladder', 'Scale dependence of gradient feasibility: the admissible fraction of '
                     'edges is nearly constant, its connectivity collapses as the cell shrinks.',
                     'ladder.png', _draw(_make_ladder)),
    'dem_map': Figure('dem_map', 'The large Idrija crop at 10 m cells with the alignment of every arm '
                      'between one origin and destination (hillshaded DEM; no existing road is drawn, '
                      'and none would be a ground truth).', 'dem_map.png', _draw(_make_dem_map)),
    'tiles': Figure('tiles', 'The real-DEM tile population, per arm: search space, length relative to '
                    'the full-grid path of the same tile, rough-axis gradient against the class limit, '
                    'and the share of tiles that meet the class.', 'tiles.png', _draw(_make_tiles)),
    'network': Figure('network', 'The portal overlay: break-even number of queries and route quality '
                      'against the portal spacing, in cells.', 'network.png', _draw(_make_network)),
    'stas': Figure('stas', 'Every STAS 863-85 parameter along the chainage of one representative '
                   'alignment (`STAS863_V40`, `TO CONFIRM`; `hag_cta`, `climb_tiebreak`, the '
                   '`baseline` enhancement arm, 2 iterations), against its class limit.',
                   'stas_od1__STAS863_V40__hag_cta__climb_tiebreak__baseline__it2.png',
                   _copy_from_results('stas_od1__STAS863_V40__hag_cta__climb_tiebreak__'
                                      'baseline__it2.png', subdir='stas/figures')),
}


# ==============================================================================================
# assembly
# ==============================================================================================

FACT_RE = re.compile(r'\{\{fact:([A-Za-z0-9_]+)\}\}')
TAB_RE = re.compile(r'\{\{tab:([A-Za-z0-9_]+)\}\}')
FIG_RE = re.compile(r'\{\{fig:([A-Za-z0-9_]+)\}\}')
CITE_RE = re.compile(r'\{\{cite:([A-Za-z0-9_,\s]+)\}\}')
TABLE_BLOCK_RE = re.compile(r'<!--\s*TABLE:([A-Za-z0-9_]+)\s*-->')
FIGURE_BLOCK_RE = re.compile(r'<!--\s*FIGURE:([A-Za-z0-9_]+)\s*-->')
REFERENCES_RE = re.compile(r'<!--\s*REFERENCES\s*-->')


class BuildError(Exception):
    pass


@dataclass
class Build:
    text: str
    todo: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    generated_placeholders: int = 0
    tables: dict = field(default_factory=dict)
    figures_missing: list = field(default_factory=list)


def _numbering(text: str) -> tuple[dict, dict, dict]:
    tabs: dict = {}
    for m in TABLE_BLOCK_RE.finditer(text):
        tabs.setdefault(m.group(1), len(tabs) + 1)
    figs: dict = {}
    for m in FIGURE_BLOCK_RE.finditer(text):
        figs.setdefault(m.group(1), len(figs) + 1)
    cites: dict = {}
    for m in CITE_RE.finditer(text):
        for key in [k.strip() for k in m.group(1).split(',') if k.strip()]:
            cites.setdefault(key, len(cites) + 1)
    return tabs, figs, cites


def build(data: Data, sources: Sequence[Path], figures_dir: Path, tables_dir: Path,
          make_figures: bool = True, strict: bool = False) -> Build:
    from article_2.references import REFERENCES

    raw = '\n\n'.join(p.read_text(encoding='utf-8').rstrip() for p in sorted(sources))
    tabs, figs, cites = _numbering(raw)
    result = Build(text='')

    def resolve_fact(name: str) -> str:
        try:
            if name in FACTS:
                return FACTS[name](data)
            value = dynamic_fact(name, data)
            if value is None:
                raise BuildError(f'unknown fact {{{{fact:{name}}}}}')
            return value
        except BuildError:
            raise
        except (Missing, KeyError, IndexError, ZeroDivisionError, ValueError, TypeError,
                StatisticsError) as error:
            result.generated_placeholders += 1
            return f'<<VALUE: {name} ({error})>>'

    def sub_facts(text: str) -> str:
        return FACT_RE.sub(lambda m: resolve_fact(m.group(1)), text)

    # -- tables -----------------------------------------------------------------------------
    def table_block(m) -> str:
        key = m.group(1)
        if key not in TABLES:
            raise BuildError(f'unknown table {key!r}')
        try:
            t = TABLES[key](data)
        except (Missing, KeyError, IndexError, ZeroDivisionError, ValueError, TypeError,
                StatisticsError) as error:
            result.generated_placeholders += 1
            return (f'**Table {tabs[key]}.** <<VALUE: table `{key}` ({error})>>')
        t.caption = sub_facts(t.caption)
        result.tables[key] = (tabs[key], t)
        return render_table(t, tabs[key])

    def figure_block(m) -> str:
        key = m.group(1)
        if key not in FIGURES:
            raise BuildError(f'unknown figure {key!r}')
        fig = FIGURES[key]
        target = figures_dir / fig.filename
        written = False
        if make_figures:
            try:
                written = bool(fig.make(data, figures_dir, data.results))
            except (Missing, KeyError, IndexError, ValueError) as error:
                written = False
                result.problems.append(f'figure {key}: {error}')
        if not written and not target.exists():
            result.figures_missing.append(key)
            result.generated_placeholders += 1
            return (f'**Figure {figs[key]}.** <<VALUE: figure `{key}` ({fig.filename} could not '
                    f'be produced and is not committed)>>')
        caption = sub_facts(fig.caption)
        return f'![{fig.filename}](figures/{fig.filename})\n\n**Figure {figs[key]}.** {caption}'

    text = TABLE_BLOCK_RE.sub(table_block, raw)
    text = FIGURE_BLOCK_RE.sub(figure_block, text)

    # -- references ------------------------------------------------------------------------
    def cite(m) -> str:
        keys = [k.strip() for k in m.group(1).split(',') if k.strip()]
        for k in keys:
            if k not in REFERENCES:
                raise BuildError(f'unknown reference {k!r}')
        return ', '.join(str(cites[k]) for k in keys)

    def reference_list(_m) -> str:
        lines = []
        for key, number in sorted(cites.items(), key=lambda kv: kv[1]):
            entry = REFERENCES[key]
            lines.append(f'{number}. {entry["text"]}')
            if entry['verified'] == 'TODO':
                result.todo.append(('refs', f'reference `{key}` is not verified: {entry["source"]}'))
        return '\n'.join(lines)

    text = REFERENCES_RE.sub(reference_list, text)
    text = CITE_RE.sub(cite, text)

    # -- numbers, table and figure references -----------------------------------------------
    text = sub_facts(text)

    def tab_ref(m) -> str:
        if m.group(1) not in tabs:
            raise BuildError(f'{{{{tab:{m.group(1)}}}}} has no <!-- TABLE:{m.group(1)} --> block')
        return f'Table {tabs[m.group(1)]}'

    def fig_ref(m) -> str:
        if m.group(1) not in figs:
            raise BuildError(f'{{{{fig:{m.group(1)}}}}} has no <!-- FIGURE:{m.group(1)} --> block')
        return f'Figure {figs[m.group(1)]}'

    text = TAB_RE.sub(tab_ref, text)
    text = FIG_RE.sub(fig_ref, text)

    banner = ('<!-- GENERATED by `python -m article_2.build_article` from article_2/src/*.md and '
              'results/. Edit the sources, never this file. -->\n\n')
    result.text = banner + text + '\n'

    # -- todo extraction ---------------------------------------------------------------------
    heading = ''
    for number, line in enumerate(result.text.splitlines(), start=1):
        if line.startswith('#'):
            heading = line.lstrip('# ').strip()
        for m in re.finditer(r'TODO\(([^)]*)\):?\s*([^`\n]*)', line):
            result.todo.append((m.group(1), f'{heading} (line {number}): {m.group(2).strip()}'))
        for m in re.finditer(r'<<VALUE:[^>]*>>', line):
            result.todo.append(('value', f'{heading} (line {number}): {m.group(0)}'))
        for m in re.finditer(r'\[PLACEHOLDER[^\]]*\]', line):
            result.todo.append(('placeholder', f'{heading} (line {number}): {m.group(0)}'))

    # -- checks ---------------------------------------------------------------------------------
    result.problems += check(result.text, strict=strict, build=result)
    return result


def check(text: str, *, strict: bool, build: Build) -> list:
    problems = []
    for m in re.finditer(r'\{\{[^}]*\}\}', text):
        problems.append(f'unresolved marker {m.group(0)}')
    if re.search(r'Table\s+Table|Figure\s+Figure', text):
        problems.append('a table or figure number is written twice ("Table Table")')
    flat = ' '.join(text.split())
    for sentence in COMMITTED_SENTENCES:
        if sentence not in flat:
            problems.append(f'committed sentence missing: {sentence[:60]}…')
    for sentence in re.split(r'(?<=[.!?])\s+', flat):
        if 'validat' in sentence.lower() and not any(n in sentence.lower() for n in NEGATIONS):
            problems.append(f'the paper says "evaluated", never "validated": {sentence[:120]}…')
    if strict:
        if build.generated_placeholders:
            problems.append(f'{build.generated_placeholders} placeholder(s) generated from missing data')
        for owner, what in build.todo:
            if owner == 'refs':
                problems.append(what)
    return problems


def write_outputs(build_result: Build, out_dir: Path, tables_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'article.md').write_text(build_result.text, encoding='utf-8')
    tables_dir.mkdir(parents=True, exist_ok=True)
    for key, (number, t) in build_result.tables.items():
        (tables_dir / f'{key}.md').write_text(render_table(t, number) + '\n', encoding='utf-8')

    lines = ['# Open items in the article draft', '',
             'Generated by `python -m article_2.build_article`. Owners: `coauthors`, `data`, '
             '`venue`, `refs`, `interpret`, `T9`, or a task id; `value` and `placeholder` are '
             'generated markers that stand where a number could not be computed.', '']
    by_owner: dict = {}
    for owner, what in build_result.todo:
        by_owner.setdefault(owner, []).append(what)
    for owner in sorted(by_owner):
        lines += [f'## {owner} ({len(by_owner[owner])})', '']
        lines += [f'- {what}' for what in by_owner[owner]]
        lines.append('')
    (out_dir / 'TODO.md').write_text('\n'.join(lines), encoding='utf-8')


#: the tables of the HAG article, in the order they appear in it
HAG_TABLES: tuple = ('terrain_classes', 'x3_by_terrain', 'dem_by_cell', 'x3_reduction', 'baseline',
                     'compute_by_terrain', 'stas', 'stas_by_cost', 'stas_dem', 'od_effect',
                     'height_interval')


def build_hag_outputs(data: 'Data', out: Path) -> int:
    """The HAG article's tables (numbered from 4, as in the article), Appendix A and the
    supplementary tables, from the result files alone."""
    tables = out / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    for number, key in enumerate(HAG_TABLES, 4):
        (tables / f'{key}.md').write_text(render_table(TABLES[key](data), number) + '\n',
                                          encoding='utf-8')
    (out / 'appendix.md').write_text(appendix_markdown(data), encoding='utf-8')
    (out / 'supplementary.md').write_text(supplementary_markdown(data), encoding='utf-8')
    print(f'{len(HAG_TABLES)} tables, appendix.md and supplementary.md written to {out}')
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--out', type=Path, default=HERE)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--no-figures', action='store_true',
                        help='do not (re)draw figures; use the committed files')
    parser.add_argument('--article', choices=('full', 'hag'), default='full',
                        help='hag: only the tables, appendix and supplement of the HAG article')
    args = parser.parse_args(argv)

    data = Data.load(args.results)
    if args.article == 'hag':
        return build_hag_outputs(data, args.out)
    sources = sorted((HERE / 'src').glob('*.md'))
    if not sources:
        print(f'no sources in {HERE / "src"}', file=sys.stderr)
        return 1
    try:
        result = build(data, sources, args.out / 'figures', args.out / 'tables',
                       make_figures=not args.no_figures, strict=args.strict)
    except BuildError as error:
        print(f'build error: {error}', file=sys.stderr)
        return 1
    write_outputs(result, args.out, args.out / 'tables')
    (args.out / 'index.md').write_text(index_markdown(data), encoding='utf-8')
    (args.out / 'appendix.md').write_text(appendix_markdown(data), encoding='utf-8')
    (args.out / 'supplementary.md').write_text(supplementary_markdown(data), encoding='utf-8')

    print(f'{args.out / "article.md"}: {len(result.text.split()):,} words, '
          f'{len(result.tables)} tables, {result.generated_placeholders} generated placeholder(s), '
          f'{len(result.todo)} open item(s) (see TODO.md)')
    for problem in result.problems:
        print(f'  PROBLEM: {problem}', file=sys.stderr)
    return 1 if result.problems else 0


if __name__ == '__main__':
    sys.exit(main())
