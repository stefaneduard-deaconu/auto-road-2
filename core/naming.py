"""The names new result files use, and how they map onto the older ones.

Every row of a new experiment carries three separate columns instead of one combined
"arm" label, so a reader never has to decode `hag_yellow_k1__gradpen_1`:

* `search_space`   which cells the detailed search may visit;
* `hag_edge_cost`  how an edge between two touching height areas is priced;
* `grid_edge_cost` how an edge between two neighbouring grid cells is priced.

"CTA" is the draft article's own term: the *candidate terrain areas* the HAG selects.
Older CSVs (`results/matrix.csv`, `results/gradient/`, `results/dem*/`, `results/stas/`)
keep their names; `LEGACY_SEARCH_ARMS` and `LEGACY_COST_ARMS` translate them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core import costs as cost_models
from core.hag import HAG_EDGE_COSTS


@dataclass(frozen=True)
class SearchSpace:
    name: str
    rule: str
    k_hops: int
    description: str


SEARCH_SPACES: dict[str, SearchSpace] = {s.name: s for s in (
    SearchSpace('full_grid', 'full', 0, 'every valid cell of the terrain'),
    SearchSpace('hag_cta', 'yellow', 0,
                'the candidate terrain areas: the areas on the cheapest HAG chain'),
    SearchSpace('hag_cta_ring1', 'yellow', 1,
                'the candidate terrain areas plus every area touching them'),
)}

#: both HAG edge costs are terrain costs; the study always uses `shared_border`.
HAG_EDGE_COST_DESCRIPTIONS: dict[str, str] = {
    'area_means': 'grid cost of (difference of mean area heights, centroid distance)',
    'shared_border': 'grid cost of (mean height step across the common contour, '
                     'centroid -> contour midpoint -> centroid)',
}
assert set(HAG_EDGE_COST_DESCRIPTIONS) == set(HAG_EDGE_COSTS)

GRID_EDGE_COST_DESCRIPTIONS: dict[str, str] = {
    'climb': '|height difference| (Equation 1)',
    'climb_tiebreak': '|height difference| + a 1e-6 per metre length tie-break (the default)',
    'length_3d': 'length along the terrain',
    'climb_plus_length': '|height difference| + planar length, equal weights',
    'climb_gradient_penalty': 'climb_tiebreak + 1 per metre per percent over i_max',
    'climb_gradient_cut': 'climb_tiebreak on a graph without the edges steeper than i_max',
}

#: the cost every experiment uses unless the grid edge cost is the variable under test
DEFAULT_GRID_EDGE_COST = 'climb_tiebreak'
#: the HAG edge cost of the study (agreed with the authors, 2026-10-01): every experiment
#: uses it; `area_means` stays configurable through `core.hag.HAG_EDGE_COSTS`.
DEFAULT_HAG_EDGE_COST = 'shared_border'


def grid_edge_cost(name: str, i_max_percent: Optional[float] = None
                   ) -> tuple[cost_models.CostSpec, Optional[float]]:
    """`(cost model, gradient cap)` for a grid edge cost name; the cap is set only by the cut."""
    if name == 'climb':
        return 'height', None
    if name == 'climb_tiebreak':
        return 'height_tiebreak', None
    if name == 'length_3d':
        return '3d', None
    if name == 'climb_plus_length':
        return cost_models.weighted_sum(height=1.0, length=1.0), None
    if name in ('climb_gradient_penalty', 'climb_gradient_cut'):
        if i_max_percent is None:
            raise ValueError(f'{name!r} needs the road class i_max_percent')
        if name == 'climb_gradient_cut':
            return 'height_tiebreak', float(i_max_percent)
        return cost_models.gradient_penalised('height_tiebreak', i_max_percent=i_max_percent,
                                              penalty_per_percent=1.0), None
    raise ValueError(f'unknown grid edge cost {name!r}; known: {sorted(GRID_EDGE_COST_DESCRIPTIONS)}')


#: old combined arm label -> (search_space, hag_edge_cost)
LEGACY_SEARCH_ARMS: dict[str, tuple[str, str]] = {
    'full_grid': ('full_grid', ''),
    'hag_yellow': ('hag_cta', 'shared_border'),
    'hag_yellow_k1': ('hag_cta_ring1', 'shared_border'),
}

#: old cost arm label (results/stas, results/gradient) -> grid edge cost
LEGACY_COST_ARMS: dict[str, str] = {
    'height': 'climb',
    'height_tiebreak': 'climb_tiebreak',
    '3d': 'length_3d',
    'weighted_h1_l1': 'climb_plus_length',
    'gradpen_1': 'climb_gradient_penalty',
    'gradcut': 'climb_gradient_cut',
}
