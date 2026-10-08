"""Is a road class's `i_max` reachable between two points at all? Screen, then certify.

The real-DEM run raised the question. On the Idrija crop the corner-to-corner pair drops
83.5 m over 791 m, so the STRAIGHT line already needs 10.6% where `RO_CLASS_V_DEAL` allows
7%. No search and no smoothing can fix that: the pipeline minimises `sum |dh|`, not the
gradient, and it never develops length (serpentines) to trade gradient for distance. A
matrix built on such pairs would report infeasibility that was decided by the endpoints,
not measured by the method.

Two steps, cheap first:

**1. Screen (`screen_grade`, O(1)).** The shortest possible alignment is the straight line,
so `|dz| / straight_distance` is the lowest average gradient any DIRECT alignment can have.
Above `i_max`, the pair needs development length: at least `|dz| / i_max` of it. This is a
necessary condition for a direct road and it costs two array lookups.

**2. Certify (`certify_grade`, one Dijkstra).** The screen is about averages, and an average
inside the limit says nothing about the ground in between. So build the grid graph with
every edge steeper than `i_max` REMOVED (`core.search.build_graph(max_abs_gradient_percent=)`)
and search it for the shortest remaining path. If one exists it is a witness: every one of
its steps holds the gradient, at cell resolution. If none exists, no alignment on this grid
does, whatever the algorithm.

**What a certificate does NOT say.** Three limits, all reported rather than hidden:

- it is a VERTICAL certificate only. The witness is free to snake along the contours, so its
  R_min is usually far below any class. `GradeCertificate.r_min_m` is measured and reported
  for exactly that reason, and `development_ratio` says how much longer than the straight
  line the witness had to be;
- it holds at CELL RESOLUTION. Algorithm 1 moves the axis in XY and re-samples Z from the
  terrain, so the smoothed alignment's gradients are not the witness's gradients. A
  certificate is necessary, not sufficient;
- gradients here are per grid step. On a 1 m DEM that is a 1 m step, which picks up LiDAR
  micro-relief; `chord_m` on a real design is much longer. A pair that fails at 1 m may pass
  on the same terrain resampled to 10 m, which is a property of the measurement scale and
  has to be stated with any number taken from here.

`find_feasible_pairs` samples pairs that pass both steps. It is a DELIBERATE selection and
is kept apart from `core.od_sampler`, whose protocol generates pairs without looking at the
outcome on purpose: filtering there would hide the failure modes
the summary is asked to count. Use this to choose a demonstration case, and say in the text
that the pair was chosen for gradient viability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Iterator, Optional, TypeAlias

import numpy as np

from core import metrics as metrics_mod
from core.costs import DEFAULT_COST, CostSpec
from core.grid import Connectivity, Grid
from core.od_sampler import MARGIN_CELLS, MIN_DIAGONAL_FRACTION
from core.search import GridGraph, build_graph, dijkstra
from data.configs.road_classes import RoadClass

#: a witness longer than this many times the straight line is reported as a snake: it holds
#: the gradient only by following the contours, which is not a road alignment any more
DEVELOPMENT_RATIO_WARNING: Final[float] = 2.0

#: a grid cell as (row, column)
Cell: TypeAlias = tuple[int, int]
#: what `grade_limited_graphs` returns: the unrestricted graph and the i_max-limited one
GradeGraphs: TypeAlias = tuple[GridGraph, GridGraph]


@dataclass(frozen=True)
class GradeScreen:
    """The O(1) necessary condition: can a DIRECT alignment hold `i_max`?"""

    start: Cell
    target: Cell
    z_start_m: float
    z_target_m: float
    dz_m: float
    straight_distance_m: float
    straight_gradient_percent: float
    i_max_percent: float
    #: the shortest alignment that can lose `dz` without ever exceeding `i_max`
    min_length_m: float
    #: `min_length_m / straight_distance_m`; 1.0 means no development length is needed
    development_needed: float
    passed: bool

    @property
    def reason(self) -> str:
        if self.passed:
            return (f"the straight line needs {self.straight_gradient_percent:.2f}%, within "
                    f"{self.i_max_percent:.1f}%")
        return (f"the straight line already needs {self.straight_gradient_percent:.2f}%, "
                f"above {self.i_max_percent:.1f}%; no alignment shorter than "
                f"{self.min_length_m:,.0f} m can hold the limit")

    def as_dict(self) -> dict:
        return {'start': list(self.start), 'target': list(self.target),
                'z_start_m': self.z_start_m, 'z_target_m': self.z_target_m,
                'dz_m': self.dz_m, 'straight_distance_m': self.straight_distance_m,
                'straight_gradient_percent': self.straight_gradient_percent,
                'i_max_percent': self.i_max_percent, 'min_length_m': self.min_length_m,
                'development_needed': self.development_needed, 'passed': self.passed}


@dataclass(frozen=True)
class GradeCertificate:
    """The exact answer, with a witness alignment when there is one."""

    start: Cell
    target: Cell
    i_max_percent: float
    reachable: bool
    #: (n, 2) grid coordinates of the shortest gradient-admissible path, or None
    path_xy: Optional[np.ndarray]
    length_m: float
    #: the witness's own worst gradient; <= `i_max_percent` by construction
    witness_i_max_percent: float
    #: the witness's smallest curve radius. NOT constrained here, and usually tiny
    r_min_m: float
    development_ratio: float
    edges_kept: int
    edges_total: int

    @property
    def edge_keep_fraction(self) -> float:
        return self.edges_kept / self.edges_total if self.edges_total else 0.0

    @property
    def snaking(self) -> bool:
        return self.reachable and self.development_ratio > DEVELOPMENT_RATIO_WARNING

    @property
    def verdict(self) -> str:
        if not self.reachable:
            return (f"NO alignment on this grid holds {self.i_max_percent:.1f}%: the "
                    f"endpoints are in different components once the steep edges are cut")
        if self.snaking:
            return (f"gradient-feasible, but the witness is {self.development_ratio:.1f}x "
                    f"the straight line: it holds the limit by following the contours")
        return (f"gradient-feasible: a {self.length_m:,.0f} m alignment holds "
                f"{self.i_max_percent:.1f}% ({self.development_ratio:.2f}x the straight line)")

    def as_dict(self) -> dict:
        return {'start': list(self.start), 'target': list(self.target),
                'i_max_percent': self.i_max_percent, 'reachable': self.reachable,
                'length_m': self.length_m,
                'witness_i_max_percent': self.witness_i_max_percent,
                'r_min_m': self.r_min_m, 'development_ratio': self.development_ratio,
                'edges_kept': self.edges_kept, 'edges_total': self.edges_total,
                'edge_keep_fraction': self.edge_keep_fraction, 'verdict': self.verdict}


def screen_grade(grid: Grid, start: Cell, target: Cell,
                 road_class: RoadClass) -> GradeScreen:
    """Step 1: the O(1) necessary condition on the straight line."""
    start, target = tuple(int(x) for x in start), tuple(int(x) for x in target)
    z0 = float(grid.surf[start])
    z1 = float(grid.surf[target])
    dz = z1 - z0
    distance = math.dist(start, target) * grid.cell_size_m
    if distance <= 0:
        raise ValueError(f"start and target are the same cell {start}")
    gradient = 100.0 * abs(dz) / distance
    min_length = abs(dz) / (road_class.i_max_percent / 100.0)
    return GradeScreen(start=start, target=target, z_start_m=z0, z_target_m=z1, dz_m=dz,
                       straight_distance_m=distance, straight_gradient_percent=gradient,
                       i_max_percent=road_class.i_max_percent, min_length_m=min_length,
                       development_needed=min_length / distance,
                       passed=gradient <= road_class.i_max_percent)


def grade_limited_graphs(grid: Grid, road_class: RoadClass,
                         connectivity: Connectivity = 8,
                         cost: CostSpec = DEFAULT_COST) -> GradeGraphs:
    """`(full graph, gradient-limited graph)`, built once and reused over many O-D pairs.

    Both are a function of the terrain and the class only, so screening a hundred candidates
    costs a hundred Dijkstras, not a hundred graph builds.
    """
    return (build_graph(grid, cost=cost, connectivity=connectivity),
            build_graph(grid, cost=cost, connectivity=connectivity,
                        max_abs_gradient_percent=road_class.i_max_percent))


def certify_grade(grid: Grid, start: Cell, target: Cell, road_class: RoadClass,
                  connectivity: Connectivity = 8,
                  cost: CostSpec = DEFAULT_COST,
                  metric_step_m: float = metrics_mod.DEFAULT_RESAMPLE_STEP_M,
                  metric_chord_m: float = metrics_mod.DEFAULT_CHORD_M,
                  graphs: Optional[GradeGraphs] = None) -> GradeCertificate:
    """Step 2: search the grid with every edge steeper than `i_max` removed.

    The witness minimises `cost` among the admissible paths, so with the default cost it is
    the one a normal run would have found had the steep ground not been there. `graphs` is
    the pair from `grade_limited_graphs`, to avoid rebuilding them per candidate.
    """
    start, target = tuple(int(x) for x in start), tuple(int(x) for x in target)
    full, limited = (grade_limited_graphs(grid, road_class, connectivity, cost)
                     if graphs is None else graphs)
    result = dijkstra(limited, grid.index(start), grid.index(target))
    straight = math.dist(start, target) * grid.cell_size_m

    if not result.reached:
        return GradeCertificate(start=start, target=target,
                                i_max_percent=road_class.i_max_percent, reachable=False,
                                path_xy=None, length_m=math.inf,
                                witness_i_max_percent=math.nan, r_min_m=math.nan,
                                development_ratio=math.inf,
                                edges_kept=limited.n_edges, edges_total=full.n_edges)

    path_xy = result.coords(grid)
    xyz = grid.path_xyz_m(path_xy.astype(float))
    measured = metrics_mod.measure(xyz, step_m=metric_step_m, chord_m=metric_chord_m)
    return GradeCertificate(start=start, target=target,
                            i_max_percent=road_class.i_max_percent, reachable=True,
                            path_xy=path_xy, length_m=measured.length_m,
                            witness_i_max_percent=measured.i_max_percent,
                            r_min_m=measured.r_min_m,
                            development_ratio=measured.length_m / straight,
                            edges_kept=limited.n_edges, edges_total=full.n_edges)


def _candidate_pairs(grid: Grid, rng: np.random.Generator, margin: int,
                     min_fraction: float,
                     max_candidates: int) -> Iterator[tuple[Cell, Cell]]:
    """The same shape of candidate as `core.od_sampler`: inset from the border, not short."""
    diagonal = math.hypot(grid.n_rows - 1, grid.n_cols - 1)
    lo_i, hi_i = margin, grid.n_rows - 1 - margin
    lo_j, hi_j = margin, grid.n_cols - 1 - margin
    if hi_i <= lo_i or hi_j <= lo_j:
        raise ValueError(f"a margin of {margin} cells leaves nothing in a {grid.shape} grid")
    for _ in range(max_candidates):
        i1 = int(rng.integers(lo_i, hi_i + 1))
        j1 = int(rng.integers(lo_j, hi_j + 1))
        i2 = int(rng.integers(lo_i, hi_i + 1))
        j2 = int(rng.integers(lo_j, hi_j + 1))
        if math.dist((i1, j1), (i2, j2)) >= min_fraction * diagonal:
            yield (i1, j1), (i2, j2)


@dataclass(frozen=True)
class FeasiblePair:
    screen: GradeScreen
    certificate: GradeCertificate
    candidates_tried: int

    @property
    def start(self) -> Cell:
        return self.screen.start

    @property
    def target(self) -> Cell:
        return self.screen.target


def find_feasible_pairs(grid: Grid, road_class: RoadClass, *, n_pairs: int = 2,
                        seed: int = 0, margin: int = MARGIN_CELLS,
                        min_diagonal_fraction: float = MIN_DIAGONAL_FRACTION,
                        max_candidates: int = 2000,
                        max_development_ratio: float = DEVELOPMENT_RATIO_WARNING,
                        min_separation_cells: float = 0.0,
                        connectivity: Connectivity = 8) -> list[FeasiblePair]:
    """Sample O-D pairs that pass the screen and then the certificate.

    Deterministic given `seed`. Candidates are screened first because the screen is free and
    the certificate costs two graph builds and a Dijkstra. `min_separation_cells` keeps the
    returned pairs from being small variations of each other.
    """
    rng = np.random.default_rng(seed)
    found: list[FeasiblePair] = []
    tried = 0
    graphs = None
    for start, target in _candidate_pairs(grid, rng, margin, min_diagonal_fraction,
                                          max_candidates):
        tried += 1
        screen = screen_grade(grid, start, target, road_class)
        if not screen.passed:
            continue
        if any(math.dist(start, p.start) < min_separation_cells
               and math.dist(target, p.target) < min_separation_cells for p in found):
            continue
        if graphs is None:          # only once a candidate has survived the free screen
            graphs = grade_limited_graphs(grid, road_class, connectivity)
        certificate = certify_grade(grid, start, target, road_class,
                                    connectivity=connectivity, graphs=graphs)
        if not certificate.reachable or certificate.development_ratio > max_development_ratio:
            continue
        found.append(FeasiblePair(screen=screen, certificate=certificate,
                                  candidates_tried=tried))
        if len(found) >= n_pairs:
            break
    return found
