"""Dijkstra on integer node ids over a CSR adjacency, with work counters.

This is the trustworthy baseline the evaluation needs. It really builds the
full-grid graph, and there is no pickle cache anywhere: a graph is
cheap to rebuild and a cache keyed on too little silently reuses another terrain's
graph.

`tests/test_core_search.py` checks the distances against `scipy.sparse.csgraph.dijkstra`
on 20 random grids, which is the acceptance criterion of T1.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core import costs as cost_models
from core.grid import Connectivity, Grid, edge_arrays

INF = float('inf')


@dataclass
class SearchStats:
    """How much work the search did. `nodes_expanded` is research step 1's measure."""

    nodes_expanded: int = 0  # nodes popped with a still-current distance and relaxed
    nodes_pushed: int = 0    # heap insertions
    nodes_reached: int = 0   # nodes that ended with a finite distance
    n_nodes: int = 0         # nodes in the graph that was searched
    n_edges: int = 0         # directed edges in that graph

    def as_dict(self) -> dict:
        return {'nodes_expanded': self.nodes_expanded, 'nodes_pushed': self.nodes_pushed,
                'nodes_reached': self.nodes_reached, 'n_nodes': self.n_nodes,
                'n_edges': self.n_edges}


@dataclass
class GridGraph:
    """A CSR adjacency over grid node ids, plus the raw edge attributes.

    `weights[k]` is the search cost of edge `k`, `lengths_m[k]` its planar length and
    `dh[k]` its height difference. `indptr`/`indices` index into those arrays, so an
    edge's attributes are recoverable from the CSR position.
    """

    grid: Grid
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    lengths_m: np.ndarray
    dh: np.ndarray
    mask: Optional[np.ndarray] = None
    cost_name: str = cost_models.DEFAULT_COST
    #: when set, edges steeper than this were dropped (see `build_graph`)
    max_abs_gradient_percent: Optional[float] = None

    @property
    def n_nodes(self) -> int:
        return self.grid.n_nodes

    @property
    def n_edges(self) -> int:
        return int(self.indices.size)

    @property
    def n_searchable_nodes(self) -> int:
        """Nodes the search may visit: the whole grid, or the cells inside `mask`."""
        return self.n_nodes if self.mask is None else int(np.count_nonzero(self.mask))


def _weights_for(model, cost_name: str, dh: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Evaluate a cost model on real edges and refuse anything Dijkstra cannot take.

    Shared by `build_graph` and `with_cost` so a graph whose weights were swapped is
    validated exactly as one built from scratch. The probe in `costs.validate` cannot see
    a real terrain; this can.
    """
    #: broadcast, so a constant-cost model (`lambda dh, length_m: 1.0`) is one line
    weights = np.broadcast_to(np.asarray(model(dh, lengths), dtype=float),
                              dh.shape).astype(float, copy=True)
    if weights.size and not np.isfinite(weights).all():
        raise ValueError(f"cost model {cost_name!r} produced a non-finite weight on this "
                         "terrain; Dijkstra needs finite weights")
    if weights.size and weights.min() < 0:
        raise ValueError(f"cost model {cost_name!r} produced a negative weight "
                         f"({weights.min()}) on this terrain; Dijkstra needs non-negative "
                         "weights")
    return weights


def build_graph(grid: Grid,
                cost: cost_models.CostSpec = cost_models.DEFAULT_COST,
                mask: Optional[np.ndarray] = None,
                connectivity: Connectivity = 8,
                max_abs_gradient_percent: Optional[float] = None) -> GridGraph:
    """Build the CSR grid graph. With `mask=None` this is the full-grid baseline.

    `cost` is a registered name or any `(dh, length_m) -> weights` callable, so an
    experiment can pass one of `core.costs`' factories or its own function without
    registering it first; `graph.cost_name` records which, for the result row.

    `mask` removes CELLS (that is how the HAG restriction works); `max_abs_gradient_percent`
    removes EDGES whose slope `|dh| / length` exceeds it. The two are independent, and the
    edge filter is what `core.od_feasibility` uses to decide whether any alignment between
    two points can hold a road class's `i_max`: a path in the filtered graph is one, edge by
    edge, and no path at all means none exists at cell resolution.
    """
    model = cost_models.resolve(cost)
    cost_name = cost_models.name_of(cost)
    u, v, lengths, dh = edge_arrays(grid, mask=mask, connectivity=connectivity)
    if max_abs_gradient_percent is not None:
        if max_abs_gradient_percent <= 0:
            raise ValueError("max_abs_gradient_percent must be positive; "
                             f"got {max_abs_gradient_percent}")
        keep = np.abs(dh) <= (max_abs_gradient_percent / 100.0) * lengths
        u, v, lengths, dh = u[keep], v[keep], lengths[keep], dh[keep]
    #: broadcast, so a constant-cost model (`lambda dh, length_m: 1.0`) is one line
    weights = _weights_for(model, cost_name, dh, lengths)
    order = np.argsort(u, kind='stable')
    u, v = u[order], v[order]
    weights, lengths, dh = weights[order], lengths[order], dh[order]
    indptr = np.zeros(grid.n_nodes + 1, dtype=np.int64)
    np.add.at(indptr, u + 1, 1)
    np.cumsum(indptr, out=indptr)
    return GridGraph(grid=grid, indptr=indptr, indices=v, weights=weights,
                     lengths_m=lengths, dh=dh, mask=mask, cost_name=cost_name,
                     max_abs_gradient_percent=max_abs_gradient_percent)


@dataclass
class SearchResult:
    path: Optional[np.ndarray]          # (k,) node ids, or None when unreachable
    cost: float                          # total edge cost of `path`
    distances: np.ndarray                # (n_nodes,) shortest cost, inf where unreached
    stats: SearchStats = field(default_factory=SearchStats)
    #: (n_nodes,) predecessor of every settled node, -1 where there is none. One sweep
    #: therefore yields a path to EVERY reached node, not only to `target`, which is what
    #: `dijkstra_many` and the portal tables of `core.network` are built on.
    previous: Optional[np.ndarray] = None

    @property
    def reached(self) -> bool:
        return self.path is not None

    def coords(self, grid: Grid) -> np.ndarray:
        """The path as an (k, 2) array of (i, j) grid coordinates."""
        if self.path is None:
            raise ValueError('no path')
        return grid.coords(self.path)


def dijkstra(graph: GridGraph, start: int, target: int,
             stop_at_target: bool = True) -> SearchResult:
    """Shortest path from `start` to `target` over `graph`.

    `stop_at_target=True` stops as soon as the target is settled, which is what an
    honest "nodes expanded" comparison between the full grid and the HAG needs; the
    distances of nodes further away than the target are then left at infinity.
    """
    n = graph.n_nodes
    start, target = int(start), int(target)
    if not (0 <= start < n and 0 <= target < n):
        raise IndexError(f"start/target outside a graph of {n} nodes")
    indptr, indices, weights = graph.indptr, graph.indices, graph.weights

    distances = np.full(n, INF, dtype=float)
    previous = np.full(n, -1, dtype=np.int64)
    distances[start] = 0.0
    stats = SearchStats(n_nodes=graph.n_searchable_nodes, n_edges=graph.n_edges)
    queue = [(0.0, start)]
    stats.nodes_pushed = 1
    settled = np.zeros(n, dtype=bool)

    while queue:
        dist, node = heapq.heappop(queue)
        if settled[node]:
            continue
        settled[node] = True
        stats.nodes_expanded += 1
        if stop_at_target and node == target:
            break
        lo, hi = int(indptr[node]), int(indptr[node + 1])
        for k in range(lo, hi):
            neighbor = int(indices[k])
            candidate = dist + float(weights[k])
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
                stats.nodes_pushed += 1

    stats.nodes_reached = int(np.count_nonzero(np.isfinite(distances)))
    if not np.isfinite(distances[target]):
        return SearchResult(path=None, cost=INF, distances=distances, stats=stats,
                            previous=previous)
    return SearchResult(path=_trace(previous, start, target), cost=float(distances[target]),
                        distances=distances, stats=stats, previous=previous)


def _trace(previous: np.ndarray, start: int, target: int) -> np.ndarray:
    """Walk the predecessor array back from `target` to `start`."""
    path = [int(target)]
    while path[-1] != start:
        path.append(int(previous[path[-1]]))
    path.reverse()
    return np.array(path, dtype=np.int64)


@dataclass
class ManyResult:
    """One sweep, every requested target answered.

    `costs[k]` and `paths[k]` belong to `targets[k]`; an unreachable target has cost `inf`
    and a `None` path. `stats` describes the single sweep, so `nodes_expanded` here is the
    honest work figure for answering ALL of the targets at once.
    """

    start: int
    targets: np.ndarray
    costs: np.ndarray
    paths: list[Optional[np.ndarray]]
    distances: np.ndarray
    previous: np.ndarray
    stats: SearchStats = field(default_factory=SearchStats)

    def cost_to(self, target: int) -> float:
        return float(self.distances[int(target)])


def dijkstra_many(graph: GridGraph, start: int, targets) -> ManyResult:
    """Shortest paths from `start` to several targets in ONE sweep.

    `core.network` needs the cost AND the geometry between every pair of portals of an
    area. Calling `dijkstra` once per pair would re-expand the same area |P| times over and
    would still not give the polylines, because `dijkstra` keeps its predecessors to
    itself. Here one sweep per source settles every target and `previous` reconstructs all
    of the paths, so an area with |P| portals costs |P| sweeps rather than |P| squared.

    The sweep stops as soon as the last target is settled, exactly as
    `dijkstra(stop_at_target=True)` does for one, so distances beyond the furthest target
    are left at infinity. A target that the graph cannot reach keeps the sweep running to
    exhaustion, which is the only honest thing to do: nothing else proves unreachability.
    """
    n = graph.n_nodes
    start = int(start)
    if not 0 <= start < n:
        raise IndexError(f"start outside a graph of {n} nodes")
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if targets.size and (targets.min() < 0 or targets.max() >= n):
        raise IndexError(f"a target is outside a graph of {n} nodes")

    indptr, indices, weights = graph.indptr, graph.indices, graph.weights
    distances = np.full(n, INF, dtype=float)
    previous = np.full(n, -1, dtype=np.int64)
    distances[start] = 0.0
    stats = SearchStats(n_nodes=graph.n_searchable_nodes, n_edges=graph.n_edges)

    wanted = np.zeros(n, dtype=bool)
    wanted[targets] = True
    remaining = int(np.count_nonzero(wanted))

    queue = [(0.0, start)]
    stats.nodes_pushed = 1
    settled = np.zeros(n, dtype=bool)
    while queue and remaining:
        dist, node = heapq.heappop(queue)
        if settled[node]:
            continue
        settled[node] = True
        stats.nodes_expanded += 1
        if wanted[node]:
            remaining -= 1
            if not remaining:
                break
        lo, hi = int(indptr[node]), int(indptr[node + 1])
        for k in range(lo, hi):
            neighbor = int(indices[k])
            candidate = dist + float(weights[k])
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
                stats.nodes_pushed += 1

    stats.nodes_reached = int(np.count_nonzero(np.isfinite(distances)))
    costs = np.array([distances[t] for t in targets], dtype=float)
    paths: list[Optional[np.ndarray]] = [
        _trace(previous, start, int(t)) if np.isfinite(distances[t]) else None
        for t in targets]
    return ManyResult(start=start, targets=targets, costs=costs, paths=paths,
                      distances=distances, previous=previous, stats=stats)


def with_cost(graph: GridGraph, cost: cost_models.CostSpec) -> GridGraph:
    """The same graph under a different cost model, without rebuilding the CSR.

    A cost model changes only what an edge is worth, never which edges exist, so
    `indptr`, `indices`, `lengths_m` and `dh` all survive a change of objective untouched.
    That is the whole saving of updating the graph when the objective function changes, and
    of the U4 update of
    `core.network_update`: re-costing is one vectorised call where rebuilding is a sort
    and a scan over every edge of the grid.

    `mask` and `max_abs_gradient_percent` are carried over, because they describe which
    edges were dropped when the CSR was built and are still true of it. A graph that was
    built gradient-limited stays gradient-limited and says so.
    """
    model = cost_models.resolve(cost)
    cost_name = cost_models.name_of(cost)
    return GridGraph(grid=graph.grid, indptr=graph.indptr, indices=graph.indices,
                     weights=_weights_for(model, cost_name, graph.dh, graph.lengths_m),
                     lengths_m=graph.lengths_m, dh=graph.dh, mask=graph.mask,
                     cost_name=cost_name,
                     max_abs_gradient_percent=graph.max_abs_gradient_percent)


def edge_sources(graph: GridGraph) -> np.ndarray:
    """The `u` column the CSR implies: one source node id per edge position."""
    return np.repeat(np.arange(graph.n_nodes, dtype=np.int64), np.diff(graph.indptr))


def subgraph(graph: GridGraph, mask: np.ndarray,
             sources: Optional[np.ndarray] = None) -> GridGraph:
    """The same graph restricted to the cells of `mask`, SLICED rather than rebuilt.

    `build_graph(mask=...)` walks the whole grid once per neighbour offset, so building one
    graph per height area costs the full grid over and over. `core.network` needs exactly
    that - a graph per area - and on a 600x600 DEM crop with a hundred areas the rebuild
    dominates everything else. Slicing an already-built CSR instead costs one pass over the
    edges plus one over the nodes.

    Node ids stay global (`indptr` keeps its full length), so a subgraph, the full graph and
    a gradient-limited graph over one `Grid` remain directly comparable, which is what makes
    the portal tables cacheable in the first place.

    The result is identical to `build_graph` with the same mask, and
    `tests/test_core_search_many.py` pins that. An existing `mask` is intersected, never
    dropped.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != graph.grid.shape:
        raise ValueError(f"mask shape {mask.shape} != grid shape {graph.grid.shape}")
    combined = mask if graph.mask is None else (mask & np.asarray(graph.mask, dtype=bool))
    flat = combined.reshape(-1)

    n = graph.n_nodes
    #: `sources` is O(edges) to derive and identical for every subgraph of one graph, so a
    #: caller slicing one graph per height area computes it once and passes it in
    if sources is None:
        sources = edge_sources(graph)
    keep = flat[sources] & flat[graph.indices]

    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(sources[keep], minlength=n))
    return GridGraph(grid=graph.grid, indptr=indptr, indices=graph.indices[keep],
                     weights=graph.weights[keep], lengths_m=graph.lengths_m[keep],
                     dh=graph.dh[keep], mask=combined, cost_name=graph.cost_name,
                     max_abs_gradient_percent=graph.max_abs_gradient_percent)


def path_cost(graph: GridGraph, path: np.ndarray) -> float:
    """Total edge cost of an explicit node path, recomputed from the graph."""
    return float(sum(_edge_attr(graph, a, b, graph.weights) for a, b in zip(path, path[1:])))


def path_length_m(graph: GridGraph, path: np.ndarray) -> float:
    return float(sum(_edge_attr(graph, a, b, graph.lengths_m) for a, b in zip(path, path[1:])))


def path_height_cost(graph: GridGraph, path: np.ndarray) -> float:
    """Equation 1: sum of |height difference| along the path."""
    return float(sum(abs(_edge_attr(graph, a, b, graph.dh)) for a, b in zip(path, path[1:])))


def _edge_attr(graph: GridGraph, u: int, v: int, attr: np.ndarray) -> float:
    lo, hi = int(graph.indptr[u]), int(graph.indptr[u + 1])
    for k in range(lo, hi):
        if int(graph.indices[k]) == int(v):
            return float(attr[k])
    raise KeyError(f"({u}, {v}) is not an edge of this graph")
