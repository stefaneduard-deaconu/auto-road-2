"""Geometry measures for a road axis. All inputs and outputs are in METERS.

Research step 2 (engineering checks) and step 3 (Algorithm 1 before/after) both need the
same numbers, so there is one implementation here and the checks in `core.checks` only
compare them against a `RoadClass`.

THE CURVATURE METHOD (bad point 8 / S2 item 6)
----------------------------------------------
A three-point circumradius applied to whatever point spacing the caller happens to have
makes `R_min` ambiguous. A
B-spline resampled at 8 points per control point gives closely spaced points, and a
three-point circumradius over closely spaced points is dominated by numerical noise, so
the reported minimum radius depends on the resampling and not on the road: a curve
smoothed with `minimal_radius = 25` can then measure 16 m.

ONE method is defined here and used everywhere:

    1. Resample the polyline at a constant arc-length step `step_m`
       (`DEFAULT_RESAMPLE_STEP_M`, 5 m), by linear interpolation along the polyline.
    2. Take every triple of resampled points separated by a fixed arc length
       `chord_m` (`DEFAULT_CHORD_M`, 20 m), i.e. (P[i], P[i+w], P[i+2w]) with
       w = round(chord_m / step_m).
    3. For each triple take the Menger circumradius R = (a*b*c) / (4*A), with a, b, c
       the side lengths and A the triangle area. Collinear triples give R = inf.
    4. `R_min` is the minimum over all such triples.

Why the fixed chord and not consecutive points: the circumradius of three points at
spacing h on a curve of radius R is controlled by a sagitta of about h^2 / (8R). At
h = 5 m and R = 650 m that sagitta is 5 mm, so a few millimetres of polyline
discretisation move the estimate by several percent, and on a near-straight stretch it
can invent a small radius out of nothing. At h = 20 m the sagitta is 8 cm and the
estimate is stable. The formula stays exact for points that really lie on a circle, at
any spacing, so the method loses no accuracy on genuine curves.

`step_m` and `chord_m` are explicit, reported parameters: a three-point circumradius is
a chord measure, so a result is only comparable to another one computed with the same
two. Both are stored in `PathMetrics` and written into every result row.

Known bias: the resampled points lie on the chords of the input polyline, i.e. inside a
convex curve, so a polyline approximating a circle measures slightly BELOW the true
radius (under 1% at the default step and chord). The bias is conservative for an
engineering check, which is the direction to be wrong in, and it is the reason the
tests compare against a band rather than an exact equality.

This is a *planimetric* radius of the road axis polyline. It is not a designed circular
arc with transition curves; the norm's "raza minima a curbei in plan" applies to a
designed arc. The evaluation therefore reports it as a lower bound on the geometry the
generated alignment offers, which is the conservative direction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from scipy.spatial.distance import directed_hausdorff

#: arc-length step the path is resampled at before measuring curvature, in meters.
DEFAULT_RESAMPLE_STEP_M = 5.0
#: arc length between the three points of a circumradius triple, in meters.
DEFAULT_CHORD_M = 20.0

INF = float('inf')


def _as_xy(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] < 2:
        raise ValueError(f"path must be (n, 2) or (n, 3), got shape {path.shape}")
    return path[:, :2]


def _drop_repeats(points: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    if len(points) < 2:
        return points
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > tol
    return points[keep]


# -- length, sinuosity -------------------------------------------------------------------

def segment_lengths_m(path: np.ndarray) -> np.ndarray:
    """Planar length of every segment, in meters."""
    xy = _as_xy(path)
    return np.linalg.norm(np.diff(xy, axis=0), axis=1)


def path_length_m(path: np.ndarray) -> float:
    """Planimetric length of the axis, in meters."""
    return float(segment_lengths_m(path).sum())


def path_length_3d_m(path: np.ndarray) -> float:
    """Length along the terrain, in meters. Needs a 3-column path."""
    xyz = np.asarray(path, dtype=float)
    if xyz.shape[1] < 3:
        raise ValueError('path_length_3d_m needs (x, y, z)')
    return float(np.linalg.norm(np.diff(xyz[:, :3], axis=0), axis=1).sum())


def straight_line_distance_m(path: np.ndarray) -> float:
    xy = _as_xy(path)
    return float(np.linalg.norm(xy[-1] - xy[0]))


#: a path whose ends are closer together than this fraction of its length counts as closed
CLOSED_PATH_TOLERANCE = 1e-9


def sinuosity(path: np.ndarray) -> float:
    """Path length divided by the straight-line distance between its ends (>= 1).

    `inf` for a closed path, i.e. one whose ends coincide to within
    `CLOSED_PATH_TOLERANCE` of its own length.
    """
    straight = straight_line_distance_m(path)
    length = path_length_m(path)
    if straight <= CLOSED_PATH_TOLERANCE * max(length, 1.0):
        return INF
    return length / straight


# -- resampling and curvature -------------------------------------------------------------

def resample_by_arclength(path: np.ndarray, step_m: float = DEFAULT_RESAMPLE_STEP_M) -> np.ndarray:
    """Points along the polyline at a constant arc-length spacing.

    The first and last points are kept exactly. A path shorter than one step is
    returned as just its two ends.
    """
    xy = _drop_repeats(_as_xy(path))
    if len(xy) < 2:
        return xy
    if step_m <= 0:
        raise ValueError('step_m must be positive')
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    total = float(s[-1])
    if total < step_m:
        return xy[[0, -1]]
    n_steps = max(int(np.floor(total / step_m)), 1)
    targets = np.linspace(0.0, total, n_steps + 1)
    return np.column_stack([np.interp(targets, s, xy[:, 0]), np.interp(targets, s, xy[:, 1])])


def menger_radii(points: np.ndarray, stride: int = 1) -> np.ndarray:
    """Circumradius of every triple (P[i], P[i+stride], P[i+2*stride]) of `points`.

    Collinear triples give inf.
    """
    points = np.asarray(points, dtype=float)
    stride = max(int(stride), 1)
    if len(points) < 2 * stride + 1:
        return np.empty(0, dtype=float)
    p1, p2, p3 = points[:-2 * stride], points[stride:-stride], points[2 * stride:]
    a = np.linalg.norm(p1 - p2, axis=1)
    b = np.linalg.norm(p2 - p3, axis=1)
    c = np.linalg.norm(p3 - p1, axis=1)
    # twice the triangle area, from the 2D cross product (exact, unlike Heron's formula
    # which loses precision on the very thin triangles a straight road produces)
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    area2 = np.abs(cross)
    with np.errstate(divide='ignore', invalid='ignore'):
        radii = np.where(area2 > 0, (a * b * c) / (2.0 * area2), INF)
    return radii


def curve_radii_m(path: np.ndarray,
                  step_m: float = DEFAULT_RESAMPLE_STEP_M,
                  chord_m: float = DEFAULT_CHORD_M) -> np.ndarray:
    """THE curvature method (see the module docstring).

    Arc-length-resample at `step_m`, then take circumradii of triples separated by
    `chord_m` of arc. On a path too short for one full triple the stride shrinks so
    that at least one triple is measured; on a path with fewer than 3 points the result
    is empty and `r_min_m` returns inf.
    """
    points = resample_by_arclength(path, step_m)
    stride = max(int(round(chord_m / step_m)), 1)
    while stride > 1 and len(points) < 2 * stride + 1:
        stride -= 1
    return menger_radii(points, stride=stride)


def r_min_m(path: np.ndarray,
            step_m: float = DEFAULT_RESAMPLE_STEP_M,
            chord_m: float = DEFAULT_CHORD_M) -> float:
    """Smallest horizontal curve radius of the axis, in meters. `inf` for a straight line."""
    radii = curve_radii_m(path, step_m, chord_m)
    return float(radii.min()) if radii.size else INF


# -- gradients ------------------------------------------------------------------------------

def gradients_percent(path_xyz: np.ndarray) -> np.ndarray:
    """Longitudinal gradient of every segment, in percent, signed.

    A segment with no horizontal extent is skipped rather than reported as an infinite
    gradient; a road axis should not contain one.
    """
    xyz = np.asarray(path_xyz, dtype=float)
    if xyz.shape[1] < 3:
        raise ValueError('gradients_percent needs (x, y, z) in meters')
    horizontal = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
    dz = np.diff(xyz[:, 2])
    ok = horizontal > 0
    return 100.0 * dz[ok] / horizontal[ok]


def i_max_percent(path_xyz: np.ndarray) -> float:
    """Largest absolute longitudinal gradient, in percent."""
    g = gradients_percent(path_xyz)
    return float(np.abs(g).max()) if g.size else 0.0


def elevation_difference_m(path_xyz: np.ndarray) -> dict:
    """Net, total ascent, total descent and span of the elevations along the path."""
    z = np.asarray(path_xyz, dtype=float)[:, 2]
    dz = np.diff(z)
    return {
        'net_m': float(z[-1] - z[0]),
        'ascent_m': float(dz[dz > 0].sum()),
        'descent_m': float(-dz[dz < 0].sum()),
        'span_m': float(z.max() - z.min()),
    }


# -- deviation between two paths ---------------------------------------------------------

def hausdorff_m(path_a: np.ndarray, path_b: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two polylines, measured on their points.

    This is the point-set Hausdorff distance, It is an upper bound on the true curve-to-curve distance: two curves
    that coincide but are sampled differently can show a non-zero value, so both paths
    are resampled at a common step before comparing.
    """
    a, b = _as_xy(path_a), _as_xy(path_b)
    return float(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]))


def deviation(path_a: np.ndarray, path_b: np.ndarray,
              step_m: float = DEFAULT_RESAMPLE_STEP_M) -> dict:
    """How far `path_a` sits from `path_b`, after resampling both at `step_m`.

    Returns the symmetric Hausdorff distance and the mean/median distance from the
    points of `path_a` to the polyline of `path_b`.
    """
    a = resample_by_arclength(path_a, step_m)
    b = resample_by_arclength(path_b, step_m)
    nearest = np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1), axis=1)
    return {
        'hausdorff_m': hausdorff_m(a, b),
        'mean_deviation_m': float(nearest.mean()),
        'median_deviation_m': float(np.median(nearest)),
        'max_deviation_m': float(nearest.max()),
    }


# -- the bundle ------------------------------------------------------------------------------

@dataclass(frozen=True)
class PathMetrics:
    """Every measure research steps 2 and 3 ask for, in meters and percent."""

    n_points: int
    step_m: float
    chord_m: float
    length_m: float
    length_3d_m: Optional[float]
    straight_line_m: float
    sinuosity: float
    r_min_m: float
    i_max_percent: Optional[float]
    i_min_signed_percent: Optional[float]
    i_max_signed_percent: Optional[float]
    elevation_net_m: Optional[float]
    elevation_ascent_m: Optional[float]
    elevation_descent_m: Optional[float]
    elevation_span_m: Optional[float]

    def as_dict(self) -> dict:
        return asdict(self)


def measure(path: np.ndarray,
            step_m: float = DEFAULT_RESAMPLE_STEP_M,
            chord_m: float = DEFAULT_CHORD_M) -> PathMetrics:
    """Measure a path. A (n, 2) path gets the planimetric measures only."""
    arr = np.asarray(path, dtype=float)
    has_z = arr.shape[1] >= 3
    gradients = gradients_percent(arr) if has_z else np.empty(0)
    elevation = elevation_difference_m(arr) if has_z else {}
    return PathMetrics(
        n_points=int(len(arr)),
        step_m=float(step_m),
        chord_m=float(chord_m),
        length_m=path_length_m(arr),
        length_3d_m=path_length_3d_m(arr) if has_z else None,
        straight_line_m=straight_line_distance_m(arr),
        sinuosity=sinuosity(arr),
        r_min_m=r_min_m(arr, step_m, chord_m),
        i_max_percent=(float(np.abs(gradients).max()) if gradients.size else 0.0) if has_z else None,
        i_min_signed_percent=(float(gradients.min()) if gradients.size else 0.0) if has_z else None,
        i_max_signed_percent=(float(gradients.max()) if gradients.size else 0.0) if has_z else None,
        elevation_net_m=elevation.get('net_m'),
        elevation_ascent_m=elevation.get('ascent_m'),
        elevation_descent_m=elevation.get('descent_m'),
        elevation_span_m=elevation.get('span_m'),
    )
