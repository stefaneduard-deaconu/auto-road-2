"""Point removal and B-spline interpolation for Algorithm 1 (`core.algorithm_1`)."""
import math

import numpy as np
from scipy.interpolate import splev, splprep

Coord = tuple[int, int]
Coord3D = tuple[float, float, float]


def is_collinear(p1: np.array, p2: np.array, p3: np.array):
    return abs(
        np.linalg.det([
            [*p1, 1],
            [*p2, 1],
            [*p3, 1],
        ])
    ) < 0.00001


def distance_to_line(point: np.array,
                     line: tuple[np.array]) -> float:
    line = np.array(line)
    line_direction = line[1] - line[0]

    # Compute the vector representing the line segment from line_point_1 to the point
    line_segment = point - line[0]

    # Compute the cross product of the two vectors. np.cross on 2D vectors was
    # removed in NumPy 2.5, so the 2D case (a scalar) is written out.
    if len(line_direction) == 2:
        cross_product = line_direction[0] * line_segment[1] - line_direction[1] * line_segment[0]
    else:
        cross_product = np.cross(line_direction, line_segment)

    # Compute the distance from the point to the line
    return np.linalg.norm(cross_product) / np.linalg.norm(line_direction)


def eucl(a: np.array, b: np.array):
    a = np.array(a)
    b = np.array(b)
    return math.sqrt(np.sum((a - b) ** 2))


def remove_bad_points(path3d: list[Coord3D], minimal_radius=15, GRID_RATIO_TO_METERS=10,
                      strict: bool = True):
    """Algorithm 1, steps 0 to 2: drop collinear, almost-collinear and tight-radius points.

    With `strict=True` (the default) two consecutive short lines raise `Exception('BAD1')`,
    which can happen on valid paths, and the marked indices are iterated in set order
    although the logic needs ascending order.
    With `strict=False` the indices are iterated in ascending order and two consecutive
    short lines are merged with the same "extract the longest almost-straight run" rule
    used everywhere else, instead of raising. `core.algorithm_1` uses `strict=False`.
    """
    bad_points = {
        'collinear': [],
        'almost_collinear': [],
        'too_small_radius': []
    }

    # Step 1. remove collinear points


    # 1. remove collinear points
    path = [path3d[0],
            *[p2
              for p1, p2, p3 in zip(path3d[0:],
                                    path3d[1:],
                                    path3d[2:])
              if not is_collinear(p1, p2, p3)],
            path3d[-1]]
    bad_points['collinear'] = [p2
                              for p1, p2, p3 in zip(path3d[0:],
                                                    path3d[1:],
                                                    path3d[2:])
                              if is_collinear(p1, p2, p3)]

    # Step 2. remove point who are outside the minimum radius
    #         of two consecutive lines


    # Step 2.
    min_radius = minimal_radius / GRID_RATIO_TO_METERS  # 15m, but each element on the grid has 5 meters
    min_diameter = 2 * min_radius

    # 1) check if radius is big enough
    short_lines = {i
                   for i in range(len(path) - 1)
                   if eucl(path[i], path[i + 1]) < min_diameter}
    long_lines = {i
                  for i in range(len(path) - 1)
                  if eucl(path[i], path[i + 1]) >= min_diameter}
    # look for consecutive short lines, and try to merge them
    # they are "mergeable" if the straight line from start to bottom, and all the short lines together,
    #  have a negligable area, as compared with the number of square to traverse.
    new_path = list(path)

    # cases: 1. long - short - long, we ignore
    #           if same directions => raise error
    def is_almost_line(start: int,
                       end: int,
                       new_path: np.array):
        for i in range(start, min(end + 1,
                                  len(new_path))):
            p = new_path[i]
            try:
                d = distance_to_line(p, (new_path[start], new_path[end + 1]))
            except IndexError:  # end + 1 is past the last point
                d = distance_to_line(p, (
                    new_path[start],
                    new_path[len(new_path) - 1]))
            if d > min_radius:
                return False  # return False if at least a point is too far from the line
        return True

    def extract_longest_line(i: int, new_path: np.array):
        start = i
        end = i + 1

        while end < len(new_path) and is_almost_line(start, end, new_path) \
                and end not in long_lines:
            end += 1
        return start, end + 1

    ignore_until = 0
    for i in (short_lines if strict else sorted(short_lines)):
        if i < ignore_until:
            continue
        if new_path[i] is None:
            continue

        if i - 1 < 0:
            # special cases:
            if i + 1 in long_lines:
                new_path[i + 1] = None
            else:
                # extract longest possible line
                start, end = extract_longest_line(i,
                                                  new_path)
                # line is from path[start] to path[end+1], so we remote path[start+1:end+1]
                bad_points['almost_collinear'].extend(new_path[start + 1:end])
                new_path[start + 1:end] = [None] * (end - (start + 1))
                # ignore the point in the big fore
                ignore_until = end
        else:
            # we have both previous and next line

            # if previous is short, strict mode gives up
            if new_path[i - 1] is not None and i - 1 in short_lines:
                if strict:
                    raise Exception('BAD1')
                # two consecutive short lines: merge them with the same rule as below
                start, end = extract_longest_line(i, new_path)
                bad_points['almost_collinear'].extend(new_path[start + 1:end])
                new_path[start + 1:end] = [None] * (end - (start + 1))
                ignore_until = end
                continue
            # if both are long, we'll ignore the second point from this line
            if i + 1 in long_lines:
                bad_points['almost_collinear'].append(new_path[i + 1])
                new_path[i + 1] = None
            else:
                # long before, short after, is the same as line 147 (first else from the for)
                # extract longest possible line
                start, end = extract_longest_line(i,
                                                  new_path)  # TODO may sometime unite a few short lines, with a long line
                # line is from path[start] to path[end+1], so we remote path[start+1:end+1]
                bad_points['almost_collinear'].extend(new_path[start + 1:end])
                new_path[start + 1:end] = [None] * (end - (start + 1))
                # ignore the point in the big fore
                ignore_until = end
    # never remove last point
    if new_path[-1] is None:
        new_path[-1] = path3d[-1]
    return np.array([x
                     for x in new_path
                     if x is not None]), \
        bad_points


def interpolate_2d_path_v2(path2d: list[Coord], multiplier: int = 10, s_value: float = 10,
                           k: int = 3) -> list[Coord]:
    """
    Interpolate a path using B-spline interpolation for smooth roads.
    :param path2d: List of (x, y) coordinates (more dimensions work too)
    :param multiplier: Number of points to generate per input point
    :param s_value: smoothing factor of splprep, 0 is an exact fit
    :param k: spline degree. `splprep` needs more points than the degree, so a short
        path must lower it; the default is 3.
    :return: Smooth interpolated list of coordinates
    """
    if len(path2d) < 2:
        return path2d  # Not enough points to interpolate

    dimensions = zip(*path2d)
    tck, u = splprep([*dimensions], s=s_value, k=k)

    u_new = np.linspace(0, 1, len(path2d) * multiplier)
    new_dimensions = splev(u_new, tck)

    smooth_path = np.array(list(zip(*new_dimensions)))
    return smooth_path
