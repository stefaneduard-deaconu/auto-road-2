"""Pure, tested primitives for the article.

Everything in here works on numpy arrays and integer node ids, imports no matplotlib and
writes no files -- except `core.figures`, which is the one plotting module and is
imported by none of the others. `core.algorithm_1` takes its point removal and its
B-spline interpolation from `core.interpolate`, with `strict=False`.
"""
