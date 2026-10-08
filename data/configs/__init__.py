# batches of terrain configurations, plus the RoadClass registry in road_classes.py
#
# `TerrainGeneratorConfig` lives in `main`, which imports matplotlib. Importing it at
# module level made `data.configs.road_classes` pull in matplotlib and the whole
# experiment runner, which `core/` must not do. The config batches are therefore built
# lazily: `from data.configs import fast_configs` still works, and
# `import data.configs.road_classes` no longer imports main.
from itertools import product
from typing import Iterable


def generate_configs(
        seeds: Iterable[int],
        grid_sizes: Iterable[int],
        scaling_arguments: Iterable[int],
        height_deltas: Iterable[int] = (1, 2, 3, 5),
) -> list:
    from main import TerrainGeneratorConfig
    return [
        TerrainGeneratorConfig(
            seed=seed,
            GRID_SIZE=(grid_size, grid_size),
            scaling_argument=(scale_arg, scale_arg),
            height_interval=(100, 150),
            height_delta=height_delta
        )
        for seed, grid_size, scale_arg, height_delta in product(
            seeds,
            grid_sizes,
            scaling_arguments,
            height_deltas,
        )
        if grid_size % scale_arg == 0
    ]


def _build(name: str):
    if name == 'over_100_seeds_for_algorithm1':
        return generate_configs(range(5, 105), [50], [2], [3])
    if name == 'fast_configs':
        return generate_configs(range(3, 10), [20, 50, 80], range(2, 5))
    if name == 'slow_configs':
        return generate_configs(range(3), [120, 150, 180], range(2, 5))
    if name == 'all_configs':
        return [*_build('fast_configs'), *_build('slow_configs')]
    if name == 'article_config':
        from main import TerrainGeneratorConfig
        return TerrainGeneratorConfig(
            seed=0, GRID_SIZE=(100, 100),
            scaling_argument=(4, 4),
            height_interval=(100, 150),
            height_delta=3
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_LAZY = ('over_100_seeds_for_algorithm1', 'fast_configs', 'slow_configs', 'all_configs',
         'article_config')


def __getattr__(name):
    if name in _LAZY:
        value = _build(name)
        globals()[name] = value  # built once, then it is an ordinary module attribute
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
