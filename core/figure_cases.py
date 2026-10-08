"""The worked examples behind the article's map figures, computed once and saved.

Figures never run a search (`core/figures*.py` only draw committed or saved data), so the
one example that needs a search on real terrain is computed here and written to
`results/figure_cases/` (git-ignored, regenerated on demand):

    python -m core.figure_cases dem-section

`dem-section`: the whole surveyed Idrija swath at 20 m cells (for the overview panel) and one
window of that swath at 10 m cells with its HAG, the candidate terrain areas (CTA), CTA + 1
ring and the full-grid, CTA and CTA + 1 ring alignments between the first O-D pair the window
protocol of the robustness study draws (the same protocol as `core.programme._window_case_pairs`).
Needs the raster (`raster/`, git-ignored).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from core import programme
from core.experiment_hag import _run_search, _select
from core.hag import build_hag
from core.naming import DEFAULT_HAG_EDGE_COST, grid_edge_cost

OUT_DIR = Path('results/figure_cases')
DEFAULT_TIF = Path('raster/Idrija_Fault_LiDAR_DEM.tif')


def dem_section(tif_path: Path = DEFAULT_TIF, *, size_m: int = 1920, k: int = 0,
                cell_m: int = 10, height_delta: float = 3.0,
                out_dir: Path = OUT_DIR) -> Path:
    """Compute the DEM overview and one worked window; save them as one `.npz`."""
    ext = programme.dem_extent(tif_path)
    whole = programme.dem_case(ext['tif'], ext['row0'], ext['col0'], ext['n_rows'],
                               ext['n_cols'], 20, 'idrija_whole')
    options = programme.RunOptions(tif=tif_path)
    case, pairs = programme._window_case_pairs(
        options, {'size': size_m, 'k': k, 'cell': cell_m, 'delta': height_delta,
                  'per_size': max(k + 1, 5)})
    pair = pairs[0]
    hag = build_hag(case.grid, height_delta, valid_mask=case.valid, native=True)
    cost, cap = grid_edge_cost('climb_tiebreak')
    start, target = case.grid.index(pair.start), case.grid.index(pair.target)
    cache: dict = {}
    saved = {}
    for space in ('full_grid', 'hag_cta', 'hag_cta_ring1'):
        _, chosen, _, _ = _select(hag, space, DEFAULT_HAG_EDGE_COST, pair.start, pair.target,
                                  cache, cost)
        if space == 'full_grid':
            mask = case.valid
        else:
            mask = hag.mask(chosen)
            mask.reshape(-1)[[start, target]] = True
            saved[f'mask_{space}'] = mask
        result = _run_search(case, mask, start, target, cost, cap, 'native')
        saved[f'path_{space}'] = result.coords(case.grid) if result.reached else np.zeros((0, 2))
    window = programme.dem_windows(ext, size_m, max(k + 1, 5), seed=7000 + size_m)[k]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'dem_section.npz'
    np.savez_compressed(
        out, whole=np.where(whole.valid, whole.grid.surf, np.nan), whole_cell_m=20.0,
        section=np.where(case.valid, case.grid.surf, np.nan), section_cell_m=float(cell_m),
        labels=hag.labels, start=np.array(pair.start), target=np.array(pair.target),
        window_rc_m=np.array([window['row0'] - ext['row0'], window['col0'] - ext['col0'],
                              size_m]),
        height_delta=height_delta, **saved)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('case', choices=('dem-section',))
    parser.add_argument('--tif', type=Path, default=DEFAULT_TIF)
    args = parser.parse_args(argv)
    print(dem_section(args.tif))
    return 0


if __name__ == '__main__':
    sys.exit(main())
