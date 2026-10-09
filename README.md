# Height Area Graphs for automated preliminary road alignment

[![DOI](https://zenodo.org/badge/1410562344.svg)](https://doi.org/10.5281/zenodo.23246417)

| | |
|---|---|
| Latest release | v0.2 (version 0.2.0), 2026-10-09: the full rerun on macOS (Apple M3 Pro) |
| Archive (DOI) | v0.1: https://doi.org/10.5281/zenodo.23250604 · v0.2: https://doi.org/10.5281/zenodo.23254011 · all versions: https://doi.org/10.5281/zenodo.23246417 |
| Repository | https://github.com/stefaneduard-deaconu/auto-road-2 (branch `article-2026`) |
| Software and data | Ioana-Alexandra Șomîtcă, Ștefan-Eduard Deaconu (see *Authors and contributions*) |
| Article | *Height Area Graphs for Automated Preliminary Road Alignment in Hilly Terrain* (Șomîtcă, Deaconu, Boitor, Dragomir), in preparation |
| Python | CPython 3.14, free-threaded build; numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2 |
| Real terrain data | Idrija Fault LiDAR DEM, OpenTopography, https://doi.org/10.5069/G9QC01Q2 (not included) |
| Licence | code: MIT (`LICENSE`); results, tables and figures: CC BY 4.0 (`LICENSES/CC-BY-4.0.txt`) |

Code and results of the article named above.

The Height Area Graph (HAG) partitions a terrain into connected areas of similar elevation and
selects a reduced search domain (the candidate terrain areas, CTA) before a Dijkstra search for a
preliminary road alignment. The alignment is then refined in plan (Algorithm 1) and checked at
every station against the Romanian road-design standard STAS 863-85.

## Contents

| Path | What it is |
|---|---|
| `core/` | the method: terrain grid, HAG construction and edge costs, search, Algorithm 1 and its spline interpolation, STAS 863-85 station checks, the experiments, and the figures |
| `perlin_numpy/` | the Perlin-noise terrain generator, from https://github.com/pvigier/perlin-numpy (MIT, Copyright (c) 2019 Pierre Vigier; its licence is `perlin_numpy/LICENSE`) |
| `data/configs/road_classes.py` | the road classes, including the two STAS 863-85 design scenarios (V = 40 and 25 km/h) |
| `article_2/` | rebuilds every table of the article, its Appendix A and the supplementary tables from `results/`; `article_2/tables/` holds the tables as built from the committed results |
| `results/` | one CSV per experiment, one row per scenario (the large synthetic file is gzipped and keeps the columns the article uses) |
| `figures/` | the figures of the article |
| `tools/` | `compare_runs.py` (is a rerun the same data as a release?) and `slim_results.py` (writes the gzipped synthetic file) |
| `tests/` | tests of the method, the table builder and the tools |

## Installation

Python 3.14 (free-threaded build) with the pinned packages of `uv.lock`:

```bash
uv sync
```

or, without uv, `setup_venv.sh` (Linux, macOS) or `setup_venv.ps1` (Windows). The compiled search
engine (`core/native/`) builds itself on first use with the `ziglang` package. Then:

```bash
python -m pytest -q                                  # the tests (7 are skipped without the DEM)
python -m article_2.build_article --article hag      # the tables, Appendix A, Tables S1-S5
python -m core.figures_article --out figures         # the figures (Figure 5 needs the DEM)
```

## Real terrain data

The real terrain is the airborne LiDAR DEM of the Idrija Fault, Slovenia (1 m cells), distributed by
OpenTopography: https://doi.org/10.5069/G9QC01Q2. It is not included. Download the GeoTIFF and
save it as `raster/Idrija_Fault_LiDAR_DEM.tif` (the default path; `raster/` is git-ignored), or
pass its path with `--tif` (`core.run`, `core.figure_cases`) or `--dem` (`core.experiment_stas`).

## Reproducing the experiments

Every result file of the article, in the order it is produced. The search counts, objectives,
selected areas and STAS checks do not depend on the machine; the times do. `--workers auto` may be
added to the `core.run` experiments; the times the article reports come from the serial
`calibrate` steps, which must run **alone, on an idle machine**.

```bash
python -m core.run X3 --tier full                         # search-domain reduction, synthetic terrain
python -m core.run calibrate X3 --tier full --sample 240  # serial timing sample (all grid sizes)
python -m core.run X3 --tier hilly                        # the same on hilly terrain (20-25 gon)
python -m core.run calibrate X3 --tier hilly              # its serial timing sample
python -m core.run X4 --tier full                         # grid resolution
python -m core.run X6 --tier full                         # the whole real DEM
python -m core.run X7 --tier full                         # windows of the real DEM
python -m core.experiment_stas --step-m 2                 # STAS 863-85 at every station, synthetic terrain
python -m core.experiment_stas --dem raster/Idrija_Fault_LiDAR_DEM.tif --step-m 2   # the same, real DEM
python -m core.experiment_baseline                        # the HAG against fixed square blocks
python -m tools.slim_results                              # the gzipped synthetic file the repository keeps
python -m core.figure_cases dem-section                   # the worked real-DEM example of Figure 5
python -m core.figures_article --out figures              # the figures
python -m article_2.build_article --article hag           # the tables
```

A stopped `core.run` experiment resumes where it stopped when the same command is issued again.

Two notes on reproducing v0.2:

- **Ties between equal-cost paths.** The objectives, path costs and selected areas are the same on
  every platform. When two paths have exactly the same cost, a different platform may pick the
  other one, so path geometry (maximum grade, minimum radius, Hausdorff distance), heap push
  counts and station-level STAS checks can differ slightly. v0.2 (macOS) reproduces v0.1
  (Windows) in this sense.
- **X6 at 1 m is not run.** The whole DEM at 1 m needs about 25 GiB in one array. v0.2 runs X6 at
  3, 5, 10 and 20 m, one resolution at a time:
  `python -m core.run X6 --tier full --only whole_c20_` (then `whole_c10_`, `whole_c5_`, `whole_c3_`).

## Checking a rerun against a release

To confirm that a rerun reproduces a release (for example on another machine):

```bash
git clone https://github.com/stefaneduard-deaconu/auto-road-2 && cd auto-road-2
mkdir results_v0.1                                        # the release's data, kept for the comparison
git archive v0.1 results | tar -x -C results_v0.1 --strip-components=1
mv results results_before_rerun                           # the rerun starts from an empty results/
# run every command of "Reproducing the experiments" (they write a new results/)
python -m tools.compare_runs results_v0.1 results --report compare.md
```

Both moved folders (`results_*/`) are git-ignored.

`compare_runs` compares every result column except provenance (time stamp, git, versions,
platform, workers) and the timing and memory measurements, at 9 significant digits, whatever the
row order. It ends with "Result: **identical**." and exit status 0 when the data are the same.
Floating-point results can differ in the last digits between operating systems (Windows and Linux
builds of numpy differ by up to one unit in the last place in some functions); between operating
systems, compare with `--digits 6` and read the differences it lists.

## Authors and contributions

The software (code, tests and build scripts) and the data (results, tables and figures) in this
repository were written and produced solely by Ioana-Alexandra Șomîtcă and Ștefan-Eduard Deaconu,
who hold all copyright in them.

Rozalia-Melania Boitor and Mihai-Liviu Dragomir are co-authors of the forthcoming article only.
They contributed road-engineering domain expertise to the article: what to measure, what to
test, and what to include in or exclude from the article. They made no contribution to the
software or the data and hold no copyright in this repository.

## Citation

See `CITATION.cff`, which names both licences (MIT and CC BY 4.0). Please cite the article and
the release you used, by its version DOI (v0.1: https://doi.org/10.5281/zenodo.23250604; v0.2:
https://doi.org/10.5281/zenodo.23254011);
https://doi.org/10.5281/zenodo.23246417 always resolves to the latest version.

## Licence

The code is under the MIT licence (`LICENSE`), except `perlin_numpy/`, which keeps its own MIT
licence (`perlin_numpy/LICENSE`). The results, tables and figures (`results/`,
`figures/`, `article_2/*.md`, `article_2/tables/`) are under the Creative Commons Attribution 4.0
International licence (`LICENSES/CC-BY-4.0.txt`): they may be reused with credit, by citing the
article and the release.
