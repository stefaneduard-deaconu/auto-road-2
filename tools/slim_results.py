"""Write the large synthetic-terrain result file in the form this repository keeps.

    python -m tools.slim_results [--results results]

`results/x3_reduction_synthetic/x3_reduction_synthetic.csv` is about 56 MB after a full run, too
large for the repository. This writes `x3_reduction_synthetic.csv.gz` next to it: only the
columns that `article_2/build_article.py` and `core/summary.py` read, without the time stamp,
numbers at 9 significant digits (the tables show at most 3 decimals), gzipped without a time
in the header, so the same data always give the same bytes. The full CSV stays on disk and is
git-ignored; the table builder reads the `.csv.gz` when the `.csv` is absent.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
#: the files slimmed, relative to the results folder
SLIM = ('x3_reduction_synthetic/x3_reduction_synthetic.csv',)
#: columns no table reads and that compress badly
DROP = {'timestamp_utc'}
DIGITS = 9


def used_columns() -> set[str]:
    """Every quoted lower-case name in the table builder and the summary code."""
    source = ''.join((ROOT / p).read_text(encoding='utf-8')
                     for p in ('article_2/build_article.py', 'core/summary.py'))
    return set(re.findall(r"'([a-z0-9_]+)'", source))


def short(value: str) -> str:
    """A float written with 9 significant digits; anything short or non-numeric as it is."""
    if len(value) > DIGITS:
        try:
            return f'{float(value):.{DIGITS}g}'
        except ValueError:
            return value
    return value


def slim(source: Path, target: Path, used: set[str]) -> int:
    with open(source, encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        columns = [c for c in reader.fieldnames if c in used and c not in DROP]
        # mtime=0: the same data give the same bytes, so a rebuild leaves git clean
        with io.TextIOWrapper(gzip.GzipFile(str(target), 'wb', mtime=0),
                              encoding='utf-8', newline='') as out:
            writer = csv.DictWriter(out, columns, extrasaction='ignore')
            writer.writeheader()
            rows = 0
            for row in reader:
                writer.writerow({k: short(v) for k, v in row.items()})
                rows += 1
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--results', type=Path, default=Path('results'))
    args = parser.parse_args(argv)
    used = used_columns()
    for rel in SLIM:
        source = args.results / rel
        if not source.exists():
            parser.error(f'{source} not found: run the experiment first')
        target = source.with_name(source.name + '.gz')
        rows = slim(source, target, used)
        print(f'{target}: {rows} rows, {target.stat().st_size / 1e6:.1f} MB')
    return 0


if __name__ == '__main__':
    sys.exit(main())
