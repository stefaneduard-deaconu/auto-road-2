"""Compare two result folders: is a rerun the same data as a release?

    python -m tools.compare_runs results_v0.1 results [--digits 9] [--report compare.md]

Every result file (`*.csv` or `*.csv.gz`) under OLD is paired with the file of the same name
under NEW; a gzipped file and a plain one of the same name pair with each other. Only the
columns both files have are compared, and never the columns that describe the run rather than
the result: provenance (time stamp, git, Python and package versions, platform, worker and
run identifiers) and every timing or memory measurement, which depend on the machine.

Numbers are compared at `--digits` significant digits (9 by default, the precision of the
gzipped synthetic-terrain file). Rows are compared as multisets, so a parallel run that writes
its rows in another order is still the same data.

Exit status 0: every file of OLD has an identical counterpart in NEW. Exit status 1 otherwise.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from collections import Counter
from pathlib import Path

#: columns that describe how a run was made, not what it found
IGNORED = re.compile(r'^(?:timestamp_utc|git_\w+|python\w*|platform|numpy|scipy|run_id|worker_id|'
                     r'n_workers|timing_mode|timing_repeats)$|time|_bytes$|memory|^wall')
#: columns that name a row, used to show which rows differ (the comparison itself uses every column)
IDENTITY = ('experiment', 'unit_id', 'scenario_id', 'case_id', 'family', 'terrain_id', 'terrain_seed',
            'window_id', 'od_id', 'od_band', 'road_class', 'search_space', 'hag_edge_cost',
            'grid_edge_cost', 'cost_model', 'algorithm_1_arm', 'engine')
EXAMPLES = 10


def result_files(folder: Path) -> dict[str, Path]:
    """`{name without .gz: path}` of every result file under `folder`."""
    files = {}
    for path in sorted(folder.rglob('*')):
        if path.is_file() and (path.name.endswith('.csv') or path.name.endswith('.csv.gz')):
            files[path.relative_to(folder).as_posix().removesuffix('.gz')] = path
    return files


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def normalise(value: str, digits: int) -> str:
    """A number at `digits` significant digits; any other value as it is."""
    try:
        return f'{float(value):.{digits}g}'
    except ValueError:
        return value


def compare_file(old: Path, new: Path, digits: int) -> dict:
    old_columns, old_rows = read_rows(old)
    new_columns, new_rows = read_rows(new)
    columns = [c for c in old_columns if c in new_columns and not IGNORED.search(c)]
    key = [c for c in IDENTITY if c in columns]

    def as_tuple(row):
        return tuple(normalise(row.get(c, ''), digits) for c in columns)

    old_count = Counter(as_tuple(r) for r in old_rows)
    new_count = Counter(as_tuple(r) for r in new_rows)
    only_old = old_count - new_count
    only_new = new_count - old_count
    result = {'rows_old': len(old_rows), 'rows_new': len(new_rows), 'columns': len(columns),
              'not_compared': sorted(set(old_columns) ^ set(new_columns)),
              'only_old': sum(only_old.values()), 'only_new': sum(only_new.values()), 'examples': []}
    # pair the differing rows by their identity columns, to say what changed
    index = [columns.index(c) for c in key]
    new_by_key = {tuple(t[i] for i in index): t for t in only_new}
    for row in list(only_old)[:EXAMPLES]:
        name = tuple(row[i] for i in index)
        label = ', '.join(f'{c}={v}' for c, v in zip(key, name)) or 'a row'
        other = new_by_key.get(name) if key else None
        if other is None:
            result['examples'].append(f'{label}: only in OLD')
        else:
            changed = [f'{c}: {a} -> {b}' for c, a, b in zip(columns, row, other) if a != b]
            result['examples'].append(f'{label}: ' + '; '.join(changed))
    return result


def compare(old_dir: Path, new_dir: Path, digits: int = 9) -> tuple[bool, list[str]]:
    old_files, new_files = result_files(old_dir), result_files(new_dir)
    lines = [f'# Comparison of `{old_dir}` (OLD) and `{new_dir}` (NEW)', '',
             f'Numbers at {digits} significant digits; provenance, timing and memory columns '
             'are not compared.', '', '| File | Rows OLD | Rows NEW | Columns compared | Result |',
             '|---|---|---|---|---|']
    details, same = [], True
    for name, old in old_files.items():
        new = new_files.get(name)
        if new is None:
            same = False
            lines.append(f'| `{name}` | | | | **missing in NEW** |')
            continue
        r = compare_file(old, new, digits)
        ok = r['only_old'] == 0 and r['only_new'] == 0
        same &= ok
        verdict = 'identical' if ok else f"**differs**: {r['only_old']} rows only in OLD, {r['only_new']} only in NEW"
        lines.append(f"| `{name}` | {r['rows_old']} | {r['rows_new']} | {r['columns']} | {verdict} |")
        if not ok:
            details += ['', f'## `{name}`', ''] + [f'- {e}' for e in r['examples']]
    extra = sorted(set(new_files) - set(old_files))
    if extra:
        details += ['', '## Only in NEW (not compared)', ''] + [f'- `{n}`' for n in extra]
    lines += details
    lines += ['', 'Result: **identical**.' if same else 'Result: **different** (see above).']
    return same, lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('old', type=Path)
    parser.add_argument('new', type=Path)
    parser.add_argument('--digits', type=int, default=9, help='significant digits compared')
    parser.add_argument('--report', type=Path, help='also write the comparison to this Markdown file')
    args = parser.parse_args(argv)
    for folder in (args.old, args.new):
        if not folder.is_dir():
            parser.error(f'{folder} is not a folder')
    same, lines = compare(args.old, args.new, args.digits)
    text = '\n'.join(lines) + '\n'
    print(text)
    if args.report:
        args.report.write_text(text, encoding='utf-8')
    return 0 if same else 1


if __name__ == '__main__':
    sys.exit(main())
