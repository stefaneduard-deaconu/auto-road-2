"""`tools.compare_runs`: the same data pass whatever the machine; a changed result fails."""
import csv
import gzip
from pathlib import Path

from tools.compare_runs import compare, main

COLUMNS = ['unit_id', 'od_id', 'search_space', 'objective_cost_sum_abs_dh_m', 'n_cells_selected',
           'search_time_s_median', 'peak_memory_bytes', 'timestamp_utc', 'git_sha', 'n_workers']
ROWS = [
    ['u1', 'od1', 'full_grid', '123.456789012', '1000', '0.52', '1048576', '2026-10-08T10:00:00', 'abc', '8'],
    ['u1', 'od1', 'cta', '130.0', '420', '0.11', '524288', '2026-10-08T10:00:01', 'abc', '8'],
    ['u2', 'od1', 'cta', '98.5', '380', '0.10', '524288', '2026-10-08T10:00:02', 'abc', '8'],
]


def write(folder: Path, rows, name='x3/x3.csv', gz=False, columns=COLUMNS):
    path = folder / (name + ('.gz' if gz else ''))
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gz else open
    with opener(path, 'wt', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


def test_identical_folders_are_the_same(tmp_path):
    write(tmp_path / 'old', ROWS)
    write(tmp_path / 'new', ROWS)
    assert main([str(tmp_path / 'old'), str(tmp_path / 'new')]) == 0


def test_times_provenance_and_row_order_are_not_compared(tmp_path):
    rerun = [r[:5] + ['9.9', '1', '2027-01-01T00:00:00', 'def', '32'] for r in reversed(ROWS)]
    write(tmp_path / 'old', ROWS)
    write(tmp_path / 'new', rerun)
    same, _ = compare(tmp_path / 'old', tmp_path / 'new')
    assert same


def test_a_changed_result_is_reported_with_its_row(tmp_path):
    changed = [list(r) for r in ROWS]
    changed[1][3] = '131.0'
    write(tmp_path / 'old', ROWS)
    write(tmp_path / 'new', changed)
    same, lines = compare(tmp_path / 'old', tmp_path / 'new')
    assert not same
    text = '\n'.join(lines)
    assert 'unit_id=u1, od_id=od1, search_space=cta' in text
    assert 'objective_cost_sum_abs_dh_m: 130 -> 131' in text
    assert main([str(tmp_path / 'old'), str(tmp_path / 'new')]) == 1


def test_numbers_are_compared_at_the_stated_precision(tmp_path):
    rounded = [list(r) for r in ROWS]
    rounded[0][3] = '123.456789'  # 9 significant digits, as the gzipped release file keeps them
    write(tmp_path / 'old', rounded)
    write(tmp_path / 'new', ROWS)
    assert compare(tmp_path / 'old', tmp_path / 'new', digits=9)[0]
    assert not compare(tmp_path / 'old', tmp_path / 'new', digits=12)[0]


def test_a_gzipped_file_pairs_with_a_plain_one(tmp_path):
    write(tmp_path / 'old', ROWS, gz=True)
    write(tmp_path / 'new', ROWS)
    same, lines = compare(tmp_path / 'old', tmp_path / 'new')
    assert same and any('`x3/x3.csv`' in line for line in lines)


def test_only_shared_columns_are_compared(tmp_path):
    """The release keeps a subset of a file's columns; the rerun writes them all."""
    keep = [0, 1, 2, 3]
    write(tmp_path / 'old', [[r[i] for i in keep] for r in ROWS], columns=[COLUMNS[i] for i in keep])
    write(tmp_path / 'new', ROWS)
    assert compare(tmp_path / 'old', tmp_path / 'new')[0]


def test_a_missing_file_fails_and_an_extra_one_is_listed(tmp_path):
    write(tmp_path / 'old', ROWS)
    write(tmp_path / 'old', ROWS, name='stas/stas.csv')
    write(tmp_path / 'new', ROWS)
    write(tmp_path / 'new', ROWS, name='extra/extra.csv')
    same, lines = compare(tmp_path / 'old', tmp_path / 'new')
    text = '\n'.join(lines)
    assert not same
    assert '`stas/stas.csv` | | | | **missing in NEW**' in text
    assert '`extra/extra.csv`' in text


def test_the_report_is_written(tmp_path):
    write(tmp_path / 'old', ROWS)
    write(tmp_path / 'new', ROWS)
    report = tmp_path / 'compare.md'
    main([str(tmp_path / 'old'), str(tmp_path / 'new'), '--report', str(report)])
    assert 'Result: **identical**.' in report.read_text(encoding='utf-8')
