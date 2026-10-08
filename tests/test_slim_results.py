"""`tools.slim_results`: the gzipped file keeps the columns the tables read, at 9 digits, and
the same data always give the same bytes."""
import csv
import gzip

from core.results_io import read_csv
from tools.slim_results import main, short, used_columns

REL = 'x3_reduction_synthetic/x3_reduction_synthetic.csv'


def write_full(results):
    path = results / REL
    path.parent.mkdir(parents=True)
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['scenario_id', 'search_space', 'timestamp_utc', 'zz_not_read_by_any_table'])
        writer.writerow(['u1', 'full_grid', '2026-10-08T10:00:00', '1'])
        writer.writerow(['u2', 'cta', '2026-10-08T10:00:01', '2'])
    return path


def test_the_columns_the_tables_read_are_known():
    used = used_columns()
    assert {'scenario_id', 'search_space'} <= used
    assert 'zz_not_read_by_any_table' not in used


def test_numbers_keep_nine_significant_digits():
    assert short('123.45678901234') == '123.456789'
    assert short('0.5') == '0.5'
    assert short('full_grid_and_more') == 'full_grid_and_more'


def test_the_slimmed_file_is_read_and_is_byte_stable(tmp_path):
    results = tmp_path / 'results'
    write_full(results)
    assert main(['--results', str(results)]) == 0
    gz = results / (REL + '.gz')
    first = gz.read_bytes()
    assert main(['--results', str(results)]) == 0
    assert gz.read_bytes() == first
    with gzip.open(gz, 'rt', encoding='utf-8') as handle:
        header = handle.readline().strip().split(',')
    assert header == ['scenario_id', 'search_space']
    assert [r['search_space'] for r in read_csv(gz)] == ['full_grid', 'cta']
