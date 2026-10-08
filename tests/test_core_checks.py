"""T6 acceptance: a table of pass flags per class, and an unmeetable class is reported
as infeasible, never relaxed."""
import numpy as np
import pytest

from core.checks import (Check, ChecksReport, check_against_classes, check_path,
                         feasible_classes, flags_table, infeasible_classes)
from data.configs.road_classes import ROAD_CLASSES, get

MIDDLE = get('RO_CLASS_IV_DEAL')  # R >= 125 m, i <= 6.5%
STRICT = get('RO_CLASS_III_DEAL')  # R >= 240 m, i <= 6%
LOOSE = get('SYNTHETIC_LOOSE')     # R >= 25 m, i <= 10%


def arc_road(radius_m: float, gradient_percent: float, arc: float = np.pi / 2,
             n: int = 2000) -> np.ndarray:
    """A circular arc of a known radius climbing at a known constant gradient."""
    t = np.linspace(0, arc, n)
    xy = np.column_stack([radius_m * np.cos(t), radius_m * np.sin(t)])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    return np.column_stack([xy, s * gradient_percent / 100.0])


def straight_road(length_m: float = 1000.0, gradient_percent: float = 0.0, n: int = 200):
    x = np.linspace(0, length_m, n)
    return np.column_stack([x, np.zeros(n), x * gradient_percent / 100.0])


# -- the two pass/fail checks ------------------------------------------------------------

def test_a_generous_road_passes_every_check():
    report = check_path(arc_road(400.0, 3.0), MIDDLE)
    assert report.feasible
    assert report.verdict == 'meets RO_CLASS_IV_DEAL'
    assert {c.name for c in report.checks} == {'R_min', 'i_max'}
    assert all(c.passed for c in report.checks)


def test_a_radius_below_the_class_fails_and_is_named():
    report = check_path(arc_road(50.0, 2.0), MIDDLE)
    assert not report.feasible
    assert 'infeasible under this class' in report.verdict
    assert 'R_min' in report.verdict
    by_name = {c.name: c for c in report.checks}
    assert by_name['R_min'].value == pytest.approx(50.0, rel=1e-2)
    assert by_name['R_min'].limit == 125.0
    assert by_name['i_max'].passed


def test_a_gradient_above_the_class_fails():
    report = check_path(straight_road(gradient_percent=9.0), MIDDLE)
    by_name = {c.name: c for c in report.checks}
    assert not by_name['i_max'].passed
    assert by_name['i_max'].value == pytest.approx(9.0)
    assert by_name['i_max'].limit == 6.5
    assert by_name['R_min'].passed  # a straight road has infinite radius


def test_the_check_is_exactly_at_the_limit():
    assert check_path(straight_road(gradient_percent=6.5), MIDDLE).feasible
    assert not check_path(straight_road(gradient_percent=6.51), MIDDLE).feasible
    # the measured radius of a polyline arc is slightly BELOW the true radius, because
    # the resampled points sit on chords inside the circle. The bias is under 1% at the
    # default step/chord and it errs on the safe side, so the check is exercised just
    # outside that band rather than exactly on the limit.
    assert check_path(arc_road(125.0 * 1.01, 0.0), MIDDLE).feasible
    assert not check_path(arc_road(125.0 * 0.99, 0.0), MIDDLE).feasible


def test_every_check_cites_the_norm():
    for check in check_path(arc_road(400.0, 3.0), MIDDLE).checks:
        assert '1296/2017' in check.source and 'Tabelul' in check.source


def test_a_two_column_path_is_checked_for_the_radius_only():
    report = check_path(arc_road(400.0, 0.0)[:, :2], MIDDLE)
    assert {c.name for c in report.checks} == {'R_min'}
    assert report.reported['elevation_net_m'] is None


# -- nothing is ever relaxed ---------------------------------------------------------------

def test_an_unmeetable_class_is_reported_not_relaxed():
    # a 1 km terrain cannot hold a 650 m radius motorway curve plus 4% gradient
    path = arc_road(60.0, 8.0)
    autostrada = get('RO_CLASS_I_DEAL_AUTOSTRADA')
    report = check_path(path, autostrada)
    assert not report.feasible
    assert report.verdict.startswith('infeasible under this class')
    # the class object itself is untouched, and so is the registry
    assert autostrada.r_min_m == 650.0 and autostrada.i_max_percent == 4.0
    assert ROAD_CLASSES['RO_CLASS_I_DEAL_AUTOSTRADA'] is autostrada


def test_checking_many_classes_does_not_pick_the_one_that_passes():
    path = arc_road(130.0, 6.0)
    reports = check_against_classes(path, ['RO_CLASS_III_DEAL', 'RO_CLASS_IV_DEAL',
                                           'SYNTHETIC_LOOSE'])
    assert set(reports) == {'RO_CLASS_III_DEAL', 'RO_CLASS_IV_DEAL', 'SYNTHETIC_LOOSE'}
    assert 'RO_CLASS_III_DEAL' in infeasible_classes(reports)   # needs R >= 240 m
    assert 'RO_CLASS_IV_DEAL' in feasible_classes(reports)      # needs R >= 125 m
    assert 'SYNTHETIC_LOOSE' in feasible_classes(reports)


# -- the serpentine note ---------------------------------------------------------------------

def test_a_serpentine_radius_still_fails_but_is_flagged():
    # class IV deal: plain minimum 125 m, serpentine minimum 30 m
    report = check_path(arc_road(60.0, 0.0), MIDDLE)
    r_check = next(c for c in report.checks if c.name == 'R_min')
    assert not r_check.passed
    assert 'serpentine' in r_check.note


def test_no_serpentine_note_when_the_radius_is_far_below_everything():
    report = check_path(arc_road(10.0, 0.0), MIDDLE)
    assert next(c for c in report.checks if c.name == 'R_min').note == ''


# -- the reported (not judged) measures ---------------------------------------------------

def test_length_sinuosity_and_elevation_are_reported_without_a_verdict():
    report = check_path(straight_road(1000.0, 4.0), MIDDLE)
    assert report.reported['length_m'] == pytest.approx(1000.0)
    assert report.reported['sinuosity'] == pytest.approx(1.0)
    assert report.reported['elevation_net_m'] == pytest.approx(40.0)
    assert report.reported['elevation_ascent_m'] == pytest.approx(40.0)
    assert report.reported['design_speed_kmh'] == MIDDLE.design_speed_kmh
    assert {c.name for c in report.checks} == {'R_min', 'i_max'}  # nothing else is judged


# -- the table ---------------------------------------------------------------------------------

def test_flags_table_lists_every_class_with_its_status():
    reports = check_against_classes(arc_road(130.0, 6.0))
    table = flags_table(reports)
    for name in ROAD_CLASSES:
        assert name in table
    assert 'TO CONFIRM' in table
    assert 'No limit is relaxed' in table
    assert 'infeasible under this class' in table


def test_flags_table_marks_pass_and_fail():
    reports = check_against_classes(arc_road(400.0, 2.0),
                                    ['RO_CLASS_IV_DEAL', 'RO_CLASS_I_DEAL_AUTOSTRADA'])
    table = flags_table(reports)
    assert 'yes' in table and 'NO' in table


def test_report_serialises():
    report = check_path(arc_road(130.0, 6.0), MIDDLE)
    data = report.as_dict()
    assert data['feasible'] in (True, False)
    assert set(data['checks']) == {'R_min', 'i_max'}
    assert data['road_class_status'] == 'TO CONFIRM'
    assert data['step_m'] > 0 and data['chord_m'] > 0
    assert isinstance(report, ChecksReport) and isinstance(report.checks[0], Check)


def test_check_prints_a_readable_line():
    report = check_path(straight_road(gradient_percent=9.0), MIDDLE)
    text = str(next(c for c in report.checks if c.name == 'i_max'))
    assert text.startswith('FAIL i_max') and '6.50 %' in text
