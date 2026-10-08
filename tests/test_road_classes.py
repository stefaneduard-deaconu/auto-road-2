"""T0: the RoadClass registry is complete and sourced; only STAS 863-85 is signed off."""
import pytest

from data.configs.road_classes import (DEFAULT_CELL_SIZE_M, PROPOSED_MATRIX_CLASSES,
                                       ROAD_CLASSES, STAS_MATRIX_CLASSES, RoadClass,
                                       UnconfirmedRoadClass, get, unconfirmed, unofficial)


def test_registry_is_not_empty_and_keys_match_names():
    assert ROAD_CLASSES
    for name, rc in ROAD_CLASSES.items():
        assert rc.name == name


def test_every_entry_cites_a_source():
    for rc in ROAD_CLASSES.values():
        assert rc.source.strip(), rc.name
        if rc.synthetic:
            assert rc.source.startswith('SYNTHETIC'), rc.name
        elif rc.official:
            assert 'STAS 863-85' in rc.source, rc.name
            assert '2026-09-28' in rc.source, rc.name
        else:
            assert 'Tabelul' in rc.source, rc.name
            assert '1296/2017' in rc.source, rc.name


def test_only_the_stas_classes_are_signed_off():
    # The two STAS 863-85 scenarios were signed off on 2026-10-08 (agreed item A2); every
    # other entry stays TO CONFIRM.
    stas = {name for name, rc in ROAD_CLASSES.items() if rc.official}
    assert stas == {'STAS863_V40', 'STAS863_V25'}
    assert unconfirmed() == sorted(set(ROAD_CLASSES) - stas)
    for name, rc in ROAD_CLASSES.items():
        if name in stas:
            assert rc.status == 'CONFIRMED'
            rc.assert_confirmed()
        else:
            assert rc.status == 'TO CONFIRM'
            with pytest.raises(UnconfirmedRoadClass):
                rc.assert_confirmed()


def test_values_are_physically_sane():
    for rc in ROAD_CLASSES.values():
        assert 0 < rc.design_speed_kmh <= 140, rc.name
        assert 0 < rc.i_max_percent <= 12, rc.name
        assert 0 < rc.r_min_m <= 2000, rc.name
        assert rc.cell_size_m > 0, rc.name


def test_there_is_a_strict_and_a_loose_synthetic_class():
    synthetic = [rc for rc in ROAD_CLASSES.values() if rc.synthetic]
    assert len(synthetic) >= 2
    norms = [rc for rc in ROAD_CLASSES.values() if not rc.synthetic]
    strict = min(synthetic, key=lambda rc: rc.i_max_percent)
    loose = max(synthetic, key=lambda rc: rc.i_max_percent)
    assert strict.i_max_percent < min(rc.i_max_percent for rc in norms)
    assert loose.i_max_percent > max(rc.i_max_percent for rc in norms)
    # SYNTHETIC_LOOSE was set to 25 m, which STAS863_V25 also uses,
    # so the radius bracket is inclusive at the loose end.
    assert loose.r_min_m <= min(rc.r_min_m for rc in norms)


def test_the_stas_classes_are_the_official_ones_and_carry_the_full_parameter_set():
    for name in STAS_MATRIX_CLASSES:
        rc = get(name)
        assert rc.official, name
        assert not rc.synthetic, name
        assert rc.has_vertical_geometry, name
        assert rc.standing == 'official', name
        assert rc.l_alignment_min_m is not None and rc.l_alignment_min_m > 0, name
        assert rc.l_design_step_min_m is not None and rc.l_design_step_min_m > 0, name
        assert rc.r_vert_concave_min_m is not None and rc.r_vert_concave_min_m > 0, name
        assert rc.r_vert_convex_min_m is not None and rc.r_vert_convex_min_m > 0, name


def test_the_stas_values_match_the_co_author_transcription():
    baseline, difficult = get('STAS863_V40'), get('STAS863_V25')
    assert (baseline.design_speed_kmh, baseline.r_min_m, baseline.i_max_percent) == (40, 60.0, 7.0)
    assert (baseline.l_alignment_min_m, baseline.l_design_step_min_m) == (56.0, 50.0)
    assert (baseline.r_vert_concave_min_m, baseline.r_vert_convex_min_m) == (1000.0, 1000.0)
    assert (difficult.design_speed_kmh, difficult.r_min_m, difficult.i_max_percent) == (25, 25.0, 8.0)
    assert (difficult.l_alignment_min_m, difficult.l_design_step_min_m) == (35.0, 50.0)
    assert (difficult.r_vert_concave_min_m, difficult.r_vert_convex_min_m) == (300.0, 500.0)


def test_every_non_stas_entry_is_unofficial_for_the_second_article():
    assert unofficial() == sorted(set(ROAD_CLASSES) - set(STAS_MATRIX_CLASSES))
    for name in unofficial():
        rc = get(name)
        assert rc.standing == 'unofficial', name
        assert not rc.has_vertical_geometry, name


def test_hilly_classes_are_present():
    hilly = [rc for rc in ROAD_CLASSES.values() if rc.terrain_category == 'deal']
    assert len(hilly) >= 4
    assert {rc.technical_class for rc in hilly} >= {'III', 'IV', 'V'}


def test_default_cell_size_matches_the_articles():
    assert DEFAULT_CELL_SIZE_M == 10.0
    assert get('RO_CLASS_IV_DEAL').r_min_cells == pytest.approx(12.5)


def test_with_cell_size_only_changes_the_resolution():
    rc = get('RO_CLASS_IV_DEAL')
    dem = rc.with_cell_size(30.0)
    assert isinstance(dem, RoadClass)
    assert dem.cell_size_m == 30.0 and dem.r_min_m == rc.r_min_m
    assert dem.r_min_cells == pytest.approx(125.0 / 30.0)


def test_proposed_matrix_classes_exist_and_are_ordered_by_strictness():
    rcs = [get(n) for n in PROPOSED_MATRIX_CLASSES]
    assert [rc.r_min_m for rc in rcs] == sorted((rc.r_min_m for rc in rcs), reverse=True)
    assert all(rc.terrain_category == 'deal' for rc in rcs)
