"""Task T8: real elevation data as a `Grid`.

Every fixture is generated in the test. No tile is downloaded and no binary lives in
the repo, so the DEM path is testable on any machine; the real N45E024 case runs
whenever someone supplies the tile.
"""
import hashlib
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from core.dem import (DEMError, SRTM_VOID, VALID_TILE_SIZES, cell_size_m,
                      coarsen_stream, coarsen_to_cell_size, coverage_mask, crop,
                      fill_voids, full_windows, infer_samples, load_dem_grid,
                      load_geotiff_grid, load_hgt, parse_tile_name, pixel_to_world,
                      read_geotiff_header, read_rect, read_window, to_square_grid)
from core.grid import Grid


def synthetic_hgt(tmp_path, samples=1201, name='N45E024.hgt', voids=None,
                  builder=None):
    """Write a big-endian int16 tile, the way a real `.hgt` file is laid out."""
    if builder is None:
        rows = np.arange(samples, dtype=np.int16).reshape(-1, 1)
        data = np.tile(rows, (1, samples)) % 500 + 200
    else:
        data = builder(samples)
    data = data.astype(np.int16)
    if voids:
        r0, c0, size = voids
        data[r0:r0 + size, c0:c0 + size] = SRTM_VOID
    path = tmp_path / name
    path.write_bytes(data.astype('>i2').tobytes())
    return path


# -- the tile size is validated, not guessed ---------------------------------------

@pytest.mark.parametrize('samples', sorted(VALID_TILE_SIZES))
def test_the_two_real_tile_layouts_are_accepted(samples):
    assert infer_samples(samples * samples * 2) == samples


@pytest.mark.parametrize('n_bytes', [12345, 0, 1201 * 1201 * 2 - 1, 999 * 999 * 2])
def test_a_file_that_is_not_a_real_tile_is_refused(n_bytes):
    """int(sqrt(n/2)) would truncate instead of complaining."""
    with pytest.raises(DEMError):
        infer_samples(n_bytes)


# -- tile names --------------------------------------------------------------------

@pytest.mark.parametrize('name,expected', [
    ('N45E024', (45, 24)), ('N45E024.hgt', (45, 24)), ('S01W003', (-1, -3)),
    ('N00E000', (0, 0)),
])
def test_parse_tile_name(name, expected):
    assert parse_tile_name(name) == expected


def test_a_malformed_tile_name_is_refused():
    with pytest.raises(DEMError):
        parse_tile_name('terrain.hgt')


# -- the non-square-cell problem ----------------------------------------------------

def test_a_cell_is_much_wider_than_tall_in_romania():
    """At 45 N a 1-arcsec cell is about 30.9 m N-S but only about 21.8 m E-W."""
    ns_m, ew_m = cell_size_m(45.0, 1)
    assert ns_m == pytest.approx(30.9, abs=0.2)
    assert ew_m == pytest.approx(21.8, abs=0.2)
    assert ns_m / ew_m == pytest.approx(1.0 / math.cos(math.radians(45.0)), rel=0.01)


def test_the_two_cell_sizes_swap_order_at_the_equator():
    """A strong oracle for the ellipsoid formulae: E-W exceeds N-S only near 0."""
    ns_eq, ew_eq = cell_size_m(0.0, 1)
    assert ew_eq > ns_eq
    ns_45, ew_45 = cell_size_m(45.0, 1)
    assert ew_45 < ns_45


def test_to_square_grid_resamples_the_columns_and_keeps_the_north_south_cell():
    surf = np.tile(np.arange(100, dtype=float), (40, 1))      # an east-west ramp
    grid, audit = to_square_grid(surf, 45.0, 1)
    assert grid.cell_size_m == pytest.approx(audit['north_south_m'])
    assert grid.n_cols < surf.shape[1]                         # columns were squeezed
    assert grid.n_rows == surf.shape[0]
    assert audit['resample_factor'] == pytest.approx(audit['east_west_m']
                                                     / audit['north_south_m'])


def test_the_east_west_gradient_survives_the_resampling():
    """The ramp rises 1 unit per sample; after squaring it must rise per METRE the same."""
    surf = np.tile(np.arange(120, dtype=float), (30, 1))
    ns_m, ew_m = cell_size_m(45.0, 1)
    grid, _ = to_square_grid(surf, 45.0, 1)
    total_rise = float(grid.surf[0, -1] - grid.surf[0, 0])
    total_run_m = (grid.n_cols - 1) * grid.cell_size_m
    expected_slope = 1.0 / ew_m                                # units per metre
    assert total_rise / total_run_m == pytest.approx(expected_slope, rel=0.02)


# -- voids -------------------------------------------------------------------------

def test_voids_become_nan_and_are_counted(tmp_path):
    path = synthetic_hgt(tmp_path, samples=1201, voids=(10, 10, 5))
    tile = load_hgt(path)
    assert tile.n_void == 25
    assert np.isnan(tile.elevation_m[12, 12])


def test_fill_voids_replaces_them_with_finite_values(tmp_path):
    path = synthetic_hgt(tmp_path, samples=1201, voids=(10, 10, 5))
    tile = load_hgt(path)
    window, _ = crop(tile, 0, 0, 40)
    filled, audit = fill_voids(window)
    assert audit['n_void'] == 25
    assert audit['largest_void_cells'] == 25
    assert np.isfinite(filled).all()


def test_a_crop_with_too_many_voids_is_refused(tmp_path):
    path = synthetic_hgt(tmp_path, samples=1201, voids=(0, 0, 30))
    with pytest.raises(DEMError, match='void'):
        load_dem_grid(path, row0=0, col0=0, size=40, max_void_fraction=0.01)


def test_a_crop_outside_the_tile_is_refused(tmp_path):
    path = synthetic_hgt(tmp_path, samples=1201)
    tile = load_hgt(path)
    with pytest.raises(DEMError):
        crop(tile, 1190, 1190, 50)


# -- the Grid guards ----------------------------------------------------------------

def test_grid_refuses_a_surface_with_voids_still_in_it():
    with pytest.raises(ValueError, match='non-finite'):
        Grid(surf=np.array([[1.0, np.nan], [2.0, 3.0]]))


def test_grid_refuses_an_unconverted_sentinel():
    grid = Grid(surf=np.full((4, 4), float(SRTM_VOID)))
    with pytest.raises(ValueError, match='sentinel'):
        grid.assert_plausible_heights()


# -- end to end ---------------------------------------------------------------------

def test_a_dem_crop_becomes_a_usable_grid(tmp_path):
    path = synthetic_hgt(tmp_path, samples=1201, voids=(20, 20, 3))
    grid, valid, provenance = load_dem_grid(path, row0=0, col0=0, size=60)
    assert isinstance(grid, Grid)
    assert np.isfinite(grid.surf).all()
    assert grid.cell_size_m == pytest.approx(provenance['cells']['north_south_m'])
    assert provenance['tile']['name'] == 'N45E024'
    assert provenance['crop']['lat_centre'] == pytest.approx(46.0, abs=0.1)
    assert provenance['voids']['n_void'] == 9
    assert valid.shape == grid.shape
    assert len(provenance['tile']['sha256']) == 64


def test_the_whole_pipeline_runs_on_a_dem(tmp_path):
    """load -> HAG -> search -> Algorithm 1 -> checks, with the cell size converted."""
    from core.algorithm_1 import before_after
    from core.checks import check_path
    from core.experiment_step1 import Step1Config, run_step1_full
    from data.configs.road_classes import get

    def hills(samples):
        i = np.arange(samples)[:, None]
        j = np.arange(samples)[None, :]
        return (300 + 40 * np.sin(i / 9.0) + 30 * np.cos(j / 7.0)).astype(np.int16)

    path = synthetic_hgt(tmp_path, samples=1201, builder=hills)
    grid, _, provenance = load_dem_grid(path, row0=0, col0=0, size=60)

    config = Step1Config(start=(5, 5), target=(grid.n_rows - 6, grid.n_cols - 6),
                         height_delta=10.0, terrain_label='DEM')
    run = run_step1_full(config, grid=grid, terrain_info=provenance)

    assert run.row['seed'] is None            # a DEM has no seed; none is fabricated
    assert run.row['config']['terrain'] is provenance
    assert run.paths_xy['full_grid'].shape[1] == 2

    road_class = get('RO_CLASS_V_DEAL').with_cell_size(grid.cell_size_m)
    report = before_after(grid, run.paths_xy['full_grid'], road_class)
    assert math.isfinite(report['after']['length_m'])

    checks = check_path(run.paths_xyz_m['full_grid'], road_class)
    assert checks.verdict


def test_algorithm_1_refuses_a_road_class_whose_cell_size_was_not_converted(tmp_path):
    """On a 10 m synthetic terrain the two cell sizes coincide; on a DEM they do not."""
    from core.algorithm_1 import before_after
    from data.configs.road_classes import get

    path = synthetic_hgt(tmp_path, samples=1201)
    grid, _, _ = load_dem_grid(path, row0=0, col0=0, size=30)
    rough = np.array([[3.0, 3.0], [10.0, 10.0], [20.0, 18.0]])
    with pytest.raises(ValueError, match='with_cell_size'):
        before_after(grid, rough, get('RO_CLASS_V_DEAL'))


# =====================================================================================
# GeoTIFF: the Idrija LiDAR DEM (uncompressed, single band, strips)
# =====================================================================================

#: the real raster, unzipped from raster.zip. Untracked and far too large to commit, so
#: every test below builds its own file and only the last one uses this when it is there.
REAL_TIF = Path(__file__).resolve().parents[1] / 'raster' / 'Idrija_Fault_LiDAR_DEM.tif'


def synthetic_geotiff(tmp_path, data, name='dem.tif', *, byte_order='<',
                      pixel_scale=(1.0, 1.0), origin=(404949.2375, 5110095.8916),
                      epsg=32633, nodata=-99999.0, rows_per_strip=1,
                      strip_order='rotated', magic=42, compression=1,
                      samples_per_pixel=1, tiled=False):
    """Write a classic single-band strip TIFF the way GDAL lays one out.

    `strip_order='rotated'` reproduces the real file's layout, where the strip table does
    NOT ascend from the first row -- the case a `base + row * row_bytes` reader gets wrong.
    """
    data = np.asarray(data)
    height, width = data.shape
    n_strips = -(-height // rows_per_strip)
    strip_bytes = rows_per_strip * width * data.dtype.itemsize

    order = list(range(n_strips))
    if strip_order == 'rotated' and n_strips > 1:
        cut = n_strips // 3 or 1
        order = order[cut:] + order[:cut]      # strip k is written at slot order.index(k)

    tags = {}
    values = bytearray()                       # the out-of-line value block

    def out_of_line(payload):
        offset = len(values)
        values.extend(payload)
        return offset

    sample_format = {'f': 3, 'i': 2, 'u': 1}[data.dtype.kind]
    tags[256] = (3, 1, [width])
    tags[257] = (3, 1, [height])
    tags[258] = (3, 1, [data.dtype.itemsize * 8])
    tags[259] = (3, 1, [compression])
    tags[262] = (3, 1, [1])
    tags[273] = (4, n_strips, None)            # StripOffsets, filled in below
    tags[277] = (3, 1, [samples_per_pixel])
    tags[278] = (3, 1, [rows_per_strip])
    #: the real bytes of each strip; the last one is short when the rows do not divide
    tags[279] = (4, n_strips, [min(rows_per_strip, height - k * rows_per_strip)
                               * width * data.dtype.itemsize for k in range(n_strips)])
    tags[284] = (3, 1, [1])
    if tiled:
        tags[322] = (3, 1, [16])
        tags[323] = (3, 1, [16])
    tags[339] = (3, 1, [sample_format])
    tags[33550] = (12, 3, [pixel_scale[0], pixel_scale[1], 0.0])
    tags[33922] = (12, 6, [0.0, 0.0, 0.0, origin[0], origin[1], 0.0])
    ascii_params = 'WGS 84 / UTM zone 33N|WGS 84|'
    tags[34735] = (3, 12, [1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, epsg])
    tags[34737] = (2, len(ascii_params) + 1, ascii_params)
    if nodata is not None:
        tags[42113] = (2, len(str(nodata)) + 1, str(nodata))

    # layout: header, [strip data], [out-of-line values], IFD -- as in the real file, whose
    # IFD sits at the very end
    data_start = 8
    slot_offset = [data_start + slot * strip_bytes for slot in range(n_strips)]
    tags[273] = (4, n_strips, [slot_offset[order.index(k)] for k in range(n_strips)])

    fmt = {1: 'B', 2: 's', 3: 'H', 4: 'I', 11: 'f', 12: 'd'}
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 11: 4, 12: 8}
    entries = []
    for tag in sorted(tags):
        field_type, count, payload = tags[tag]
        if field_type == 2:
            raw = payload.encode('ascii') + b'\x00'
        else:
            raw = struct.pack(byte_order + f'{count}{fmt[field_type]}', *payload)
        if len(raw) > 4:
            entries.append((tag, field_type, count, out_of_line(raw)))
        else:
            entries.append((tag, field_type, count, raw.ljust(4, b'\x00')))

    payload_start = data_start + n_strips * strip_bytes
    body = bytearray()
    for slot in order:                          # slot s holds strip `order[s]`
        chunk = data[slot * rows_per_strip:(slot + 1) * rows_per_strip]
        body.extend(np.ascontiguousarray(chunk.astype(byte_order + data.dtype.str[1:])).tobytes()
                    .ljust(strip_bytes, b'\x00'))
    ifd_offset = payload_start + len(values)

    ifd = struct.pack(byte_order + 'H', len(entries))
    for tag, field_type, count, value in entries:
        if isinstance(value, int):
            value = struct.pack(byte_order + 'I', payload_start + value)
        ifd += struct.pack(byte_order + 'HHI', tag, field_type, count) + value

    path = tmp_path / name
    with open(path, 'wb') as handle:
        handle.write((b'II' if byte_order == '<' else b'MM')
                     + struct.pack(byte_order + 'HI', magic, ifd_offset))
        handle.write(bytes(body))
        handle.write(bytes(values))
        handle.write(ifd)
    return path


def ramp(height=40, width=50):
    """height == 100 * row + col, so a scrambled row order is impossible to miss."""
    i = np.arange(height, dtype=np.float32)[:, None]
    j = np.arange(width, dtype=np.float32)[None, :]
    return (100.0 + i * 10.0 + j * 0.1).astype(np.float32)


def test_the_header_is_read(tmp_path):
    path = synthetic_geotiff(tmp_path, ramp())
    tif = read_geotiff_header(path, sha256=False)
    assert (tif.height, tif.width) == (40, 50)
    assert tif.dtype.itemsize == 4 and tif.dtype.kind == 'f'
    assert tif.pixel_scale_m == (1.0, 1.0)
    assert tif.epsg == 32633
    assert 'UTM zone 33N' in tif.crs_name
    assert tif.nodata == -99999.0


def test_a_window_reads_back_the_written_values(tmp_path):
    surf = ramp()
    path = synthetic_geotiff(tmp_path, surf)
    tif = read_geotiff_header(path, sha256=False)
    np.testing.assert_allclose(read_window(tif, 0, 0, 40), surf[:40, :40], rtol=0, atol=1e-4)
    np.testing.assert_allclose(read_window(tif, 7, 11, 9), surf[7:16, 11:20], rtol=0, atol=1e-4)


def test_a_rotated_strip_table_is_followed_not_assumed(tmp_path):
    """The real file's layout: `data_start + row * row_bytes` returns a scrambled DEM."""
    surf = ramp()
    path = synthetic_geotiff(tmp_path, surf, strip_order='rotated')
    tif = read_geotiff_header(path, sha256=False)
    assert not np.all(np.diff(tif.strip_offsets) > 0), 'the fixture is not rotated'
    np.testing.assert_allclose(read_window(tif, 0, 0, 40), surf[:, :40], rtol=0, atol=1e-4)

    naive = np.frombuffer(path.read_bytes()[8:8 + surf.size * 4], dtype='<f4')
    assert not np.allclose(naive.reshape(surf.shape)[:, :40], surf[:, :40])


def test_several_rows_per_strip(tmp_path):
    surf = ramp(height=40)
    path = synthetic_geotiff(tmp_path, surf, rows_per_strip=8)
    tif = read_geotiff_header(path, sha256=False)
    assert tif.rows_per_strip == 8 and tif.strip_offsets.size == 5
    np.testing.assert_allclose(read_window(tif, 3, 0, 30), surf[3:33, :30], rtol=0, atol=1e-4)


def test_a_partial_last_strip(tmp_path):
    """45 rows at 8 rows per strip: the last strip holds 5, and its byte count says so."""
    surf = ramp(height=45)
    path = synthetic_geotiff(tmp_path, surf, rows_per_strip=8)
    tif = read_geotiff_header(path, sha256=False)
    assert tif.strip_offsets.size == 6
    np.testing.assert_allclose(read_window(tif, 5, 0, 40), surf[5:45, :40], rtol=0, atol=1e-4)


def test_padded_strips_are_refused_rather_than_read_by_arithmetic(tmp_path):
    """Rows are addressed inside a strip by arithmetic, so padding has to be refused."""
    surf = ramp()
    path = synthetic_geotiff(tmp_path, surf, rows_per_strip=8)
    raw = bytearray(path.read_bytes())
    ifd_start = struct.unpack('<I', raw[4:8])[0]
    (n_entries,) = struct.unpack('<H', raw[ifd_start:ifd_start + 2])
    for i in range(n_entries):
        head = ifd_start + 2 + 12 * i
        tag, _, count = struct.unpack('<HHI', raw[head:head + 8])
        if tag == 279:                                   # StripByteCounts
            offset = struct.unpack('<I', raw[head + 8:head + 12])[0]
            raw[offset:offset + 4] = struct.pack('<I', 8 * 50 * 4 + 16)
            break
    path.write_bytes(bytes(raw))
    with pytest.raises(DEMError, match='padded'):
        read_geotiff_header(path, sha256=False)


def test_big_endian_reads_the_same(tmp_path):
    surf = ramp()
    little = read_geotiff_header(synthetic_geotiff(tmp_path, surf, name='le.tif'), sha256=False)
    big = read_geotiff_header(synthetic_geotiff(tmp_path, surf, name='be.tif',
                                                byte_order='>'), sha256=False)
    np.testing.assert_array_equal(read_window(little, 0, 0, 40), read_window(big, 0, 0, 40))


def test_pixel_to_world_walks_north_to_south(tmp_path):
    tif = read_geotiff_header(synthetic_geotiff(tmp_path, ramp()), sha256=False)
    assert pixel_to_world(tif, 0, 0) == (404949.2375, 5110095.8916)
    east, north = pixel_to_world(tif, 10, 20)
    assert east == pytest.approx(404969.2375) and north == pytest.approx(5110085.8916)


def test_nodata_becomes_nan_and_is_filled(tmp_path):
    surf = ramp()
    surf[5:8, 6:9] = -99999.0
    tif = read_geotiff_header(synthetic_geotiff(tmp_path, surf), sha256=False)
    window = read_window(tif, 0, 0, 40)
    assert np.isnan(window[5:8, 6:9]).all()
    assert int((~np.isfinite(window)).sum()) == 9

    grid, _, provenance = load_geotiff_grid(tif.path, row0=0, col0=0, size=40,
                                            sha256=False, max_void_fraction=0.02)
    assert np.isfinite(grid.surf).all()
    assert provenance['voids']['n_void'] == 9


def test_a_nodata_that_float32_cannot_hold_exactly_is_still_matched(tmp_path):
    """GDAL_NODATA is an ASCII decimal; -9999.9 is stored as -9999.900390625 in float32."""
    surf = ramp()
    surf[2:4, 2:4] = np.float32(-9999.9)
    tif = read_geotiff_header(synthetic_geotiff(tmp_path, surf, nodata=-9999.9), sha256=False)
    window = read_window(tif, 0, 0, 10)
    assert np.isnan(window[2:4, 2:4]).all()
    assert int((~np.isfinite(window)).sum()) == 4


def test_a_window_with_too_many_voids_is_refused(tmp_path):
    surf = ramp()
    surf[:20] = -99999.0
    tif = read_geotiff_header(synthetic_geotiff(tmp_path, surf), sha256=False)
    with pytest.raises(DEMError, match='void'):
        load_geotiff_grid(tif.path, row0=0, col0=0, size=40, sha256=False)


def test_a_window_outside_the_raster_is_refused(tmp_path):
    tif = read_geotiff_header(synthetic_geotiff(tmp_path, ramp()), sha256=False)
    with pytest.raises(DEMError, match='outside'):
        read_window(tif, 30, 0, 40)
    with pytest.raises(DEMError, match='positive'):
        read_window(tif, 0, 0, 0)


@pytest.mark.parametrize('kwargs,message', [
    ({'magic': 43}, 'BigTIFF'),
    ({'compression': 5}, 'compression'),
    ({'samples_per_pixel': 3}, 'samples per pixel'),
    ({'tiled': True}, 'TILED'),
])
def test_what_the_reader_does_not_handle_is_refused_by_name(tmp_path, kwargs, message):
    path = synthetic_geotiff(tmp_path, ramp(), **kwargs)
    with pytest.raises(DEMError, match=message):
        read_geotiff_header(path, sha256=False)


def test_a_non_square_pixel_is_refused_rather_than_averaged(tmp_path):
    """A geographic raster has to go through to_square_grid; averaging would hide the error."""
    path = synthetic_geotiff(tmp_path, ramp(), pixel_scale=(1.0, 1.4))
    with pytest.raises(DEMError, match='not square'):
        load_geotiff_grid(path, row0=0, col0=0, size=20, sha256=False)


def test_a_file_that_is_not_a_tiff_is_refused(tmp_path):
    path = tmp_path / 'not.tif'
    path.write_bytes(b'XY' + b'\x00' * 30)
    with pytest.raises(DEMError, match='byte order'):
        read_geotiff_header(path, sha256=False)


def test_a_geotiff_window_becomes_a_usable_grid(tmp_path):
    grid, valid, provenance = load_geotiff_grid(synthetic_geotiff(tmp_path, ramp()),
                                                row0=2, col0=3, size=20, sha256=False)
    assert isinstance(grid, Grid)
    assert grid.cell_size_m == 1.0                       # native, not resampled
    assert provenance['cells']['resample_factor'] == 1.0
    assert provenance['crop']['epsg'] == 32633
    assert provenance['crop']['easting_west'] == pytest.approx(404952.2375)
    assert provenance['crop']['northing_north'] == pytest.approx(5110093.8916)
    assert valid.shape == grid.shape


def test_the_sha256_is_recorded_when_asked(tmp_path):
    path = synthetic_geotiff(tmp_path, ramp())
    assert read_geotiff_header(path, sha256=False).sha256 is None
    digest = read_geotiff_header(path, sha256=True).sha256
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


# -- the real raster, when it has been unzipped ----------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not REAL_TIF.exists(), reason='raster.zip has not been unzipped')
def test_the_real_idrija_raster_loads():
    """Cross-checked against docs/MetaData_IdrijaLiDAR.pdf: UTM 33N, 1 m, 404949..423752 E."""
    tif = read_geotiff_header(REAL_TIF, sha256=False)
    assert (tif.height, tif.width) == (15200, 18803)
    assert tif.epsg == 32633 and tif.pixel_scale_m == (1.0, 1.0)
    assert tif.nodata == -99999.0
    assert not np.all(np.diff(tif.strip_offsets) > 0)    # the rotated table, in the wild

    # a window inside the surveyed swath; most of the raster is nodata
    grid, _, provenance = load_geotiff_grid(REAL_TIF, row0=1400, col0=150, size=200,
                                            sha256=False, tif=tif)
    assert grid.shape == (200, 200)
    assert grid.cell_size_m == 1.0
    assert np.isfinite(grid.surf).all()
    assert 200.0 < grid.surf.min() and grid.surf.max() < 1100.0
    assert provenance['crop']['crs_name'] == 'WGS 84 / UTM zone 33N'


# -- read_rect: the rectangular counterpart of read_window ------------------------------

def test_read_rect_matches_read_window_on_a_square(tmp_path):
    surf = ramp()
    path = synthetic_geotiff(tmp_path, surf)
    tif = read_geotiff_header(path, sha256=False)
    np.testing.assert_array_equal(read_rect(tif, 3, 4, 10, 10), read_window(tif, 3, 4, 10))


def test_read_rect_reads_a_genuine_rectangle(tmp_path):
    """height == 100*row + col (ramp()), so a wrong row/col stride is easy to catch."""
    surf = ramp(height=40, width=50)
    path = synthetic_geotiff(tmp_path, surf)
    tif = read_geotiff_header(path, sha256=False)
    np.testing.assert_allclose(read_rect(tif, 5, 2, 30, 6), surf[5:35, 2:8],
                               rtol=0, atol=1e-4)


def test_read_rect_follows_the_rotated_strip_table(tmp_path):
    surf = ramp(height=40, width=50)
    path = synthetic_geotiff(tmp_path, surf, strip_order='rotated')
    tif = read_geotiff_header(path, sha256=False)
    np.testing.assert_allclose(read_rect(tif, 0, 0, 40, 30), surf[:, :30], rtol=0, atol=1e-4)


def test_read_rect_marks_nodata_as_nan(tmp_path):
    surf = ramp()
    surf[4:6, 5:7] = -99999.0
    tif = read_geotiff_header(synthetic_geotiff(tmp_path, surf), sha256=False)
    window = read_rect(tif, 0, 0, 40, 50)
    assert np.isnan(window[4:6, 5:7]).all()
    assert int((~np.isfinite(window)).sum()) == 4


def test_read_rect_outside_the_raster_is_refused(tmp_path):
    tif = read_geotiff_header(synthetic_geotiff(tmp_path, ramp()), sha256=False)
    with pytest.raises(DEMError, match='outside'):
        read_rect(tif, 35, 0, 10, 10)
    with pytest.raises(DEMError, match='positive'):
        read_rect(tif, 0, 0, 0, 10)


# -- coverage_mask and full_windows: finding crops without loading the raster -----------

def _brute_force_full_windows(tif, size):
    """The definition full_windows(stride=1) is checked against: read every window for real."""
    found = []
    for r in range(tif.height - size + 1):
        for c in range(tif.width - size + 1):
            if np.isfinite(read_window(tif, r, c, size)).all():
                found.append((r, c))
    return found


def _swath(height=30, width=40, margin=6):
    """A raster mostly nodata, with a valid diagonal 'swath' band -- like the real file."""
    surf = ramp(height=height, width=width)
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    band_centre = cols * (height / width)
    outside = np.abs(rows - band_centre) > margin
    surf = surf.copy()
    surf[outside] = -99999.0
    return surf


def test_coverage_mask_at_stride_1_matches_isfinite_of_the_whole_raster(tmp_path):
    surf = _swath()
    tif = read_geotiff_header(synthetic_geotiff(tmp_path, surf), sha256=False)
    mask = coverage_mask(tif, stride=1)
    assert mask.shape == surf.shape
    np.testing.assert_array_equal(mask, surf != -99999.0)


def test_coverage_mask_stride_decimates_rows_and_columns(tmp_path):
    surf = _swath(height=20, width=30)
    tif = read_geotiff_header(synthetic_geotiff(tmp_path, surf), sha256=False)
    mask = coverage_mask(tif, stride=5)
    assert mask.shape == (4, 6)                       # ceil(20/5), ceil(30/5)
    np.testing.assert_array_equal(mask, (surf != -99999.0)[::5, ::5])


def test_full_windows_at_stride_1_matches_a_brute_force_scan(tmp_path):
    surf = _swath(height=22, width=26, margin=5)
    tif = read_geotiff_header(synthetic_geotiff(tmp_path, surf), sha256=False)
    size = 6
    candidates = {(w.row0, w.col0) for w in full_windows(tif, size, stride=1)}
    expected = set(_brute_force_full_windows(tif, size))
    assert candidates == expected
    assert len(expected) > 0, 'the fixture must actually contain some full windows'


def test_full_windows_rejects_a_size_not_a_multiple_of_stride(tmp_path):
    tif = read_geotiff_header(synthetic_geotiff(tmp_path, ramp()), sha256=False)
    with pytest.raises(DEMError, match='multiple'):
        full_windows(tif, 7, stride=3)


def test_full_windows_decimated_scan_is_a_superset_of_the_exact_one(tmp_path):
    """A stride>1 candidate list may over-approve (samples between the stride can be void)
    but must never miss a window that IS fully covered -- every exact hit is a decimated hit
    too, since the decimated grid is a subset of the same valid cells."""
    surf = _swath(height=30, width=40, margin=6)
    tif = read_geotiff_header(synthetic_geotiff(tmp_path, surf), sha256=False)
    size = 12
    exact = {(w.row0, w.col0) for w in full_windows(tif, size, stride=1)}
    # a stride that divides size, coarse enough to be a real decimation
    candidate = {(w.row0, w.col0) for w in full_windows(tif, size, stride=2)}
    assert exact <= candidate


def test_full_windows_relief_filter(tmp_path):
    """ramp() height is 100*row + col, so relief grows with window size, predictably."""
    surf = ramp(height=30, width=30)
    tif = read_geotiff_header(synthetic_geotiff(tmp_path, surf), sha256=False)
    size = 10
    all_windows = full_windows(tif, size, stride=1)
    assert all(w.relief_m == pytest.approx(9 * 10.0 + 9 * 0.1, abs=0.05) for w in all_windows)
    assert full_windows(tif, size, stride=1, min_relief_m=1000.0) == []


def test_full_windows_returns_nothing_when_the_raster_has_no_valid_window(tmp_path):
    surf = np.full((20, 20), -99999.0, dtype=np.float32)
    surf[5, 5] = 12.0                          # one valid cell, not enough for any window
    tif = read_geotiff_header(synthetic_geotiff(tmp_path, surf), sha256=False)
    assert full_windows(tif, 3, stride=1) == []


# -- coarsen_stream -----------------------------------------------------------------------

def test_coarsen_stream_matches_coarsen_to_cell_size_with_no_voids(tmp_path):
    surf = ramp(height=24, width=18)
    tif = read_geotiff_header(synthetic_geotiff(tmp_path, surf), sha256=False)
    factor = 3

    streamed, audit = coarsen_stream(tif, row0=0, col0=0, n_rows=24, n_cols=18,
                                     factor=factor, band_rows=7)   # not a multiple of factor
    fine = Grid(surf=read_rect(tif, 0, 0, 24, 18), cell_size_m=1.0)
    reference, _ = coarsen_to_cell_size(fine, target_cell_m=float(factor))

    assert audit['shape_after'] == [8, 6]
    assert audit['n_void_fine_cells'] == 0
    np.testing.assert_allclose(streamed, reference.surf, rtol=0, atol=1e-9)


def test_coarsen_stream_bands_do_not_change_the_result(tmp_path):
    """band_rows is a performance knob; a small band and a huge one must agree exactly."""
    surf = ramp(height=40, width=20)
    tif = read_geotiff_header(synthetic_geotiff(tmp_path, surf), sha256=False)
    small_band, _ = coarsen_stream(tif, row0=0, col0=0, n_rows=40, n_cols=20,
                                   factor=4, band_rows=5)
    big_band, _ = coarsen_stream(tif, row0=0, col0=0, n_rows=40, n_cols=20,
                                 factor=4, band_rows=1000)
    np.testing.assert_array_equal(small_band, big_band)


def test_coarsen_stream_a_wholly_void_block_becomes_nan_and_is_counted(tmp_path):
    surf = ramp(height=12, width=12)
    surf[0:3, 0:3] = -99999.0                  # exactly one 3x3 block at factor=3
    tif = read_geotiff_header(synthetic_geotiff(tmp_path, surf), sha256=False)
    coarse, audit = coarsen_stream(tif, row0=0, col0=0, n_rows=12, n_cols=12, factor=3)
    assert np.isnan(coarse[0, 0])
    assert np.isfinite(coarse[0, 1])            # neighbouring blocks are untouched
    assert audit['n_void_fine_cells'] == 9
    assert audit['void_fraction'] == pytest.approx(9 / (12 * 12))


def test_coarsen_stream_a_partly_void_block_averages_the_rest(tmp_path):
    surf = np.zeros((4, 4), dtype=np.float32)
    surf[0, 0] = -99999.0                       # one void cell of a 2x2 block, others 0
    tif = read_geotiff_header(synthetic_geotiff(tmp_path, surf), sha256=False)
    coarse, audit = coarsen_stream(tif, row0=0, col0=0, n_rows=4, n_cols=4, factor=2)
    assert coarse[0, 0] == pytest.approx(0.0)   # mean of the three finite zeros
    assert audit['n_void_fine_cells'] == 1


def test_coarsen_stream_rejects_a_size_not_a_multiple_of_factor(tmp_path):
    tif = read_geotiff_header(synthetic_geotiff(tmp_path, ramp()), sha256=False)
    with pytest.raises(DEMError, match='multiple'):
        coarsen_stream(tif, row0=0, col0=0, n_rows=10, n_cols=7, factor=3)


@pytest.mark.slow
@pytest.mark.skipif(not REAL_TIF.exists(), reason='raster.zip has not been unzipped')
def test_full_windows_census_on_the_real_idrija_raster():
    """The reproducible replacement for the ad hoc, unsaved '102 336 windows' figure.

    A coarse stride keeps this affordable in a test run; core.experiment_dem's swath
    census uses the same function at the stride it reports.
    """
    tif = read_geotiff_header(REAL_TIF, sha256=False)
    windows = full_windows(tif, 600, stride=25)
    assert len(windows) > 0
    # every candidate must itself be a valid crop location
    for w in windows[:5]:
        assert 0 <= w.row0 <= tif.height - 600
        assert 0 <= w.col0 <= tif.width - 600
