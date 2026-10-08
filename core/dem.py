"""Real elevation data (SRTM/Copernicus `.hgt` tiles) as a `Grid`. Task T8.

Three things a real DEM has that a Perlin terrain does not, and how each is handled:

**The tile size is validated, not guessed.** Inferring it as `int(sqrt(len(data) / 2))`
silently truncates for any file that is not a perfect square of 2-byte samples. Here only the two real SRTM layouts are accepted,
1201 (3 arcsec) and 3601 (1 arcsec), and anything else raises.

**Voids.** SRTM marks missing data with -32768. Nothing downstream survives it:
`core.hag.height_buckets` takes `surf.min()`, so one void turns a 30 m bucket into
about 11000 buckets. Voids become NaN at load, are filled from the nearest valid cell
(or masked out of the search), and the void fraction is reported and capped.

**Cells are not square.** `.hgt` samples sit on a geographic grid, so at 45 degrees N a
1-arcsec cell is about 30.9 m north-south but only about 21.8 m east-west -- a 29%
difference. `Grid` carries a single scalar `cell_size_m`, and feeding it an average
would put that error into every length, every gradient and every radius, which is fatal
for a paper that reports R_min against a norm. So the DEM is resampled to an isotropic
metric grid BEFORE it becomes a `Grid`: the column axis is resampled by
`ew_m / ns_m`, and the resulting square cell is the north-south size.

The residual: the cell size is evaluated at the crop's centre latitude. Over a 500-cell
crop (about 15 km) the latitude moves by roughly 0.14 degrees, so cos(lat) varies by
under 0.25%. That bound is the accuracy claim, and the audit dict carries both metre
values and the resample factor so the choice is auditable.
"""
from __future__ import annotations

import hashlib
import math
import re
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage

#: The real DEM of the study (`raster/Idrija_Fault_LiDAR_DEM.tif`, git-ignored): the Idrija
#: Fault LiDAR DEM from OpenTopography (portal.opentopography.org), 1 m cells, EPSG:32633.
#: Cite it with its DOI; the article's entry is `moulin2024` in article_2/references.py.
IDRIJA_DEM_SOURCE = 'OpenTopography (portal.opentopography.org)'
IDRIJA_DEM_DOI = '10.5069/G9QC01Q2'
IDRIJA_DEM_CITATION = ('Moulin, A. Digital Terrain Model of the Idrija Fault in Northwest '
                       'Slovenia, 2004; distributed by OpenTopography. '
                       f'https://doi.org/{IDRIJA_DEM_DOI}')

from core.grid import Grid

SRTM_VOID = -32768
#: samples per side -> arcseconds per sample. Only the real SRTM layouts.
VALID_TILE_SIZES = {1201: 3, 3601: 1}
#: WGS84
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

_TILE_NAME = re.compile(r'^([NS])(\d{2})([EW])(\d{3})', re.IGNORECASE)


class DEMError(ValueError):
    """The DEM cannot be used as given: wrong size, unreadable name, too many voids."""


@dataclass(frozen=True)
class HgtTile:
    name: str
    lat0: int
    lon0: int
    arcsec: int
    samples: int
    #: (samples, samples) float64, north at row 0, voids as NaN
    elevation_m: np.ndarray
    n_void: int
    sha256: str
    source_path: Optional[Path] = None

    @property
    def void_fraction(self) -> float:
        return self.n_void / float(self.elevation_m.size)

    def as_dict(self) -> dict:
        return {'name': self.name, 'lat0': self.lat0, 'lon0': self.lon0,
                'arcsec': self.arcsec, 'samples': self.samples, 'n_void': self.n_void,
                'void_fraction': self.void_fraction, 'sha256': self.sha256,
                'source_path': str(self.source_path) if self.source_path else None}


def infer_samples(n_bytes: int) -> int:
    """Samples per side for a `.hgt` file of `n_bytes`, or raise."""
    for samples in sorted(VALID_TILE_SIZES):
        if n_bytes == samples * samples * 2:
            return samples
    raise DEMError(
        f"{n_bytes} bytes is not a known SRTM tile. Expected "
        + " or ".join(f"{s * s * 2} ({s}x{s})" for s in sorted(VALID_TILE_SIZES))
        + ". The size is validated rather than inferred from sqrt(n/2), which "
          "silently truncates for a wrong or truncated file.")


def parse_tile_name(name: str) -> tuple[int, int]:
    """`'N45E024'` -> `(45, 24)`; `'S01W003'` -> `(-1, -3)`."""
    match = _TILE_NAME.match(Path(name).stem)
    if not match:
        raise DEMError(f"{name!r} is not an SRTM tile name such as N45E024")
    ns, lat, ew, lon = match.groups()
    lat_v = int(lat) * (1 if ns.upper() == 'N' else -1)
    lon_v = int(lon) * (1 if ew.upper() == 'E' else -1)
    return lat_v, lon_v


def cell_size_m(lat_deg: float, arcsec: int) -> tuple[float, float]:
    """`(north_south_m, east_west_m)` for one sample at `lat_deg`, on WGS84."""
    lat = math.radians(lat_deg)
    w = 1.0 - WGS84_E2 * math.sin(lat) ** 2
    meridian = WGS84_A * (1.0 - WGS84_E2) / w ** 1.5        # metres per radian, N-S
    normal = WGS84_A / math.sqrt(w)                          # metres per radian, E-W
    step = math.radians(arcsec / 3600.0)
    return meridian * step, normal * math.cos(lat) * step


def load_hgt(path, *, samples: Optional[int] = None, arcsec: Optional[int] = None,
             void_value: int = SRTM_VOID) -> HgtTile:
    """Read a big-endian int16 `.hgt` tile. North ends up at row 0, voids are NaN."""
    path = Path(path)
    data = path.read_bytes()
    samples = samples or infer_samples(len(data))
    arcsec = arcsec or VALID_TILE_SIZES.get(samples)
    if arcsec is None:
        raise DEMError(f"no arcsecond spacing known for a {samples}x{samples} tile")

    raw = np.frombuffer(data, dtype='>i2').reshape(samples, samples)
    elevation = raw.astype(np.float64)
    voids = raw == void_value
    elevation[voids] = np.nan

    try:
        lat0, lon0 = parse_tile_name(path.name)
    except DEMError:
        lat0, lon0 = 0, 0

    return HgtTile(name=path.stem, lat0=lat0, lon0=lon0, arcsec=arcsec,
                   samples=samples, elevation_m=elevation, n_void=int(voids.sum()),
                   sha256=hashlib.sha256(data).hexdigest(), source_path=path)


def crop(tile: HgtTile, row0: int, col0: int, size: int) -> tuple[np.ndarray, dict]:
    """A `size x size` window, with the geographic bounds of what was taken."""
    if row0 < 0 or col0 < 0 or row0 + size > tile.samples or col0 + size > tile.samples:
        raise DEMError(f"crop ({row0}, {col0}) + {size} is outside a "
                       f"{tile.samples}x{tile.samples} tile")
    window = tile.elevation_m[row0:row0 + size, col0:col0 + size]
    degrees = tile.arcsec / 3600.0
    #: row 0 is the northern edge, so latitude decreases with the row index
    lat_north = tile.lat0 + 1.0 - row0 * degrees
    bounds = {'row0': row0, 'col0': col0, 'size': size,
              'lat_north': lat_north, 'lat_south': lat_north - size * degrees,
              'lon_west': tile.lon0 + col0 * degrees,
              'lon_east': tile.lon0 + (col0 + size) * degrees}
    bounds['lat_centre'] = 0.5 * (bounds['lat_north'] + bounds['lat_south'])
    return np.ascontiguousarray(window), bounds


def fill_voids(surf: np.ndarray, method: str = 'nearest') -> tuple[np.ndarray, dict]:
    """Fill NaN cells from the nearest valid cell. Returns the filled array and an audit."""
    missing = ~np.isfinite(surf)
    audit = {'n_void': int(missing.sum()),
             'void_fraction': float(missing.mean()),
             'method': method}
    if not missing.any():
        audit['largest_void_cells'] = 0
        return surf.copy(), audit
    if missing.all():
        raise DEMError('every cell is a void')

    labels, _ = ndimage.label(missing)
    sizes = np.bincount(labels.ravel())[1:]
    audit['largest_void_cells'] = int(sizes.max()) if sizes.size else 0

    if method != 'nearest':
        raise DEMError(f"unknown void-fill method {method!r}")
    _, indices = ndimage.distance_transform_edt(missing, return_indices=True)
    return surf[tuple(indices)], audit


def to_square_grid(surf: np.ndarray, lat_deg: float, arcsec: int) -> tuple[Grid, dict]:
    """Resample the columns so the cells are square, then wrap it in a `Grid`."""
    ns_m, ew_m = cell_size_m(lat_deg, arcsec)
    if ew_m <= 0:
        raise DEMError(f"degenerate east-west cell size at latitude {lat_deg}")
    factor = ew_m / ns_m                      # < 1 away from the equator
    n_rows, n_cols = surf.shape
    new_cols = max(2, int(round(n_cols * factor)))

    columns = np.linspace(0, n_cols - 1, new_cols)
    rows = np.arange(n_rows)
    mesh_r, mesh_c = np.meshgrid(rows, columns, indexing='ij')
    resampled = ndimage.map_coordinates(surf, [mesh_r, mesh_c], order=1, mode='nearest')

    audit = {'lat_centre': lat_deg, 'arcsec': arcsec,
             'north_south_m': ns_m, 'east_west_m': ew_m,
             'resample_factor': factor, 'cell_size_m': ns_m,
             'shape_before': [int(n_rows), int(n_cols)],
             'shape_after': [int(resampled.shape[0]), int(resampled.shape[1])]}
    return Grid(surf=resampled, cell_size_m=ns_m), audit


def coarsen_to_cell_size(grid: Grid, target_cell_m: float) -> tuple[Grid, dict]:
    """Block-average a fine DEM onto a coarser square cell, e.g. 1 m LiDAR onto a 10 m grid.

    Needed because **gradient feasibility is scale dependent**. On the Idrija raster, the
    fraction of 7%-admissible steps barely moves with the cell size (62% at 1 m, 61% at 10 m)
    but their CONNECTIVITY collapses: the admissible subgraph of a gentle 600 m crop has
    16531 components at 1 m, whose largest holds 4.5% of the cells, against 83 components and
    67% at 10 m. At 1 m the admissible cells are disconnected slivers between micro-relief
    the road would simply cut and fill through, so an `i_max` test at native resolution
    answers a question about the ground texture, not about the alignment.

    `RoadClass.cell_size_m` is the design resolution the classes are stated for, which is
    why `core.od_feasibility` is meant to be run on a grid coarsened to it. The mean is the
    right reduction here: it is what a cut-and-fill surface approximates, and it preserves
    the macro slope that the gradient check is about.

    Trailing cells that do not fill a whole block are dropped, and the audit says how many.
    """
    factor = target_cell_m / grid.cell_size_m
    if factor < 1 or abs(factor - round(factor)) > 1e-9:
        raise DEMError(f"target cell {target_cell_m} m is not a whole multiple of the grid's "
                       f"{grid.cell_size_m} m (factor {factor:.4f}); block averaging needs one")
    factor = int(round(factor))
    if factor == 1:
        return Grid(surf=grid.surf.copy(), cell_size_m=grid.cell_size_m), {
            'factor': 1, 'cell_size_m': grid.cell_size_m, 'dropped_rows': 0,
            'dropped_cols': 0, 'shape_before': list(grid.shape), 'shape_after': list(grid.shape)}

    n_rows = (grid.n_rows // factor) * factor
    n_cols = (grid.n_cols // factor) * factor
    if n_rows == 0 or n_cols == 0:
        raise DEMError(f"a {grid.shape} grid has no whole {factor}x{factor} block")
    blocks = grid.surf[:n_rows, :n_cols].reshape(n_rows // factor, factor,
                                                 n_cols // factor, factor)
    coarse = Grid(surf=blocks.mean(axis=(1, 3)), cell_size_m=grid.cell_size_m * factor)
    audit = {'factor': factor, 'cell_size_m': coarse.cell_size_m, 'method': 'block mean',
             'dropped_rows': int(grid.n_rows - n_rows), 'dropped_cols': int(grid.n_cols - n_cols),
             'shape_before': list(grid.shape), 'shape_after': list(coarse.shape)}
    return coarse, audit


def load_dem_grid(path, *, row0: int, col0: int, size: int,
                  samples: Optional[int] = None, arcsec: Optional[int] = None,
                  void_policy: str = 'fill_nearest',
                  max_void_fraction: float = 0.01) -> tuple[Grid, np.ndarray, dict]:
    """`.hgt` file -> a square-celled `Grid`, a validity mask, and the provenance.

    `void_policy='fill_nearest'` fills the voids; `'exclude'` leaves them filled but
    returns a mask with them False, so `core.search.build_graph(mask=...)` can keep the
    search out of them. Which policy was used goes into the provenance and must be
    reported (article.md 5.5).
    """
    tile = load_hgt(path, samples=samples, arcsec=arcsec)
    window, bounds = crop(tile, row0, col0, size)

    valid = np.isfinite(window)
    void_fraction = float((~valid).mean())
    if void_fraction > max_void_fraction:
        raise DEMError(f"{void_fraction:.1%} of the crop is void, above the "
                       f"{max_void_fraction:.1%} limit. Choose another crop or raise "
                       f"the limit deliberately.")

    filled, void_audit = fill_voids(window)
    grid, cell_audit = to_square_grid(filled, bounds['lat_centre'], tile.arcsec)

    #: the mask has to be resampled the same way the heights were
    if void_policy == 'exclude':
        mask_grid, _ = to_square_grid(valid.astype(float), bounds['lat_centre'],
                                      tile.arcsec)
        valid_resampled = mask_grid.surf > 0.999
    elif void_policy == 'fill_nearest':
        valid_resampled = np.ones(grid.shape, dtype=bool)
    else:
        raise DEMError(f"unknown void policy {void_policy!r}")

    grid.assert_plausible_heights()
    provenance = {'tile': tile.as_dict(), 'crop': bounds, 'voids': void_audit,
                  'cells': cell_audit, 'void_policy': void_policy,
                  'note': 'cell size taken at the crop centre latitude; over a '
                          '500-cell crop cos(lat) varies by under 0.25%'}
    return grid, valid_resampled, provenance


# =====================================================================================
# GeoTIFF (uncompressed, single band, strip based)
# =====================================================================================
#
# A fourth thing a real DEM has: it may not be an `.hgt` tile at all. The Idrija LiDAR DEM
# is an 18803 x 15200 float32 GeoTIFF in EPSG:32633 (WGS 84 / UTM zone 33N) with 1 m square
# cells, so the geographic resampling above is not needed -- the cells are already metric
# and square, and `to_square_grid` must NOT be applied to it.
#
# The reader below is hand written and deliberately narrow, for two reasons:
#
# **No new dependency.** `setup_venv` installs with `--only-binary=:all:` on CPython 3.14
# free-threaded, so every dependency has to publish a cp314t wheel. Reading one window of a
# 1.14 GB raster needs a few hundred seeks, not GDAL.
#
# **Row order comes from the strip table, never from arithmetic.** The real file's
# `StripOffsets` array is *rotated*: it starts at 1128602808, runs to the end of the file
# and wraps round to 121960. A reader that assumes `data_start + row * row_bytes` -- the
# obvious implementation -- returns a silently scrambled DEM. Every row here is addressed
# through `strip_offsets`.
#
# Everything this reader does not accept raises `DEMError` naming the limitation, in the
# same spirit as `infer_samples`: validated, not guessed.

#: TIFF tag numbers used here
_TAG = {'ImageWidth': 256, 'ImageLength': 257, 'BitsPerSample': 258, 'Compression': 259,
        'PhotometricInterpretation': 262, 'StripOffsets': 273, 'SamplesPerPixel': 277,
        'RowsPerStrip': 278, 'StripByteCounts': 279, 'PlanarConfiguration': 284,
        'TileWidth': 322, 'TileLength': 323, 'TileOffsets': 324, 'SampleFormat': 339,
        'ModelPixelScale': 33550, 'ModelTiepoint': 33922, 'ModelTransformation': 34264,
        'GeoKeyDirectory': 34735, 'GeoAsciiParams': 34737, 'GDALNoData': 42113}
#: TIFF field type -> (struct code, byte size)
_FIELD_TYPES = {1: ('B', 1), 2: ('s', 1), 3: ('H', 2), 4: ('I', 4), 5: ('I', 8),
                6: ('b', 1), 7: ('B', 1), 8: ('h', 2), 9: ('i', 4), 10: ('i', 8),
                11: ('f', 4), 12: ('d', 8)}
#: (SampleFormat, BitsPerSample) -> numpy dtype character. 1 = uint, 2 = int, 3 = float.
_SAMPLE_DTYPES = {(1, 16): 'u2', (1, 32): 'u4', (2, 16): 'i2', (2, 32): 'i4',
                  (3, 32): 'f4', (3, 64): 'f8'}
#: GeoTIFF key ids
_PROJECTED_CS_TYPE = 3072
_GEOGRAPHIC_TYPE = 2048
_PCS_CITATION = 3073
_GT_CITATION = 1026


@dataclass(frozen=True)
class GeoTiff:
    """The header of a single-band, uncompressed, strip-based GeoTIFF."""

    path: Path
    width: int
    height: int
    dtype: np.dtype
    #: (east_west_m, north_south_m) from ModelPixelScale
    pixel_scale_m: tuple[float, float]
    #: world coordinates of the upper-left corner of pixel (0, 0)
    origin_m: tuple[float, float]
    epsg: Optional[int]
    crs_name: str
    nodata: Optional[float]
    #: file offset of each strip, IN STRIP ORDER -- not necessarily ascending
    strip_offsets: np.ndarray
    strip_byte_counts: np.ndarray
    rows_per_strip: int
    sha256: Optional[str] = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def as_dict(self) -> dict:
        return {'name': self.path.stem, 'width': self.width, 'height': self.height,
                'dtype': str(self.dtype), 'pixel_scale_m': list(self.pixel_scale_m),
                'origin_m': list(self.origin_m), 'epsg': self.epsg,
                'crs_name': self.crs_name, 'nodata': self.nodata,
                'rows_per_strip': self.rows_per_strip,
                'n_strips': int(self.strip_offsets.size),
                'strip_offsets_ascending': bool(np.all(np.diff(self.strip_offsets) > 0)),
                'sha256': self.sha256, 'source_path': str(self.path)}


def _read_entry_values(handle, byte_order: str, field_type: int, count: int, payload: bytes):
    """One IFD entry's value, following the offset in `payload` when it does not fit in 4 bytes."""
    code, size = _FIELD_TYPES[field_type]
    total = size * count
    if total > 4:
        offset = struct.unpack(byte_order + 'I', payload)[0]
        handle.seek(offset)
        raw = handle.read(total)
        if len(raw) != total:
            raise DEMError(f"IFD value at offset {offset} is truncated")
    else:
        raw = payload[:total]
    if field_type == 2:
        return raw.split(b'\x00')[0].decode('ascii', 'replace')
    if field_type in (5, 10):  # rationals, stored as numerator/denominator pairs
        pairs = struct.unpack(byte_order + f'{2 * count}{code}', raw)
        return [pairs[i] / pairs[i + 1] if pairs[i + 1] else math.nan
                for i in range(0, len(pairs), 2)]
    return list(struct.unpack(byte_order + f'{count}{code}', raw))


def _read_ifd(handle, byte_order: str, offset: int) -> dict:
    """Tag number -> list of values, for the IFD at `offset`."""
    handle.seek(offset)
    (n_entries,) = struct.unpack(byte_order + 'H', handle.read(2))
    entries = handle.read(12 * n_entries)
    if len(entries) != 12 * n_entries:
        raise DEMError('the IFD is truncated')
    tags = {}
    for i in range(n_entries):
        tag, field_type, count = struct.unpack(byte_order + 'HHI', entries[12 * i:12 * i + 8])
        if field_type not in _FIELD_TYPES:
            continue  # an unknown field type is not a reason to refuse the whole file
        tags[tag] = _read_entry_values(handle, byte_order, field_type, count,
                                       entries[12 * i + 8:12 * i + 12])
    return tags


def _geo_keys(tags: dict) -> dict:
    """GeoKeyDirectory as `key id -> (tag location, count, value)`."""
    directory = tags.get(_TAG['GeoKeyDirectory'])
    if not directory or len(directory) < 4:
        return {}
    keys = {}
    for i in range(directory[3]):
        head = 4 + 4 * i
        if head + 4 > len(directory):
            break
        key_id, location, count, value = directory[head:head + 4]
        keys[key_id] = (location, count, value)
    return keys


def _crs(tags: dict) -> tuple[Optional[int], str]:
    """`(EPSG code, human readable name)`, as far as the GeoTIFF keys say."""
    keys = _geo_keys(tags)
    epsg = None
    for key in (_PROJECTED_CS_TYPE, _GEOGRAPHIC_TYPE):
        location, _, value = keys.get(key, (None, None, None))
        if location == 0 and value not in (None, 0, 32767):
            epsg = int(value)
            break
    ascii_params = tags.get(_TAG['GeoAsciiParams'], '')
    name = ''
    for key in (_PCS_CITATION, _GT_CITATION):
        location, count, value = keys.get(key, (None, None, None))
        if location == _TAG['GeoAsciiParams'] and ascii_params:
            name = ascii_params[value:value + count].strip('|').strip()
            break
    if not name and ascii_params:
        name = ascii_params.split('|')[0]
    return epsg, name


def _file_sha256(path: Path, chunk: int = 1 << 23) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def read_geotiff_header(path, *, sha256: bool = True) -> GeoTiff:
    """Read and validate the header of a single-band uncompressed strip GeoTIFF."""
    path = Path(path)
    with open(path, 'rb') as handle:
        header = handle.read(8)
        if len(header) < 8:
            raise DEMError(f"{path.name} is too short to be a TIFF")
        if header[:2] == b'II':
            byte_order = '<'
        elif header[:2] == b'MM':
            byte_order = '>'
        else:
            raise DEMError(f"{header[:2]!r} is not a TIFF byte order mark (II or MM)")
        (magic,) = struct.unpack(byte_order + 'H', header[2:4])
        if magic == 43:
            raise DEMError("BigTIFF (magic 43) is not supported by this reader, only "
                           "classic TIFF (magic 42). Convert the file or extend the reader.")
        if magic != 42:
            raise DEMError(f"TIFF magic {magic} is neither 42 (classic) nor 43 (BigTIFF)")
        (ifd_offset,) = struct.unpack(byte_order + 'I', header[4:8])
        tags = _read_ifd(handle, byte_order, ifd_offset)

    def one(name, default=None):
        values = tags.get(_TAG[name])
        return default if not values else values[0]

    if _TAG['TileWidth'] in tags or _TAG['TileOffsets'] in tags:
        raise DEMError(f"{path.name} is a TILED TIFF; this reader only handles strips.")
    compression = one('Compression', 1)
    if compression != 1:
        raise DEMError(f"{path.name} uses TIFF compression {compression}; this reader only "
                       "handles uncompressed data (compression 1).")
    samples_per_pixel = one('SamplesPerPixel', 1)
    if samples_per_pixel != 1:
        raise DEMError(f"{path.name} has {samples_per_pixel} samples per pixel; a DEM is "
                       "a single band.")
    planar = one('PlanarConfiguration', 1)
    if planar != 1:
        raise DEMError(f"{path.name} uses planar configuration {planar}; only 1 (chunky).")
    if _TAG['ModelTransformation'] in tags and _TAG['ModelPixelScale'] not in tags:
        raise DEMError(f"{path.name} is georeferenced by a ModelTransformation matrix "
                       "(rotated or sheared); this reader needs ModelPixelScale plus "
                       "ModelTiepoint.")

    bits = one('BitsPerSample', 32)
    sample_format = one('SampleFormat', 1)
    dtype_char = _SAMPLE_DTYPES.get((sample_format, bits))
    if dtype_char is None:
        raise DEMError(f"{path.name} stores {bits}-bit samples of format {sample_format}; "
                       f"supported (SampleFormat, BitsPerSample): {sorted(_SAMPLE_DTYPES)}")

    width, height = one('ImageWidth'), one('ImageLength')
    if not width or not height:
        raise DEMError(f"{path.name} declares no image size")
    rows_per_strip = min(int(one('RowsPerStrip', height) or height), int(height))

    offsets = tags.get(_TAG['StripOffsets'])
    byte_counts = tags.get(_TAG['StripByteCounts'])
    if not offsets:
        raise DEMError(f"{path.name} has no StripOffsets")
    expected_strips = -(-height // rows_per_strip)
    if len(offsets) != expected_strips:
        raise DEMError(f"{path.name} lists {len(offsets)} strips but {expected_strips} are "
                       f"implied by {height} rows at {rows_per_strip} rows per strip")

    #: `read_window` addresses a row inside its strip as `within * width * itemsize`, so a
    #: strip must be exactly its rows, unpadded. StripByteCounts is what says whether it is.
    item = np.dtype(dtype_char).itemsize
    if byte_counts:
        rows_in = [min(rows_per_strip, height - k * rows_per_strip)
                   for k in range(len(offsets))]
        expected = [rows * width * item for rows in rows_in]
        if list(byte_counts) != expected:
            raise DEMError(f"{path.name} has padded or otherwise unexpected strip sizes "
                           f"(first {byte_counts[0]} bytes, expected {expected[0]}); this "
                           "reader addresses rows inside a strip by arithmetic.")

    scale = tags.get(_TAG['ModelPixelScale'], [1.0, 1.0, 0.0])
    tiepoint = tags.get(_TAG['ModelTiepoint'], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    if len(tiepoint) < 6:
        raise DEMError(f"{path.name} has a ModelTiepoint of {len(tiepoint)} values, not 6")
    #: the tiepoint maps a raster point to a world point; a non-zero raster point shifts it
    origin = (tiepoint[3] - tiepoint[0] * scale[0], tiepoint[4] + tiepoint[1] * scale[1])

    nodata_text = tags.get(_TAG['GDALNoData'])
    try:
        nodata = float(nodata_text) if nodata_text else None
    except ValueError:
        nodata = None

    epsg, crs_name = _crs(tags)
    return GeoTiff(path=path, width=int(width), height=int(height),
                   dtype=np.dtype(byte_order + dtype_char),
                   pixel_scale_m=(float(scale[0]), float(scale[1])),
                   origin_m=(float(origin[0]), float(origin[1])),
                   epsg=epsg, crs_name=crs_name, nodata=nodata,
                   strip_offsets=np.asarray(offsets, dtype=np.int64),
                   strip_byte_counts=np.asarray(byte_counts or [], dtype=np.int64),
                   rows_per_strip=rows_per_strip,
                   sha256=_file_sha256(path) if sha256 else None)


def read_window(tif: GeoTiff, row0: int, col0: int, size: int) -> np.ndarray:
    """A `size x size` window as float64, nodata as NaN. Row 0 of the file is north.

    Every row is addressed through `tif.strip_offsets`; the strip table of a real GDAL file
    is not necessarily in file order, so `data_start + row * row_bytes` is wrong.
    """
    if size <= 0:
        raise DEMError(f"window size {size} must be positive")
    if row0 < 0 or col0 < 0 or row0 + size > tif.height or col0 + size > tif.width:
        raise DEMError(f"window ({row0}, {col0}) + {size} is outside a "
                       f"{tif.height}x{tif.width} raster")

    item = tif.dtype.itemsize
    out = np.empty((size, size), dtype=np.float64)
    with open(tif.path, 'rb') as handle:
        for i in range(size):
            row = row0 + i
            strip = row // tif.rows_per_strip
            within = row - strip * tif.rows_per_strip
            handle.seek(int(tif.strip_offsets[strip]) + (within * tif.width + col0) * item)
            raw = handle.read(size * item)
            if len(raw) != size * item:
                raise DEMError(f"row {row} is truncated in {tif.path.name}")
            out[i] = np.frombuffer(raw, dtype=tif.dtype).astype(np.float64)

    if tif.nodata is not None:
        #: GDAL_NODATA is an ASCII decimal, so it has to be compared in the FILE's dtype:
        #: a float32 raster whose nodata is -9999.9 stores -9999.900390625, and comparing
        #: the widened value against the decimal would silently match nothing.
        out[out == np.asarray(tif.nodata, dtype=tif.dtype).astype(np.float64)] = np.nan
    return out


def pixel_to_world(tif: GeoTiff, row: float, col: float) -> tuple[float, float]:
    """`(easting, northing)` of a pixel's upper-left corner. Northing decreases with the row."""
    return (tif.origin_m[0] + col * tif.pixel_scale_m[0],
            tif.origin_m[1] - row * tif.pixel_scale_m[1])


def load_geotiff_grid(path, *, row0: int, col0: int, size: int,
                      void_policy: str = 'fill_nearest',
                      max_void_fraction: float = 0.01,
                      sha256: bool = True,
                      tif: Optional[GeoTiff] = None) -> tuple[Grid, np.ndarray, dict]:
    """GeoTIFF window -> a square-celled `Grid`, a validity mask, and the provenance.

    The same contract as `load_dem_grid`, void policies included, but with no geographic
    resampling: a projected DEM already has square metric cells, so `cell_size_m` is read
    straight from `ModelPixelScale` and a non-square scale is refused rather than averaged.

    `tif` reuses an already-read header, so a script that loads several windows of a 1.14 GB
    file does not re-hash it every time.
    """
    tif = read_geotiff_header(path, sha256=sha256) if tif is None else tif
    scale_x, scale_y = tif.pixel_scale_m
    if scale_x <= 0 or scale_y <= 0:
        raise DEMError(f"degenerate pixel scale {tif.pixel_scale_m}")
    if abs(scale_x - scale_y) > 1e-9 * scale_x:
        raise DEMError(f"pixel scale {tif.pixel_scale_m} is not square. A projected DEM is "
                       "expected to be; a geographic one has to go through to_square_grid.")

    window = read_window(tif, row0, col0, size)
    valid = np.isfinite(window)
    void_fraction = float((~valid).mean())
    if void_fraction > max_void_fraction:
        raise DEMError(f"{void_fraction:.1%} of the crop is void, above the "
                       f"{max_void_fraction:.1%} limit. Choose another crop or raise "
                       f"the limit deliberately.")

    filled, void_audit = fill_voids(window)
    grid = Grid(surf=filled, cell_size_m=scale_x)

    if void_policy == 'exclude':
        valid_mask = valid
    elif void_policy == 'fill_nearest':
        valid_mask = np.ones(grid.shape, dtype=bool)
    else:
        raise DEMError(f"unknown void policy {void_policy!r}")

    grid.assert_plausible_heights()
    north_west = pixel_to_world(tif, row0, col0)
    south_east = pixel_to_world(tif, row0 + size, col0 + size)
    bounds = {'row0': row0, 'col0': col0, 'size': size,
              'easting_west': north_west[0], 'northing_north': north_west[1],
              'easting_east': south_east[0], 'northing_south': south_east[1],
              'epsg': tif.epsg, 'crs_name': tif.crs_name}
    provenance = {'tiff': tif.as_dict(), 'crop': bounds, 'voids': void_audit,
                  'cells': {'cell_size_m': scale_x, 'source': 'ModelPixelScale',
                            'resample_factor': 1.0,
                            'shape_before': [size, size],
                            'shape_after': [int(grid.n_rows), int(grid.n_cols)]},
                  'void_policy': void_policy,
                  'note': 'projected CRS, square metric cells, read at native resolution; '
                          'no geographic resampling was applied'}
    return grid, valid_mask, provenance


# =====================================================================================
# Finding and reading crops of a raster too large to load whole
# =====================================================================================
#
# The Idrija swath is about 2.2 km wide and 23 km long, and only the swath itself is
# valid: the rest of the 18803x15200 raster is nodata. `load_geotiff_grid` refuses a
# window with too many voids, which answers "is THIS crop usable" but not "which crops
# ARE usable", and it reads a SQUARE window into memory whole, which does not fit the
# swath's 10:1 aspect or its ~23 km length at native resolution. The three functions
# below answer those questions without ever materialising the whole file.


def read_rect(tif: GeoTiff, row0: int, col0: int, n_rows: int, n_cols: int) -> np.ndarray:
    """A `n_rows x n_cols` window as float64, nodata as NaN. Row 0 of the file is north.

    The rectangular counterpart of `read_window`, which is square only. Same strip-table
    addressing (the file's `StripOffsets` is rotated, see the module note above
    `read_window`), same nodata handling; kept as a separate implementation rather than a
    generalisation of `read_window` so neither risks the other's tests.
    """
    if n_rows <= 0 or n_cols <= 0:
        raise DEMError(f"window {n_rows}x{n_cols} must be positive")
    if row0 < 0 or col0 < 0 or row0 + n_rows > tif.height or col0 + n_cols > tif.width:
        raise DEMError(f"window ({row0}, {col0}) + {n_rows}x{n_cols} is outside a "
                       f"{tif.height}x{tif.width} raster")

    item = tif.dtype.itemsize
    out = np.empty((n_rows, n_cols), dtype=np.float64)
    with open(tif.path, 'rb') as handle:
        for i in range(n_rows):
            row = row0 + i
            strip = row // tif.rows_per_strip
            within = row - strip * tif.rows_per_strip
            handle.seek(int(tif.strip_offsets[strip]) + (within * tif.width + col0) * item)
            raw = handle.read(n_cols * item)
            if len(raw) != n_cols * item:
                raise DEMError(f"row {row} is truncated in {tif.path.name}")
            out[i] = np.frombuffer(raw, dtype=tif.dtype).astype(np.float64)

    if tif.nodata is not None:
        out[out == np.asarray(tif.nodata, dtype=tif.dtype).astype(np.float64)] = np.nan
    return out


def _decimated_scan(tif: GeoTiff, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """One pass over the raster, keeping every `stride`-th row and column.

    Reads a whole row per seek (one strip-table lookup, `width` samples), rather than one
    cell at a time, so scanning the whole raster costs `O(height / stride)` seeks of
    `O(width)` bytes each, not `O(height * width)`. At `stride=1` this reads every row and
    is exact; at a larger stride it is a candidate scan (see `full_windows`).

    Returns `(heights, valid)`, both `(ceil(height/stride), ceil(width/stride))`: `heights`
    is the sampled value with nodata cells as NaN, `valid` is `np.isfinite(heights)`.
    """
    if stride < 1:
        raise DEMError(f"stride must be >= 1, got {stride}")
    rows = np.arange(0, tif.height, stride, dtype=np.int64)
    cols = np.arange(0, tif.width, stride, dtype=np.int64)
    item = tif.dtype.itemsize
    nodata = (np.asarray(tif.nodata, dtype=tif.dtype).astype(np.float64)
              if tif.nodata is not None else None)

    heights = np.empty((rows.size, cols.size), dtype=np.float64)
    with open(tif.path, 'rb') as handle:
        for k, row in enumerate(rows):
            strip = int(row) // tif.rows_per_strip
            within = int(row) - strip * tif.rows_per_strip
            handle.seek(int(tif.strip_offsets[strip]) + within * tif.width * item)
            raw = handle.read(tif.width * item)
            if len(raw) != tif.width * item:
                raise DEMError(f"row {row} is truncated in {tif.path.name}")
            full_row = np.frombuffer(raw, dtype=tif.dtype).astype(np.float64)
            heights[k] = full_row[cols]

    valid = np.isfinite(heights)
    if nodata is not None:
        valid &= heights != nodata
    return np.where(valid, heights, np.nan), valid


def coverage_mask(tif: GeoTiff, *, stride: int = 50) -> np.ndarray:
    """A decimated validity mask of the whole raster, without loading it.

    `stride=1` reads every cell -- exact, but then the mask is the size of the raster, so
    only affordable on a small file or a small crop. A larger stride is a candidate scan:
    `full_windows` builds on it to find crops worth verifying exactly, not to certify one
    on its own. See `_decimated_scan`.
    """
    return _decimated_scan(tif, stride)[1]


@dataclass(frozen=True)
class WindowCandidate:
    """One candidate crop from `full_windows`: coverage checked, not yet verified exactly."""

    row0: int
    col0: int
    size: int
    #: max - min of the DECIMATED samples inside the window: a LOWER BOUND on the true
    #: relief, since the true extremes can fall between the samples that were read.
    relief_m: float

    def as_dict(self) -> dict:
        return {'row0': self.row0, 'col0': self.col0, 'size': self.size,
                'relief_m': self.relief_m}


def full_windows(tif: GeoTiff, size: int, *, stride: int = 1,
                 min_relief_m: Optional[float] = None) -> list[WindowCandidate]:
    """Every `size x size` window whose DECIMATED samples are all valid.

    Built with a summed-area table over `coverage_mask`'s grid, so the whole raster is
    scanned once regardless of how many candidate windows exist, rather than once per
    window. `size` must be a whole multiple of `stride`.

    **This is a candidate list, not a certificate of coverage.** A stride greater than 1
    only samples every stride-th cell, so a void strictly between two samples is invisible
    to it; a caller that needs a guarantee reads the window for real before using it as
    data (`read_window`/`read_rect` plus `np.isfinite`, or `load_geotiff_grid(...,
    max_void_fraction=0.0)`, which raises on the first void found). At `stride=1` every
    cell is checked and the list IS exact -- `tests/test_dem.py` cross-checks that case
    against a brute-force scan, which is also the proof that the decimated form at
    `stride > 1` is the same algorithm, just faster and approximate. This two-stage design
    (cheap decimated candidates, then an exact check of the ones actually used) is what
    replaces the "102 336 fully covered 600 m windows" figure that an earlier session
    computed ad hoc and did not save: that number is not reproducible and must not be
    quoted; a count from this function, with its stride stated, is.

    `relief_m`, when reported, is the max-min of the DECIMATED heights inside the window,
    which understates the true relief for the same sampling reason: it ranks candidates,
    it does not measure the terrain.
    """
    if size <= 0:
        raise DEMError(f"size must be positive, got {size}")
    if size % stride != 0:
        raise DEMError(f"size {size} must be a whole multiple of stride {stride}")

    heights, valid = _decimated_scan(tif, stride)
    win = size // stride
    if win > valid.shape[0] or win > valid.shape[1]:
        return []

    sat = np.zeros((valid.shape[0] + 1, valid.shape[1] + 1), dtype=np.int64)
    sat[1:, 1:] = np.cumsum(np.cumsum(valid.astype(np.int64), axis=0), axis=1)
    total = sat[win:, win:] - sat[:-win, win:] - sat[win:, :-win] + sat[:-win, :-win]
    rows_dec, cols_dec = np.nonzero(total == win * win)

    out = []
    for r, c in zip(rows_dec.tolist(), cols_dec.tolist()):
        block = heights[r:r + win, c:c + win]
        relief = float(np.nanmax(block) - np.nanmin(block)) if block.size else 0.0
        if min_relief_m is not None and relief < min_relief_m:
            continue
        out.append(WindowCandidate(row0=r * stride, col0=c * stride, size=size,
                                   relief_m=relief))
    return out


def coarsen_stream(tif: GeoTiff, *, row0: int, col0: int, n_rows: int, n_cols: int,
                   factor: int, band_rows: int = 4000,
                   strict: bool = False) -> tuple[np.ndarray, dict]:
    """Block-mean a big rectangular crop onto a coarser cell, without holding it fine.

    `load_geotiff_grid` reads a crop whole and `coarsen_to_cell_size` then averages it,
    which is fine for a few hundred cells square but not for the whole ~23 km swath: at
    1 m that crop is tens of millions of float64 cells (`load_geotiff_grid` also refuses
    any void at all above `max_void_fraction`, which the swath's own nodata border makes
    unusable here). This instead reads the crop in horizontal BANDS of up to `band_rows`
    fine rows (rounded down to a whole multiple of `factor`), block-means each band on its
    own with `read_rect`, and stacks the coarse bands, so peak memory is one band, not the
    whole crop.

    A block whose fine cells are ALL void averages to NaN (`np.nanmean` of an empty slice);
    a block with SOME void cells averages the rest, which is what block averaging with
    missing data means. This function does not fill voids -- the caller decides how, via
    `fill_voids` on the returned array or by excluding the NaNs from a search mask -- and
    the audit reports how many fine cells were void so the choice is made knowingly.
    `strict=True` makes a block with ANY void fine cell NaN, so no coarse cell is an
    average over a partly surveyed block (the policy of the whole-DEM experiments).
    """
    if factor < 1:
        raise DEMError(f"factor must be >= 1, got {factor}")
    if n_rows % factor or n_cols % factor:
        raise DEMError(f"{n_rows}x{n_cols} is not a whole multiple of factor {factor}")

    band_rows = max(factor, (band_rows // factor) * factor)
    coarse_rows, coarse_cols = n_rows // factor, n_cols // factor
    out = np.empty((coarse_rows, coarse_cols), dtype=np.float64)
    n_void = 0

    for band_start in range(0, n_rows, band_rows):
        rows_here = min(band_rows, n_rows - band_start)
        rows_here -= rows_here % factor
        if rows_here == 0:
            break
        band = read_rect(tif, row0 + band_start, col0, rows_here, n_cols)
        n_void += int((~np.isfinite(band)).sum())
        blocks = band.reshape(rows_here // factor, factor, coarse_cols, factor)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            coarse_band = np.nanmean(blocks, axis=(1, 3))
        if strict:
            coarse_band[np.isnan(blocks).any(axis=(1, 3))] = np.nan
        dest0 = band_start // factor
        out[dest0:dest0 + rows_here // factor] = coarse_band

    audit = {'factor': factor, 'shape_before': [n_rows, n_cols],
             'shape_after': [coarse_rows, coarse_cols], 'n_void_fine_cells': n_void,
             'void_fraction': (n_void / (n_rows * n_cols)) if n_rows * n_cols else 0.0,
             'method': 'block mean, streamed in bands, NaN-aware'
                       + (', any void -> void' if strict else '')}
    return out, audit
