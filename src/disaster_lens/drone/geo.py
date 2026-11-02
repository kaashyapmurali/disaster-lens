"""Read GeoTIFFs onto a common metric grid (UTM), using built-in overviews for large mosaics."""

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject


def utm_crs(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def read_geotiff(path, bounds: tuple[float, float, float, float], crs, res: float) -> np.ndarray:
    """Resample the first three bands of a GeoTIFF onto the grid `bounds` (in `crs`) at `res`; uint8 HxWx3.

    Reads the coarsest overview that is still fine enough, so 10+ GB orthomosaics load quickly. Single-band images
    are repeated to grey RGB.
    """
    width = int(round((bounds[2] - bounds[0]) / res))
    height = int(round((bounds[3] - bounds[1]) / res))
    transform = from_origin(bounds[0], bounds[3], res, res)
    with rasterio.open(path) as base:
        src_res_m = abs(base.transform.a) * (111_320 if base.crs.is_geographic else 1)
        level = None
        for i, factor in enumerate(base.overviews(1)):
            if src_res_m * factor <= res * 0.75:
                level = i
    with rasterio.open(path, **({"overview_level": level} if level is not None else {})) as src:
        bands = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
        data = src.read(bands)
        out = np.zeros((3, height, width), dtype=np.uint8)
        for k in range(3):
            reproject(data[k], out[k], src_transform=src.transform, src_crs=src.crs, dst_transform=transform,
                      dst_crs=crs, resampling=Resampling.bilinear, src_nodata=0, dst_nodata=0)
    return out.transpose(1, 2, 0)
