"""Earth Engine helpers shared by the Kharif benchmark scripts.

Data sources (all public Earth Engine catalog assets):
  COPERNICUS/S2_HARMONIZED          Sentinel-2 L1C, 13 bands (same product level as SEN12MS-CR)
  COPERNICUS/S2_CLOUD_PROBABILITY   s2cloudless probability, joined on system:index
  COPERNICUS/S1_GRD                 Sentinel-1 GRD IW, VV + VH in dB (as SEN12MS-CR)
  ESA/WorldCover/v200               10 m land cover (cropland = class 40), 2021
  FAO/GAUL/2015/level2              district boundaries
"""
import io
import math
from pathlib import Path

import yaml

S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
NODATA_S1 = -9999.0


def load_cfg(path=None):
    path = Path(path or Path(__file__).with_name("kharif.yaml"))
    return yaml.safe_load(path.read_text())


def init(project):
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    return ee


def aoi_geometry(ee, aoi):
    if "bbox" in aoi:
        return ee.Geometry.Rectangle(aoi["bbox"])
    gaul = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM0_NAME", "India"))
    fc = gaul.filter(ee.Filter.eq("ADM1_NAME", aoi["state"])).filter(ee.Filter.eq("ADM2_NAME", aoi["district"]))
    if fc.size().getInfo() == 0:
        names = gaul.filter(ee.Filter.eq("ADM1_NAME", aoi["state"])).aggregate_array("ADM2_NAME").getInfo()
        raise ValueError(f"district {aoi['district']!r} not in GAUL for state {aoi['state']!r}; "
                         f"available: {sorted(set(names))}")
    return fc.geometry()


def s2_with_clouds(ee, geom, start, end):
    """S2 L1C images over geom with the s2cloudless probability attached as band 'prob'."""
    s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(geom).filterDate(start, end)
    prob = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(geom).filterDate(start, end)
    joined = ee.Join.saveFirst("cloud").apply(
        s2, prob, ee.Filter.equals(leftField="system:index", rightField="system:index"))
    return ee.ImageCollection(joined).map(
        lambda img: ee.Image(img).addBands(ee.Image(ee.Image(img).get("cloud")).select("probability").rename("prob")))


def cloud_and_shadow(ee, img, prob_threshold=40, nir_dark=0.15, max_dist_km=1.0):
    """Binary cloud and projected-shadow bands for one S2 L1C image (after the Earth Engine
    s2cloudless tutorial: dark NIR pixels inside the cloud projection along the sun azimuth)."""
    cloud = img.select("prob").gt(prob_threshold).rename("cloud")
    ndwi = img.normalizedDifference(["B3", "B8"])
    dark = img.select("B8").lt(nir_dark * 1e4).And(ndwi.lt(0))
    azimuth = ee.Number(90).subtract(ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE")))
    proj = (cloud.directionalDistanceTransform(azimuth, max_dist_km * 10)
            .reproject(crs=img.select("B2").projection(), scale=100)
            .select("distance").mask())
    shadow = proj.And(dark).And(cloud.Not()).rename("shadow")
    return cloud, shadow


def nearest_s1(ee, geom, date, max_days):
    """Closest Sentinel-1 IW VV+VH acquisition (as a same-day mosaic) within max_days of date."""
    d = ee.Date(date)
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(geom)
           .filterDate(d.advance(-max_days, "day"), d.advance(max_days + 1, "day"))
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
           .map(lambda im: im.set("dt", ee.Number(im.date().difference(d, "day")).abs())))
    return col.sort("dt")


def utm_epsg(lon, lat):
    zone = int(math.floor((lon + 180) / 6) + 1)
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def grid_cells(bounds_lonlat, epsg, size_m):
    """Square cells (x0, y0 = upper-left corner in UTM metres) aligned to multiples of size_m."""
    from pyproj import Transformer
    w, s, e, n = bounds_lonlat
    tr = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    xs, ys = zip(*[tr.transform(x, y) for x, y in ((w, s), (w, n), (e, s), (e, n))])
    x0 = math.floor(min(xs) / size_m) * size_m
    y1 = math.ceil(max(ys) / size_m) * size_m
    cells = []
    for x in range(int(x0), int(max(xs)), size_m):
        for y in range(int(y1), int(min(ys)), -size_m):
            cells.append((x, y))
    return cells


def cell_id(epsg, x, y):
    return f"{epsg.split(':')[1]}_{x}_{y}"


def cell_rect(ee, epsg, x, y, size_m):
    return ee.Geometry.Rectangle([x, y - size_m, x + size_m, y], ee.Projection(epsg), False)


def compute_pixels(ee, image, epsg, x, y, size_px=256, scale=10):
    """Exact size_px x size_px GeoTIFF bytes on the UTM grid with upper-left corner (x, y)."""
    return ee.data.computePixels({
        "expression": image,
        "fileFormat": "GEO_TIFF",
        "grid": {
            "dimensions": {"width": size_px, "height": size_px},
            "affineTransform": {"scaleX": scale, "shearX": 0, "translateX": x,
                                "shearY": 0, "scaleY": -scale, "translateY": y},
            "crsCode": epsg,
        },
    })


def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data if isinstance(data, (bytes, bytearray)) else io.BytesIO(data).getvalue())
    tmp.replace(path)
