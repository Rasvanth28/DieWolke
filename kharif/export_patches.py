"""Step 2 of the Kharif benchmark: download 256 x 256 patches (10 m, UTM grid) from Earth Engine.

    python kharif/export_patches.py --config kharif/kharif.yaml [--roles clear donor] [--dry_run]

Reads <raw_dir>/scenes.csv from find_scenes.py. For each scene it scores every grid cell over the
district (cloud, shadow, valid data, cropland share), then downloads
  clear patches: cloud + shadow <= clear_max_cloud and cropland >= cropland_min
  donor patches: cloud in donor_cloud (their real monsoon clouds are transplanted later)
Each patch gives <date>_<cell>_s2.tif (16 x uint16: 13 S2 bands, s2cloudless prob in %, shadow,
WorldCover class) and <date>_<cell>_s1.tif (VV, VH in dB, -9999 = no data), plus a line in
<raw_dir>/manifest.jsonl. Re-running skips patches already on disk.
"""
import argparse
import csv
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gee_common as g  # noqa: E402

_lock = threading.Lock()


def scene_image(ee, scene_id, cfg):
    img = ee.Image(f"COPERNICUS/S2_HARMONIZED/{scene_id}")
    prob = ee.Image(f"COPERNICUS/S2_CLOUD_PROBABILITY/{scene_id}").select("probability").rename("prob")
    img = img.addBands(prob)
    cloud, shadow = g.cloud_and_shadow(ee, img, cfg["cloud_prob_threshold"])
    lc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("lc")
    return img, cloud, shadow, lc


def score_cells(ee, geom, epsg, cells, img, cloud, shadow, lc, size_m):
    feats = [ee.Feature(g.cell_rect(ee, epsg, x, y, size_m), {"cell": g.cell_id(epsg, x, y), "x": x, "y": y})
             for x, y in cells]
    fc = ee.FeatureCollection(feats).filterBounds(geom).filterBounds(img.geometry())
    valid = ee.Image(1).updateMask(img.select("B2").mask()).unmask(0).rename("valid")
    stack = ee.Image.cat([cloud.unmask(0), shadow.unmask(0), lc.eq(40).rename("crop"), valid])
    stats = stack.reduceRegions(fc, ee.Reducer.mean(), scale=20, crs=epsg)
    return [f["properties"] for f in stats.getInfo()["features"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--roles", nargs="+", default=["clear", "donor"])
    ap.add_argument("--aois", nargs="*", default=None, help="restrict to these AOI names")
    ap.add_argument("--dry_run", action="store_true", help="score cells and report counts, no downloads")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = g.load_cfg(args.config)
    ee = g.init(cfg["project"])
    raw = Path(cfg["raw_dir"])
    size_m = cfg["grid_m"]
    rng = random.Random(args.seed)
    scenes = list(csv.DictReader(open(raw / "scenes.csv")))
    manifest = raw / "manifest.jsonl"
    done = set()
    if manifest.exists():
        done = {json.loads(l)["key"] for l in manifest.read_text().splitlines() if l.strip()}

    aois = {a["name"]: a for a in cfg["aois"] if not args.aois or a["name"] in args.aois}
    geoms, grids = {}, {}
    for name, aoi in aois.items():
        geoms[name] = g.aoi_geometry(ee, aoi)
        ring = geoms[name].bounds().coordinates().getInfo()[0]
        lons, lats = [p[0] for p in ring], [p[1] for p in ring]
        w, s, e, n = min(lons), min(lats), max(lons), max(lats)
        lon, lat = (w + e) / 2, (s + n) / 2
        epsg = g.utm_epsg(lon, lat)
        grids[name] = (epsg, g.grid_cells((w, s, e, n), epsg, size_m))
        print(f"[+] {name}: {epsg}, {len(grids[name][1])} grid cells in the bounding box")

    by_aoi_year = {}
    for sc in scenes:
        if sc["aoi"] in aois:
            by_aoi_year.setdefault((sc["aoi"], sc["year"]), []).append(sc)

    jobs = []
    for (aoi, year), group in sorted(by_aoi_year.items()):
        group = sorted(group, key=lambda r: float(r["aoi_cloud"] or 1))[:cfg["max_scenes_per_aoi_year"]]
        epsg, cells = grids[aoi]
        for sc in group:
            c = float(sc["aoi_cloud"] or 1)
            want_clear = "clear" in args.roles and c <= cfg["scene_max_cloud"]
            want_donor = "donor" in args.roles and c >= cfg["donor_cloud"][0]
            if not (want_clear or want_donor):
                continue
            img, cloud, shadow, lc = scene_image(ee, sc["id"], cfg)
            try:
                stats = g.retry(lambda: score_cells(ee, geoms[aoi], epsg, cells, img, cloud, shadow, lc, size_m))
            except Exception as exc:  # noqa: BLE001 - a bad scene should not stop the export
                print(f"  [!] {sc['id']}: {exc}")
                continue
            ok = [s for s in stats if s.get("valid", 0) > 0.999]
            clear = [s for s in ok if s["cloud"] + s["shadow"] <= cfg["clear_max_cloud"]
                     and s["crop"] >= cfg["cropland_min"]] if want_clear else []
            donor = [s for s in ok if cfg["donor_cloud"][0] <= s["cloud"] <= cfg["donor_cloud"][1]] if want_donor else []
            rng.shuffle(clear)
            rng.shuffle(donor)
            clear, donor = clear[:cfg["max_clear_per_scene"]], donor[:cfg["max_donor_per_scene"]]
            print(f"  {aoi} {sc['date']} ({sc['id']}): aoi cloud {c:.2f}, "
                  f"{len(clear)} clear and {len(donor)} donor patches", flush=True)
            if not clear and not donor:
                continue
            s1col = g.nearest_s1(ee, geoms[aoi], sc["date"], cfg["s1_max_days"])
            first = s1col.first()
            s1_info = g.retry(lambda: ee.Dictionary({"id": first.get("system:index"),
                                                     "date": first.date().format("YYYY-MM-dd")}).getInfo()
                              if s1col.size().getInfo() else None)
            if s1_info is None:
                print(f"    [!] no Sentinel-1 within {cfg['s1_max_days']} days; skipped")
                continue
            d = ee.Date(s1_info["date"])
            s1 = s1col.filterDate(d, d.advance(1, "day")).mosaic().select(["VV", "VH"]).unmask(g.NODATA_S1).toFloat()
            s2 = ee.Image.cat([img.select(g.S2_BANDS), img.select("prob"), shadow.unmask(0), lc]).unmask(0).toUint16()
            for role, cells_ in (("clear", clear), ("donor", donor)):
                for s in cells_:
                    key = f"{role}/{aoi}/{sc['date']}_{s['cell']}"
                    if key not in done:
                        jobs.append((key, role, aoi, sc, s1_info, s, epsg, s2, s1))

    print(f"[+] {len(jobs)} patches to download")
    if args.dry_run or not jobs:
        return

    def fetch(job):
        key, role, aoi, sc, s1_info, s, epsg, s2, s1 = job
        base = raw / key
        try:
            g.write_bytes(f"{base}_s2.tif", g.retry(lambda: g.compute_pixels(ee, s2, epsg, s["x"], s["y"])))
            g.write_bytes(f"{base}_s1.tif", g.retry(lambda: g.compute_pixels(ee, s1, epsg, s["x"], s["y"])))
        except Exception as exc:  # noqa: BLE001
            return f"[!] {key}: {exc}"
        row = {"key": key, "role": role, "aoi": aoi, "s2_id": sc["id"], "date": sc["date"],
               "tile": sc["tile"], "s1_id": s1_info["id"], "s1_date": s1_info["date"], "cell": s["cell"],
               "epsg": epsg, "x": s["x"], "y": s["y"], "cloud": s["cloud"], "shadow": s["shadow"],
               "crop": s["crop"], "s2": f"{key}_s2.tif", "s1": f"{key}_s1.tif"}
        with _lock, open(manifest, "a") as f:
            f.write(json.dumps(row) + "\n")
        return None

    with ThreadPoolExecutor(cfg.get("workers", 8)) as pool:
        for i, err in enumerate(pool.map(fetch, jobs), 1):
            if err:
                print("  " + err)
            if i % 50 == 0:
                print(f"  downloaded {i}/{len(jobs)}", flush=True)
    print(f"[+] done; manifest at {manifest}")


if __name__ == "__main__":
    main()
