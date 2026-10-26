"""Step 1 of the Kharif benchmark: list Sentinel-2 scenes per district and season with their
cloud cover over the district, and check how many could hold clear patches.

    python kharif/find_scenes.py --config kharif/kharif.yaml
Writes <raw_dir>/scenes.csv and prints, per district and year, how many scenes are mostly clear,
partly clear and cloudy. Run this first: the benchmark is only viable if enough July-October
scenes have clear cropland patches (the "rare clear Kharif scenes" risk in NOVELTY.md).
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gee_common as g  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = g.load_cfg(args.config)
    ee = g.init(cfg["project"])
    out = Path(args.out or Path(cfg["raw_dir"]) / "scenes.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    thr = cfg["cloud_prob_threshold"]
    rows = []
    for aoi in cfg["aois"]:
        geom = g.aoi_geometry(ee, aoi)
        for year in cfg["years"]:
            start, end = f"{year}-{cfg['season']['start']}", f"{year}-{cfg['season']['end']}"
            col = g.s2_with_clouds(ee, geom, start, end)

            def stats(img):
                frac = img.select("prob").gt(thr).reduceRegion(ee.Reducer.mean(), geom, 100, maxPixels=1e9, bestEffort=True)
                valid = ee.Image(1).updateMask(img.select("B2").mask()).unmask(0).rename("B2")
                cover = valid.reduceRegion(ee.Reducer.mean(), geom, 100, maxPixels=1e9, bestEffort=True)
                return ee.Feature(None, {
                    "id": img.get("system:index"), "date": img.date().format("YYYY-MM-dd"),
                    "tile": img.get("MGRS_TILE"), "aoi_cloud": frac.get("prob"),
                    "aoi_coverage": cover.get("B2"), "granule_cloud": img.get("CLOUDY_PIXEL_PERCENTAGE")})

            feats = g.retry(lambda: ee.FeatureCollection(col.map(stats)).getInfo())["features"]
            n = {"clear": 0, "partly": 0, "cloudy": 0}
            for f in feats:
                p = f["properties"]
                c = p.get("aoi_cloud")
                if c is None:
                    continue
                kind = "clear" if c < 0.05 else "partly" if c < cfg["scene_max_cloud"] else "cloudy"
                n[kind] += 1
                rows.append({"aoi": aoi["name"], "year": year, **p, "kind": kind})
            print(f"{aoi['name']:>12} {year}: {len(feats)} scenes, {n['clear']} clear, "
                  f"{n['partly']} partly clear, {n['cloudy']} cloudy", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, ["aoi", "year", "id", "date", "tile", "aoi_cloud", "aoi_coverage", "granule_cloud", "kind"])
        w.writeheader()
        w.writerows(rows)
    print(f"[+] wrote {len(rows)} scenes to {out}")


if __name__ == "__main__":
    main()
