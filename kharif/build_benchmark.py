"""Step 3 of the Kharif benchmark: turn downloaded patches into evaluation splits.

    python kharif/build_benchmark.py --config kharif/kharif.yaml [--labels crops.tif ...]

Writes, in the same cache format as SEN12MS-CR (so evaluate.py / predict tools work unchanged),
into <benchmark_dir>:
  kharif_synth   clear Kharif cropland patches with real monsoon clouds (and their cast
                 shadows) transplanted from donor patches of another date and place. Exact
                 ground truth and an exact mask (field "mask").
  kharif_real    real cloudy/clear pairs of the same cell from two dates within --pair_days
                 (SEN12MS-CR-style protocol; mask from s2cloudless + shadow, ground truth is
                 another date). Only built when such pairs exist.
Each split also gets per-pixel "labels" (uint8, 0 = unlabelled) and <split>_meta.json with the
tile IDs, dates, cells, donor and classifier group of every patch, which is what gets released.

Labels: by default ESA WorldCover 2021 classes (a land-cover PROXY for the downstream task).
For real crop classification pass --labels with one or more crop-type rasters (GeoTIFF, any CRS,
integer classes, 0 = unknown); they are resampled (nearest) onto each patch.
"""
import argparse
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gee_common as g  # noqa: E402
from diewolke.bands import normalize_s1  # noqa: E402
from diewolke.data.clouds import MASK_THRESHOLD, real_mask  # noqa: E402

WORLDCOVER = {10: "tree", 20: "shrub", 30: "grass", 40: "cropland", 50: "built", 60: "bare",
              70: "snow", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}
WC_TO_ID = {code: i + 1 for i, code in enumerate(WORLDCOVER)}


def read(path):
    import rasterio
    with rasterio.open(path) as src:
        return src.read(), src.transform, src.crs


def labels_from_rasters(rasters, transform, crs, shape):
    import rasterio
    from rasterio.warp import Resampling, reproject
    out = np.zeros(shape, np.uint8)
    for path in rasters:
        with rasterio.open(path) as src:
            tmp = np.zeros(shape, np.int32)
            reproject(rasterio.band(src, 1), tmp, dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.nearest, dst_nodata=0)
        out = np.where(out == 0, np.clip(tmp, 0, 255).astype(np.uint8), out)
    return out


def load_patch(raw, row, label_rasters):
    s2, transform, crs = read(raw / row["s2"])
    s1, _, _ = read(raw / row["s1"])
    if (s1 == g.NODATA_S1).mean() > 0.01 or (s2[:13] == 0).all(0).mean() > 0.001:
        return None
    lc = s2[15]
    if label_rasters:
        labels = labels_from_rasters(label_rasters, transform, crs, lc.shape)
    else:
        labels = np.vectorize(lambda v: WC_TO_ID.get(int(v), 0), otypes=[np.uint8])(lc)
    s1 = np.where(s1 == g.NODATA_S1, np.nan, s1).astype(np.float32)
    return {"s2": s2[:13].astype(np.uint16), "prob": s2[13].astype(np.float32) / 100,
            "shadow": (s2[14] > 0).astype(np.uint8), "s1": normalize_s1(s1), "labels": labels}


def transplant(clear, donor, shadow_strength=0.6):
    """Real monsoon cloud + its projected shadow from a donor patch onto a clear patch (DN)."""
    alpha = np.clip((donor["prob"] - 0.2) / 0.5, 0, 1)
    shadow = donor["shadow"].astype(np.float32) * (1 - alpha)
    c = clear["s2"].astype(np.float32)
    cloudy = (c * (1 - alpha) + donor["s2"].astype(np.float32) * alpha) * (1 - shadow_strength * shadow)
    mask = ((alpha + shadow) > MASK_THRESHOLD).astype(np.uint8)
    return np.clip(cloudy, 0, 10000).astype(np.uint16), mask, shadow


def write_split(out_dir, split, records, meta):
    n = len(records)
    if n == 0:
        print(f"[!] {split}: no patches")
        return
    fields = {"s1": np.float16, "clear": np.uint16, "cloudy": np.uint16, "prob": np.float16,
              "prob_clear": np.float16, "shadow": np.uint8, "labels": np.uint8}
    if "mask" in records[0]:
        fields["mask"] = np.uint8
    for k, dt in fields.items():
        arr = np.lib.format.open_memmap(out_dir / f"{split}_{k}.npy", "w+", dt, (n,) + records[0][k].shape)
        for i, r in enumerate(records):
            arr[i] = r[k]
        arr.flush()
    (out_dir / f"{split}_keys.json").write_text(json.dumps([m["id"] for m in meta]))
    (out_dir / f"{split}_meta.json").write_text(json.dumps(meta, indent=1))
    (out_dir / f"{split}.done").touch()
    print(f"[+] {split}: {n} patches")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--labels", nargs="*", default=None, help="crop-type label rasters")
    ap.add_argument("--pair_days", type=int, default=30, help="max date gap for kharif_real pairs")
    ap.add_argument("--block", type=int, default=4, help="classifier group = block x block grid cells")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = g.load_cfg(args.config)
    raw, out_dir = Path(cfg["raw_dir"]), Path(cfg["benchmark_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = [json.loads(l) for l in (raw / "manifest.jsonl").read_text().splitlines() if l.strip()]
    clear_rows = [r for r in rows if r["role"] == "clear"]
    donor_rows = [r for r in rows if r["role"] == "donor"]
    print(f"[+] manifest: {len(clear_rows)} clear, {len(donor_rows)} donor patches")

    cache = OrderedDict()  # small LRU: patches are ~2.5 MB each

    def get(row):
        if row["key"] in cache:
            cache.move_to_end(row["key"])
        else:
            cache[row["key"]] = load_patch(raw, row, args.labels)
            if len(cache) > 400:
                cache.popitem(last=False)
        return cache[row["key"]]

    def group(row):
        b = cfg["grid_m"] * args.block
        return f"{row['aoi']}_{row['x'] // b}_{row['y'] // b}"

    # ---- kharif_synth: exact ground truth with transplanted real monsoon clouds
    records, meta, skipped = [], [], 0
    donors_by_aoi = defaultdict(list)
    for d in donor_rows:
        donors_by_aoi[d["aoi"]].append(d)
    for r in clear_rows:
        c = get(r)
        if c is None:
            skipped += 1
            continue
        # prefer a donor from another district, never the same date and cell
        pool = [d for a, ds in donors_by_aoi.items() if a != r["aoi"] for d in ds] or \
               [d for d in donor_rows if not (d["date"] == r["date"] and d["cell"] == r["cell"])]
        donor = None
        for j in rng.permutation(len(pool))[:10]:
            donor = get(pool[j])
            if donor is not None:
                dr = pool[j]
                break
        if donor is None:
            skipped += 1
            continue
        cloudy, mask, shadow = transplant(c, donor)
        records.append({"s1": c["s1"].astype(np.float16), "clear": c["s2"], "cloudy": cloudy,
                        "prob": donor["prob"].astype(np.float16),
                        "prob_clear": c["prob"].astype(np.float16), "shadow": (shadow > 0.3).astype(np.uint8),
                        "mask": mask, "labels": c["labels"]})
        meta.append({"id": f"{r['aoi']}_{r['date']}_{r['cell']}", "aoi": r["aoi"], "cell": r["cell"],
                     "epsg": r["epsg"], "x": r["x"], "y": r["y"], "s2_id": r["s2_id"], "date": r["date"],
                     "s1_id": r["s1_id"], "s1_date": r["s1_date"], "donor_key": dr["key"],
                     "donor_s2_id": dr["s2_id"], "donor_date": dr["date"], "cloud_frac": float(mask.mean()),
                     "group": group(r)})
    print(f"  kharif_synth: {len(records)} patches ({skipped} skipped for missing data or donors)")
    write_split(out_dir, "kharif_synth", records, meta)

    # ---- kharif_real: same cell, cloudy date vs nearest clear date
    from datetime import date as _date
    to_day = lambda s: _date.fromisoformat(s).toordinal()  # noqa: E731
    clear_by_cell = defaultdict(list)
    for r in clear_rows:
        clear_by_cell[(r["aoi"], r["cell"])].append(r)
    records, meta = [], []
    for d in donor_rows:
        cands = [r for r in clear_by_cell.get((d["aoi"], d["cell"]), [])
                 if 0 < abs(to_day(r["date"]) - to_day(d["date"])) <= args.pair_days]
        if not cands:
            continue
        r = min(cands, key=lambda r: abs(to_day(r["date"]) - to_day(d["date"])))
        cd, cc = get(d), get(r)
        if cd is None or cc is None:
            continue
        records.append({"s1": cd["s1"].astype(np.float16), "clear": cc["s2"], "cloudy": cd["s2"],
                        "prob": cd["prob"].astype(np.float16), "prob_clear": cc["prob"].astype(np.float16),
                        "shadow": cd["shadow"], "labels": cc["labels"]})
        meta.append({"id": f"{d['aoi']}_{d['date']}_{d['cell']}", "aoi": d["aoi"], "cell": d["cell"],
                     "epsg": d["epsg"], "x": d["x"], "y": d["y"], "s2_id": d["s2_id"], "date": d["date"],
                     "s1_id": d["s1_id"], "s1_date": d["s1_date"], "clear_s2_id": r["s2_id"],
                     "clear_date": r["date"], "group": group(d),
                     "cloud_frac": float(real_mask(cd["prob"], cd["shadow"]).mean())})
    write_split(out_dir, "kharif_real", records, meta)
    names = {i: n for n, i in zip(WORLDCOVER.values(), WC_TO_ID.values())} if not args.labels else None
    (out_dir / "label_names.json").write_text(json.dumps(names or {"source": args.labels}, indent=1))


if __name__ == "__main__":
    main()
