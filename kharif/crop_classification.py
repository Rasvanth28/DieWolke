"""Step 4: downstream classification on reconstructed imagery (the C4 crop-classification test).

A per-pixel gradient-boosted classifier is trained on CLEAR ground-truth imagery of the training
groups, then applied to the held-out groups using different inputs:
    gt        the clear image (upper bound)
    cloudy    the cloudy input (lower bound)
    <name>    each reconstruction passed with --preds name=pred.npy (from evaluate.py
              --save_predictions or baselines/xgb_reconstruct.py)
Accuracy is reported on all labelled pixels and on cloud-covered pixels only (the pixels that
were actually reconstructed). For predictions with a scale .npy, it also reports accuracy and
coverage after discarding the most uncertain cloud pixels.

    python kharif/crop_classification.py --cache BENCH --split kharif_synth \
        --preds full=EVAL/pred_full.npy,EVAL/scale_full.npy xgb=EVAL/pred_xgb.npy --out EVAL/crops
Groups (spatial blocks of grid cells, from <split>_meta.json) never mix between train and test.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diewolke.bands import ndmi, ndvi  # noqa: E402
from diewolke.data import Cache, CloudPairs  # noqa: E402


def pixel_features(img):
    """(13, H, W) reflectance -> (H*W, 15): bands + NDVI + NDMI."""
    f = np.concatenate([img, ndvi(img)[None], ndmi(img)[None]], 0)
    return f.reshape(f.shape[0], -1).T.astype(np.float32)


def make_classifier(n_classes, n_jobs):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8,
                             tree_method="hist", n_jobs=n_jobs)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300)


def scores(y, p, n_classes):
    from sklearn.metrics import cohen_kappa_score, f1_score
    if len(y) == 0:
        return {"n": 0}
    return {"n": int(len(y)), "oa": float((y == p).mean()),
            "macro_f1": float(f1_score(y, p, average="macro", labels=list(range(n_classes)), zero_division=0)),
            "kappa": float(cohen_kappa_score(y, p))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="kharif_synth")
    ap.add_argument("--mode", default="real", help="dataset mode used when the predictions were made")
    ap.add_argument("--preds", nargs="*", default=[], metavar="NAME=PRED.npy[,SCALE.npy]")
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--train_pixels", type=int, default=400000)
    ap.add_argument("--min_class_pixels", type=int, default=500)
    ap.add_argument("--n_jobs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cache = Cache.load(args.cache, args.split)
    if "labels" not in cache:
        raise SystemExit("this split has no labels (build it with kharif/build_benchmark.py)")
    meta_path = Path(args.cache) / f"{args.split}_meta.json"
    groups = [m["group"] for m in json.loads(meta_path.read_text())] if meta_path.exists() else \
        [str(i) for i in range(len(cache))]
    uniq = sorted(set(groups))
    test_groups = set(rng.permutation(uniq)[:max(1, int(round(args.test_frac * len(uniq))))])
    test_idx = [i for i, gr in enumerate(groups) if gr in test_groups]
    train_idx = [i for i, gr in enumerate(groups) if gr not in test_groups]
    ds = CloudPairs(cache, mode=args.mode, train=False)  # gives the same cloudy input and mask as evaluation

    # classes: keep those with enough training pixels, remap to 0..K-1 (unlabelled = 0 dropped)
    lab = np.asarray(cache["labels"])
    counts = np.bincount(lab[train_idx].ravel(), minlength=256)
    classes = [c for c in range(1, 256) if counts[c] >= args.min_class_pixels]
    remap = np.full(256, -1, np.int64)
    remap[classes] = np.arange(len(classes))
    k = len(classes)
    print(f"[+] {len(train_idx)} train / {len(test_idx)} test patches, {k} classes {classes}")

    xs, ys = [], []
    per_patch = max(1, args.train_pixels // max(1, len(train_idx)))
    for i in train_idx:
        y = remap[lab[i].ravel()]
        ok = np.flatnonzero(y >= 0)
        if len(ok) == 0:
            continue
        pick = rng.choice(ok, min(per_patch, len(ok)), replace=False)
        xs.append(pixel_features(np.asarray(cache["clear"][i], np.float32) / 10000)[pick])
        ys.append(y[pick])
    y_train = np.concatenate(ys)
    present = np.unique(y_train)  # xgboost needs labels 0..n-1 without gaps
    enc = np.full(k, -1, np.int64)
    enc[present] = np.arange(len(present))
    clf = make_classifier(len(present), args.n_jobs).fit(np.concatenate(xs), enc[y_train])

    def classify(feats):
        return present[np.asarray(clf.predict(feats)).astype(np.int64).ravel()]

    sources = {"gt": None, "cloudy": None}
    for item in args.preds:
        name, _, paths = item.partition("=")
        pred, _, scale = paths.partition(",")
        sources[name] = (np.load(pred, mmap_mode="r"), np.load(scale, mmap_mode="r") if scale else None)

    results = {}
    for name, src in sources.items():
        y_all, p_all, y_cl, p_cl, u_cl = [], [], [], [], []
        for i in test_idx:
            b = ds[i]
            if name == "gt":
                img = b["clear"].numpy()
            elif name == "cloudy":
                img = b["cloudy"].numpy()
            else:
                img = np.asarray(src[0][i], np.float32)
                img = img / 10000 if src[0].dtype == np.uint16 else img
            y = remap[lab[i].ravel()]
            ok = y >= 0
            p = classify(pixel_features(img))
            m = b["mask"].numpy()[0].ravel() > 0.5
            y_all.append(y[ok]), p_all.append(p[ok])
            y_cl.append(y[ok & m]), p_cl.append(p[ok & m])
            if name not in ("gt", "cloudy") and src[1] is not None:
                u_cl.append(np.asarray(src[1][i], np.float32).mean(0).ravel()[ok & m])
        r = {"all": scores(np.concatenate(y_all), np.concatenate(p_all), k),
             "cloud": scores(np.concatenate(y_cl), np.concatenate(p_cl), k)}
        if u_cl:
            u, yc, pc = np.concatenate(u_cl), np.concatenate(y_cl), np.concatenate(p_cl)
            r["cloud_by_uncertainty"] = {}
            for q in (0.9, 0.75, 0.5):
                keep = u <= np.quantile(u, q)
                r["cloud_by_uncertainty"][f"keep_{int(q * 100)}pct"] = scores(yc[keep], pc[keep], k)
        results[name] = r
        print(f"  {name:>10}: OA all {r['all'].get('oa', float('nan')):.4f}, "
              f"OA cloud {r['cloud'].get('oa', float('nan')):.4f}, macro-F1 cloud {r['cloud'].get('macro_f1', float('nan')):.4f}")
    (out / "crop_classification.json").write_text(json.dumps(
        {"classes": classes, "train_patches": len(train_idx), "test_patches": len(test_idx), "results": results}, indent=1))
    lines = ["| input | OA (all) | macro-F1 (all) | OA (cloud px) | macro-F1 (cloud px) | kappa (cloud px) |",
             "|---|---|---|---|---|---|"]
    for name, r in results.items():
        a, c = r["all"], r["cloud"]
        lines.append(f"| {name} | {a.get('oa', 0):.4f} | {a.get('macro_f1', 0):.4f} | {c.get('oa', 0):.4f} | "
                     f"{c.get('macro_f1', 0):.4f} | {c.get('kappa', 0):.4f} |")
    (out / "crop_classification.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
