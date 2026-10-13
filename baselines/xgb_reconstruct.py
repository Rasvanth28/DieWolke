"""Classical baseline: per-pixel gradient-boosted regression from SAR to the 13 S2 bands.

For each test patch, a regressor is fitted on that patch's clear pixels (features: VV, VH, their
difference, 5x5 local mean and std of each, and pixel coordinates) and predicts the cloudy
pixels; clear pixels are kept from the input. Patches with too few clear pixels fall back to a
pooled model trained on clear pixels from the training split.

This is our approximation of a "classical ML reconstruction" in the spirit of Gupta & Verma
(2025, XGBoost); their exact features and protocol are not public, so describe it as
"XGBoost SAR-regression gap filling", not as their method.

    python baselines/xgb_reconstruct.py --cache /content/cache --split test --mode real \
        --out /content/drive/MyDrive/dieWolke/eval/pred_xgb.npy
    python -m diewolke.evaluate --preds xgb=/content/.../pred_xgb.npy --cache /content/cache ...
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diewolke.data import Cache, CloudPairs, fake_cache  # noqa: E402


def features(s1):
    """(2, H, W) normalised S1 -> (H*W, 10) features."""
    vv, vh = s1[0], s1[1]
    feats = [vv, vh, vv - vh]
    for band in (vv, vh):
        m = uniform_filter(band, 5)
        feats += [m, np.sqrt(np.maximum(uniform_filter(band ** 2, 5) - m ** 2, 0))]
    h, w = vv.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    feats += [yy / h, xx / w]
    return np.stack(feats, -1).reshape(-1, len(feats)).astype(np.float32)


def make_regressor(n_estimators, max_depth, n_jobs):
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1,
                            subsample=0.8, tree_method="hist", n_jobs=n_jobs, multi_strategy="one_output_per_tree")
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor
        return MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=n_estimators, max_depth=max_depth))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="/content/cache")
    ap.add_argument("--split", default="test")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--train_cache", default=None, help="cache holding --train_split (default: --cache)")
    ap.add_argument("--mode", default="real", choices=["real", "synthetic"],
                    help="must match the evaluate.py --mode you will score with")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_clear", type=float, default=0.2, help="clear share needed for a per-patch fit")
    ap.add_argument("--max_fit_pixels", type=int, default=20000)
    ap.add_argument("--n_estimators", type=int, default=150)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--n_jobs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(0)

    test = fake_cache(8, 64, seed=5) if args.fake else Cache.load(args.cache, args.split)
    ds = CloudPairs(test, mode=args.mode, train=False)  # same clouds/masks as evaluate.py
    n = len(ds) if args.limit is None else min(args.limit, len(ds))

    # pooled fallback model on clear target pixels of the training split
    train = fake_cache(16, 64, seed=1) if args.fake else Cache.load(args.train_cache or args.cache, args.train_split)
    xs, ys = [], []
    for i in rng.choice(len(train["clear"]), min(200, len(train["clear"])), replace=False):
        f = features(np.asarray(train["s1"][i], np.float32))
        y = (np.asarray(train["clear"][i], np.float32) / 10000).reshape(13, -1).T
        pick = rng.choice(len(f), min(2000, len(f)), replace=False)
        xs.append(f[pick]), ys.append(y[pick])
    pooled = make_regressor(args.n_estimators, args.max_depth, args.n_jobs).fit(np.concatenate(xs), np.concatenate(ys))

    first = ds[0]
    h, w = first["clear"].shape[-2:]
    out = np.lib.format.open_memmap(args.out, "w+", np.float16, (len(ds), 13, h, w))
    t0, fallback = time.time(), 0
    for k in range(len(ds)):
        b = ds[k]
        cloudy, mask = b["cloudy"].numpy(), b["mask"].numpy()[0] > 0.5
        if k >= n:
            out[k] = cloudy.astype(np.float16)  # beyond --limit: leave the input unchanged
            continue
        f = features(b["s1"].numpy())
        target = cloudy.reshape(13, -1).T
        clear_px = np.flatnonzero(~mask.reshape(-1))
        if mask.any():
            if len(clear_px) >= args.min_clear * mask.size:
                pick = rng.choice(clear_px, min(args.max_fit_pixels, len(clear_px)), replace=False)
                model = make_regressor(args.n_estimators, args.max_depth, args.n_jobs).fit(f[pick], target[pick])
            else:
                model, fallback = pooled, fallback + 1
            cloud_px = np.flatnonzero(mask.reshape(-1))
            pred = target.copy()
            pred[cloud_px] = model.predict(f[cloud_px])
            out[k] = np.clip(pred.T.reshape(13, h, w), 0, 1).astype(np.float16)
        else:
            out[k] = cloudy.astype(np.float16)
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{n} ({time.time() - t0:.0f}s, {fallback} pooled fallbacks)", flush=True)
    out.flush()
    print(f"[+] wrote {args.out} ({fallback} of {n} patches used the pooled model)")


if __name__ == "__main__":
    main()
