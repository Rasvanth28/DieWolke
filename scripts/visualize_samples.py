"""Plot S1 / cloudy S2 / clear S2 / mask for a few patches, from tifs or from a cache.

    python scripts/visualize_samples.py --data_root /content/data/sen12ms_cr --n 4 --out samples.png
    python scripts/visualize_samples.py --cache /content/cache --split test --n 4 --out samples.png
    # add reconstructions saved by evaluate.py --save_predictions:
    python scripts/visualize_samples.py --cache /content/cache --split test --pred EVAL/pred_full.npy --out recon.png
Triplets are matched by (roi, scene, patch) parsed from the file name, which fixes the path bug in
the original datasets/ notebook.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diewolke.bands import RGB  # noqa: E402
from diewolke.data import Cache, CloudPairs, find_triplets  # noqa: E402
from diewolke.data.sen12mscr import read_s1, read_s2  # noqa: E402


def rgb(x, gain=3.3):
    return np.clip(x[RGB].transpose(1, 2, 0) * gain, 0, 1)


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root")
    ap.add_argument("--cache")
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", default="real")
    ap.add_argument("--pred", default=None)
    ap.add_argument("--scale", default=None, help="scale_<run>.npy to show the uncertainty map")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="samples.png")
    args = ap.parse_args(argv)

    rows = []
    if args.data_root:
        for key, p in find_triplets(args.data_root)[args.start:args.start + args.n]:
            rows.append({"title": f"{key[0]} {key[1]} p{key[2]}", "s1": read_s1(p["s1"]),
                         "cloudy": read_s2(p["s2_cloudy"]) / 1e4, "clear": read_s2(p["s2"]) / 1e4})
    else:
        cache = Cache.load(args.cache, args.split)
        ds = CloudPairs(cache, mode=args.mode, train=False)
        pred = np.load(args.pred, mmap_mode="r") if args.pred else None
        scale = np.load(args.scale, mmap_mode="r") if args.scale else None
        for i in range(args.start, min(args.start + args.n, len(ds))):
            b = ds[i]
            r = {"title": cache.ids[i], "s1": b["s1"].numpy(), "cloudy": b["cloudy"].numpy(),
                 "clear": b["clear"].numpy(), "mask": b["mask"].numpy()[0]}
            if pred is not None:
                r["recon"] = np.asarray(pred[i], np.float32)
            if scale is not None:
                r["uncertainty"] = np.asarray(scale[i], np.float32).mean(0)
            rows.append(r)
    if not rows:
        raise SystemExit("nothing to show")
    cols = [c for c in ("s1", "cloudy", "mask", "recon", "uncertainty", "clear") if c in rows[0]]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3 * len(cols), 3 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        for c, name in enumerate(cols):
            ax = axes[r, c]
            v = row[name]
            if name == "s1":
                ax.imshow(v[0], cmap="gray")
            elif name in ("mask",):
                ax.imshow(v, cmap="gray", vmin=0, vmax=1)
            elif name == "uncertainty":
                ax.imshow(v, cmap="magma")
            else:
                ax.imshow(rgb(v))
            ax.set_title((f"{row['title']}\n" if c == 0 else "") + name, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"[+] wrote {args.out}")


if __name__ == "__main__":
    main()
