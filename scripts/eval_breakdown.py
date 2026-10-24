"""Break an evaluation down by cloud cover and by scene, from evaluate.py's per-image CSVs.

    python scripts/eval_breakdown.py EVAL/test_real_baselines --reference full

Prints Markdown tables: mean metrics per cloud-fraction bin and per scene, and paired differences
(run minus reference) with 95% bootstrap CIs inside each cloud bin. The test split has few scenes,
so per-scene tables show how much rankings vary beyond the per-image CIs.
"""
import argparse
import csv
import random
import statistics as st
from pathlib import Path

BINS = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01)]


def load(d):
    runs = {}
    for p in sorted(Path(d).glob("per_image_*.csv")):
        runs[p.stem.removeprefix("per_image_")] = {r["id"]: r for r in csv.DictReader(open(p))}
    return runs


def boot(diffs, n=4000, seed=0):
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return st.mean(diffs), means[int(0.025 * n)], means[int(0.975 * n)]


def table(runs, groups, metric):
    names = list(runs)
    lines = [f"| {metric} | n | " + " | ".join(names) + " |", "|---" * (len(names) + 2) + "|"]
    for label, ids in groups:
        if ids:
            vals = [f"{st.mean(float(runs[r][i][metric]) for i in ids):.4g}" for r in names]
            lines.append(f"| {label} | {len(ids)} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--reference", default="full")
    ap.add_argument("--metrics", nargs="*", default=["psnr", "ndvi_rmse", "sam"])
    ap.add_argument("--split_at", type=float, default=0.9, help="cloud fraction for the paired comparison")
    args = ap.parse_args(argv)
    runs = load(args.eval_dir)
    ids = list(runs["input"])
    cf = {i: float(runs["input"][i]["cloud_frac"]) for i in ids}
    by_bin = [(f"{lo:.1f}–{min(hi, 1):.1f}", [i for i in ids if lo <= cf[i] < hi]) for lo, hi in BINS]
    scenes = {}
    for i in ids:
        scenes.setdefault(i.rsplit("_p", 1)[0], []).append(i)
    by_scene = [(f"{s} (cloud {st.mean(cf[i] for i in v):.2f})", v) for s, v in scenes.items()]

    for m in args.metrics:
        print(f"\n### {m} by cloud fraction\n\n{table(runs, by_bin, m)}")
    print(f"\n### psnr by scene\n\n{table(runs, by_scene, 'psnr')}")
    for name, sel in [(f"cloud >= {args.split_at}", [i for i in ids if cf[i] >= args.split_at]),
                      (f"cloud < {args.split_at}", [i for i in ids if cf[i] < args.split_at])]:
        print(f"\n### {name} (n={len(sel)}): run minus {args.reference}, 95% bootstrap CI\n")
        print("| run | " + " | ".join(args.metrics) + " |\n" + "|---" * (len(args.metrics) + 1) + "|")
        for r in runs:
            if r in ("input", args.reference):
                continue
            cells = []
            for m in args.metrics:
                mean, lo, hi = boot([float(runs[r][i][m]) - float(runs[args.reference][i][m]) for i in sel])
                cells.append(f"{mean:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            print(f"| {r} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
