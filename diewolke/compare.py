"""Paired comparison of two evaluated runs from their per-image CSVs (same split and mode).

    python -m diewolke.compare EVAL/per_image_full.csv EVAL/per_image_matched_baseline.csv
Prints mean difference (A - B), 95% bootstrap CI and the share of images where A is better.
"""
import argparse
import csv

import numpy as np

from .bands import S2_BANDS
from .metrics import HEADLINE, paired_difference


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {r["index"]: r for r in rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--per_band", action="store_true", help="also compare per-band MAE")
    args = ap.parse_args(argv)
    a, b = read(args.a), read(args.b)
    common = sorted(set(a) & set(b), key=int)
    if len(common) < len(a) or len(common) < len(b):
        print(f"[!] {len(a)} vs {len(b)} images; comparing the {len(common)} in both")
    metrics = list(HEADLINE) + ([(f"mae_{band}", False) for band in S2_BANDS] if args.per_band else [])
    for k, hib in metrics:
        if k not in a[common[0]]:
            continue
        va = np.array([float(a[i][k]) for i in common])
        vb = np.array([float(b[i][k]) for i in common])
        d = paired_difference(va, vb, hib)
        if d:
            print(f"{k:>16}: {d['mean']:+.5f} [{d['ci95'][0]:+.5f}, {d['ci95'][1]:+.5f}]  "
                  f"A better on {100 * d['frac_images_better']:.0f}% of {d['n']}")


if __name__ == "__main__":
    main()
