"""Stage 4 validation harness: evaluate one or more trained runs on a cached split.

    python -m diewolke.evaluate --runs RUNS/full RUNS/matched_baseline \
        --cache /content/cache --split test --mode real --out RUNS/eval_test_real

Writes to --out:
  per_image_<run>.csv   every metric for every image (input to compare.py)
  summary.json          means with 95% bootstrap CIs, per-band errors, calibration statistics
  report.md             headline table, per-band table, paired differences vs the first run,
                        calibration and uncertainty-filtering results
  pred_<run>.npy        (with --save_predictions) float16 reconstructions in cache order,
                        plus scale_<run>.npy for models with an uncertainty head
The cloudy input itself is always evaluated as the "input" row (the floor).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import calibration as cal
from .bands import N_S2, S2_BANDS, ndvi
from .data import Cache, CloudPairs, fake_cache
from .engine import Pipeline, load_run
from .metrics import HEADLINE, PixelSampler, bootstrap_ci, image_metrics, paired_difference
from .utils import get_device


class ExternalPredictions:
    """Scores predictions computed elsewhere (XGBoost baseline, GLF-CR, UnCRtainTS, ...).

    path: .npy of shape (N, 13, H, W) in cache order, reflectance in [0, 1] (float) or DN (uint16).
    An optional scale .npy (same shape) is treated as a Laplace scale for calibration statistics.
    """

    kind = "external"

    def __init__(self, path, scale_path=None):
        self.pred = np.load(path, mmap_mode="r")
        self.scale = np.load(scale_path, mmap_mode="r") if scale_path else None
        self.cfg = {"name": Path(path).stem.removeprefix("pred_")}

    def predict(self, batch, **_):
        idx = batch["index"].cpu().numpy()
        p = np.asarray(self.pred[idx], dtype=np.float32)
        if self.pred.dtype == np.uint16:
            p = p / 10000.0
        dev = batch["cloudy"].device
        out = {"mean": torch.from_numpy(p).to(dev), "scale": None, "sample_std": None}
        if self.scale is not None:
            out["scale"] = torch.from_numpy(np.asarray(self.scale[idx], dtype=np.float32)).to(dev)
        return out


def make_eval_loader(cache, mode, batch, workers, seed=0, crop=None):
    ds = CloudPairs(cache, mode=mode, crop=crop, train=False, seed=seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)


@torch.no_grad()
def run_one(pipe, loader, device, steps, eta, n_samples, save_to=None, n_total=None, sampler_seed=0):
    """Evaluate one pipeline (or the cloudy input when pipe is None)."""
    per_image = defaultdict(list)
    ids = []
    sampler = PixelSampler(seed=sampler_seed)
    preds = scales = None
    offset = 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        if pipe is None:
            out = {"mean": b["cloudy"], "scale": None, "sample_std": None}
        else:
            out = pipe.predict(b, steps=steps, eta=eta, n_samples=n_samples)
        pred = out["mean"].float().clamp(0, 1)
        for k, v in image_metrics(pred, b["clear"], b["mask"]).items():
            per_image[k].append(v.cpu().numpy())
        ids += b["index"].cpu().tolist()
        err = (pred - b["clear"]).abs()
        nd_err = (ndvi(pred) - ndvi(b["clear"])).abs()[:, None]
        arrays = {"err": err, "mask": b["mask"], "ndvi_err": nd_err}
        if out.get("scale") is not None:
            arrays["scale"] = out["scale"].float()
        if out.get("sample_std") is not None:
            arrays["sample_std"] = out["sample_std"].float()
        sampler.add(**arrays)
        if save_to is not None:
            if preds is None:
                shape = (n_total, N_S2) + tuple(pred.shape[-2:])
                preds = np.lib.format.open_memmap(save_to[0], "w+", np.float16, shape)
                if out.get("scale") is not None:
                    scales = np.lib.format.open_memmap(save_to[1], "w+", np.float16, shape)
            n = pred.shape[0]
            preds[offset:offset + n] = pred.cpu().numpy().astype(np.float16)
            if scales is not None:
                scales[offset:offset + n] = out["scale"].float().cpu().numpy().astype(np.float16)
            offset += n
    for a in (preds, scales):
        if a is not None:
            a.flush()
    return {k: np.concatenate(v) for k, v in per_image.items()}, ids, sampler


def summarize_metrics(per_image):
    out = {}
    for k, v in per_image.items():
        mean, ci = bootstrap_ci(v)
        out[k] = {"mean": mean, "ci95": ci}
    return out


def uncertainty_report(sampler, temperature=None):
    err, scale = sampler.get("err"), sampler.get("scale")
    if scale is None:
        return None
    rep = {"raw": cal.summarize(err, scale)}
    if temperature is not None:
        rep["calibrated"] = cal.summarize(err, scale, temperature)
        rep["temperature"] = temperature.tolist()
    # does dropping uncertain pixels improve NDVI? (pooled pixels, uncertainty averaged over bands)
    u = (scale if temperature is None else scale * temperature[None]).mean(1)
    nd = sampler.get("ndvi_err")[:, 0]
    rep["ndvi_rmse_kept"] = {}
    for q in (1.0, 0.9, 0.75, 0.5):
        keep = u <= np.quantile(u, q)
        rep["ndvi_rmse_kept"][f"keep_{int(q * 100)}pct"] = float(np.sqrt((nd[keep] ** 2).mean()))
    ss = sampler.get("sample_std")
    if ss is not None:
        rep["sample_spread_spearman"] = float(cal.spearmanr(ss.mean(1), err.mean(1)).statistic)
    return rep


def write_csv(path, ids, per_image, names=None):
    keys = list(per_image)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "id"] + keys)
        for i, idx in enumerate(ids):
            w.writerow([idx, names[idx] if names else idx] + [f"{per_image[k][i]:.6g}" for k in keys])


def fmt(v, k):
    return f"{v:.3f}" if k in ("psnr", "psnr_cloud", "sam") else f"{v:.4f}"


def report_md(results, reference, args):
    lines = [f"# Evaluation: split `{args.split}`, mode `{args.mode}`, {args.steps} bridge step(s)", ""]
    names = list(results)
    n_img = len(next(iter(results.values()))["per_image"]["psnr"])
    lines += [f"{n_img} images. Mean over images; 95% bootstrap CI in the JSON.", "",
              "| run | " + " | ".join(k for k, _ in HEADLINE) + " |",
              "|---" * (len(HEADLINE) + 1) + "|"]
    for n in names:
        s = results[n]["summary"]
        lines.append(f"| {n} | " + " | ".join(fmt(s[k]["mean"], k) if k in s else "-" for k, _ in HEADLINE) + " |")
    lines += ["", "## Per-band MAE (reflectance)", "",
              "| run | " + " | ".join(S2_BANDS) + " |", "|---" * (len(S2_BANDS) + 1) + "|"]
    for n in names:
        s = results[n]["summary"]
        lines.append(f"| {n} | " + " | ".join(f"{s[f'mae_{b}']['mean']:.4f}" for b in S2_BANDS) + " |")
    if len(names) > 1:
        lines += ["", f"## Paired differences vs `{reference}` (run minus reference)", ""]
        for n in names:
            if n == reference or (n == "input" and reference != "input"):
                continue
            lines.append(f"**{n}**")
            for k, hib in HEADLINE:
                if k not in results[n]["per_image"]:
                    continue
                d = paired_difference(results[n]["per_image"][k], results[reference]["per_image"][k], hib)
                if d:
                    lines.append(f"- {k}: {d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}], "
                                 f"better on {100 * d['frac_images_better']:.0f}% of {d['n']} images")
            lines.append("")
    for n in names:
        u = results[n].get("uncertainty")
        if not u:
            continue
        lines += [f"## Uncertainty: {n}", ""]
        for kind in ("raw", "calibrated"):
            if kind in u:
                r = u[kind]
                lines.append(f"- {kind}: NLL {r['nll']:.4f}, calibration error {r['calibration_error']:.4f}, "
                             f"Spearman(uncertainty, error) {r['spearman_uncertainty_error']:.3f}, "
                             f"AUSE {r['sparsification']['ause']:.5f}")
        lines.append("- NDVI RMSE on pixels kept after dropping the most uncertain: "
                     + ", ".join(f"{k} {v:.4f}" for k, v in u["ndvi_rmse_kept"].items()))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], help="run directories or .pt checkpoints")
    ap.add_argument("--preds", nargs="*", default=[], metavar="NAME=PRED.npy[,SCALE.npy]",
                    help="external predictions in cache order (see ExternalPredictions)")
    ap.add_argument("--cache", default="/content/cache")
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", default="real", choices=["real", "synthetic"])
    ap.add_argument("--steps", type=int, default=None, help="bridge steps (default: run config)")
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--masker", default=None, help="masker run, for runs with mask_source=masker")
    ap.add_argument("--calibrate_on", default="val", help="split used to fit the uncertainty temperature ('' to skip)")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N images")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save_predictions", action="store_true")
    ap.add_argument("--which", default="best", choices=["best", "last"])
    ap.add_argument("--fake", action="store_true", help="random data (pipeline check)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    device = get_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = fake_cache(8, 64, seed=5) if args.fake else Cache.load(args.cache, args.split)
    if args.limit:
        cache = Cache({k: v[:args.limit] for k, v in cache.items()})
        cache.ids = list(range(args.limit))
    names = getattr(cache, "ids", None)
    loader = make_eval_loader(cache, args.mode, args.batch, args.workers)
    masker = load_run(args.masker, device)[1] if args.masker else None

    results = {}
    per_image, ids, _ = run_one(None, loader, device, None, None, 1)
    results["input"] = {"per_image": per_image, "summary": summarize_metrics(per_image)}
    write_csv(out / "per_image_input.csv", ids, per_image, names)
    run_names = []
    for path in args.runs:
        pipe = Pipeline.from_run(path, device, masker=masker, which=args.which)
        name = pipe.cfg.get("name", Path(path).stem)
        run_names.append(name)
        steps = args.steps if args.steps is not None else pipe.cfg.get("eval", {}).get("steps", 1)
        print(f"[+] evaluating {name} ({pipe.kind}, {steps} step(s)) on {len(loader.dataset)} images")
        temperature = None
        if args.calibrate_on and pipe.kind == "bridge" and pipe.cfg["model"].get("uncertainty") and not args.fake:
            val = Cache.load(args.cache, args.calibrate_on)
            _, _, vs = run_one(pipe, make_eval_loader(val, args.mode, args.batch, args.workers), device,
                               steps, args.eta, args.n_samples)
            temperature = cal.fit_temperature(vs.get("err"), vs.get("scale"))
        save = (out / f"pred_{name}.npy", out / f"scale_{name}.npy") if args.save_predictions else None
        per_image, ids, sampler = run_one(pipe, loader, device, steps, args.eta, args.n_samples, save,
                                          len(loader.dataset))
        results[name] = {"per_image": per_image, "summary": summarize_metrics(per_image),
                         "uncertainty": uncertainty_report(sampler, temperature), "steps": steps}
        write_csv(out / f"per_image_{name}.csv", ids, per_image, names)
    for item in args.preds:
        name, _, paths = item.partition("=")
        pred_path, _, scale_path = paths.partition(",")
        pipe = ExternalPredictions(pred_path, scale_path or None)
        print(f"[+] evaluating external predictions {name} from {pred_path}")
        per_image, ids, sampler = run_one(pipe, loader, device, None, None, 1)
        results[name] = {"per_image": per_image, "summary": summarize_metrics(per_image),
                         "uncertainty": uncertainty_report(sampler)}
        write_csv(out / f"per_image_{name}.csv", ids, per_image, names)
        run_names.append(name)
    if not run_names:
        raise SystemExit("nothing to evaluate: give --runs and/or --preds")
    reference = run_names[0] if len(run_names) > 1 else "input"
    (out / "summary.json").write_text(json.dumps(
        {n: {k: v for k, v in r.items() if k != "per_image"} for n, r in results.items()}, indent=1))
    args.steps = args.steps if args.steps is not None else "config"
    text = report_md(results, reference, args)
    (out / "report.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
