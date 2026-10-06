"""Masking ablation: does a separate Stage 1 cloud mask help Stage 2 reconstruction?

Three variants, same reconstructor size:
  single  one U-Net: [S1, cloudy S2] -> clear S2
  two     masker U-Net predicts the mask from cloudy S2; reconstructor gets
          [S1, cloudy S2 with masked pixels zeroed, mask] -> clear S2
  oracle  the same reconstructor fed the true mask (upper bound for "two")
"input" (the cloudy image returned unchanged) is reported as a floor.

Training inputs are made by transplanting real clouds (s2cloudless probability
maps of real cloudy patches) onto clear patches, so the true mask is known.
Evaluation runs on held-out scenes, twice:
  synthetic  transplanted clouds, exact ground truth
  real       SEN12MS-CR cloudy/clear pairs; the clear target is a different
             date and the mask is s2cloudless (no shadows), so treat as indicative

Colab (GPU runtime, dataset extracted to /content/data/sen12ms_cr):
    !pip install -q s2cloudless
    !python mask_ablation.py --data_root /content/data/sen12ms_cr \
        --out /content/drive/MyDrive/dieWolke/mask_ablation
Pipeline check on random data (no dataset or GPU needed):
    python mask_ablation.py --smoke --out /tmp/mask_ablation_smoke
As notebook cells: paste this file into one cell, then in the next cell run
    main(["--smoke", "--out", "/tmp/mask_ablation_smoke"])
"""
import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

S1_FLOOR = np.array([-25.0, -32.5], dtype=np.float32).reshape(2, 1, 1)  # VV, VH clip in dB
RED, NIR = 3, 7  # B4, B8 in the 13-band S2 order
RGB = [3, 2, 1]  # B4, B3, B2
CLOUD_PROB = 0.4  # s2cloudless default threshold
MASK_THRESHOLD = 0.3  # cloud alpha + shadow above this counts as masked
# e.g. ROIs1158_spring_s2_cloudy_102_p30.tif -> (ROIs1158_spring, s2_cloudy, 102, 30)
NAME_RE = re.compile(r"^(ROIs\d+_\w+?)_(s1|s2_cloudy|s2)_(\d+)_p(\d+)\.tiff?$", re.IGNORECASE)
VARIANTS = ["input", "single", "two", "oracle"]
METRICS = [("psnr", "PSNR ↑"), ("psnr_cloud", "PSNR cloud ↑"), ("ssim", "SSIM ↑"),
           ("sam_deg", "SAM° ↓"), ("ndvi_rmse", "NDVI RMSE ↓"), ("ndvi_rmse_cloud", "NDVI RMSE cloud ↓")]


# ----------------------------------------------------------------------------- data

def find_triplets(root):
    """Match S1 / cloudy S2 / clear S2 patches by (roi, scene, patch) parsed from file names."""
    found = defaultdict(dict)
    unmatched = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() not in (".tif", ".tiff"):
            continue
        m = NAME_RE.match(p.name)
        if m:
            roi, modality, scene, patch = m.groups()
            found[(roi, int(scene), int(patch))][modality.lower()] = str(p)
        else:
            unmatched.append(p)
    triplets = [(k, found[k]) for k in sorted(found) if len(found[k]) == 3]
    if not triplets:
        counts = defaultdict(int)
        for paths in found.values():
            for modality in paths:
                counts[modality] += 1
        print(f"[!] no complete triplets under {root}")
        print(f"    tif files matching the SEN12MS-CR name pattern, per modality: {dict(counts)}")
        print(f"    tif files not matching the pattern: {len(unmatched)}; examples: {[str(p) for p in unmatched[:5]]}")
        print(f"    top-level entries: {[p.name for p in sorted(Path(root).glob('*'))[:10]]}")
    return triplets


def split_by_scene(triplets, n_train, n_test, seed):
    """Whole scenes go to one split so test patches never overlap training areas."""
    scenes = sorted({k[:2] for k, _ in triplets})
    rng = random.Random(seed)
    rng.shuffle(scenes)
    test_scenes = set(scenes[: max(1, int(0.15 * len(scenes)))])
    splits = {"train": [], "test": []}
    for k, paths in triplets:
        splits["test" if k[:2] in test_scenes else "train"].append(paths)
    for name, limit in (("train", n_train), ("test", n_test)):
        rng.shuffle(splits[name])
        splits[name] = splits[name][:limit]
    return splits


def read_s1(path):
    import rasterio
    with rasterio.open(path) as src:
        x = src.read().astype(np.float32)
    x = np.where(np.isnan(x), S1_FLOOR, x)
    return (np.clip(x, S1_FLOOR, 0) - S1_FLOOR) / -S1_FLOOR


def read_s2(path):
    import rasterio
    with rasterio.open(path) as src:
        return np.clip(src.read(), 0, 10000).astype(np.uint16)


_detector = None


def cloud_probability(s2):
    """(n, 13, H, W) uint16 -> (n, H, W) cloud probability in [0, 1]."""
    global _detector
    refl = s2.astype(np.float32) / 10000
    try:
        from s2cloudless import S2PixelCloudDetector
    except ImportError:
        if _detector is None:
            print("[!] s2cloudless not installed; falling back to a brightness threshold")
            _detector = "brightness"
        return np.clip((refl[:, [1, 2, 3]].mean(1) - 0.15) / 0.2, 0, 1)
    if _detector is None:
        _detector = S2PixelCloudDetector(threshold=CLOUD_PROB, average_over=4, dilation_size=2, all_bands=True)
    return _detector.get_cloud_probability_maps(refl.transpose(0, 2, 3, 1)).astype(np.float32)


def build_cache(items, cache_dir, split, chunk=64):
    """Read tifs once into .npy memmaps; reruns reuse them."""
    names = {"s1": np.float16, "clear": np.uint16, "cloudy": np.uint16, "prob": np.float16}
    files = {k: cache_dir / f"{split}_{k}.npy" for k in names}
    done = cache_dir / f"{split}.done"
    if not done.exists():
        n = len(items)
        size = read_s2(items[0]["s2"]).shape[-1]
        arrays = {k: np.lib.format.open_memmap(files[k], "w+", dt, (n, 2, size, size) if k == "s1"
                                               else (n, size, size) if k == "prob" else (n, 13, size, size))
                  for k, dt in names.items()}
        t0 = time.time()
        for start in range(0, n, chunk):
            batch = items[start:start + chunk]
            arrays["s1"][start:start + len(batch)] = np.stack([read_s1(p["s1"]) for p in batch])
            arrays["clear"][start:start + len(batch)] = np.stack([read_s2(p["s2"]) for p in batch])
            cloudy = np.stack([read_s2(p["s2_cloudy"]) for p in batch])
            arrays["cloudy"][start:start + len(batch)] = cloudy
            arrays["prob"][start:start + len(batch)] = cloud_probability(cloudy)
            print(f"  cached {split} {start + len(batch)}/{n} ({time.time() - t0:.0f}s)")
        for a in arrays.values():
            a.flush()
        done.touch()
    return {k: np.load(f, mmap_mode="r") for k, f in files.items()}


def fake_cache(n, size, seed):
    g = torch.Generator().manual_seed(seed)

    def smooth(ch):
        return F.interpolate(torch.rand(n, ch, size // 16, size // 16, generator=g), size=size,
                             mode="bilinear", align_corners=False).numpy()
    clear = smooth(13) * 3000 + 500
    prob = smooth(1)[:, 0]
    cloudy = clear * (1 - prob[:, None]) + 8000 * prob[:, None]
    return {"s1": smooth(2).astype(np.float16), "clear": clear.astype(np.uint16),
            "cloudy": cloudy.astype(np.uint16), "prob": prob.astype(np.float16)}


def synthesize(clear, donor_cloudy, donor_prob, rng):
    """Blend a real cloud (with soft edges and a cast shadow) onto a clear patch."""
    alpha = np.clip((donor_prob - 0.2) / 0.5, 0, 1)
    dy, dx = rng.integers(-20, 21, size=2)
    shadow = np.roll(alpha, (dy, dx), axis=(0, 1)) * (1 - alpha)
    cloudy = (clear * (1 - alpha) + donor_cloudy * alpha) * (1 - 0.6 * shadow)
    mask = ((alpha + shadow) > MASK_THRESHOLD).astype(np.float32)
    return cloudy.astype(np.float32), mask


class SyntheticClouds(torch.utils.data.Dataset):
    def __init__(self, cache, crop=None, train=True, seed=0):
        self.c, self.crop, self.train, self.seed = cache, crop, train, seed
        frac = np.array([(cache["prob"][i] > CLOUD_PROB).mean() for i in range(len(cache["prob"]))])
        donors = np.flatnonzero((frac > 0.1) & (frac < 0.8))
        self.donors = donors if len(donors) else np.arange(len(frac))

    def __len__(self):
        return len(self.c["clear"])

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.train else self.seed + i)
        j = rng.choice(self.donors)
        size = self.c["clear"].shape[-1]
        s = self.crop or size
        y0, x0, y1, x1 = rng.integers(0, size - s + 1, size=4)
        s1 = self.c["s1"][i, :, y0:y0 + s, x0:x0 + s].astype(np.float32)
        clear = self.c["clear"][i, :, y0:y0 + s, x0:x0 + s].astype(np.float32) / 10000
        donor = self.c["cloudy"][j, :, y1:y1 + s, x1:x1 + s].astype(np.float32) / 10000
        prob = self.c["prob"][j, y1:y1 + s, x1:x1 + s].astype(np.float32)
        cloudy, mask = synthesize(clear, donor, prob, rng)
        arrays = {"s1": s1, "cloudy": cloudy, "clear": clear, "mask": mask[None]}
        if self.train:
            k, flip = rng.integers(4), rng.integers(2)
            for name, a in arrays.items():
                a = np.rot90(a, k, axes=(-2, -1))
                arrays[name] = np.flip(a, -1) if flip else a
        return {k: torch.from_numpy(np.ascontiguousarray(a)) for k, a in arrays.items()}


class RealClouds(torch.utils.data.Dataset):
    def __init__(self, cache):
        self.c = cache

    def __len__(self):
        return len(self.c["clear"])

    def __getitem__(self, i):
        return {"s1": torch.from_numpy(self.c["s1"][i].astype(np.float32)),
                "cloudy": torch.from_numpy(self.c["cloudy"][i].astype(np.float32) / 10000),
                "clear": torch.from_numpy(self.c["clear"][i].astype(np.float32) / 10000),
                "mask": torch.from_numpy((self.c["prob"][i] > CLOUD_PROB).astype(np.float32)[None])}


# ----------------------------------------------------------------------------- model

def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, cin, cout, base=32, depth=4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth)]
        self.downs, self.ups, self.decs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        prev = cin
        for c in chs:
            self.downs.append(conv_block(prev, c))
            prev = c
        self.mid = conv_block(prev, prev * 2)
        prev *= 2
        for c in reversed(chs):
            self.ups.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
            self.decs.append(conv_block(2 * c, c))
            prev = c
        self.head = nn.Conv2d(prev, cout, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            x = dec(torch.cat([up(x), skip], 1))
        return self.head(x)


def single_inputs(b):
    return torch.cat([b["s1"], b["cloudy"]], 1)


def recon_inputs(b, mask):
    return torch.cat([b["s1"], b["cloudy"] * (1 - mask), mask], 1)


LOSSES = {
    "single": lambda m, b: F.l1_loss(m(single_inputs(b)), b["clear"]),
    "recon": lambda m, b: F.l1_loss(m(recon_inputs(b, b["mask"])), b["clear"]),
    "masker": lambda m, b: F.binary_cross_entropy_with_logits(m(b["cloudy"]), b["mask"]),
}


def train(model, name, loader, steps, lr, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    model.train()
    step, t0, running = 0, time.time(), 0.0
    while step < steps:
        for b in loader:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = LOSSES[name](model, b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            running += loss.item()
            if step % 200 == 0 or step == steps:
                print(f"  {name} step {step}/{steps} loss {running / (step % 200 or 200):.4f} ({time.time() - t0:.0f}s)")
                running = 0.0
            if step >= steps:
                break


# ----------------------------------------------------------------------------- evaluation

def ssim(x, y):
    """Mean Gaussian-window SSIM over channels, per image. Inputs (B, C, H, W) in [0, 1]."""
    c = x.shape[1]
    g = torch.exp(-(torch.arange(11, device=x.device, dtype=x.dtype) - 5) ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).repeat(c, 1, 1, 1)

    def f(t):
        return F.conv2d(t, w, groups=c)
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return s.mean((1, 2, 3))


def ndvi(x):
    return (x[:, NIR] - x[:, RED]) / (x[:, NIR] + x[:, RED] + 1e-6)


def image_metrics(pred, gt, mask):
    """Per-image metrics; *_cloud are NaN for images with no masked pixels."""
    pred = pred.clamp(0, 1)
    m = mask[:, 0]
    n_cloud = m.sum((1, 2))
    nan = torch.full_like(n_cloud, float("nan"))
    err = ((pred - gt) ** 2).mean(1)
    cos = (pred * gt).sum(1) / (pred.norm(dim=1) * gt.norm(dim=1) + 1e-8)
    dn = (ndvi(pred) - ndvi(gt)) ** 2
    mse_cloud = torch.where(n_cloud > 0, (err * m).sum((1, 2)) / n_cloud.clamp(min=1), nan)
    ndvi_cloud = torch.where(n_cloud > 0, (dn * m).sum((1, 2)) / n_cloud.clamp(min=1), nan)
    return {"psnr": -10 * torch.log10(err.mean((1, 2)) + 1e-10),
            "psnr_cloud": -10 * torch.log10(mse_cloud + 1e-10),
            "ssim": ssim(pred, gt),
            "sam_deg": torch.rad2deg(torch.acos(cos.clamp(-1, 1))).mean((1, 2)),
            "ndvi_rmse": dn.mean((1, 2)).sqrt(),
            "ndvi_rmse_cloud": ndvi_cloud.sqrt()}


def paired_difference(a, b, higher_is_better, seed=0, n_boot=2000):
    """Mean of (a - b) per image with a 95% bootstrap CI, and how often a is better."""
    d = a - b
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return None
    boots = np.random.default_rng(seed).choice(d, (n_boot, len(d))).mean(1)
    better = d > 0 if higher_is_better else d < 0
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "frac_images_better": float(better.mean()), "n": int(len(d))}


@torch.no_grad()
def evaluate(single, masker, recon, loader, device, sample_path):
    for m in (single, masker, recon):
        m.eval()
    per_image = defaultdict(lambda: defaultdict(list))
    inter = union = 0.0
    for bi, b in enumerate(loader):
        b = {k: v.to(device) for k, v in b.items()}
        pred_mask = (torch.sigmoid(masker(b["cloudy"])) > 0.5).float()
        preds = {"input": b["cloudy"], "single": single(single_inputs(b)),
                 "two": recon(recon_inputs(b, pred_mask)), "oracle": recon(recon_inputs(b, b["mask"]))}
        for name, p in preds.items():
            for k, v in image_metrics(p, b["clear"], b["mask"]).items():
                per_image[name][k].append(v.cpu().numpy())
        inter += (pred_mask * b["mask"]).sum().item()
        union += ((pred_mask + b["mask"]) > 0).sum().item()
        if bi == 0:
            save_samples(b, preds, pred_mask, sample_path)
    per_image = {v: {k: np.concatenate(x) for k, x in d.items()} for v, d in per_image.items()}
    summary = {"variants": {v: {k: float(np.nanmean(x)) for k, x in d.items()} for v, d in per_image.items()},
               "masker_iou": inter / max(union, 1), "n_images": len(per_image["single"]["psnr"]),
               "two_minus_single": {}}
    for k, label in METRICS:
        summary["two_minus_single"][k] = paired_difference(
            per_image["two"][k], per_image["single"][k], higher_is_better="↑" in label)
    return summary


def save_samples(b, preds, pred_mask, path, n=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rgb(t):
        return (t[RGB].permute(1, 2, 0).float().cpu().numpy() / 0.3).clip(0, 1)
    cols = [("cloudy input", b["cloudy"]), ("single", preds["single"]), ("two-stage", preds["two"]),
            ("oracle mask", preds["oracle"]), ("clear target", b["clear"])]
    n = min(n, len(b["clear"]))
    fig, axes = plt.subplots(n, len(cols) + 2, figsize=(2.2 * (len(cols) + 2), 2.2 * n), squeeze=False)
    for r in range(n):
        for c, (title, t) in enumerate(cols):
            axes[r, c].imshow(rgb(t[r]))
            axes[r, c].set_title(title if r == 0 else "", fontsize=8)
        axes[r, -2].imshow(b["mask"][r, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[r, -1].imshow(pred_mask[r, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        if r == 0:
            axes[r, -2].set_title("true mask", fontsize=8)
            axes[r, -1].set_title("predicted mask", fontsize=8)
        for ax in axes[r]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def report(results):
    lines = []
    for split in ("synthetic", "real"):
        s = results[split]
        lines += [f"## {split} test ({s['n_images']} images, masker IoU {s['masker_iou']:.3f})", "",
                  "| variant | " + " | ".join(label for _, label in METRICS) + " |",
                  "|---" * (len(METRICS) + 1) + "|"]
        for v in VARIANTS:
            lines.append(f"| {v} | " + " | ".join(f"{s['variants'][v][k]:.4f}" for k, _ in METRICS) + " |")
        lines += ["", "two-stage minus single (mean, 95% CI, share of images where two-stage is better):", ""]
        for k, label in METRICS:
            d = s["two_minus_single"][k]
            if d:
                lines.append(f"- {label}: {d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}], "
                             f"better on {100 * d['frac_images_better']:.0f}% of {d['n']}")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main

def main(argv=None):
    """argv: list of CLI arguments, e.g. main(["--smoke"]) from a notebook; None reads sys.argv."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="/content/data/sen12ms_cr")
    ap.add_argument("--cache", default="/content/cache", help="local disk for .npy memmaps (~11 GB at defaults)")
    ap.add_argument("--out", default="mask_ablation_out")
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8000, help="training steps for single and reconstructor")
    ap.add_argument("--mask_steps", type=int, default=3000)
    ap.add_argument("--base", type=int, default=32, help="reconstructor / single U-Net width")
    ap.add_argument("--mask_base", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrain", action="store_true", help="ignore saved checkpoints")
    ap.add_argument("--smoke", action="store_true", help="tiny random-data run to check the pipeline")
    args = ap.parse_args(argv)
    if args.smoke:
        args.n_train, args.n_test, args.crop, args.batch = 24, 8, 32, 4
        args.steps, args.mask_steps, args.base, args.mask_base, args.workers = 4, 4, 8, 8, 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[+] device {device}")

    if args.smoke:
        caches = {"train": fake_cache(args.n_train, 64, 1), "test": fake_cache(args.n_test, 64, 2)}
    else:
        triplets = find_triplets(args.data_root)
        if not triplets:
            raise SystemExit(f"No S1/S2/S2-cloudy triplets found under {args.data_root}")
        splits = split_by_scene(triplets, args.n_train, args.n_test, args.seed)
        print(f"[+] {len(triplets)} triplets; using {len(splits['train'])} train / {len(splits['test'])} test")
        cache_dir = Path(args.cache)
        cache_dir.mkdir(parents=True, exist_ok=True)
        caches = {name: build_cache(items, cache_dir, name) for name, items in splits.items()}

    loader_kw = {"num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_loader = torch.utils.data.DataLoader(SyntheticClouds(caches["train"], args.crop, train=True),
                                               batch_size=args.batch, shuffle=True, drop_last=True,
                                               persistent_workers=args.workers > 0, **loader_kw)
    test_loaders = {
        "synthetic": torch.utils.data.DataLoader(SyntheticClouds(caches["test"], train=False, seed=args.seed),
                                                 batch_size=8, **loader_kw),
        "real": torch.utils.data.DataLoader(RealClouds(caches["test"]), batch_size=8, **loader_kw),
    }

    models = {"single": UNet(15, 13, args.base), "recon": UNet(16, 13, args.base),
              "masker": UNet(13, 1, args.mask_base)}
    params = {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}
    print(f"[+] parameters {params}")
    for name, model in models.items():
        model.to(device)
        ckpt = out / f"{name}.pt"
        if ckpt.exists() and not args.retrain and not args.smoke:
            model.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"[+] loaded {ckpt}")
            continue
        print(f"[+] training {name}")
        train(model, name, train_loader, args.mask_steps if name == "masker" else args.steps, args.lr, device)
        torch.save(model.state_dict(), ckpt)

    results = {"config": vars(args), "parameters": params}
    for split, loader in test_loaders.items():
        print(f"[+] evaluating {split}")
        results[split] = evaluate(models["single"], models["masker"], models["recon"], loader, device,
                                  out / f"samples_{split}.png")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    text = report(results)
    (out / "results.md").write_text(text)
    print("\n" + text)


# In a notebook, sys.argv holds Jupyter's kernel flags; call main([...]) from a cell instead.
if __name__ == "__main__" and "ipykernel" not in sys.modules:
    main()
