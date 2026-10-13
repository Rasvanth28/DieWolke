"""Evaluation metrics. All inputs are reflectance tensors in [0, 1], shape (B, 13, H, W).

Conventions follow SEN12MS-CR papers (DSen2-CR, UnCRtainTS, DB-CR): metrics over all 13 bands,
data range 1, SAM in degrees, per image then averaged. "*_cloud" variants use only masked pixels
and are NaN for images without cloud.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .bands import INDICES, S2_BANDS


def ssim(x, y, window=11, sigma=1.5):
    """Mean Gaussian-window SSIM over channels, per image (valid region only)."""
    c = x.shape[1]
    g = torch.exp(-(torch.arange(window, device=x.device, dtype=x.dtype) - window // 2) ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(c, 1, window, window).contiguous()

    def f(t):
        return F.conv2d(t, w, groups=c)
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return s.mean((1, 2, 3))


def _masked_mean(v, m):
    """v, m: (B, H, W). NaN where an image has no masked pixels."""
    n = m.sum((1, 2))
    out = (v * m).sum((1, 2)) / n.clamp(min=1)
    return torch.where(n > 0, out, torch.full_like(out, float("nan")))


@torch.no_grad()
def image_metrics(pred, gt, mask=None):
    """Per-image metrics -> dict of (B,) tensors, plus per-band arrays under 'mae_<band>'/'rmse_<band>'."""
    pred, gt = pred.float().clamp(0, 1), gt.float()
    err = pred - gt
    se = (err ** 2).mean(1)
    out = {
        "psnr": -10 * torch.log10(se.mean((1, 2)) + 1e-10),
        "ssim": ssim(pred, gt),
        "mae": err.abs().mean((1, 2, 3)),
        "rmse": se.mean((1, 2)).sqrt(),
    }
    cos = (pred * gt).sum(1) / (pred.norm(dim=1) * gt.norm(dim=1) + 1e-8)
    sam = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
    out["sam"] = sam.mean((1, 2))
    for name, f in INDICES.items():
        d = (f(pred) - f(gt)) ** 2
        out[f"{name}_rmse"] = d.mean((1, 2)).sqrt()
        if mask is not None:
            out[f"{name}_rmse_cloud"] = _masked_mean(d, mask[:, 0]).sqrt()
    for b, band in enumerate(S2_BANDS):
        out[f"mae_{band}"] = err[:, b].abs().mean((1, 2))
        out[f"rmse_{band}"] = (err[:, b] ** 2).mean((1, 2)).sqrt()
    if mask is not None:
        m = mask[:, 0]
        out["psnr_cloud"] = -10 * torch.log10(_masked_mean(se, m) + 1e-10)
        out["mae_cloud"] = _masked_mean(err.abs().mean(1), m)
        out["sam_cloud"] = _masked_mean(sam, m)
        out["cloud_frac"] = m.mean((1, 2))
    return out


# Headline metrics shown in tables, with direction (True = higher is better).
HEADLINE = [("psnr", True), ("ssim", True), ("mae", False), ("rmse", False), ("sam", False),
            ("ndvi_rmse", False), ("psnr_cloud", True), ("mae_cloud", False), ("ndvi_rmse_cloud", False)]


class PixelSampler:
    """Collects a fixed random subset of pixels per image for global (pooled) statistics."""

    def __init__(self, per_image=2048, seed=0):
        self.per_image, self.rng = per_image, np.random.default_rng(seed)
        self.store = {}

    def add(self, **arrays):
        """arrays: name -> (B, C, H, W) tensors with identical B, H, W."""
        first = next(iter(arrays.values()))
        b, _, h, w = first.shape
        k = min(self.per_image, h * w)
        for i in range(b):
            idx = torch.from_numpy(self.rng.choice(h * w, k, replace=False)).to(first.device)
            for name, a in arrays.items():
                v = a[i].reshape(a.shape[1], -1)[:, idx].T.float().cpu().numpy()  # (k, C)
                self.store.setdefault(name, []).append(v)

    def get(self, name):
        return np.concatenate(self.store[name]) if name in self.store else None


def bootstrap_ci(values, n_boot=2000, seed=0):
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), [float("nan"), float("nan")]
    boots = np.random.default_rng(seed).choice(v, (n_boot, len(v))).mean(1)
    return float(v.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def paired_difference(a, b, higher_is_better, n_boot=2000, seed=0):
    """Mean of (a - b) over images, 95% bootstrap CI, and the share of images where a wins."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return None
    mean, ci = bootstrap_ci(d, n_boot, seed)
    wins = d > 0 if higher_is_better else d < 0
    return {"mean": mean, "ci95": ci, "frac_images_better": float(wins.mean()), "n": int(len(d))}
