"""Training losses.

reconstruction: L1 on all 13 bands (as DB-CR), optionally up-weighted inside the cloud mask.
ndvi (C2):      L1 between NDVI of the prediction and of the target (optionally NDMI too), on
                pixels where the target's denominator (e.g. NIR + red) is at least
                `index_floor`. On darker pixels (water, shadow, no-data) the ratio's gradient
                grows like 1 / denominator: with the plain 1e-6 epsilon this drowned the L1 term
                and drove every Kaggle run with C2 to NaN.
uncertainty (C3): Laplace negative log-likelihood of the target under (mean, b). The mean is
                detached in this term, so switching the head on or off does not change what the
                mean is trained on; the head only learns to explain the remaining error.
"""
import torch
import torch.nn.functional as F

from .bands import INDICES, NIR, RED, SWIR1

# normalised-difference indices as (a, b) band pairs: (a - b) / (a + b)
ND_PAIRS = {"ndvi": (NIR, RED), "ndmi": (NIR, SWIR1)}


def index_l1(pred, target, name, floor=0.1):
    """Per-pixel |index(pred) - index(target)| as (B, 1, H, W), and the valid-pixel mask.

    The prediction's denominator is clamped at `floor`, so the gradient per pixel is at most
    2 / floor; pixels whose target denominator is below `floor` are left out.
    """
    if name not in ND_PAIRS:  # e.g. EVI, whose denominator stays near 1
        f = INDICES[name]
        return (f(pred) - f(target)).abs()[:, None], None
    a, b = ND_PAIRS[name]
    p, t = pred.clamp(min=0), target.clamp(min=0)
    t_den = t[:, a] + t[:, b]
    valid = (t_den >= floor).float()[:, None]
    p_idx = (p[:, a] - p[:, b]) / (p[:, a] + p[:, b]).clamp(min=floor)
    t_idx = (t[:, a] - t[:, b]) / t_den.clamp(min=floor)
    return (p_idx - t_idx).abs()[:, None] * valid, valid


def _weights(mask, cloud_weight):
    if mask is None or cloud_weight == 1.0:
        return None
    return 1.0 + (cloud_weight - 1.0) * mask


def _wmean(x, w):
    return x.mean() if w is None else (x * w).sum() / (w.sum() * x.shape[1] / w.shape[1]).clamp(min=1)


def laplace_nll(target, mean, log_scale):
    return (target - mean).abs() * torch.exp(-log_scale) + log_scale


def bridge_loss(out, batch, cfg):
    """out: {"mean", "log_scale"}; batch: dict with clear, mask. cfg: the `loss` config section."""
    target, mask = batch["clear"], batch.get("mask")
    w = _weights(mask, cfg.get("cloud_weight", 1.0))
    mean = out["mean"].float()
    terms = {"l1": _wmean((mean - target).abs(), w)}
    lam_ndvi = cfg.get("ndvi", 0.0)
    if lam_ndvi > 0:
        idx = 0.0
        for name in cfg.get("indices", ["ndvi"]):
            err, valid = index_l1(mean, target, name, cfg.get("index_floor", 0.1))
            # mean over valid pixels only (all pixels for EVI)
            wv = valid if w is None else (w if valid is None else w * valid)
            idx = idx + (err.mean() if wv is None else (err * wv).sum() / wv.sum().clamp(min=1))
        terms["ndvi"] = idx
    lam_unc = cfg.get("uncertainty", 0.0)
    if lam_unc > 0 and out.get("log_scale") is not None:
        terms["nll"] = _wmean(laplace_nll(target, mean.detach(), out["log_scale"].float()), w)
    total = terms["l1"] + lam_ndvi * terms.get("ndvi", 0.0) + lam_unc * terms.get("nll", 0.0)
    return total, {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in terms.items()}


def carl_loss(pred, batch, lam=1.0):
    """Cloud-adaptive regularised loss of DSen2-CR: keep clear pixels, fix cloudy ones."""
    m = batch["mask"]
    keep = ((1 - m) * (pred - batch["cloudy"])).abs() + (m * (pred - batch["clear"])).abs()
    total = keep.mean() + lam * (pred - batch["clear"]).abs().mean()
    return total, {"carl": float(total.detach())}


def masker_loss(logits, batch, dice_weight=0.5):
    target = batch["mask"]
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    total = bce + dice_weight * dice.mean()
    return total, {"bce": float(bce.detach()), "dice": float(dice.mean().detach())}
