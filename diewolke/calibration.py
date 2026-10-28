"""Uncertainty calibration for the Laplace head (contribution C3).

Inputs are pooled pixel samples: err (N, 13) absolute errors and scale (N, 13) Laplace b.
  fit_temperature   one scalar s per band so that b' = s * b maximises validation likelihood
                    (closed form for the Laplace: s = mean(|e| / b))
  coverage_curve    observed vs nominal coverage of central intervals |e| <= -b ln(1 - p)
  sparsification    error left after removing the most uncertain pixels, vs the oracle order (AUSE)
"""
import numpy as np
from scipy.stats import spearmanr

LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])


def fit_temperature(err, scale, per_band=True):
    ratio = err / np.maximum(scale, 1e-8)
    s = ratio.mean(0) if per_band else np.full(err.shape[1], ratio.mean())
    return s.astype(np.float32)


def nll(err, scale):
    scale = np.maximum(scale, 1e-8)
    return float((err / scale + np.log(2 * scale)).mean())


def coverage_curve(err, scale, levels=LEVELS):
    half_width = -np.log(1 - levels)[None, None, :] * scale[..., None]
    observed = (err[..., None] <= half_width).mean((0, 1))
    return observed


def calibration_error(err, scale, levels=LEVELS):
    return float(np.abs(coverage_curve(err, scale, levels) - levels).mean())


def sparsification(err, scale, fractions=np.linspace(0, 0.95, 20)):
    """MAE of the kept pixels after removing a fraction, ordered by uncertainty and by oracle error."""
    e, u = err.mean(1), scale.mean(1)  # per pixel, averaged over bands
    by_u, by_e = e[np.argsort(-u)], e[np.argsort(-e)]
    n = len(e)
    curve_u = np.array([by_u[int(f * n):].mean() for f in fractions])
    curve_e = np.array([by_e[int(f * n):].mean() for f in fractions])
    ause = float(np.trapezoid(curve_u - curve_e, fractions)) if hasattr(np, "trapezoid") else \
        float(np.trapz(curve_u - curve_e, fractions))
    return {"fractions": fractions.tolist(), "mae_by_uncertainty": curve_u.tolist(),
            "mae_oracle": curve_e.tolist(), "ause": ause}


def summarize(err, scale, temperature=None):
    """All calibration statistics for one set of pooled pixels (optionally after scaling)."""
    if temperature is not None:
        scale = scale * temperature[None]
    rho = spearmanr(scale.mean(1), err.mean(1)).statistic
    return {
        "nll": nll(err, scale),
        "calibration_error": calibration_error(err, scale),
        "coverage": dict(zip([f"{p:.2f}" for p in LEVELS], coverage_curve(err, scale).round(4).tolist())),
        "spearman_uncertainty_error": float(rho),
        "sparsification": sparsification(err, scale),
    }
