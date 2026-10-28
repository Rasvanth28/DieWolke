"""Cloud probability, a cloud-shadow heuristic, and synthetic cloud transplanting."""
import numpy as np
from scipy.ndimage import binary_dilation, maximum_filter, median_filter

from ..bands import BLUE, NIR, SWIR1, S2_MAX

CLOUD_PROB = 0.4      # s2cloudless default threshold
MASK_THRESHOLD = 0.3  # cloud alpha + shadow above this counts as masked in synthetic data

_detector = None


def cloud_probability(s2):
    """(n, 13, H, W) S2 DN -> (n, H, W) float32 cloud probability in [0, 1].

    Uses s2cloudless when installed; otherwise a crude brightness proxy (and says so once).
    """
    global _detector
    refl = s2.astype(np.float32) / S2_MAX
    if _detector is None:
        try:
            from s2cloudless import S2PixelCloudDetector
            _detector = S2PixelCloudDetector(threshold=CLOUD_PROB, average_over=4, dilation_size=2,
                                             all_bands=True)
        except ImportError:
            print("[!] s2cloudless not installed; using a brightness threshold as cloud probability")
            _detector = "brightness"
    if _detector == "brightness":
        return np.clip((refl[:, [1, 2, 3]].mean(1) - 0.15) / 0.2, 0, 1).astype(np.float32)
    return _detector.get_cloud_probability_maps(refl.transpose(0, 2, 3, 1)).astype(np.float32)


def shadow_mask(s2, cloud_prob=None):
    """Heuristic cloud-shadow mask for one (13, H, W) S2 DN patch -> (H, W) uint8.

    Shadows are dark in NIR + SWIR1 and not bright in blue (spectral-index thresholds after
    Zhai et al. 2018, also used by Meraner et al. 2020 for SEN12MS-CR). Thresholds are relative to
    the patch statistics, so this flags dark land as well on cloud-free patches; it is only used
    together with a cloud mask, never alone.
    """
    x = s2.astype(np.float32) / S2_MAX
    csi = (x[NIR] + x[SWIR1]) / 2
    blue = x[BLUE]
    t_csi = csi.min() + 0.75 * (csi.mean() - csi.min())
    t_blue = blue.min() + (5 / 6) * (blue.mean() - blue.min())
    m = (csi < t_csi) & (blue < t_blue)
    if cloud_prob is not None:
        m &= cloud_prob < CLOUD_PROB
    return median_filter(m.astype(np.uint8), size=5)


def real_mask(prob, shadow=None, threshold=CLOUD_PROB, dilate=2, use_shadow=True):
    """Binary cloud(+shadow) mask for a real cloudy patch. prob: (H, W)."""
    m = prob > threshold
    if dilate:
        m = binary_dilation(m, iterations=dilate)
    if use_shadow and shadow is not None and m.any():
        # only keep shadow pixels near clouds; isolated dark fields are not shadows
        near = maximum_filter(m, size=81)
        m |= (shadow > 0) & near
    return m.astype(np.float32)


def synthesize(clear, donor_cloudy, donor_prob, rng, max_shift=20, shadow_strength=0.6):
    """Blend a real cloud (soft edges plus a displaced cast shadow) onto a clear patch.

    clear, donor_cloudy: (13, H, W) reflectance in [0, 1]; donor_prob: (H, W).
    Returns (cloudy, mask, alpha), where mask is the exact binary cloud+shadow mask.
    """
    alpha = np.clip((donor_prob - 0.2) / 0.5, 0, 1).astype(np.float32)
    dy, dx = rng.integers(-max_shift, max_shift + 1, size=2)
    shadow = np.roll(alpha, (dy, dx), axis=(0, 1)) * (1 - alpha)
    cloudy = (clear * (1 - alpha) + donor_cloudy * alpha) * (1 - shadow_strength * shadow)
    mask = ((alpha + shadow) > MASK_THRESHOLD).astype(np.float32)
    return cloudy.astype(np.float32), mask, alpha


def donor_indices(prob, lo=0.1, hi=0.8, threshold=CLOUD_PROB):
    """Indices of patches whose cloud cover is in (lo, hi): good cloud donors."""
    frac = np.array([(np.asarray(prob[i]) > threshold).mean() for i in range(len(prob))])
    idx = np.flatnonzero((frac > lo) & (frac < hi))
    return idx if len(idx) else np.arange(len(frac)), frac
