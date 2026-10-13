"""Sentinel band conventions shared by every module.

SEN12MS-CR stores Sentinel-2 L1C as 13 bands, all resampled to 10 m, in this order:
B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12 (0-indexed below). Sentinel-1 is VV, VH in dB.
"""
import numpy as np
import torch

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
S1_BANDS = ["VV", "VH"]
N_S2, N_S1 = len(S2_BANDS), len(S1_BANDS)
IDX = {name: i for i, name in enumerate(S2_BANDS)}

BLUE, GREEN, RED, NIR, SWIR1 = IDX["B02"], IDX["B03"], IDX["B04"], IDX["B08"], IDX["B11"]
RGB = [RED, GREEN, BLUE]

# Band groups for band-group attention (contribution C1). B01 (coastal aerosol) sits with the
# visible bands; B09 (water vapour, 945 nm) with red-edge/NIR; B10 (cirrus, 1375 nm) with SWIR.
BAND_GROUPS = {
    "visible": [IDX[b] for b in ("B01", "B02", "B03", "B04")],
    "rededge_nir": [IDX[b] for b in ("B05", "B06", "B07", "B08", "B8A", "B09")],
    "swir": [IDX[b] for b in ("B10", "B11", "B12")],
}
assert sorted(sum(BAND_GROUPS.values(), [])) == list(range(N_S2))

# Value ranges, following SEN12MS-CR / UnCRtainTS conventions.
S2_MAX = 10000.0                  # S2 DN clipped to [0, 10000], then / 10000 -> [0, 1]
S1_DB_MIN = np.array([-25.0, -32.5], dtype=np.float32)  # VV, VH clipped to [min, 0] dB -> [0, 1]


def normalize_s1(x):
    """(2, H, W) dB array -> [0, 1]. NaNs become the floor value."""
    floor = S1_DB_MIN.reshape(2, 1, 1)
    x = np.where(np.isnan(x), floor, x).astype(np.float32)
    return (np.clip(x, floor, 0) - floor) / -floor


def to_reflectance(dn):
    """S2 DN (uint16 or float) -> float32 reflectance in [0, 1]."""
    return np.clip(np.asarray(dn, dtype=np.float32), 0, S2_MAX) / S2_MAX


def _nd(a, b, eps=1e-6):
    return (a - b) / (a + b + eps)


def ndvi(x):
    """Normalised difference vegetation index. x: (B, 13, H, W) tensor or (13, H, W) array."""
    x = x.clamp(min=0) if torch.is_tensor(x) else np.clip(x, 0, None)
    return _nd(x[..., NIR, :, :], x[..., RED, :, :])


def ndmi(x):
    """Normalised difference moisture index (NIR vs SWIR1)."""
    x = x.clamp(min=0) if torch.is_tensor(x) else np.clip(x, 0, None)
    return _nd(x[..., NIR, :, :], x[..., SWIR1, :, :])


def evi(x):
    """Enhanced vegetation index on reflectance in [0, 1]."""
    x = x.clamp(min=0) if torch.is_tensor(x) else np.clip(x, 0, None)
    n, r, b = x[..., NIR, :, :], x[..., RED, :, :], x[..., BLUE, :, :]
    den = n + 6 * r - 7.5 * b + 1
    return 2.5 * (n - r) / (den + (1e-6 if torch.is_tensor(den) else 1e-6))


INDICES = {"ndvi": ndvi, "ndmi": ndmi, "evi": evi}
