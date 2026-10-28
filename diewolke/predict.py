"""Reconstruct cloudy Sentinel-2 GeoTIFFs with a trained run, keeping georeferencing.

    # every SEN12MS-CR-style triplet under a folder
    python -m diewolke.predict --run RUNS/full --data_root /content/data/some_rois --out /content/recon
    # one scene of any size (tiled with overlap)
    python -m diewolke.predict --run RUNS/full --s1 s1.tif --s2_cloudy s2_cloudy.tif --out /content/recon

For each input writes <name>_recon.tif (13-band uint16 DN, like the input), and for runs with an
uncertainty head <name>_uncertainty.tif (13-band float32 Laplace scale b, reflectance units,
multiplied by the calibration temperature if --temperature is given) and <name>_mask.tif.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .bands import S2_MAX, normalize_s1, to_reflectance
from .data.clouds import cloud_probability, real_mask, shadow_mask
from .data.sen12mscr import find_triplets, patch_id
from .engine import Pipeline, load_run
from .utils import get_device


def _windows(n, tile, stride):
    starts = list(range(0, max(n - tile, 0) + 1, stride))
    if starts[-1] + tile < n:
        starts.append(n - tile)
    return starts


@torch.no_grad()
def reconstruct(pipe, s1, cloudy_dn, device, tile=256, overlap=32, steps=None, mask=None):
    """s1: (2, H, W) normalised; cloudy_dn: (13, H, W) DN. Returns (recon refl, scale or None, mask)."""
    h, w = cloudy_dn.shape[-2:]
    if mask is None and pipe.mask_source == "data":
        prob = cloud_probability(cloudy_dn[None])[0]
        mask = real_mask(prob, shadow_mask(cloudy_dn, prob))
    cloudy = to_reflectance(cloudy_dn)
    mult = 2 ** (getattr(pipe.model, "n_levels", 4) - 1)
    tile = max(mult, (min(tile, max(h, w)) // mult) * mult)
    ph, pw = max(0, tile - h), max(0, tile - w)  # pad small inputs up to one tile
    pad = lambda a: np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, ph), (0, pw)], mode="reflect")
    s1p, cp = pad(s1), pad(cloudy)
    mp = pad(mask[None])[0] if mask is not None else None
    H, W = cp.shape[-2:]
    acc = np.zeros((13, H, W), np.float32)
    acc_s = np.zeros((13, H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    ramp = np.minimum(np.arange(tile) + 1, np.arange(tile)[::-1] + 1).astype(np.float32)
    weight = np.minimum(np.minimum.outer(ramp, ramp), overlap + 1) if overlap else np.ones((tile, tile), np.float32)
    has_scale, used_mask = False, np.zeros((H, W), np.float32)
    stride = tile - overlap
    for y in _windows(H, tile, stride):
        for x in _windows(W, tile, stride):
            sl = (slice(y, y + tile), slice(x, x + tile))
            batch = {"s1": torch.from_numpy(s1p[(slice(None),) + sl])[None].to(device),
                     "cloudy": torch.from_numpy(cp[(slice(None),) + sl])[None].to(device)}
            batch["mask"] = torch.from_numpy(mp[sl] if mp is not None else np.zeros((tile, tile), np.float32))[None, None].to(device)
            out = pipe.predict(batch, steps=steps)
            acc[(slice(None),) + sl] += out["mean"][0].float().cpu().numpy() * weight
            if out.get("scale") is not None:
                has_scale = True
                acc_s[(slice(None),) + sl] += out["scale"][0].float().cpu().numpy() * weight
            if out.get("mask") is not None:
                used_mask[sl] = np.maximum(used_mask[sl], out["mask"][0, 0].cpu().numpy())
            wsum[sl] += weight
    recon = (acc / wsum)[:, :h, :w]
    scale = (acc_s / wsum)[:, :h, :w] if has_scale else None
    return np.clip(recon, 0, 1), scale, used_mask[:h, :w]


def write_like(src_path, dst_path, array, dtype):
    import rasterio
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
    profile.update(count=array.shape[0], dtype=dtype, compress="deflate", nodata=None)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(array.astype(dtype))


def main(argv=None):
    import rasterio
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--masker", default=None)
    ap.add_argument("--data_root")
    ap.add_argument("--s1")
    ap.add_argument("--s2_cloudy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--temperature", default=None, help="summary.json from evaluate.py, to calibrate the scale")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    device = get_device()
    masker = load_run(args.masker, device)[1] if args.masker else None
    pipe = Pipeline.from_run(args.run, device, masker=masker)
    temp = None
    if args.temperature:
        summ = json.loads(Path(args.temperature).read_text())
        u = summ.get(pipe.cfg.get("name"), {}).get("uncertainty") or {}
        temp = np.array(u["temperature"], np.float32) if "temperature" in u else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.data_root:
        jobs = [(patch_id(k), p["s1"], p["s2_cloudy"]) for k, p in find_triplets(args.data_root)]
    elif args.s1 and args.s2_cloudy:
        jobs = [(Path(args.s2_cloudy).stem, args.s1, args.s2_cloudy)]
    else:
        raise SystemExit("give --data_root, or --s1 and --s2_cloudy")
    for name, s1_path, s2_path in jobs[:args.limit]:
        with rasterio.open(s1_path) as f:
            s1 = normalize_s1(f.read().astype(np.float32))
        with rasterio.open(s2_path) as f:
            cloudy = np.clip(np.nan_to_num(f.read()), 0, S2_MAX).astype(np.uint16)
        recon, scale, mask = reconstruct(pipe, s1, cloudy, device, args.tile, args.overlap, args.steps)
        write_like(s2_path, out / f"{name}_recon.tif", np.round(recon * S2_MAX), "uint16")
        write_like(s2_path, out / f"{name}_mask.tif", mask[None], "uint8")
        if scale is not None:
            if temp is not None:
                scale = scale * temp[:, None, None]
            write_like(s2_path, out / f"{name}_uncertainty.tif", scale, "float32")
        print(f"  wrote {name}")


if __name__ == "__main__":
    main()
