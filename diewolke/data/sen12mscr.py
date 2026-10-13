"""SEN12MS-CR file discovery, train/val/test splits and the on-disk .npy cache.

Layout on disk (any of the four seasons may be present):
    <root>/ROIs1158_spring_s1/s1_<scene>/ROIs1158_spring_s1_<scene>_p<patch>.tif
    <root>/ROIs1158_spring_s2/s2_<scene>/ROIs1158_spring_s2_<scene>_p<patch>.tif
    <root>/ROIs1158_spring_s2_cloudy/s2_cloudy_<scene>/ROIs1158_spring_s2_cloudy_<scene>_p<patch>.tif
Triplets are matched on (roi, scene, patch) parsed from the file name, never by path substitution.
"""
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..bands import N_S1, N_S2, S2_MAX, normalize_s1

# e.g. ROIs1158_spring_s2_cloudy_102_p30.tif -> (ROIs1158_spring, s2_cloudy, 102, 30)
NAME_RE = re.compile(r"^(ROIs\d+_[a-z]+?)_(s1|s2_cloudy|s2)_(\d+)_p(\d+)\.tiff?$", re.IGNORECASE)

# Official SEN12MS-CR scene split, as used by the UnCRtainTS reference data loader
# (github.com/PatrickTUM/UnCRtainTS, data/dataLoader.py, class SEN12MSCR). Keys are (roi, scene).
OFFICIAL_TEST = {("ROIs1158_spring", 106), ("ROIs1158_spring", 123), ("ROIs1158_spring", 140),
                 ("ROIs1158_spring", 31), ("ROIs1158_spring", 44), ("ROIs1868_summer", 119),
                 ("ROIs1868_summer", 73), ("ROIs1970_fall", 139), ("ROIs2017_winter", 108),
                 ("ROIs2017_winter", 63)}
OFFICIAL_VAL = {("ROIs2017_winter", 22), ("ROIs1868_summer", 19), ("ROIs1970_fall", 65),
                ("ROIs1158_spring", 17), ("ROIs2017_winter", 107), ("ROIs1868_summer", 80),
                ("ROIs1868_summer", 127), ("ROIs2017_winter", 130), ("ROIs1868_summer", 17),
                ("ROIs2017_winter", 84)}


def patch_id(key):
    roi, scene, patch = key
    return f"{roi}_{scene}_p{patch}"


def find_triplets(root, verbose=True):
    """Return sorted [(key, {"s1": path, "s2": path, "s2_cloudy": path})] for complete triplets."""
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
    if verbose and not triplets:
        counts = defaultdict(int)
        for paths in found.values():
            for modality in paths:
                counts[modality] += 1
        print(f"[!] no complete triplets under {root}")
        print(f"    files per modality matching the SEN12MS-CR pattern: {dict(counts)}")
        print(f"    unmatched tif files: {len(unmatched)}; e.g. {[str(p) for p in unmatched[:5]]}")
    return triplets


def split_triplets(triplets, scheme="official", val_frac=0.05, test_frac=0.1, seed=0):
    """Split by whole scenes so no test patch overlaps a training area.

    scheme="official": the official SEN12MS-CR test/val scenes; everything else is train.
        With only the spring ROI downloaded this gives 5 test scenes and 1 val scene.
    scheme="random": a seeded random scene split (useful when the official scenes are missing).
    """
    scenes = sorted({k[:2] for k, _ in triplets})
    if scheme == "official":
        test, val = OFFICIAL_TEST, OFFICIAL_VAL
        if not any(s in test for s in scenes):
            raise ValueError("none of the official test scenes are present; use scheme='random'")
    elif scheme == "random":
        rng = random.Random(seed)
        shuffled = scenes[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(test_frac * len(scenes)))
        n_val = max(1, round(val_frac * len(scenes)))
        test, val = set(shuffled[:n_test]), set(shuffled[n_test:n_test + n_val])
    else:
        raise ValueError(f"unknown split scheme {scheme!r}")
    splits = {"train": [], "val": [], "test": []}
    for k, paths in triplets:
        s = k[:2]
        splits["test" if s in test else "val" if s in val else "train"].append((k, paths))
    return splits


def subsample(items, limit, seed=0):
    if limit is None or limit <= 0 or len(items) <= limit:
        return items
    rng = random.Random(seed)
    return sorted(rng.sample(items, limit))


def read_tif(path):
    import rasterio
    with rasterio.open(path) as src:
        return src.read()


def read_s1(path):
    return normalize_s1(read_tif(path).astype(np.float32))


def read_s2(path):
    return np.clip(np.nan_to_num(read_tif(path)), 0, S2_MAX).astype(np.uint16)


CACHE_FIELDS = {
    # name: (dtype, channels or None for a single (H, W) plane)
    "s1": (np.float16, N_S1),
    "clear": (np.uint16, N_S2),
    "cloudy": (np.uint16, N_S2),
    "prob": (np.float16, None),        # s2cloudless cloud probability of the cloudy image
    "prob_clear": (np.float16, None),  # same for the "clear" target (it is not always clear)
    "shadow": (np.uint8, None),        # heuristic cloud-shadow mask of the cloudy image
}


# Optional per-split fields: an exact cloud+shadow mask (uint8, H x W) for benchmarks with
# transplanted clouds, and per-pixel class labels (uint8, 0 = unlabelled) for the downstream task.
OPTIONAL_FIELDS = ("mask", "labels")


def build_cache(items, cache_dir, split, chunk=64, verbose=True):
    """Read every tif once into .npy memmaps under cache_dir; later runs reuse them.

    items: [(key, paths)] from split_triplets. Returns a Cache.
    """
    from .clouds import cloud_probability, shadow_mask

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    done = cache_dir / f"{split}.done"
    files = {k: cache_dir / f"{split}_{k}.npy" for k in CACHE_FIELDS}
    if not done.exists():
        n = len(items)
        if n == 0:
            raise ValueError(f"split {split!r} is empty")
        size = read_s2(items[0][1]["s2"]).shape[-1]
        arrays = {}
        for k, (dt, ch) in CACHE_FIELDS.items():
            shape = (n, size, size) if ch is None else (n, ch, size, size)
            arrays[k] = np.lib.format.open_memmap(files[k], "w+", dt, shape)
        t0 = time.time()
        for start in range(0, n, chunk):
            batch = items[start:start + chunk]
            sl = slice(start, start + len(batch))
            arrays["s1"][sl] = np.stack([read_s1(p["s1"]) for _, p in batch])
            clear = np.stack([read_s2(p["s2"]) for _, p in batch])
            cloudy = np.stack([read_s2(p["s2_cloudy"]) for _, p in batch])
            arrays["clear"][sl] = clear
            arrays["cloudy"][sl] = cloudy
            arrays["prob"][sl] = cloud_probability(cloudy)
            arrays["prob_clear"][sl] = cloud_probability(clear)
            arrays["shadow"][sl] = np.stack([shadow_mask(c) for c in cloudy])
            if verbose:
                print(f"  cached {split} {start + len(batch)}/{n} ({time.time() - t0:.0f}s)", flush=True)
        for a in arrays.values():
            a.flush()
        (cache_dir / f"{split}_keys.json").write_text(json.dumps([patch_id(k) for k, _ in items]))
        done.touch()
    return Cache.load(cache_dir, split)


class Cache(dict):
    """dict of memmapped arrays plus .ids (patch ids, same order as the arrays)."""

    @classmethod
    def load(cls, cache_dir, split):
        cache_dir = Path(cache_dir)
        c = cls({k: np.load(cache_dir / f"{split}_{k}.npy", mmap_mode="r") for k in CACHE_FIELDS})
        for k in OPTIONAL_FIELDS:  # written by kharif/build_benchmark.py
            f = cache_dir / f"{split}_{k}.npy"
            if f.exists():
                c[k] = np.load(f, mmap_mode="r")
        keys = cache_dir / f"{split}_keys.json"
        c.ids = json.loads(keys.read_text()) if keys.exists() else [str(i) for i in range(len(c["clear"]))]
        return c

    def __len__(self):
        return len(self["clear"])


def fake_cache(n, size=64, seed=0):
    """Random, spatially smooth data with the cache's layout, for smoke tests.

    The optical bands are a fixed mix of the two SAR channels plus an independent component, so
    a model can learn to use S1 under clouds (a weak stand-in for the real SAR-optical relation).
    """
    from scipy.ndimage import uniform_filter
    rng = np.random.default_rng(seed)
    mix = np.random.default_rng(1234).uniform(0.2, 1.0, (N_S2, N_S1)).astype(np.float32)

    def smooth(ch, cells=4):
        low = rng.random((n, ch, cells, cells), dtype=np.float32)
        rep = -(-size // cells)
        up = np.repeat(np.repeat(low, rep, -2), rep, -1)[..., :size, :size]
        return uniform_filter(up, size=(1, 1, 9, 9))

    s1 = smooth(N_S1)
    clear = 300 + 2500 * (np.einsum("bc,ncyx->nbyx", mix, s1) / mix.sum(1)[None, :, None, None]) \
        + 600 * smooth(N_S2)
    prob = np.clip((smooth(1)[:, 0] - 0.35) * 2.5, 0, 1)
    cloudy = clear * (1 - prob[:, None]) + 7000 * prob[:, None]
    c = Cache({"s1": s1.astype(np.float16), "clear": clear.astype(np.uint16),
               "cloudy": cloudy.astype(np.uint16), "prob": prob.astype(np.float16),
               "prob_clear": np.zeros_like(prob, dtype=np.float16),
               "shadow": np.zeros(prob.shape, dtype=np.uint8)})
    c.ids = [f"fake_{seed}_p{i}" for i in range(n)]
    return c


def prepare_caches(cfg, splits=("train", "val", "test"), verbose=True):
    """Build or load the caches named in a config's `data` section. Returns {split: Cache}."""
    d = cfg["data"]
    if d.get("fake"):
        sizes = {"train": d.get("n_train", 32), "val": d.get("n_val", 8), "test": d.get("n_test", 8)}
        return {s: fake_cache(sizes[s], d.get("fake_size", 64), seed=i) for i, s in enumerate(splits)}
    cache_dir = Path(d["cache"])
    info = {k: d.get(k) for k in ("split", "seed", "n_train", "n_val", "n_test")}
    info_file = cache_dir / "cache_info.json"
    if all((cache_dir / f"{s}.done").exists() for s in splits):
        if info_file.exists() and json.loads(info_file.read_text()) != info:
            print(f"[!] {cache_dir} was built with {info_file.read_text()}, config asks for {info}; "
                  "using the existing cache (delete it to rebuild)")
        return {s: Cache.load(cache_dir, s) for s in splits}
    triplets = find_triplets(d["root"], verbose=verbose)
    if not triplets:
        raise SystemExit(f"No S1/S2/S2-cloudy triplets under {d['root']}")
    parts = split_triplets(triplets, d.get("split", "official"), seed=d.get("seed", 0))
    limits = {"train": d.get("n_train"), "val": d.get("n_val"), "test": d.get("n_test")}
    if verbose:
        print(f"[+] {len(triplets)} triplets; split sizes "
              + ", ".join(f"{s} {len(parts[s])}" for s in parts))
    caches = {s: build_cache(subsample(parts[s], limits[s], d.get("seed", 0)), cache_dir, s, verbose=verbose)
              for s in splits}
    info_file.write_text(json.dumps(info))
    return caches
