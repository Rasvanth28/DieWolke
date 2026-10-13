"""Fixtures: a tiny fake SEN12MS-CR tif tree and a fake Kharif raw download."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kharif"))

SIZE = 64


def _smooth(rng, ch, size=SIZE, cells=4):
    low = rng.random((ch, cells, cells), dtype=np.float32)
    rep = size // cells
    return np.repeat(np.repeat(low, rep, -2), rep, -1)


def _write(path, arr, dtype, crs="EPSG:32643", x0=500000, y0=3400000):
    import rasterio
    from rasterio.transform import from_origin
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[-1], height=arr.shape[-2], count=arr.shape[0],
                       dtype=dtype, crs=crs, transform=from_origin(x0, y0, 10, 10)) as dst:
        dst.write(arr.astype(dtype))


def make_sen12mscr(root, scenes=((106, 3), (17, 2), (9, 4), (1, 4)), roi="ROIs1158_spring", seed=0):
    """Scenes 106 (official test) and 17 (official val) plus two training scenes."""
    rng = np.random.default_rng(seed)
    for scene, n_patches in scenes:
        for p in range(1, n_patches + 1):
            clear = _smooth(rng, 13) * 3000 + 300
            cloud = np.clip((_smooth(rng, 1)[0] - 0.4) * 3, 0, 1)
            cloudy = clear * (1 - cloud) + 6000 * cloud
            s1 = -25 + _smooth(rng, 2) * 20
            _write(root / f"{roi}_s1" / f"s1_{scene}" / f"{roi}_s1_{scene}_p{p}.tif", s1, "float32")
            _write(root / f"{roi}_s2" / f"s2_{scene}" / f"{roi}_s2_{scene}_p{p}.tif", clear, "uint16")
            _write(root / f"{roi}_s2_cloudy" / f"s2_cloudy_{scene}" / f"{roi}_s2_cloudy_{scene}_p{p}.tif", cloudy, "uint16")
    return root


@pytest.fixture(scope="session")
def sen12_root(tmp_path_factory):
    return make_sen12mscr(tmp_path_factory.mktemp("sen12ms_cr"))


def make_kharif_raw(raw, n_clear=6, n_donor=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for role, n in (("clear", n_clear), ("donor", n_donor)):
        for i in range(n):
            aoi = ["ludhiana", "karnal"][i % 2]
            date = f"2021-0{7 + (i % 3)}-1{i % 9}"
            cell = f"32643_{500000 + 2560 * (i % 3)}_{3400000 - 2560 * (i // 3)}"
            key = f"{role}/{aoi}/{date}_{cell}"
            s2 = _smooth(rng, 13) * 3000 + 300
            prob = np.zeros((1, SIZE, SIZE)) if role == "clear" else np.clip(_smooth(rng, 1) * 100, 0, 100)
            shadow = np.zeros((1, SIZE, SIZE))
            lc = rng.choice([40, 40, 40, 10, 50], size=(1, 4, 4)).repeat(SIZE // 4, 1).repeat(SIZE // 4, 2)
            _write(raw / f"{key}_s2.tif", np.concatenate([s2, prob, shadow, lc]), "uint16")
            _write(raw / f"{key}_s1.tif", -25 + _smooth(rng, 2) * 20, "float32")
            rows.append({"key": key, "role": role, "aoi": aoi, "s2_id": f"S2_{role}_{i}", "date": date,
                         "tile": "43RFQ", "s1_id": f"S1_{i}", "s1_date": date, "cell": cell, "epsg": "EPSG:32643",
                         "x": 500000 + 2560 * (i % 3), "y": 3400000 - 2560 * (i // 3), "cloud": 0.0,
                         "shadow": 0.0, "crop": 0.6, "s2": f"{key}_s2.tif", "s1": f"{key}_s1.tif"})
    (raw / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return raw
