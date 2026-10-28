"""Integration tests on fake GeoTIFFs: data discovery, caching, training, evaluation, prediction,
the XGBoost baseline and the Kharif benchmark tools."""
import json

import numpy as np
import pytest
import torch

from conftest import make_kharif_raw
from diewolke.config import load_config
from diewolke.data import CloudPairs, find_triplets, prepare_caches, split_triplets
from diewolke.data.sen12mscr import NAME_RE


def test_name_regex():
    m = NAME_RE.match("ROIs1158_spring_s2_cloudy_102_p30.tif")
    assert m.groups() == ("ROIs1158_spring", "s2_cloudy", "102", "30")
    assert NAME_RE.match("ROIs1868_summer_s1_7_p1.tif").groups()[1] == "s1"


def test_discovery_and_official_split(sen12_root):
    trip = find_triplets(sen12_root)
    assert len(trip) == 13
    parts = split_triplets(trip, "official")
    assert {k[1] for k, _ in parts["test"]} == {106} and {k[1] for k, _ in parts["val"]} == {17}
    assert {k[1] for k, _ in parts["train"]} == {1, 9}


@pytest.fixture(scope="module")
def setup(sen12_root, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("work")
    over = [f"data.root={sen12_root}", f"data.cache={tmp / 'cache'}", "data.fake=false",
            f"train.out_dir={tmp / 'runs'}", "data.crop=32", "data.max_target_cloud=null"]
    return tmp, over


def test_cache_and_datasets(setup):
    tmp, over = setup
    cfg = load_config("configs/smoke.yaml", over)
    caches = prepare_caches(cfg)
    assert len(caches["train"]) == 8 and len(caches["test"]) == 3
    real = CloudPairs(caches["train"], mode="real", crop=32)
    b = real[0]
    assert b["cloudy"].shape == (13, 32, 32) and b["s1"].shape == (2, 32, 32) and b["mask"].shape == (1, 32, 32)
    assert 0 <= float(b["s1"].min()) and float(b["s1"].max()) <= 1
    syn = CloudPairs(caches["test"], mode="synthetic", train=False)
    b1, b2 = syn[0], syn[0]
    assert torch.equal(b1["cloudy"], b2["cloudy"])  # evaluation clouds are deterministic
    keep = b1["mask"][0] == 0
    # outside the exact mask the synthetic input is close to the target (only faint cloud edges)
    assert float((b1["cloudy"][:, keep] - b1["clear"][:, keep]).abs().mean()) < 0.05


@pytest.mark.parametrize("config", ["smoke", "masker", "dsen2cr", "end_to_end"])
def test_training_runs_and_resumes(setup, config):
    from diewolke.train import run
    tmp, over = setup
    small = ["train.steps=3", "train.batch=2", "train.workers=0", "train.val_every=3", "train.val_batches=1",
             "train.log_every=1", "train.amp=false", "train.warmup=1", "train.val_batch=2"]
    model_small = ["model.base=16", "model.sar_ch=16", "model.heads=2"] if config != "dsen2cr" else \
        ["model.features=16", "model.blocks=2"]
    if config == "masker":
        model_small = ["model.base=8"]
    cfg = load_config(f"configs/{config}.yaml", over + small + model_small)
    out = run(cfg)
    assert (out / "best.pt").exists() and (out / "log.csv").exists()
    cfg["train"]["steps"] = 4
    run(cfg)
    st = torch.load(out / "last.pt", weights_only=False)
    assert st["step"] == 4


def test_stop_at_pauses_and_speed_switches_run(setup):
    from diewolke.train import run
    tmp, over = setup
    small = ["model.base=16", "model.sar_ch=16", "model.heads=2", "train.batch=2", "train.workers=0",
             "train.val_every=100", "train.val_batches=1", "train.log_every=1", "train.amp=false",
             "train.warmup=1", "train.val_batch=2", "train.steps=6", "train.channels_last=true", "name=pause"]
    cfg = load_config("configs/full.yaml", over + small + ["train.stop_at=3"])
    out = run(cfg)
    assert torch.load(out / "last.pt", weights_only=False)["step"] == 3
    cfg["train"]["stop_at"] = None
    run(cfg)
    assert torch.load(out / "last.pt", weights_only=False)["step"] == 6


def test_two_stage_uses_the_trained_masker(setup):
    from diewolke.engine import Pipeline
    from diewolke.train import run
    tmp, over = setup
    runs = tmp / "runs"
    assert (runs / "masker" / "best.pt").exists(), "runs after test_training_runs_and_resumes[masker]"
    cfg = load_config("configs/two_stage.yaml", over + [
        f"model.masker_ckpt={runs / 'masker'}", "model.base=16", "model.sar_ch=16", "model.heads=2",
        "train.steps=2", "train.batch=2", "train.workers=0", "train.val_every=2", "train.val_batches=1",
        "train.amp=false", "train.warmup=1", "train.val_batch=2"])
    out = run(cfg)
    pipe = Pipeline.from_run(out)
    assert pipe.mask_source == "masker" and pipe.masker is not None
    b = {"cloudy": torch.rand(1, 13, 32, 32), "s1": torch.rand(1, 2, 32, 32), "mask": torch.zeros(1, 1, 32, 32)}
    res = pipe.predict(b)
    assert res["mask"] is not None and res["mean"].shape == (1, 13, 32, 32)


def test_evaluate_predict_and_xgb(setup, sen12_root):
    from baselines import xgb_reconstruct
    from diewolke import evaluate, predict
    tmp, over = setup
    runs = tmp / "runs"
    ev = tmp / "eval"
    xgb = ev / "pred_xgb.npy"
    ev.mkdir(exist_ok=True)
    xgb_reconstruct.main(["--cache", str(tmp / "cache"), "--split", "test", "--out", str(xgb), "--n_estimators", "5"])
    evaluate.main(["--runs", str(runs / "smoke"), str(runs / "dsen2cr"), "--preds", f"xgb={xgb}",
                   "--cache", str(tmp / "cache"), "--split", "test", "--workers", "0", "--batch", "2",
                   "--save_predictions", "--out", str(ev)])
    evaluate.main(["--runs", str(runs / "smoke"), "--masker", str(runs / "masker"), "--mask_source", "masker",
                   "--cache", str(tmp / "cache"), "--split", "test", "--workers", "0", "--batch", "2",
                   "--calibrate_on", "", "--out", str(ev / "masker_masks")])
    assert "smoke+masker_mask" in json.loads((ev / "masker_masks" / "summary.json").read_text())
    summary = json.loads((ev / "summary.json").read_text())
    assert {"input", "smoke", "dsen2cr", "xgb"} <= set(summary)
    assert summary["smoke"]["uncertainty"]["temperature"] is not None
    pred = np.load(ev / "pred_smoke.npy")
    assert pred.shape == (3, 13, 64, 64)
    md = (ev / "report.md").read_text()
    assert "Paired differences" in md and "Uncertainty: smoke" in md

    out = tmp / "recon"
    predict.main(["--run", str(runs / "smoke"), "--data_root", str(sen12_root), "--out", str(out), "--limit", "2",
                  "--tile", "32", "--overlap", "8", "--temperature", str(ev / "summary.json")])
    import rasterio
    files = sorted(out.glob("*_recon.tif"))
    assert len(files) == 2 and len(list(out.glob("*_uncertainty.tif"))) == 2
    with rasterio.open(files[0]) as f:
        assert f.count == 13 and f.crs.to_epsg() == 32643 and f.width == 64


def test_kharif_benchmark_and_crop_classification(tmp_path):
    import build_benchmark
    import crop_classification
    import gee_common as g
    import yaml
    raw = make_kharif_raw(tmp_path / "raw")
    bench = tmp_path / "bench"
    cfg = g.load_cfg()
    cfg.update(raw_dir=str(raw), benchmark_dir=str(bench))
    (tmp_path / "k.yaml").write_text(yaml.safe_dump(cfg))
    build_benchmark.main(["--config", str(tmp_path / "k.yaml"), "--pair_days", "40"])
    meta = json.loads((bench / "kharif_synth_meta.json").read_text())
    assert len(meta) == 6 and all("donor_s2_id" in m for m in meta)
    mask = np.load(bench / "kharif_synth_mask.npy")
    assert mask.shape == (6, 64, 64) and mask.max() == 1
    crop_classification.main(["--cache", str(bench), "--split", "kharif_synth", "--out", str(tmp_path / "crops"),
                              "--min_class_pixels", "10", "--test_frac", "0.4"])
    res = json.loads((tmp_path / "crops" / "crop_classification.json").read_text())
    assert res["results"]["gt"]["all"]["oa"] >= res["results"]["cloudy"]["all"]["oa"] - 0.05


def test_gee_retry_waits_on_rate_limits_only(monkeypatch):
    import gee_common as g
    import pytest
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    calls = []

    def busy_twice():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("Too many concurrent aggregations.")
        return "ok"
    assert g.retry(busy_twice) == "ok" and len(calls) == 3
    with pytest.raises(ValueError):
        g.retry(lambda: (_ for _ in ()).throw(ValueError("bad band")))  # not a rate limit: raised at once


def test_grid_helpers():
    import gee_common as g
    assert g.utm_epsg(75.85, 30.9) == "EPSG:32643"
    cells = g.grid_cells((75.8, 30.8, 75.9, 30.9), "EPSG:32643", 2560)
    assert len(cells) > 10 and all(x % 2560 == 0 and y % 2560 == 0 for x, y in cells)
