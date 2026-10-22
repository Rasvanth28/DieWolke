"""Kaggle helpers: overlay, cache detection, the run queue (resume and skip) and session restore."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))
import kaggle_tools as kt  # noqa: E402
from diewolke.config import load_config  # noqa: E402
from diewolke.data.sen12mscr import cache_ready  # noqa: E402


def test_overlay_sets_paths_but_keeps_run_name():
    cfg = load_config(ROOT / "configs/no_bga.yaml", ["train.steps=7"], [str(ROOT / "kaggle/kaggle.yaml")])
    assert cfg["name"] == "no_bga" and cfg["model"]["use_bga"] is False
    assert cfg["train"]["out_dir"] == "/kaggle/working/runs" and cfg["train"]["precision"] == "fp16"
    assert cfg["train"]["steps"] == 7  # overrides win over the overlay


def test_parse_spec():
    assert kt.parse_spec("full@40000") == {"config": "full", "name": "full", "steps": 40000}
    assert kt.parse_spec("full:full_20k@20000") == {"config": "full", "name": "full_20k", "steps": 20000}
    assert kt.parse_spec("no_bga")["steps"] is None


def test_find_cache_and_ready_without_markers(tmp_path):
    d = tmp_path / "input" / "my-dataset" / "cache"
    d.mkdir(parents=True)
    for k in ("s1", "clear", "cloudy", "prob", "prob_clear", "shadow"):
        np.save(d / f"train_{k}.npy", np.zeros(1))
    assert kt.find_cache(tmp_path / "input") == str(d)
    assert cache_ready(d, "train") and not cache_ready(d, "val")  # no train.done marker needed


def test_queue_runs_resumes_skips_and_restore(tmp_path, capsys):
    out = tmp_path / "working" / "runs"
    common = ["--cache", str(tmp_path), "--out_dir", str(out), "--gpus", "0", "--workers", "0",
              "--set", "train.val_every=3", "train.val_batches=1", "train.log_every=1", "train.warmup=1",
              "train.amp=false"]
    kt.main(["queue", "smoke@3", "smoke:smoke_b@2", *common])
    assert kt.run_step(out / "smoke") == 3 and kt.run_step(out / "smoke_b") == 2
    best = torch.load(out / "smoke" / "best.pt", weights_only=False)
    assert "opt" not in best and "model" in best  # weights-only best.pt
    kt.main(["queue", "smoke@5", *common])  # longer target: resumes from 3
    assert kt.run_step(out / "smoke") == 5
    kt.main(["queue", "smoke@5", *common])
    assert "finished (5/5), skipped" in capsys.readouterr().out
    # next session: earlier output appears under /kaggle/input/<version>/runs
    new = tmp_path / "session2" / "runs"
    kt.main(["restore", "--src", str(tmp_path / "working"), "--dst", str(new)])
    assert kt.run_step(new / "smoke") == 5 and kt.run_step(new / "smoke_b") == 2


def test_queue_stops_when_session_time_is_up(tmp_path, capsys):
    import time
    out = tmp_path / "runs"
    kt.main(["queue", "smoke@3", "--cache", str(tmp_path), "--out_dir", str(out), "--gpus", "0",
             "--session_start", str(time.time() - 11.9 * 3600)])
    assert "min left in this session" in capsys.readouterr().out
    assert kt.run_step(out / "smoke") == 0


def test_publish_dry_run_writes_metadata(tmp_path):
    np.save(tmp_path / "train_clear.npy", np.zeros(3))
    kt.main(["publish", "--dir", str(tmp_path), "--user", "someone", "--dry_run"])
    meta = json.loads((tmp_path / "dataset-metadata.json").read_text())
    assert meta["id"] == "someone/diewolke-cache"


def test_prepare_streams_archives_and_keeps_only_selected_patches(tmp_path):
    import tarfile
    from conftest import make_sen12mscr
    src = make_sen12mscr(tmp_path / "src")
    arch = tmp_path / "server"
    arch.mkdir()
    for mod in ("s1", "s2", "s2_cloudy"):
        with tarfile.open(arch / f"ROIs1158_spring_{mod}.tar.gz", "w:gz") as t:
            t.add(src / f"ROIs1158_spring_{mod}", arcname=f"ROIs1158_spring_{mod}")
    data, cache = tmp_path / "data", tmp_path / "cache"
    kt.main(["prepare", "--base_url", arch.as_uri(), "--data", str(data), "--out", str(cache),
             "--n_train", "3", "--n_test", "2", "--spare", "0", "--set", "data.n_val=1"])
    # train/val/test limits from the official split: train 3 of 8, val 1 of 2, test 2 of 3 (+ margin)
    s2 = list(data.rglob("ROIs1158_spring_s2_*_p*.tif"))
    s2 = [p for p in s2 if "cloudy" not in p.name]
    assert 0 < len(s2) < 13  # only the selected patches were extracted from the S2 archive
    s1 = list(data.rglob("ROIs1158_spring_s1_*.tif"))
    assert len(s1) == len(s2)  # unused S1 files were removed
    from diewolke.data import Cache
    assert len(Cache.load(cache, "train")) == 3 and len(Cache.load(cache, "test")) == 2
    assert len(Cache.load(cache, "val")) == 1
    assert not list(data.rglob("*.tar.gz"))  # archives were never stored
