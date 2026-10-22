"""External baselines. The GLF-CR layer replacement is always tested; the full adapters run only
when the repositories are cloned and their paths are set:
    DIEWOLKE_UNCRTAINTS_REPO=/opt/UnCRtainTS DIEWOLKE_GLFCR_REPO=/opt/GLF-CR python -m pytest tests/test_external.py
(random weights; this checks the plumbing, not the published numbers)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import kernelconv2d_torch  # noqa: E402

UNCR = os.environ.get("DIEWOLKE_UNCRTAINTS_REPO")
GLF = os.environ.get("DIEWOLKE_GLFCR_REPO")


def test_kernelconv2d_matches_the_cuda_formula():
    torch.manual_seed(0)
    b, c, h, w, k = 2, 3, 6, 7, 5
    x, ker = torch.randn(b, c, h, w), torch.randn(b, c * k * k, h, w)
    pad = torch.nn.ReplicationPad2d([2, 2, 2, 2])(x)
    ref = torch.zeros(b, c, h, w)
    for ch in range(c):
        for i in range(k):
            for j in range(k):
                ref[:, ch] += pad[:, ch, i:i + h, j:j + w] * ker[:, ch * k * k + i * k + j]
    assert torch.allclose(kernelconv2d_torch.KernelConv2D(k)(x, ker), ref, atol=1e-5)


def test_s1_back_to_db():
    from run_external import s1_to_db
    s1 = torch.tensor([0.0, 1.0]).view(1, 2, 1, 1).expand(1, 2, 1, 2).clone()
    s1[0, :, 0, 1] = torch.tensor([1.0, 0.0])
    db = s1_to_db(s1)
    assert torch.allclose(db[0, :, 0, 0], torch.tensor([-25.0, 0.0]))
    assert torch.allclose(db[0, :, 0, 1], torch.tensor([0.0, -32.5]))


@pytest.mark.skipif(not UNCR, reason="set DIEWOLKE_UNCRTAINTS_REPO to a clone of PatrickTUM/UnCRtainTS")
def test_uncrtaints_adapter_writes_predictions_and_scales(tmp_path):
    import run_external
    run_external.main(["uncrtaints", "--repo", UNCR, "--fake", "--device", "cpu", "--out", str(tmp_path / "pred_u.npy")])
    pred, scale = np.load(tmp_path / "pred_u.npy"), np.load(tmp_path / "scale_u.npy")
    assert pred.shape == scale.shape == (4, 13, 64, 64) and np.isfinite(pred).all() and (scale >= 0).all()


@pytest.mark.skipif(not GLF, reason="set DIEWOLKE_GLFCR_REPO to a clone of xufangchn/GLF-CR")
def test_glfcr_adapter_tiles_full_patches(tmp_path):
    import run_external
    run_external.main(["glfcr", "--repo", GLF, "--fake", "--device", "cpu", "--batch", "2", "--limit", "1",
                       "--out", str(tmp_path / "pred_g.npy")])
    pred = np.load(tmp_path / "pred_g.npy")
    assert pred.shape == (4, 13, 256, 256) and np.isfinite(pred).all()
