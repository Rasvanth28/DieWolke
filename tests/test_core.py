"""Unit tests: bridge maths, metrics, calibration, models, configs."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from diewolke import calibration as cal
from diewolke.bands import BAND_GROUPS, N_S2, ndvi
from diewolke.bridge import BrownianBridge
from diewolke.config import load_config
from diewolke.metrics import image_metrics, paired_difference, ssim
from diewolke.models import BridgeNet, build_model
from diewolke.models.bga import GroupHeads

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))


def test_bridge_marginal_and_reverse_step_are_consistent():
    torch.manual_seed(0)
    br = BrownianBridge(sigma=0.5)
    x0, y = torch.zeros(20000, 1, 1, 1), torch.ones(20000, 1, 1, 1)
    t, s = torch.full((20000,), 0.7), torch.full((20000,), 0.3)
    xt = br.q_sample(x0, y, t)
    assert abs(xt.mean() - 0.7) < 0.01
    assert abs(xt.std() - 0.5 * math.sqrt(0.7 * 0.3)) < 0.01
    xs = br.step(xt, x0, t, s, eta=1.0)  # with the true x0, the step must reproduce q(x_s)
    assert abs(xs.mean() - 0.3) < 0.01
    assert abs(xs.std() - 0.5 * math.sqrt(0.3 * 0.7)) < 0.01


def test_bridge_t1_is_the_cloudy_image_and_single_step_returns_prediction():
    br = BrownianBridge()
    x0, y = torch.rand(2, 13, 8, 8), torch.rand(2, 13, 8, 8)
    assert torch.allclose(br.q_sample(x0, y, torch.ones(2)), y)
    seen = []

    def fn(x, t):
        seen.append(float(t[0]))
        return {"mean": x * 0 + 0.5, "log_scale": None}
    out = br.sample(fn, y, steps=1)
    assert seen == [1.0] and torch.allclose(out["mean"], torch.full_like(y, 0.5))
    seen.clear()
    br.sample(fn, y, steps=4)
    assert seen == [1.0, 0.75, 0.5, 0.25]


def test_metrics_basic_values():
    gt = torch.rand(3, 13, 32, 32) * 0.5 + 0.1
    m = image_metrics(gt.clone(), gt)
    assert torch.all(m["psnr"] > 90) and torch.allclose(m["ssim"], torch.ones(3), atol=1e-4)
    assert torch.all(m["sam"] < 0.1) and torch.all(m["mae"] == 0)
    pred = (gt + 0.1).clamp(0, 1)
    mse = ((pred - gt) ** 2).mean((1, 2, 3))
    assert torch.allclose(image_metrics(pred, gt)["psnr"], -10 * torch.log10(mse), atol=1e-3)
    mask = torch.zeros(3, 1, 32, 32)
    mask[0, :, :8] = 1
    mm = image_metrics(pred, gt, mask)
    assert not torch.isnan(mm["psnr_cloud"][0]) and torch.isnan(mm["psnr_cloud"][1])
    assert ssim(gt, gt * 0.5).mean() < 0.99


def test_ndvi_range_and_sign():
    x = torch.zeros(1, 13, 2, 2)
    x[:, 7], x[:, 3] = 0.4, 0.1  # NIR > red: vegetation
    assert torch.allclose(ndvi(x), torch.full((1, 2, 2), 0.6), atol=1e-4)


def test_ndvi_loss_gradient_is_bounded_on_dark_pixels():
    # dark and no-data target pixels with a near-zero prediction gave ~1/1e-6 gradients before
    from diewolke.losses import bridge_loss
    target = torch.full((1, 13, 4, 4), 0.3)
    target[..., :2, :] = 0.0  # no-data rows
    target[..., 2, :] = 0.01  # water: NIR + red = 0.02
    mean = torch.full_like(target, 1e-4).requires_grad_()
    _, terms = bridge_loss({"mean": mean, "log_scale": None}, {"clear": target},
                           {"ndvi": 1.0, "uncertainty": 0.0})
    (g,) = torch.autograd.grad(_ndvi_term(mean, target), mean)
    assert torch.isfinite(g).all() and g.abs().max() <= 2 / 0.1 / 4 + 1e-6  # <= 2/floor per valid pixel
    assert g[..., :3, :].abs().max() == 0  # dark rows are left out entirely
    assert math.isfinite(terms["ndvi"])


def _ndvi_term(mean, target):
    from diewolke.losses import bridge_loss
    total, _ = bridge_loss({"mean": mean, "log_scale": None}, {"clear": target}, {"ndvi": 1.0, "uncertainty": 0.0})
    return total - (mean - target).abs().mean()  # the index part only


def test_paired_difference():
    d = paired_difference(np.array([2.0, 3.0, 4.0]), np.array([1.0, 1.0, 1.0]), True)
    assert d["mean"] == pytest.approx(2.0) and d["frac_images_better"] == 1.0


def test_calibration_on_true_laplace_errors():
    rng = np.random.default_rng(0)
    b = rng.uniform(0.01, 0.1, (50000, 13))
    err = np.abs(rng.laplace(0, b))
    assert cal.calibration_error(err, b) < 0.01
    assert np.allclose(cal.fit_temperature(err, b / 3), 3, rtol=0.05)  # recovers a 3x under-estimate
    s = cal.summarize(err, b)
    assert s["spearman_uncertainty_error"] > 0.2 and s["sparsification"]["ause"] >= 0


@pytest.mark.parametrize("use_bga", [True, False])
def test_bridgenet_shapes_and_zero_init(use_bga):
    m = BridgeNet(base=16, sar_ch=16, heads=2, use_bga=use_bga)
    x, c = torch.rand(2, 13, 32, 32), torch.rand(2, 13, 32, 32)
    out = m(x, torch.rand(2), c, torch.rand(2, 2, 32, 32), torch.rand(2, 1, 32, 32))
    assert out["mean"].shape == (2, 13, 32, 32) and out["log_scale"].shape == (2, 13, 32, 32)
    assert torch.allclose(out["mean"], c)  # residual on the condition, zero-initialised heads
    (out["mean"].sum() + out["log_scale"].sum()).backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_bga_and_plain_models_have_similar_size():
    a = sum(p.numel() for p in BridgeNet(base=48, use_bga=True).parameters())
    b = sum(p.numel() for p in BridgeNet(base=48, use_bga=False).parameters())
    assert abs(a - b) / b < 0.1


def test_group_heads_route_bands_to_the_right_channels():
    h = GroupHeads(4, 8, out_per_band=1)
    for (name, idx), head in zip(BAND_GROUPS.items(), h.heads):
        torch.nn.init.zeros_(head[-1].weight)
        torch.nn.init.constant_(head[-1].bias, float(len(name)))
    out = h(torch.rand(1, 4, 4, 4))
    for name, idx in BAND_GROUPS.items():
        assert torch.all(out[0, idx] == len(name))
    assert out.shape[1] == N_S2


def test_group_heads_under_autocast_with_fp32_input():
    # on CUDA, GroupNorm hands the heads fp32 features while the convs return fp16 (crashed on a Kaggle T4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = GroupHeads(4, 8)(torch.rand(1, 4, 4, 4))
    assert out.dtype == torch.bfloat16 and out.shape[1] == 2 * N_S2


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_every_config_builds(path):
    cfg = load_config(path)
    if cfg["model"]["type"] == "bridge":
        cfg["model"]["base"], cfg["model"]["sar_ch"] = 16, 16
    model = build_model(cfg)
    assert sum(p.numel() for p in model.parameters()) > 0
