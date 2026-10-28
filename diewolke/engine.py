"""Glue between models, the bridge process and batches: training steps and inference."""
from pathlib import Path

import torch

from .bridge import BrownianBridge
from .config import load_config
from .losses import bridge_loss, carl_loss, masker_loss
from .models import build_model
from .utils import load_checkpoint


def load_run(path, device="cpu", which="best", use_ema=True):
    """Load a trained run. path: a run directory (with config.yaml and checkpoints) or a .pt file."""
    path = Path(path)
    if path.is_dir():
        ckpt = path / f"{which}.pt"
        if not ckpt.exists():
            ckpt = path / "last.pt"
    else:
        ckpt = path
    state = load_checkpoint(ckpt, device)
    cfg = state["config"]
    model = build_model(cfg).to(device)
    weights = state.get("ema") if use_ema and state.get("ema") is not None else state["model"]
    model.load_state_dict(weights)
    model.eval()
    return cfg, model, state


class Pipeline:
    """Everything needed to turn a batch into a reconstruction for one trained configuration."""

    def __init__(self, cfg, model, masker=None, device="cpu"):
        self.cfg, self.model, self.device = cfg, model, device
        self.kind = cfg["model"]["type"]
        m = cfg["model"]
        self.mask_source = m.get("mask_source", "data")
        self.cond_kind = m.get("cond", "masked")
        self.mask_threshold = m.get("mask_threshold", 0.5)
        b = cfg.get("bridge", {})
        self.bridge = BrownianBridge(b.get("sigma", 0.5), b.get("t_min", 1e-3), b.get("p_t1", 0.25))
        self.source = b.get("source", "cloudy")
        self.masker = masker
        if self.mask_source == "masker" and masker is None:
            path = m.get("masker_ckpt")
            if not path:
                raise ValueError("model.mask_source=masker needs model.masker_ckpt")
            _, self.masker, _ = load_run(path, device)
        if self.masker is not None:
            self.masker.eval()
            for p in self.masker.parameters():
                p.requires_grad_(False)

    @classmethod
    def from_run(cls, path, device="cpu", masker=None, **kw):
        cfg, model, _ = load_run(path, device, **kw)
        return cls(cfg, model, masker=masker, device=device)

    # ------------------------------------------------------------------ inputs
    def get_mask(self, batch):
        if self.mask_source == "none":
            return None
        if self.mask_source == "masker":
            with torch.no_grad():
                return (torch.sigmoid(self.masker(batch["cloudy"])) > self.mask_threshold).float()
        return batch["mask"]

    def condition(self, batch, mask):
        if mask is None or self.cond_kind == "cloudy":
            return batch["cloudy"]
        return batch["cloudy"] * (1 - mask)

    # ------------------------------------------------------------------ training
    def training_loss(self, batch):
        cfg_loss = self.cfg.get("loss", {})
        if self.kind == "masker":
            return masker_loss(self.model(batch["cloudy"]), batch, cfg_loss.get("dice", 0.5))
        if self.kind == "dsen2cr":
            return carl_loss(self.model(batch["cloudy"], batch["s1"]), batch, cfg_loss.get("carl_lambda", 1.0))
        mask = self.get_mask(batch)
        cond = self.condition(batch, mask)
        y = cond if self.source == "masked" else batch["cloudy"]
        t = self.bridge.sample_t(y.shape[0], y.device)
        x_t = self.bridge.q_sample(batch["clear"], y, t)
        out = self.model(x_t, t, cond, batch["s1"], mask)
        # the loss always weights by the data mask (exact for synthetic clouds), not the input mask
        return bridge_loss(out, batch, cfg_loss)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(self, batch, steps=None, eta=None, n_samples=1, generator=None):
        """Returns {"mean": (B,13,H,W), "scale": Laplace b or None, "mask": used mask or None,
        "sample_std": spread over n_samples stochastic samples or None}."""
        if self.kind == "masker":
            logits = self.model(batch["cloudy"])
            return {"mask_prob": torch.sigmoid(logits)}
        if self.kind == "dsen2cr":
            return {"mean": self.model(batch["cloudy"], batch["s1"]), "scale": None,
                    "mask": batch.get("mask"), "sample_std": None}
        ev = self.cfg.get("eval", {})
        steps = ev.get("steps", 1) if steps is None else steps
        eta = ev.get("eta", 0.0) if eta is None else eta
        mask = self.get_mask(batch)
        cond = self.condition(batch, mask)
        y = cond if self.source == "masked" else batch["cloudy"]

        def fn(x, t):
            return self.model(x, t, cond, batch["s1"], mask)

        outs = [self.bridge.sample(fn, y, steps, eta, generator) for _ in range(max(1, n_samples))]
        means = torch.stack([o["mean"] for o in outs])
        mean = means.mean(0)
        scale = None
        if outs[0].get("log_scale") is not None:
            scale = torch.stack([o["log_scale"].exp() for o in outs]).mean(0)
        return {"mean": mean, "scale": scale, "mask": mask,
                "sample_std": means.std(0) if len(outs) > 1 else None}


def load_config_or_run(path, overrides=None):
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        return load_config(path, overrides)
    return load_checkpoint(path if path.is_file() else path / "last.pt")["config"]
