"""Seeding, EMA, checkpointing (safe on Google Drive), and a CSV logger."""
import copy
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class EMA:
    """Exponential moving average of model weights (used for evaluation)."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step=None):
        d = self.decay if step is None else min(self.decay, (1 + step) / (10 + step))
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(p.detach(), 1 - d)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


def save_checkpoint(path, **state):
    """Write to a temp file then rename, so a Colab disconnect never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class CSVLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.fields = None
        if self.path.exists():
            with open(self.path) as f:
                header = f.readline().strip()
                self.fields = header.split(",") if header else None

    def log(self, row):
        row = {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()}
        if self.fields is None:
            self.fields = list(row)
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, self.fields).writeheader()
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, self.fields, extrasaction="ignore").writerow(row)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def minutes(self):
        return (time.time() - self.t0) / 60
