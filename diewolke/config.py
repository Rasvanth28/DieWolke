"""YAML configs with `base:` inheritance and dotted command-line overrides.

    cfg = load_config("configs/full.yaml", ["train.steps=2000", "data.root=/content/data"])
"""
import copy
from pathlib import Path

import yaml


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    bases = cfg.pop("base", None)
    if bases is None:
        return cfg
    merged = {}
    for b in [bases] if isinstance(bases, str) else bases:
        merged = deep_merge(merged, _load(path.parent / b))
    return deep_merge(merged, cfg)


def apply_overrides(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override {item!r} is not key=value")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        value = yaml.safe_load(raw)
        if isinstance(value, str):  # YAML 1.1 reads "5e-4" as a string
            try:
                value = float(value)
            except ValueError:
                pass
        node[parts[-1]] = value
    return cfg


def load_config(path, overrides=None):
    cfg = apply_overrides(_load(path), overrides)
    cfg.setdefault("name", Path(path).stem)
    return cfg


def dump_config(cfg, path):
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))
