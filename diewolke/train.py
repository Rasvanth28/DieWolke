"""Train the masker, the bridge reconstructor or the DSen2-CR baseline from a config.

    python -m diewolke.train --config configs/full.yaml --set data.root=/content/data/sen12ms_cr
    python -m diewolke.train --config configs/smoke.yaml          # random data, CPU, ~1 min

The run directory (train.out_dir / name) gets config.yaml, log.csv, last.pt and best.pt.
Re-running the same command resumes from last.pt, so a Colab disconnect only costs the steps
since the last checkpoint. train.max_minutes stops cleanly (with a checkpoint) before a
session limit.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from .config import dump_config, load_config
from .data import make_loader, prepare_caches
from .engine import Pipeline
from .metrics import image_metrics
from .models import build_model
from .utils import CSVLogger, EMA, Timer, count_parameters, get_device, load_checkpoint, save_checkpoint, seed_everything


def lr_at(step, cfg_t):
    warm, total, lr = cfg_t.get("warmup", 500), cfg_t["steps"], cfg_t["lr"]
    if step < warm:
        return lr * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return lr * (cfg_t.get("min_lr_frac", 0.05) + (1 - cfg_t.get("min_lr_frac", 0.05)) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def validate(pipe, loader, device, max_batches):
    """Quick validation: mean PSNR / NDVI RMSE for reconstructors, IoU for the masker."""
    pipe.model.eval()
    scores = {}
    inter = union = 0.0
    for bi, b in enumerate(loader):
        if bi >= max_batches:
            break
        b = {k: v.to(device) for k, v in b.items()}
        out = pipe.predict(b)
        if pipe.kind == "masker":
            pred = (out["mask_prob"] > 0.5).float()
            inter += (pred * b["mask"]).sum().item()
            union += ((pred + b["mask"]) > 0).sum().item()
            continue
        m = image_metrics(out["mean"], b["clear"], b["mask"])
        for k in ("psnr", "ssim", "mae", "ndvi_rmse", "psnr_cloud"):
            scores.setdefault(k, []).append(m[k].cpu().numpy())
    pipe.model.train()
    if pipe.kind == "masker":
        iou = inter / max(union, 1)
        return {"val_iou": iou, "score": iou}
    res = {f"val_{k}": float(np.nanmean(np.concatenate(v))) for k, v in scores.items()}
    res["score"] = res["val_psnr"]
    return res


def run(cfg):
    t_cfg = cfg["train"]
    seed_everything(t_cfg.get("seed", 0))
    device = get_device(t_cfg.get("device", "auto"))
    out = Path(t_cfg.get("out_dir", "runs")) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    print(f"[+] run {cfg['name']} -> {out}  (device {device})")

    caches = prepare_caches(cfg, splits=("train", "val"))
    workers = t_cfg.get("workers", 2)
    train_loader = make_loader(caches["train"], cfg["data"], "train", t_cfg["batch"], workers)
    val_loader = make_loader(caches["val"], cfg["data"], "val", t_cfg.get("val_batch", 8), 0,
                             mode=cfg["data"].get("val_mode", cfg["data"].get("mode", "real")))

    if len(train_loader) == 0:
        raise SystemExit(f"training set ({len(train_loader.dataset)} samples) is smaller than one batch")
    model = build_model(cfg).to(device)
    pipe = Pipeline(cfg, model, device=device)
    n_params = count_parameters(model)
    print(f"[+] {cfg['model']['type']} with {n_params / 1e6:.2f}M parameters; "
          f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val samples")
    opt = torch.optim.AdamW(model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg.get("weight_decay", 1e-4),
                            betas=(0.9, 0.99))
    amp = device.type == "cuda" and t_cfg.get("amp", True)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    ema = EMA(model, t_cfg.get("ema", 0.999)) if t_cfg.get("ema", 0.999) else None

    step, best = 0, -float("inf")
    if (out / "last.pt").exists() and not t_cfg.get("restart", False):
        st = load_checkpoint(out / "last.pt", device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        if ema is not None and st.get("ema") is not None:
            ema.shadow.load_state_dict(st["ema"])
        step, best = st["step"], st.get("best", best)
        print(f"[+] resumed from step {step}")
    dump_config(cfg, out / "config.yaml")
    logger = CSVLogger(out / "log.csv")

    def checkpoint(name):
        save_checkpoint(out / name, model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                        ema=ema.shadow.state_dict() if ema else None, step=step, best=best, config=cfg,
                        n_params=n_params)

    accum = t_cfg.get("accum", 1)
    timer, t_last, running = Timer(), time.time(), {}
    model.train()
    stop = False
    while step < t_cfg["steps"] and not stop:
        for b in train_loader:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            for g in opt.param_groups:
                g["lr"] = lr_at(step, t_cfg)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss, terms = pipe.training_loss(b)
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.get("grad_clip", 1.0))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model, step)
            step += 1
            for k, v in {"loss": float(loss.detach()), **terms}.items():
                running[k] = running.get(k, 0.0) + v
            every = t_cfg.get("log_every", 100)
            if step % every == 0 or step == t_cfg["steps"]:
                n = step % every or every
                sec = (time.time() - t_last) / n
                t_last = time.time()
                row = {"step": step, "lr": opt.param_groups[0]["lr"], "sec_per_step": sec,
                       **{k: v / n for k, v in running.items()}}
                running = {}
                eta_min = sec * (t_cfg["steps"] - step) / 60
                print(f"  step {step}/{t_cfg['steps']} " + " ".join(f"{k} {v:.4f}" for k, v in row.items()
                                                                    if k not in ("step",)) + f" | ~{eta_min:.0f} min left",
                      flush=True)
                logger.log(row)
            if step % t_cfg.get("val_every", 1000) == 0 or step == t_cfg["steps"]:
                eval_pipe = Pipeline(cfg, ema.shadow if ema else model, masker=pipe.masker, device=device)
                res = validate(eval_pipe, val_loader, device, t_cfg.get("val_batches", 25))
                improved = res["score"] > best
                best = max(best, res["score"])
                print("  [val] " + " ".join(f"{k} {v:.4f}" for k, v in res.items()) + (" *" if improved else ""))
                logger.log({"step": step, **res})
                checkpoint("last.pt")
                if improved:
                    checkpoint("best.pt")
            elif step % t_cfg.get("ckpt_every", 500) == 0:
                checkpoint("last.pt")
            if t_cfg.get("max_minutes") and timer.minutes() > t_cfg["max_minutes"]:
                print(f"[!] max_minutes reached at step {step}; checkpoint saved, rerun to resume")
                stop = True
            if step >= t_cfg["steps"] or stop:
                break
    checkpoint("last.pt")
    if not (out / "best.pt").exists():
        checkpoint("best.pt")
    print(f"[+] done: step {step}, best val score {best:.4f}, {timer.minutes():.1f} min this session")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = ap.parse_args(argv)
    return run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
