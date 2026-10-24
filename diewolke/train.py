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
    # speed switches (train.cudnn_benchmark, train.tf32, train.precision, train.channels_last, train.compile)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = t_cfg.get("cudnn_benchmark", True)  # fixed crop size -> pick fastest conv kernels
        if t_cfg.get("tf32", True):  # Ampere+: TF32 for the fp32 parts (A4000 yes, T4 ignores it)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    channels_last = bool(t_cfg.get("channels_last", False))
    model = build_model(cfg).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    n_params = count_parameters(model)
    print(f"[+] {cfg['model']['type']} with {n_params / 1e6:.2f}M parameters; "
          f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val samples")
    # fused AdamW: one CUDA kernel for the whole update (same maths, less launch overhead)
    opt = torch.optim.AdamW(model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg.get("weight_decay", 1e-4),
                            betas=(0.9, 0.99), fused=device.type == "cuda" and t_cfg.get("fused_adam", True))
    precision = t_cfg.get("precision", "fp16") if t_cfg.get("amp", True) else "fp32"
    amp = device.type == "cuda" and precision in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    # bf16 has fp32's range, so it needs no loss scaling (Ampere and newer, e.g. the A4000; not the T4)
    scaler = torch.amp.GradScaler(device.type, enabled=amp and precision == "fp16")
    ema = EMA(model, t_cfg.get("ema", 0.999)) if t_cfg.get("ema", 0.999) else None
    # torch.compile wraps the model for training only; checkpoints and EMA keep the plain module
    train_model = torch.compile(model) if t_cfg.get("compile", False) else model
    pipe = Pipeline(cfg, train_model, device=device)
    print(f"[+] precision {precision if amp else 'fp32'}, channels_last {channels_last}, "
          f"compile {bool(t_cfg.get('compile', False))}")

    step, best = 0, -float("inf")
    if (out / "last.pt").exists() and not t_cfg.get("restart", False):
        st = load_checkpoint(out / "last.pt", device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if scaler.is_enabled() and st.get("scaler"):  # also lets a run switch between fp16 and bf16
            scaler.load_state_dict(st["scaler"])
        if ema is not None and st.get("ema") is not None:
            ema.shadow.load_state_dict(st["ema"])
        step, best = st["step"], st.get("best", best)
        print(f"[+] resumed from step {step}")
    dump_config(cfg, out / "config.yaml")
    logger = CSVLogger(out / "log.csv")

    def checkpoint(name):
        # a diverged model must not overwrite the last good checkpoint (it would be resumed from)
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise SystemExit(f"[!] weights became NaN/inf before step {step}; stopping. {out / 'last.pt'} "
                             "keeps the last good state; lower train.lr or check the loss before resuming")
        # best.pt is for evaluation only, so it skips the optimizer state (about half the file size;
        # matters where output space is limited, e.g. Kaggle). Resuming always uses last.pt.
        extra = {} if name == "best.pt" else {"opt": opt.state_dict(), "scaler": scaler.state_dict()}
        save_checkpoint(out / name, model=model.state_dict(), ema=ema.shadow.state_dict() if ema else None,
                        step=step, best=best, config=cfg, n_params=n_params, **extra)

    accum = t_cfg.get("accum", 1)
    # pause at this step; the learning-rate schedule still runs over train.steps
    stop_at = min(t_cfg.get("stop_at") or t_cfg["steps"], t_cfg["steps"])
    timer, t_last, running = Timer(), time.time(), {}
    # time spent waiting for the next batch (CPU data loading) vs. the whole step; float(loss) below
    # synchronises with the GPU every step, so the split is accurate
    wait, t_ready = 0.0, time.time()
    model.train()
    stop = False
    skipped = 0  # optimizer steps skipped for a non-finite gradient since the last log line

    def to_device(v):
        if channels_last and v.dim() == 4:
            return v.to(device, non_blocking=True, memory_format=torch.channels_last)
        return v.to(device, non_blocking=True)

    interrupted = False
    try:
        while step < stop_at and not stop:
            for b in train_loader:
                wait += time.time() - t_ready
                b = {k: to_device(v) for k, v in b.items()}
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, t_cfg)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
                    loss, terms = pipe.training_loss(b)
                scaler.scale(loss / accum).backward()
                if (step + 1) % accum == 0:
                    scaler.unscale_(opt)
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.get("grad_clip", 1.0))
                    # never apply a NaN/inf update (fp16 already skips these; fp32/bf16 would not)
                    if torch.isfinite(gnorm):
                        scaler.step(opt)
                    else:
                        skipped += 1
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    if ema is not None:
                        ema.update(model, step)
                step += 1
                for k, v in {"loss": float(loss.detach()), **terms}.items():
                    running[k] = running.get(k, 0.0) + v
                every = t_cfg.get("log_every", 100)
                if step % every == 0 or step == stop_at:
                    n = step % every or every
                    sec = (time.time() - t_last) / n
                    t_last = time.time()
                    data_wait = min(1.0, wait / max(1e-9, sec * n))
                    wait = 0.0
                    row = {"step": step, "lr": opt.param_groups[0]["lr"], "sec_per_step": sec, "data_wait": data_wait,
                           "skipped": skipped / n, **{k: v / n for k, v in running.items()}}
                    running, skipped = {}, 0
                    eta_min = sec * (stop_at - step) / 60
                    print(f"  step {step}/{t_cfg['steps']} " + " ".join(f"{k} {v:.4f}" for k, v in row.items()
                                                                        if k != "step") + f" | ~{eta_min:.0f} min left",
                          flush=True)
                    if data_wait > 0.2 and step > 2 * every:
                        print(f"  [!] {100 * data_wait:.0f}% of each step is spent waiting for data; "
                              "raise train.workers or keep the cache on a faster disk / in RAM")
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
                t_ready = time.time()  # validation/checkpoint time is not counted as data wait
                if step >= stop_at or stop:
                    break
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[!] interrupted at step {step}; saving a checkpoint, rerun the same command to resume")
    checkpoint("last.pt")
    if not (out / "best.pt").exists():
        checkpoint("best.pt")
    if step < t_cfg["steps"] and not interrupted and not stop:
        print(f"[+] paused at train.stop_at={stop_at}; rerun without stop_at (or with a later one) to continue")
    print(f"[+] done: step {step}/{t_cfg['steps']}, best val score {best:.4f}, {timer.minutes():.1f} min this session")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--overlay", action="append", default=[], help="platform config merged over --config "
                                                                   "(e.g. kaggle/kaggle.yaml); repeatable")
    args = ap.parse_args(argv)
    return run(load_config(args.config, args.set, args.overlay))


if __name__ == "__main__":
    main()
