"""Run published pretrained cloud-removal models on our cached split, for evaluate.py --preds.

Inference only, no training. Inputs come from the same CloudPairs dataset as evaluate.py (same
cloudy image, same mask, same order), so the saved arrays line up with our metrics.

UnCRtainTS (Ebel et al., CVPRW 2023; MIT licence)
    git clone https://github.com/PatrickTUM/UnCRtainTS /opt/UnCRtainTS
    pip install fvcore
    # checkpoints: pCloud link in the UnCRtainTS README; each folder has conf.json + model.pth.tar.
    # Use a folder whose conf.json has "pretrain": true (mono-temporal SEN12MS-CR). Folders such
    # as diagonal_1 are multi-temporal SEN12MS-CR-TS models and are refused unless
    # --allow_multitemporal (then they see one date only, which is not how they were trained).
    python baselines/run_external.py uncrtaints --repo /opt/UnCRtainTS --weights /opt/uncr_ckpt \
        --experiment <folder name> --cache /data/cache --split test --out EVAL/pred_uncrtaints.npy

GLF-CR (Xu et al., ISPRS 2022; the repository has no licence file, so it is imported from your
clone and none of its code is copied here)
    git clone https://github.com/xufangchn/GLF-CR /opt/GLF-CR
    pip install timm
    # CR_net.pth: Google Drive link in the GLF-CR README
    python baselines/run_external.py glfcr --repo /opt/GLF-CR --ckpt /opt/GLF-CR/ckpt/CR_net.pth \
        --cache /data/cache --split test --out EVAL/pred_glfcr.npy
    GLF-CR's compiled CUDA layer is replaced by an equivalent pure-PyTorch one
    (baselines/kernelconv2d_torch.py); pass --cuda_ext to use your compiled build instead.
    GLF-CR was trained and tested on 128 x 128 crops, so 256 x 256 patches are processed as
    overlapping 128 x 128 tiles and averaged.

Then:
    python -m diewolke.evaluate --runs RUNS/full --preds uncrtaints=EVAL/pred_uncrtaints.npy,EVAL/scale_uncrtaints.npy \
        glfcr=EVAL/pred_glfcr.npy --cache /data/cache --split test --mode real --out EVAL/test_real_external

Reporting: these checkpoints were trained by their authors on their own data and settings,
probably all four SEN12MS-CR seasons. If our models are trained on spring only, the published
models have seen more data; say so in the paper.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diewolke.bands import S1_DB_MIN  # noqa: E402
from diewolke.data import Cache, CloudPairs, fake_cache  # noqa: E402


def s1_to_db(s1):
    """Our normalised S1 (VV clipped at -25 dB, VH at -32.5 dB, mapped to [0, 1]) back to dB."""
    floor = torch.as_tensor(S1_DB_MIN, device=s1.device).view(1, 2, 1, 1)
    return s1 * (-floor) + floor


# ----------------------------------------------------------------------------- UnCRtainTS

class UnCRtainTSRunner:
    def __init__(self, repo, weights, experiment, device, resume_at=0, allow_multitemporal=False,
                 config_args=None):
        repo = Path(repo).resolve()
        sys.path[:0] = [str(repo), str(repo / "model")]
        from model.parse_args import create_parser
        from model.src.model_utils import get_model, load_checkpoint
        from model.src.utils import str2list

        parser = create_parser(mode="test")
        if config_args is not None:  # tests: build from flags instead of a checkpoint's conf.json
            config, _ = parser.parse_known_args(config_args)
        else:
            conf = json.loads((Path(weights) / experiment / "conf.json").read_text())
            keep = {"pid", "device", "resume_at", "trained_checkp", "res_dir", "weight_folder", "root1", "root2",
                    "root3", "max_samples_count", "batch_size", "display_step", "plot_every", "export_every",
                    "input_t", "region", "min_cov", "max_cov", "f"}
            ns = argparse.Namespace(**{k: v for k, v in conf.items() if k not in keep})
            config, _ = parser.parse_known_args([], namespace=ns)
            config.weight_folder, config.resume_at = str(weights), resume_at
        config = str2list(config, ["encoder_widths", "decoder_widths", "out_conv"])
        config.device = str(device)
        config.experiment_name = experiment
        self.mono = bool(getattr(config, "pretrain", False))
        if not self.mono and not allow_multitemporal:
            raise SystemExit(f"{experiment} is a multi-temporal (SEN12MS-CR-TS) checkpoint (pretrain=false); "
                             "pick a mono-temporal one or pass --allow_multitemporal")
        self.method = getattr(config, "rescale_method", "default")
        self.use_sar = bool(getattr(config, "use_sar", True))
        self.model = get_model(config).to(device)
        if config_args is None:
            name = f"model_epoch_{resume_at}" if resume_at > 0 else "model"
            load_checkpoint(config, str(weights), self.model, name)
        self.model.eval()
        self.config, self.device = config, device

    def _inputs(self, b):
        cloudy, clear = b["cloudy"], b["clear"]
        if self.method == "resnet":  # S2 / 2000, S1 to [0, 2] with -25 / -32.5 dB floors
            s2_scale, s1 = 5.0, 2 * b["s1"]
        else:  # default: S2 / 10000; S1 both bands clipped to [-25, 0] dB, mapped to [0, 1]
            s2_scale, s1 = 1.0, (s1_to_db(b["s1"]).clamp(-25, 0) + 25) / 25
        x = cloudy * s2_scale
        if self.use_sar:
            x = torch.cat([s1, x], 1)
        dates = None if self.mono else torch.zeros(x.shape[0], 1, device=x.device)
        return {"A": x[:, None], "B": (clear * s2_scale)[:, None], "dates": dates, "masks": b["mask"]}, s2_scale

    @torch.no_grad()
    def __call__(self, b):
        inputs, s2_scale = self._inputs(b)
        m = self.model
        m.set_input(inputs)
        m.forward()
        m.get_loss_G()  # this is where UnCRtainTS turns its raw outputs into (co)variances
        m.rescale()
        mean = m.fake_B[:, 0, :13] / s2_scale
        var = getattr(m.netG, "variance", None)
        scale = None
        if var is not None:
            if var.dim() > 5:  # (B, 1, 13, 13, H, W) covariance -> diagonal
                var = var.diagonal(dim1=2, dim2=3).moveaxis(-1, 2)
            var = var[:, 0] / s2_scale ** 2
            scale = (var.clamp(min=0) / 2).sqrt()  # Laplace b with the same variance as the Gaussian
        m.netG.variance = None
        return mean, scale


# ----------------------------------------------------------------------------- GLF-CR

class GLFCRRunner:
    TILE = 128

    def __init__(self, repo, ckpt, device, cuda_ext=False, stride=64):
        codes = Path(repo).resolve() / "codes"
        if not cuda_ext:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import kernelconv2d_torch
            kernelconv2d_torch.install()
        sys.path.insert(0, str(codes))
        from net_CR_RDN import RDN_residual_CR
        self.net = RDN_residual_CR(self.TILE).to(device)
        if ckpt is not None:
            state = torch.load(ckpt, map_location=device, weights_only=False)
            self.net.load_state_dict(state["network"] if "network" in state else state)
        self.net.eval()
        self.stride = stride

    def _starts(self, n):
        t = self.TILE
        s = list(range(0, max(n - t, 0) + 1, self.stride))
        return s if s[-1] + t >= n else s + [n - t]

    @torch.no_grad()
    def __call__(self, b):
        # GLF-CR's own normalisation matches ours: S2 / 10000, S1 VV/VH clipped at -25/-32.5 dB -> [0, 1]
        cloudy, s1 = b["cloudy"], b["s1"]
        _, _, h, w = cloudy.shape
        if h < self.TILE or w < self.TILE:
            raise ValueError(f"GLF-CR needs at least {self.TILE} x {self.TILE} inputs, got {h} x {w}")
        out = torch.zeros_like(cloudy)
        count = torch.zeros_like(cloudy[:, :1])
        for y in self._starts(h):
            for x in self._starts(w):
                sl = (..., slice(y, y + self.TILE), slice(x, x + self.TILE))
                out[sl] += self.net(cloudy[sl].contiguous(), s1[sl].contiguous())
                count[sl] += 1
        return out / count, None


# ----------------------------------------------------------------------------- main

def run(runner, cache, mode, out_path, batch, device, limit=None):
    ds = CloudPairs(cache, mode=mode, train=False)
    n = len(ds) if limit is None else min(limit, len(ds))
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(n)), batch_size=batch, shuffle=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = ds[0]
    shape = (len(ds),) + tuple(first["clear"].shape)
    pred = np.lib.format.open_memmap(out_path, "w+", np.float16, shape)
    scale_path = out_path.with_name(out_path.name.replace("pred_", "scale_", 1)) if "pred_" in out_path.name \
        else out_path.with_name("scale_" + out_path.name)
    scales = None
    t0, k = time.time(), 0
    for b in loader:
        b = {key: v.to(device) for key, v in b.items()}
        mean, scale = runner(b)
        nb = mean.shape[0]
        pred[k:k + nb] = mean.clamp(0, 1).float().cpu().numpy().astype(np.float16)
        if scale is not None:
            if scales is None:
                scales = np.lib.format.open_memmap(scale_path, "w+", np.float16, shape)
            scales[k:k + nb] = scale.float().cpu().numpy().astype(np.float16)
        k += nb
        if k % (batch * 25) == 0:
            print(f"  {k}/{n} ({time.time() - t0:.0f}s)", flush=True)
    for i in range(n, len(ds)):  # beyond --limit: the cloudy input, so the file still lines up
        pred[i] = ds[i]["cloudy"].numpy().astype(np.float16)
    pred.flush()
    if scales is not None:
        scales.flush()
    print(f"[+] wrote {out_path}" + (f" and {scale_path}" if scales is not None else "")
          + f" ({n} patches, {time.time() - t0:.0f}s)")
    return out_path, (scale_path if scales is not None else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="model", required=True)
    for name in ("uncrtaints", "glfcr"):
        p = sub.add_parser(name)
        p.add_argument("--repo", required=True, help="path to your clone of the model's repository")
        p.add_argument("--cache", default="/content/cache")
        p.add_argument("--split", default="test")
        p.add_argument("--mode", default="real", choices=["real", "synthetic"],
                       help="must match the evaluate.py --mode you will score with")
        p.add_argument("--out", required=True, help="pred_<name>.npy; a scale_<name>.npy is written next to it if available")
        p.add_argument("--batch", type=int, default=4)
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--fake", action="store_true", help="random data and random weights (pipeline check)")
        if name == "uncrtaints":
            p.add_argument("--weights", help="folder that contains <experiment>/conf.json and model.pth.tar")
            p.add_argument("--experiment", help="checkpoint folder name")
            p.add_argument("--resume_at", type=int, default=0)
            p.add_argument("--allow_multitemporal", action="store_true")
        else:
            p.add_argument("--ckpt", help="CR_net.pth")
            p.add_argument("--cuda_ext", action="store_true", help="use your compiled KernelConv2D CUDA layer")
            p.add_argument("--stride", type=int, default=64, help="tile stride (tiles are 128 x 128)")
    args = ap.parse_args(argv)
    device = torch.device(args.device)

    if args.model == "uncrtaints":
        flags = None
        if args.fake:
            flags = ["--model", "uncrtaints", "--pretrain", "--use_sar", "--device", str(device), "--loss", "MGNLL",
                     "--covmode", "diag", "--input_t", "1", "--n_head", "16", "--out_conv", "[26]"]
        elif not (args.weights and args.experiment):
            raise SystemExit("give --weights and --experiment")
        runner = UnCRtainTSRunner(args.repo, args.weights, args.experiment, device, args.resume_at,
                                  args.allow_multitemporal, config_args=flags)
    else:
        if not args.fake and not args.ckpt:
            raise SystemExit("give --ckpt (CR_net.pth)")
        runner = GLFCRRunner(args.repo, None if args.fake else args.ckpt, device, args.cuda_ext, args.stride)
    cache = fake_cache(4, 256 if args.model == "glfcr" else 64, seed=5) if args.fake else Cache.load(args.cache, args.split)
    run(runner, cache, args.mode, args.out, args.batch, device, args.limit)


if __name__ == "__main__":
    main()
