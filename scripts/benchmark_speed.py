"""Time training steps under different speed settings and print which is fastest.

    python scripts/benchmark_speed.py --config configs/full.yaml \
        --set data.root=/data/sen12ms_cr data.cache=/data/cache train.workers=6
Each setting trains a scratch copy of the model for --steps steps (default 300) into a temporary
folder. Your real runs are not touched. The first --skip steps are ignored, because
torch.compile and cudnn.benchmark spend them compiling and choosing kernels. Needs the
train/val caches (they are built on first use, like a normal run).

Default settings compared (any train.* overrides work, separated by spaces, one setting per --try):
    fp16 | bf16 | bf16 + compile | bf16 + compile + channels_last
"""
import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diewolke.config import load_config  # noqa: E402
from diewolke.train import run  # noqa: E402

DEFAULT_TRIES = [
    "train.precision=fp16",
    "train.precision=bf16",
    "train.precision=bf16 train.compile=true",
    "train.precision=bf16 train.compile=true train.channels_last=true",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides applied to every setting (paths, workers, ...)")
    ap.add_argument("--try", dest="tries", action="append", default=None,
                    help='one setting, e.g. --try "train.precision=bf16 train.compile=true"; repeatable')
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--skip", type=int, default=100, help="warm-up steps excluded from the timing")
    ap.add_argument("--full_steps", type=int, default=None, help="steps of the real run, for the time estimate "
                                                                 "(default: train.steps of the config)")
    args = ap.parse_args(argv)
    tries = args.tries or DEFAULT_TRIES
    every = 50
    if args.skip % every or args.steps <= args.skip:
        raise SystemExit(f"--skip must be a multiple of {every} and smaller than --steps")
    scratch = Path(tempfile.mkdtemp(prefix="diewolke_bench_"))
    results = []
    try:
        for i, t in enumerate(tries):
            over = args.set + t.split() + [
                f"train.out_dir={scratch}", f"name=bench{i}", f"train.steps={args.steps}", "train.restart=true",
                f"train.log_every={every}", f"train.val_every={10 ** 9}", f"train.ckpt_every={10 ** 9}",
                "train.val_batches=1", "train.warmup=10", "train.max_minutes=null", "train.stop_at=null"]
            cfg = load_config(args.config, over)
            full_steps = args.full_steps or load_config(args.config, args.set)["train"]["steps"]
            print(f"\n=== [{i + 1}/{len(tries)}] {t}")
            try:
                run(cfg)
            except Exception as exc:  # noqa: BLE001 - e.g. compile unavailable; report and continue
                print(f"[!] failed: {exc}")
                results.append((t, None, None, None))
                continue
            rows = [r for r in csv.DictReader(open(scratch / f"bench{i}" / "log.csv"))
                    if r.get("sec_per_step") and int(r["step"]) > args.skip]
            sec = sum(float(r["sec_per_step"]) for r in rows) / len(rows)
            wait = sum(float(r.get("data_wait") or 0) for r in rows) / len(rows)
            results.append((t, sec, wait, sec * full_steps / 3600))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"\n{'setting':<62} {'s/step':>8} {'data wait':>10} {'full run':>9}")
    for t, sec, wait, hours in results:
        if sec is None:
            print(f"{t:<62} {'failed':>8}")
        else:
            print(f"{t:<62} {sec:>8.3f} {100 * wait:>9.0f}% {hours:>8.2f}h")
    ok = [r for r in results if r[1] is not None]
    if ok:
        best = min(ok, key=lambda r: r[1])
        print(f"\nfastest: {best[0]}  (add it to every training command)")
        if best[2] > 0.2:
            print("note: >20% data wait - raise train.workers or move the cache to faster storage before "
                  "trusting these numbers")


if __name__ == "__main__":
    main()
