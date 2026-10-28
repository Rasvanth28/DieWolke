"""Get SEN12MS-CR onto a Colab VM and build the .npy caches. Replaces datasets/VisuzalizeDataset.py.

    # 1. archives -> Drive (once; resumable). SEN12MSCR_URL is the archive folder URL, or a template
    #    with {name} in place of the archive name (see archive_url). It comes from the
    #    SEN12MS-CR page on mediaTUM (https://mediatum.ub.tum.de/1554803); pass it, do not commit it.
    python scripts/prepare_sen12mscr.py download --base_url "$SEN12MSCR_URL" --drive /content/drive/MyDrive/sen12ms_cr
    # 2. Drive -> local disk (every new Colab session)
    python scripts/prepare_sen12mscr.py extract --drive /content/drive/MyDrive/sen12ms_cr --local /content/data/sen12ms_cr
    # 3. tifs -> caches for a config (every new session, unless the cache is kept on Drive)
    python scripts/prepare_sen12mscr.py cache --config configs/full.yaml
Seasons: spring (ROIs1158) is the default; --seasons spring summer fall winter for all four.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEASONS = {"spring": "ROIs1158_spring", "summer": "ROIs1868_summer", "fall": "ROIs1970_fall", "winter": "ROIs2017_winter"}
MODALITIES = ["s1", "s2", "s2_cloudy"]
MIN_BYTES = 100 * 1024 * 1024  # anything smaller is a failed download


def archive_url(base, name):
    """Download URL for one archive. `base` is either a folder URL (the name is appended) or a
    template containing {name}, for servers whose links put the file name in a query string, e.g.
    https://dataserv.ub.tum.de/index.php/s/<share>/download?path=%2F&files={name}"""
    return base.format(name=name) if "{name}" in base else f"{base.rstrip('/')}/{name}"


def archives(seasons):
    return [f"{SEASONS[s]}_{m}.tar.gz" for s in seasons for m in MODALITIES]


def download(args):
    base = args.base_url or os.environ.get("SEN12MSCR_URL")
    if not base:
        raise SystemExit("give --base_url or set SEN12MSCR_URL (the archive folder from the mediaTUM page)")
    drive = Path(args.drive)
    drive.mkdir(parents=True, exist_ok=True)
    for name in archives(args.seasons):
        dst = drive / name
        if dst.exists() and dst.stat().st_size > MIN_BYTES and not args.force:
            print(f"[=] {name} present ({dst.stat().st_size / 1e9:.1f} GB)")
            continue
        print(f"[+] downloading {name}")
        subprocess.run(["wget", "-c", "-q", "--show-progress", "-O", str(dst), archive_url(base, name)], check=True)


def extract(args):
    drive, local = Path(args.drive), Path(args.local)
    local.mkdir(parents=True, exist_ok=True)
    for name in archives(args.seasons):
        src = drive / name
        marker = local / f".{name}.extracted"
        if marker.exists():
            print(f"[=] {name} already extracted")
            continue
        if not src.exists() or src.stat().st_size < MIN_BYTES:
            raise SystemExit(f"{src} missing or truncated; run the download step")
        print(f"[+] extracting {name} -> {local}", flush=True)
        # pigz (parallel gunzip) is much faster on Colab when available
        cmd = f"tar -I pigz -xf '{src}' -C '{local}'" if subprocess.run("which pigz", shell=True,
                                                                         capture_output=True).returncode == 0 \
            else f"tar -xzf '{src}' -C '{local}'"
        subprocess.run(cmd, shell=True, check=True)
        marker.touch()
    from diewolke.data import find_triplets
    print(f"[+] {len(find_triplets(local))} complete triplets under {local}")


def cache(args):
    from diewolke.config import load_config
    from diewolke.data import prepare_caches
    cfg = load_config(args.config, args.set)
    prepare_caches(cfg, splits=tuple(args.splits))
    print(f"[+] caches ready in {cfg['data']['cache']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--base_url")
    d.add_argument("--drive", default="/content/drive/MyDrive/sen12ms_cr")
    d.add_argument("--seasons", nargs="+", default=["spring"], choices=list(SEASONS))
    d.add_argument("--force", action="store_true")
    e = sub.add_parser("extract")
    e.add_argument("--drive", default="/content/drive/MyDrive/sen12ms_cr")
    e.add_argument("--local", default="/content/data/sen12ms_cr")
    e.add_argument("--seasons", nargs="+", default=["spring"], choices=list(SEASONS))
    c = sub.add_parser("cache")
    c.add_argument("--config", default="configs/full.yaml")
    c.add_argument("--set", nargs="*", default=[])
    c.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args(argv)
    {"download": download, "extract": extract, "cache": cache}[args.cmd](args)


if __name__ == "__main__":
    main()
