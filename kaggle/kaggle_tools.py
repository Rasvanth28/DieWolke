"""Helpers for running dieWolke on Kaggle notebooks (12 h sessions, T4 / T4 x2 / P100 GPUs).

    python kaggle/kaggle_tools.py env                     # GPUs, CPUs, RAM, free disk; what to watch out for
    python kaggle/kaggle_tools.py find-cache              # locate the train/val/test cache under /kaggle/input
    python kaggle/kaggle_tools.py restore                 # copy runs from earlier notebook versions to /kaggle/working/runs
    python kaggle/kaggle_tools.py queue full@40000 matched_baseline@40000 --session_start $T0
    python kaggle/kaggle_tools.py prepare --out /kaggle/working/cache   # build the cache on Kaggle (CPU notebook);
                                                                        # streams the archives, keeps only cached patches
    python kaggle/kaggle_tools.py publish --dir /data/cache --slug diewolke-cache   # upload a cache from any machine

Run specs for `queue`: CONFIG[:NAME][@STEPS], e.g. `full@40000`, `full:full_20k@20000`,
`no_bga@20000`. CONFIG is a file in configs/. Runs are spread over the visible GPUs (one lane per
GPU, runs in a lane one after another). Each run gets train.max_minutes = the session time left,
so it checkpoints cleanly before Kaggle stops the session. Finished runs are skipped, and stopped
ones resume. Run the same queue in the next session, after `restore`.
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "kaggle" / "kaggle.yaml"
SESSION_MINUTES = 12 * 60
MB_PER_PATCH = 4.0  # cache size per patch (see diewolke/data/sen12mscr.py CACHE_FIELDS)


def _gb(n):
    return f"{n / 1e9:.1f} GB"


# ----------------------------------------------------------------------------- env

def gpus():
    try:
        import torch
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        return []


def ram_bytes():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def cmd_env(args):
    g = gpus()
    print(f"GPUs:  {g or 'none (CPU session)'}")
    print(f"CPUs:  {os.cpu_count()}")
    ram = ram_bytes()
    print(f"RAM:   {_gb(ram) if ram else 'unknown'}")
    for path in ("/kaggle/working", "/kaggle/temp", "/tmp", "/kaggle/input"):
        if os.path.exists(path):
            u = shutil.disk_usage(path)
            print(f"disk {path:<16} free {_gb(u.free)} of {_gb(u.total)}")
    if g and any("P100" in x for x in g):
        print("[!] P100: no tensor cores, so fp16 gains little; prefer the T4 x2 accelerator if offered")
    if len(g) > 1:
        print(f"[+] {len(g)} GPUs: `queue` runs one training job per GPU")
    cache = find_cache(args.input) if os.path.exists(args.input) else None
    print(f"cache: {cache or 'not found under ' + args.input}")


# ----------------------------------------------------------------------------- cache

def find_cache(root="/kaggle/input", split="train"):
    """Directory under root that holds <split>_clear.npy (first match, shortest path)."""
    hits = sorted(Path(root).rglob(f"{split}_clear.npy"), key=lambda p: len(p.parts)) if Path(root).exists() else []
    return str(hits[0].parent) if hits else None


def cmd_find_cache(args):
    c = find_cache(args.input, args.split)
    if not c:
        raise SystemExit(f"no {args.split}_clear.npy under {args.input}; add the cache dataset to this notebook")
    print(c)


# ----------------------------------------------------------------------------- restore

def run_step(run_dir):
    """Step saved in run_dir/last.pt (0 if none)."""
    ck = Path(run_dir) / "last.pt"
    if not ck.exists():
        return 0
    import torch
    return int(torch.load(ck, map_location="cpu", weights_only=False).get("step", 0))


def run_target(run_dir):
    import yaml
    cfg = Path(run_dir) / "config.yaml"
    return yaml.safe_load(cfg.read_text())["train"]["steps"] if cfg.exists() else None


def cmd_restore(args):
    """Copy run folders found anywhere under --src (earlier notebook versions added as inputs)."""
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    found = 0
    for ck in sorted(Path(args.src).rglob("last.pt")):
        src = ck.parent
        if not (src / "config.yaml").exists():
            continue
        found += 1
        target = dst / src.name
        s_step, d_step = run_step(src), run_step(target)
        if s_step > d_step:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
            print(f"[+] {src.name}: restored step {s_step} from {src}")
        else:
            print(f"[=] {src.name}: kept local copy (step {d_step} >= {s_step})")
    for name in args.extra or []:  # e.g. eval folders
        for d in Path(args.src).rglob(name):
            if d.is_dir() and not (Path(args.dst).parent / name).exists():
                shutil.copytree(d, Path(args.dst).parent / name)
                print(f"[+] restored {name} from {d}")
                break
    if not found:
        print(f"[=] no runs under {args.src} (first session?)")


# ----------------------------------------------------------------------------- queue

def parse_spec(spec):
    cfg, _, steps = spec.partition("@")
    cfg, _, name = cfg.partition(":")
    return {"config": cfg, "name": name or cfg, "steps": int(steps) if steps else None}


def minutes_left(session_start, session_minutes):
    return session_minutes - (time.time() - session_start) / 60


def _lane(gpu, runs, args, cache, out_dir, workers, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)) if gpu is not None else dict(os.environ)
    for r in runs:
        run_dir = out_dir / r["name"]
        target = r["steps"] or run_target(run_dir)
        done = run_step(run_dir)
        if target and done >= target:
            log(f"[=] {r['name']}: finished ({done}/{target}), skipped")
            continue
        left = minutes_left(args.session_start, args.session_minutes) - args.margin
        if left < args.min_minutes:
            log(f"[!] {r['name']}: only {left:.0f} min left in this session; run the queue again next session")
            continue
        sets = [f"data.cache={cache}", f"train.out_dir={out_dir}", f"name={r['name']}",
                f"train.workers={workers}", f"train.max_minutes={int(left)}"]
        if r["steps"]:
            sets.append(f"train.steps={r['steps']}")
        cmd = [sys.executable, "-m", "diewolke.train", "--config", str(ROOT / "configs" / f"{r['config']}.yaml"),
               "--overlay", str(OVERLAY), "--set", *sets, *args.set]
        log(f"[>] {r['name']} on GPU {gpu}: {left:.0f} min budget, from step {done}")
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "stdout.log", "a") as f:
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                f.write(line)
                if args.verbose or "[val]" in line or "[!]" in line or "[+] done" in line or "Error" in line:
                    log(f"  [{r['name']}] {line.rstrip()}")
            p.wait()
        status = "ok" if p.returncode == 0 else f"exit code {p.returncode} (see {run_dir / 'stdout.log'})"
        log(f"[<] {r['name']}: {status}, now at step {run_step(run_dir)}")


def cmd_queue(args):
    runs = [parse_spec(s) for s in args.runs]
    cache = args.cache if args.cache != "auto" else find_cache(args.input)
    if not cache:
        raise SystemExit(f"no cache found under {args.input}; pass --cache")
    n_gpu = len(gpus()) if args.gpus == "auto" else int(args.gpus)
    lanes = [[] for _ in range(max(1, n_gpu))]
    for i, r in enumerate(runs):
        lanes[i % len(lanes)].append(r)
    workers = args.workers if args.workers is not None else max(1, (os.cpu_count() or 2) // len(lanes))
    out_dir = Path(args.out_dir)
    lock = threading.Lock()

    def log(msg):
        with lock:
            print(msg, flush=True)

    log(f"[+] cache {cache}; {len(lanes)} lane(s), {workers} data workers each; "
        f"{minutes_left(args.session_start, args.session_minutes):.0f} min left in the session")
    threads = [threading.Thread(target=_lane, args=(g if n_gpu else None, lane, args, cache, out_dir, workers, log))
               for g, lane in enumerate(lanes)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("[+] queue finished for this session; status:")
    for r in runs:
        d = out_dir / r["name"]
        log(f"    {r['name']:<20} step {run_step(d)}/{r['steps'] or run_target(d)}")


# ----------------------------------------------------------------------------- prepare (on Kaggle)

def _prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_sen12mscr", ROOT / "scripts" / "prepare_sen12mscr.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TIF_MB = {"s1": 2 * 256 * 256 * 4 / 1e6, "s2": 13 * 256 * 256 * 2 / 1e6}  # uncompressed patch sizes


def stream_extract(url, dest, keep=None, retries=3, label=""):
    """Stream a .tar.gz from `url` straight into `dest`: the archive itself never touches the disk.

    keep(file_name) -> bool selects which files to write (None = all). On a network error the
    archive is streamed again from the start (already written files are simply overwritten).
    Returns (files written, files seen).
    """
    import tarfile
    import urllib.request
    dest = Path(dest)
    for attempt in range(1, retries + 1):
        written = seen = 0
        t0 = time.time()
        try:
            with urllib.request.urlopen(url, timeout=300) as resp, \
                    tarfile.open(fileobj=resp, mode="r|gz") as tar:
                for m in tar:
                    if not m.isfile():
                        continue
                    seen += 1
                    if keep is not None and not keep(Path(m.name).name):
                        continue
                    rel = Path(*[q for q in Path(m.name).parts if q not in ("/", "..", ".")])
                    target = dest / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(m) as src, open(target, "wb") as f:
                        shutil.copyfileobj(src, f, 1 << 20)
                    written += 1
                    if seen % 2000 == 0:
                        print(f"    {label}: {seen} files read, {written} kept ({time.time() - t0:.0f}s)", flush=True)
            return written, seen
        except (OSError, EOFError, tarfile.TarError) as exc:
            print(f"[!] {label}: attempt {attempt}/{retries} failed after {seen} files: {exc}", flush=True)
    raise SystemExit(f"could not download {label}")


def select_keys(data, cfg, splits, margin=1.05, extra=10):
    """Choose the patches to cache from the S1 file names alone (same split and seed as the cache).

    A few extra per split (`margin`) cover patches whose S2 files turn out to be missing; the cache
    step then subsamples to the exact limits."""
    from diewolke.data.sen12mscr import NAME_RE, split_triplets, subsample
    d = cfg["data"]
    keys = []
    for p in Path(data).rglob("*.tif"):
        m = NAME_RE.match(p.name)
        if m and m.group(2).lower() == "s1":
            keys.append((m.group(1), int(m.group(3)), int(m.group(4))))
    parts = split_triplets([(k, {}) for k in sorted(keys)], d.get("split", "official"), seed=d.get("seed", 0))
    limits = {"train": d.get("n_train"), "val": d.get("n_val"), "test": d.get("n_test")}
    chosen = {}
    for s in splits:
        lim = limits[s]
        lim = None if lim is None else int(lim * margin) + extra
        chosen[s] = [k for k, _ in subsample(parts[s], lim, d.get("seed", 0))]
    return len(keys), chosen


def _key_filter(selected):
    from diewolke.data.sen12mscr import NAME_RE

    def keep(fname):
        m = NAME_RE.match(fname)
        return bool(m) and (m.group(1), int(m.group(3)), int(m.group(4))) in selected
    return keep


def cmd_prepare(args):
    """Stream S1 -> choose patches -> stream only those S2 / cloudy-S2 patches -> build the cache.

    Peak disk is about the extracted S1 archive plus the chosen S2 patches; no archive is ever
    stored. --full extracts every patch instead (much more disk)."""
    from diewolke.config import load_config
    from diewolke.data import prepare_caches
    from diewolke.data.sen12mscr import NAME_RE
    prep = _prepare_module()
    base = args.base_url or os.environ.get("SEN12MSCR_URL")
    data = Path(args.data)
    data.mkdir(parents=True, exist_ok=True)
    sets = [f"data.root={data}", f"data.cache={args.out}"] + args.set
    if args.n_train is not None:
        sets.append(f"data.n_train={args.n_train}")
    if args.n_test is not None:
        sets.append(f"data.n_test={args.n_test}")
    cfg = load_config(ROOT / "configs" / "full.yaml", sets, [str(OVERLAY)])
    names = prep.archives(args.seasons)
    s1_names = [n for n in names if n.endswith("_s1.tar.gz")]
    s2_names = [n for n in names if n not in s1_names]

    def fetch(name, keep=None):
        marker = data / f".{name}.extracted"
        if marker.exists():
            print(f"[=] {name} already extracted")
            return
        if not base:
            raise SystemExit("set SEN12MSCR_URL (Kaggle: Add-ons > Secrets) or pass --base_url")
        print(f"[+] {name}: streaming into {data} ({_gb(shutil.disk_usage(data).free)} free)", flush=True)
        written, seen = stream_extract(prep.archive_url(base, name), data, keep, label=name)
        print(f"[+] {name}: {written} of {seen} files kept", flush=True)
        marker.touch()

    if not args.skip_download:
        for n in s1_names:
            fetch(n)
    keep_fn = None
    if not args.full:
        sel_file = data / "selected_keys.json"
        if sel_file.exists():
            chosen = {s: [tuple(k) for k in v] for s, v in json.loads(sel_file.read_text()).items()}
            n_all = None
        else:
            n_all, chosen = select_keys(data, cfg, args.splits, extra=args.spare)
            sel_file.write_text(json.dumps(chosen))
        selected = {k for v in chosen.values() for k in v}
        print(f"[+] selected {len(selected)} patches" + (f" of {n_all}" if n_all else "") + ": "
              + ", ".join(f"{s} {len(v)}" for s, v in chosen.items()))
        # drop the S1 files that will not be used, before the big S2 archives arrive
        removed = 0
        for p in data.rglob("*.tif"):
            m = NAME_RE.match(p.name)
            if m and m.group(2).lower() == "s1" and (m.group(1), int(m.group(3)), int(m.group(4))) not in selected:
                p.unlink()
                removed += 1
        need = len(selected) * (2 * TIF_MB["s2"] + MB_PER_PATCH) * 1e6
        free = shutil.disk_usage(data).free
        print(f"[+] removed {removed} unused S1 files; S2 patches + cache need ~{_gb(need)}, {_gb(free)} free")
        if need > free * 0.95 and not args.force:
            raise SystemExit("not enough disk: lower --n_train / --n_test (or pass --force)")

        keep_fn = _key_filter(selected)
    if not args.skip_download:
        for n in s2_names:
            fetch(n, keep_fn)
    caches = prepare_caches(cfg, splits=tuple(args.splits))
    size = sum(f.stat().st_size for f in Path(args.out).glob("*") if f.is_file())
    print(f"[+] cache in {args.out}: " + ", ".join(f"{s} {len(c)}" for s, c in caches.items()) + f"; {_gb(size)}")
    if str(args.out).startswith("/kaggle/working") and size > args.output_limit_gb * 1e9:
        print(f"[!] larger than ~{args.output_limit_gb} GB, the /kaggle/working limit; "
              "lower --n_train / --n_test, or build the cache on another machine and use `publish`")
    if args.delete_tifs:
        shutil.rmtree(data, ignore_errors=True)


# ----------------------------------------------------------------------------- publish

def cmd_publish(args):
    """Create (or add a version to) a private Kaggle dataset from a cache directory, via the kaggle CLI."""
    d = Path(args.dir)
    files = [f for f in d.iterdir() if f.is_file()]
    if not any(f.name.endswith("_clear.npy") for f in files):
        raise SystemExit(f"{d} does not look like a cache (no *_clear.npy)")
    total = sum(f.stat().st_size for f in files)
    big = max(files, key=lambda f: f.stat().st_size)
    print(f"[+] {len(files)} files, {_gb(total)} total; largest {big.name} {_gb(big.stat().st_size)}")
    print("    check the current Kaggle dataset size limits for your account before uploading this much")
    user = args.user or os.environ.get("KAGGLE_USERNAME")
    if not user:
        cred = Path.home() / ".kaggle" / "kaggle.json"
        user = json.loads(cred.read_text())["username"] if cred.exists() else None
    if not user:
        raise SystemExit("no Kaggle username: set KAGGLE_USERNAME or put kaggle.json in ~/.kaggle")
    meta = d / "dataset-metadata.json"
    meta.write_text(json.dumps({"title": args.title, "id": f"{user}/{args.slug}",
                                "licenses": [{"name": "other"}]}, indent=1))
    if args.dry_run:
        print(f"[=] dry run: would upload {d} as {user}/{args.slug}")
        return
    cmd = ["kaggle", "datasets", "version", "-p", str(d), "-m", args.message] if args.version else \
        ["kaggle", "datasets", "create", "-p", str(d)]
    print("[>] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[+] uploaded; it stays private. Add it to the notebook: Add Input > Your Datasets > {args.slug}")


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("env")
    e.add_argument("--input", default="/kaggle/input")
    f = sub.add_parser("find-cache")
    f.add_argument("--input", default="/kaggle/input")
    f.add_argument("--split", default="train")
    r = sub.add_parser("restore")
    r.add_argument("--src", default="/kaggle/input")
    r.add_argument("--dst", default="/kaggle/working/runs")
    r.add_argument("--extra", nargs="*", default=["eval"], help="other folders to restore once (e.g. eval)")
    q = sub.add_parser("queue")
    q.add_argument("runs", nargs="+", help="CONFIG[:NAME][@STEPS]")
    q.add_argument("--cache", default="auto")
    q.add_argument("--input", default="/kaggle/input")
    q.add_argument("--out_dir", default="/kaggle/working/runs")
    q.add_argument("--gpus", default="auto", help="number of GPUs to use (auto = all visible)")
    q.add_argument("--workers", type=int, default=None, help="data workers per run (default: CPUs / lanes)")
    q.add_argument("--session_start", type=float, default=time.time(),
                   help="time.time() at the top of the notebook; default: now")
    q.add_argument("--session_minutes", type=float, default=SESSION_MINUTES)
    q.add_argument("--margin", type=float, default=20, help="minutes kept free for saving the notebook output")
    q.add_argument("--min_minutes", type=float, default=15, help="do not start a run with less time than this")
    q.add_argument("--set", nargs="*", default=[], help="extra overrides for every run, e.g. train.compile=true")
    q.add_argument("--verbose", action="store_true", help="stream every training log line")
    p = sub.add_parser("prepare")
    p.add_argument("--base_url", default=None)
    p.add_argument("--seasons", nargs="+", default=["spring"])
    p.add_argument("--data", default="/kaggle/temp/sen12ms_cr")
    p.add_argument("--out", default="/kaggle/working/cache")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--n_test", type=int, default=None)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--skip_download", action="store_true", help="use tifs already in --data")
    p.add_argument("--spare", type=int, default=10, help="extra patches per split (+5%%) in case S2 files are missing")
    p.add_argument("--full", action="store_true", help="extract every patch, not only the ones to cache")
    p.add_argument("--force", action="store_true", help="continue even if the disk estimate says it will not fit")
    p.add_argument("--delete_tifs", action="store_true")
    p.add_argument("--output_limit_gb", type=float, default=19.5)
    u = sub.add_parser("publish")
    u.add_argument("--dir", required=True)
    u.add_argument("--slug", default="diewolke-cache")
    u.add_argument("--title", default="dieWolke SEN12MS-CR cache")
    u.add_argument("--user", default=None)
    u.add_argument("--version", action="store_true", help="add a new version to an existing dataset")
    u.add_argument("--message", default="update")
    u.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    {"env": cmd_env, "find-cache": cmd_find_cache, "restore": cmd_restore, "queue": cmd_queue,
     "prepare": cmd_prepare, "publish": cmd_publish}[args.cmd](args)


if __name__ == "__main__":
    main()
