# Running dieWolke on Kaggle

This folder adds what Kaggle needs on top of the normal codebase. Model, training and evaluation
code are unchanged.

| File | What it is |
|---|---|
| `01_prepare_data.ipynb` | CPU notebook: builds the data cache on Kaggle (route B below) |
| `02_train.ipynb` | GPU notebook: restores earlier sessions, then runs the training queue on one or two GPUs |
| `03_evaluate.ipynb` | GPU notebook: all evaluations, the two-stage (C5) test, published baselines, XGBoost |
| `kaggle_tools.py` | `env`, `find-cache`, `restore`, `queue`, `prepare`, `publish` (run with `--help`) |
| `kaggle.yaml` | Config overlay: Kaggle paths, fp16, few data workers, stop before the 12 h limit |

## How Kaggle differs, and what handles it

- **12-hour sessions.** `queue` gives every run `train.max_minutes` equal to the time left in the
  session, minus 20 minutes for saving. Runs stop with a checkpoint. The next session adds the
  previous notebook version as an input; `restore` copies the runs back and the same queue
  continues. Finished runs are skipped.
- **Only `/kaggle/working` survives a session**, and users report a ~20 GB limit for it.
  Checkpoints are small: `best.pt` is now weights-only (~130 MB for the full model: weights plus their EMA), and `last.pt`
  is ~270 MB. Seven runs fit comfortably.
- **T4 or P100 GPUs, no bf16.** The overlay uses fp16. Pick **GPU T4 x2** if it's offered: the
  P100 has no tensor cores.
- **Two GPUs.** `queue` runs one job per GPU and splits the CPU cores between their data
  loaders. Check `data_wait` in the logs: Kaggle VMs have few cores.
- **Data is read-only under `/kaggle/input`.** Training only reads the cache.
  `find-cache` locates it inside whatever dataset or notebook output you attached. A missing
  `.done` marker no longer matters: some uploads drop empty files, and the arrays alone now count.
- **`/kaggle/input` is network-backed and too slow to train from.** Measured on a T4 session:
  a random 128 px crop of one array took 0.225 s, and the first benchmark had not reached step
  50 after 17 minutes. Sequential reads run at ~150 MB/s, so notebooks 2 and 3 first copy the
  cache to local disk (`find-cache --copy_to /kaggle/temp/cache`, ~2 min for 17 GB) and train
  from there.

## Getting the data in

The cache is ~4 MB per patch. The defaults (3,000 train, 300 val, full spring test) come to
roughly 15–30 GB, depending on the test split size.

- **Route A (recommended): build the cache on another machine and upload it.**
  ```
  python scripts/prepare_sen12mscr.py cache --config configs/full.yaml --set data.root=/data/sen12ms_cr data.cache=/data/cache
  pip install kaggle      # API token from Kaggle > Settings > API, saved as ~/.kaggle/kaggle.json
  python kaggle/kaggle_tools.py publish --dir /data/cache --slug diewolke-cache
  ```
  The dataset stays private. Kaggle's dataset size limits aren't something I could confirm, so
  check your account's quota before uploading this much.
- **Route B: build it on Kaggle** with `01_prepare_data.ipynb` (CPU, Internet on). It downloads
  over rsync (`prepare --rsync`), with the password from the mediaTUM page in `RSYNC_PASSWORD`.
  TUM's HTTPS share (`--base_url`, secret `SEN12MSCR_URL`) still works but is throttled to
  ~0.1 MB/s, far too slow for 70 GB; it also returned 503 for a while. Measured from Kaggle, rsync
  gave ~2.6 MB/s per connection and ~8.7 MB/s with three, so all archives download in parallel:
  about 3.5 h for spring. FTP does not work from Kaggle. `kaggle_tools.py prepare` then:
  1. downloads the archives to `/kaggle/temp` (resuming dropped connections, checking TUM's
     `checksums.sha512`), extracts each one and deletes it;
  2. extracts S1 fully and picks the train/val/test patches from its file names (same official
     split and seed as the cache);
  3. deletes the unused S1 files and checks free disk;
  4. keeps only the chosen patches from the two large S2 archives;
  5. builds the cache in `/kaggle/working/cache`.

  Peak disk is the S1 data plus the chosen patches. The defaults (2,500 train, 300 val, 1,500
  test) keep the cache at about 17 GB, under the 20 GB `/kaggle/working` limit. Everything has to
  finish in one session, because the extracted files in `/kaggle/temp` are not saved. For the
  full test split, run it again with `--splits test` and attach both outputs.

## The code itself

Upload the `DieWolke` folder (without data, runs or `.venv`) as a private dataset, or set
`REPO_URL` in the notebooks if the repository is pushed somewhere. Training needs no extra
packages, because Kaggle's image has PyTorch, NumPy, SciPy and PyYAML. Building the cache needs
`rasterio` and `s2cloudless` (notebook 1 installs them). The published baselines need `fvcore`
and `timm` (notebook 3).

## Time on Kaggle (estimates)

A T4 is roughly half the speed of the RTX A4000. Option A (masker, `full` and `matched_baseline`
at 40k steps, four ablations at 20k) comes to about 10–18 GPU hours. With T4 x2 that's roughly
5–9 hours of wall-clock time: one or two sessions, within the commonly reported ~30 GPU hours a
week. Run `scripts/benchmark_speed.py` from notebook 2 once to replace these guesses with
measurements.

## Not on Kaggle

- **The Kharif benchmark (`kharif/`)** needs an Earth Engine login. Build it elsewhere, upload
  `kharif_benchmark/` as a dataset, and point `evaluate.py --cache` at it.
- **Colab** keeps using `notebooks/colab_runbook.ipynb`.
