# dieWolke

Cloud removal for all 13 Sentinel-2 bands, conditioned on Sentinel-1 SAR. The pipeline follows
`Architecture.jpeg`:

1. A **Stage 1 U-Net masker** finds clouds and shadows.
2. **Stage 2** is a SAR-conditioned diffusion-bridge reconstructor with band-group attention
   and a joint aleatoric uncertainty head.
3. **Stage 4** is a validation harness: pixel metrics, NDVI error and downstream classification.
   It runs on SEN12MS-CR and on a new Indian Kharif monsoon benchmark.

The research positioning is in [NOVELTY.md](NOVELTY.md), which defines the contributions C1–C5
used below.

> **Status:** the code is complete and tested end to end on synthetic data (CPU). Nothing has
> been trained on SEN12MS-CR yet, so there are no results. Every number in this repository so
> far comes from smoke tests on random data.

## What maps to what

| Piece | Where | Switch / config |
|---|---|---|
| Stage 1 masker (U-Net, BCE + Dice, trained on exact synthetic masks) | `diewolke/models/unet.py` | `configs/masker.yaml` |
| Stage 2 bridge reconstructor | `diewolke/models/reconstructor.py`, `diewolke/bridge.py` | `configs/full.yaml` |
| C1 band-group attention (visible / red-edge+NIR / SWIR stems, attention streams with SAR cross-attention, heads) | `diewolke/models/bga.py` | `model.use_bga` → `no_bga.yaml` |
| C2 NDVI-consistency loss | `diewolke/losses.py` | `loss.ndvi` → `no_ndvi.yaml` |
| C3 calibrated per-pixel uncertainty (Laplace head, temperature fitted on val, reliability, AUSE) | `losses.py`, `calibration.py`, `evaluate.py` | `model.uncertainty` → `no_unc.yaml` |
| C4 Kharif benchmark + downstream classification | `kharif/` | `kharif/kharif.yaml` |
| C5 separate mask vs. end-to-end | `experiments/mask_ablation.py` (quick U-Net version); `two_stage.yaml` / `end_to_end.yaml` (real model) | `model.mask_source` |
| Matched baseline (C1–C3 all off) | same model | `matched_baseline.yaml` |
| DSen2-CR-style baseline (CARL loss) | `diewolke/models/dsen2cr.py` | `dsen2cr.yaml` |
| Classical XGBoost SAR-regression gap filling | `baselines/xgb_reconstruct.py` | |
| GLF-CR, UnCRtainTS | their public code, scored with `evaluate.py --preds` | `baselines/external.md` |
| Stage 3 temporal conditioning | not implemented; postponed in NOVELTY.md | |

## Quick start

```
pip install -e ".[clouds,dev]"
python -m pytest -q                                   # ~30 tests on fake GeoTIFFs, < 1 min on CPU
python -m diewolke.train --config configs/smoke.yaml  # random-data training run
```

Real work happens on Colab. Open [`notebooks/colab_runbook.ipynb`](notebooks/colab_runbook.ipynb)
and run it top to bottom: setup, data, timing check, masker, the full model and ablations,
evaluation, baselines, then Kharif. Every training cell resumes from its last checkpoint on
Drive, and `train.max_minutes` ends a session cleanly before Colab's limits.

## Main commands

```
# data (Colab): archives -> Drive once; extract + cache every session
python scripts/prepare_sen12mscr.py download --base_url "$SEN12MSCR_URL"
python scripts/prepare_sen12mscr.py extract
python scripts/prepare_sen12mscr.py cache --config configs/full.yaml

# train (any config; --set overrides any key)
python -m diewolke.train --config configs/full.yaml --set train.max_minutes=200

# evaluate several runs on the official test split, real clouds and synthetic clouds
python -m diewolke.evaluate --runs RUNS/full RUNS/matched_baseline --cache /content/cache --split test --mode real --out EVAL/real
python -m diewolke.evaluate --runs RUNS/full RUNS/matched_baseline --cache /content/cache --split test --mode synthetic --out EVAL/synth
python -m diewolke.compare EVAL/real/per_image_full.csv EVAL/real/per_image_matched_baseline.csv --per_band

# reconstruct GeoTIFFs (keeps georeferencing; tiles large scenes)
python -m diewolke.predict --run RUNS/full --s1 s1.tif --s2_cloudy s2.tif --out recon/
```

`evaluate.py` writes `report.md`, which contains:
- the headline table: PSNR, SSIM, MAE, RMSE, SAM, NDVI RMSE, and cloud-only variants;
- per-band MAE for B01–B12;
- paired differences against the first run, with 95% bootstrap CIs and the share of images
  where the run is better;
- for uncertainty runs: NLL, calibration error, Spearman(uncertainty, error), AUSE, and NDVI
  error after dropping the most uncertain pixels.

It also writes `summary.json`, per-image CSVs, and (with `--save_predictions`) the
reconstructions as `.npy` for figures and the crop task.

## Model in brief

The bridge runs between the clear image x0 (t = 0) and the cloudy image y (t = 1):
x_t = (1−t)·x0 + t·y + σ√(t(1−t))·ε, with σ = 0.5.

- **Network inputs:** x_t, t, the masked cloudy image, the mask and S1.
- **Network output:** a per-band Laplace mean, as a residual on the condition, and a log-scale.
- **Sampling:** starts from the cloudy image. One step (the DB-CR regime) is the default;
  `--steps N --eta 1 --n_samples K` gives multi-step and stochastic sampling, plus a
  sample-spread uncertainty.

Default size: 16.7M parameters at width 48. `no_bga` has 16.2M, so the C1 comparison is at
matched size.

**Training losses:**
- L1 on all bands;
- 0.1 × NDVI L1 (C2);
- 0.05 × Laplace NLL (C3). It is computed on the *detached* mean, so the head never changes
  what the mean learns, which keeps the C3 ablation clean.

**Training data:** real SEN12MS-CR pairs, which is the standard protocol. The masks are
s2cloudless plus a shadow heuristic. Pairs whose "clear" target is more than 5 % cloudy are
skipped. `data.mode=synthetic|mixed` adds transplanted real clouds, whose ground truth and mask
are exact.

## Compute (estimates, not measured)

FLOP counts at 128 × 128 crops: the full model is about 1.1 TFLOP per training step at batch 8.
On a T4, that suggests roughly 0.1–0.3 s/step, or about 2–5 h for the default 60k steps, if data
loading keeps up. The masker takes well under an hour. Measure with the notebook's timing cell
before planning sessions. Caches take ~4 MB per patch, so the default 3000 training patches use
~12 GB of local disk.

## Reporting caveats

- **SEN12MS-CR comparability.** DB-CR Table I uses all four seasons. Spring-only results
  (ROIs1158) are not directly comparable; say which seasons were used. The split is the official
  scene split, taken from the UnCRtainTS loader.
- **Real-cloud evaluation.** The target is a different date and the mask is estimated. Use the
  synthetic-cloud evaluation for exact-ground-truth claims, and report the gap between the two
  (NOVELTY.md ablation 3).
- **Re-implementations.** The DSen2-CR model here is a smaller re-implementation, so don't
  attach the paper's numbers to it. The XGBoost baseline is our approximation of classical
  reconstruction, not Gupta & Verma's exact method.
- **Kharif labels.** Without a crop-type raster, the downstream labels are ESA WorldCover
  land-cover classes. Call that a land-cover proxy until real crop labels are added
  (`kharif/README.md`).

## Repository map

```
diewolke/        package: bands, data/, models/, bridge, losses, metrics, calibration,
                 engine, train, evaluate, compare, predict, config, utils
configs/         base.yaml + one file per run (full, ablations, baselines, masker, smoke)
baselines/       xgb_reconstruct.py, external.md (GLF-CR / UnCRtainTS)
kharif/          Earth Engine scene search, patch export, benchmark build, crop classification
experiments/     mask_ablation.py (+ notebook wrapper)
scripts/         prepare_sen12mscr.py, visualize_samples.py
notebooks/       colab_runbook.ipynb
tests/           pytest suite on fake GeoTIFFs
datasets/        original exploration notebooks (superseded by scripts/; kept for reference)
```
