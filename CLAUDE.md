# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

dieWolke does cloud removal for all 13 Sentinel-2 bands using Sentinel-1 SAR. The full code
pipeline exists: data, Stage 1 masker, Stage 2 bridge model, baselines, evaluation, the Kharif
benchmark tools and a Colab runbook. It has only been run on synthetic random data on CPU (the
test suite and smoke configs). **Nothing has been trained on SEN12MS-CR yet, and there are no
results.** Do not report numbers that do not exist.

`NOVELTY.md` records the agreed research positioning: contributions C1–C5, what is already
published, baseline numbers and required ablations. Check it before proposing architecture
changes. `Plan.pptx` is the earlier plan; its PSNR/SSIM targets are superseded by NOVELTY.md.
The two `*_review.md` files are AI-assisted literature searches, and some of their citations are
unverified. `.venv/` is a local virtualenv; don't edit it.

## Commands

```
pip install -e ".[clouds,dev]"         # Colab: also .[kharif] for the benchmark
python -m pytest -q                    # ~30 tests, CPU, under a minute; uses fake GeoTIFFs
python -m diewolke.train --config configs/smoke.yaml       # random-data pipeline check
python -m diewolke.train --config configs/full.yaml --set data.root=... train.out_dir=...
python -m diewolke.evaluate --runs RUNS/full RUNS/matched_baseline --cache CACHE --split test --mode real --out EVAL
```
The local Windows machine has no PyTorch, very little disk and no dataset. Real runs happen on
Colab via `notebooks/colab_runbook.ipynb`.

## Layout

- `diewolke/bands.py`: band order (13 S2 bands B01–B12 incl. B8A, B10), band groups for C1,
  normalisation (S2 /10000 clipped to [0, 1]; S1 VV/VH clipped to [-25, 0] / [-32.5, 0] dB → [0, 1]),
  and the NDVI/NDMI/EVI functions.
- `diewolke/data/`
  - `sen12mscr.py`: triplet matching by `NAME_RE` (never path substitution), the official
    scene split (from the UnCRtainTS loader) and `.npy` memmap caches (~4 MB/patch).
  - `clouds.py`: s2cloudless probability, the shadow heuristic and synthetic cloud transplanting.
  - `datasets.py`: `CloudPairs` in real/synthetic/mixed modes.
- `diewolke/models/`
  - `reconstructor.py`: `BridgeNet` (Stage 2).
  - `bga.py`: band-group attention, stems and heads (C1).
  - `unet.py`: `Masker` (Stage 1).
  - `dsen2cr.py`: baseline.
  - `build_model()` takes only the keys each constructor accepts, so configs can inherit the
    bridge keys from `base.yaml`.
- `diewolke/bridge.py`: Brownian bridge with x_t = (1-t)x0 + t·y + σ√(t(1-t))ε. t = 1 is
  exactly the cloudy image, so 1-step sampling equals the prediction at t = 1.
- `diewolke/losses.py`: L1, NDVI-consistency (C2), and Laplace NLL on a *detached* mean (C3),
  plus CARL and the masker loss.
- `diewolke/engine.py`: `Pipeline` covers the mask source (data | masker | none), the condition,
  training loss and `predict()`. `load_run()` loads a run directory, using EMA weights.
- `diewolke/train.py`: resumable trainer (last.pt/best.pt, atomic saves, `train.max_minutes`,
  `train.stop_at`, checkpoint on Ctrl+C) with speed switches (precision fp16/bf16, compile,
  channels_last, cudnn benchmark, TF32, fused AdamW); see README "Making training faster".
- `diewolke/evaluate.py`: Stage 4 harness. Covers per-image metrics, per-band errors, bootstrap
  CIs, paired differences, uncertainty calibration (temperature fitted on val), the
  uncertainty-filtered NDVI, and `--preds` for external predictions.
- `diewolke/calibration.py`, `metrics.py`, `compare.py`, `predict.py` (GeoTIFF output with georef).
- `configs/`: `base.yaml` plus one file per run: full, matched_baseline, no_bga, no_ndvi,
  no_unc, two_stage, end_to_end, masker, dsen2cr, smoke. Overrides use `--set a.b=value`.
- `baselines/xgb_reconstruct.py`: classical SAR→S2 gradient-boosting gap filling.
  `baselines/run_external.py` runs the authors' pretrained UnCRtainTS / GLF-CR on a cached split and
  saves cache-ordered predictions for `evaluate.py --preds` (`baselines/external.md`). GLF-CR's
  compiled CUDA layer is replaced by `baselines/kernelconv2d_torch.py`. The GLF-CR repo has no
  licence, so never copy its code here; import it from a clone.
- `scripts/benchmark_speed.py`: compares speed settings; training logs report `data_wait`.
- `kaggle/`: Kaggle notebooks plus `kaggle_tools.py` (env, find-cache, restore across sessions, a
  time-budgeted run queue over one or two GPUs, prepare, publish) and the `kaggle.yaml` overlay
  (`train.py --overlay`). Configs accept overlays; `best.pt` is weights-only; a cache split counts
  as ready if its arrays exist even without the `.done` marker.
- `kharif/`: Earth Engine scene search → patch export → benchmark build → crop
  classification. See `kharif/README.md`.
- `experiments/mask_ablation.py`: standalone quick C5 ablation with plain U-Nets. Its notebook
  is a wrapper that embeds the script; regenerate the notebook if the script changes.
- `scripts/prepare_sen12mscr.py` (download/extract/cache) and `scripts/visualize_samples.py`
  supersede `datasets/VisuzalizeDataset.py` and `datasets/visualizeDataset.ipynb`. Those old
  files have a path-matching bug and a hardcoded FTP credential. Don't copy that credential
  anywhere; the new script reads the URL from `--base_url` or `SEN12MSCR_URL`.

## Conventions and caveats

- Metrics follow the SEN12MS-CR papers: all 13 bands, data range 1, SAM in degrees, per image
  then averaged. Spring-only training and testing is **not** comparable with DB-CR Table I,
  which uses all seasons. Say so wherever numbers are reported.
- Real-pair masks are s2cloudless (> 0.4, dilated) plus a shadow heuristic, so they are
  estimates. Synthetic-cloud masks are exact.
- The DSen2-CR model here is a smaller re-implementation, so don't quote the paper's numbers
  for it. The XGBoost baseline is "in the spirit of" Gupta & Verma, not their method.
- Kharif downstream labels default to ESA WorldCover classes, which are a land-cover proxy and
  not crop types.
- Stage 3 (temporal conditioning) is not implemented; NOVELTY.md postpones it.
