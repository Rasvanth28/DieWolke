# Results: SEN12MS-CR spring, first full evaluation

*Evaluated 2026-10-25 on Kaggle (`kaggle/03_evaluate.ipynb`, T4). Numbers are copied from the
evaluation output (`eval_results.zip`, extracted locally to `kaggle_logs/eval/`, not in git).*

## Setup and caveats

- **Data.** SEN12MS-CR spring (ROIs1158) only, official scene split. Training used 2,434 patches
  (the cached subset), validation 300, test 1,500 patches at 256 × 256.
  **Spring-only training and testing is not comparable with DB-CR Table I, which uses all seasons.**
- **Metrics.** All 13 bands, data range 1, SAM in degrees, per image then averaged. `_cloud`
  metrics use cloudy pixels only. 95% CIs are paired bootstrap over test images.
- **Real vs synthetic.** "Real" uses the dataset's cloudy images with estimated masks
  (s2cloudless > 0.4 dilated, plus a shadow heuristic). "Synthetic" transplants clouds onto clear
  images, so ground truth and masks are exact.
- **One training seed per run.** The CIs cover variation over test images, not over training runs.
  Gaps of a few tenths of a dB between our own runs may not survive another seed.
- **Inference.** Bridge models use 1 sampling step (deterministic), EMA weights, `best.pt` by
  validation PSNR.
- **Published baselines** are the authors' checkpoints, run unchanged: UnCRtainTS
  `monotemporalL2` (L2-trained, no uncertainty output) and GLF-CR `CR_net.pth`. They were trained on
  all of SEN12MS-CR (all seasons, the full training split), far more data than our 2,434 patches.
- **XGBoost** is "in the spirit of" Gupta & Verma (2025), not their method. 837 of 1,500 test
  patches had too few clear pixels and used the pooled model trained on the training split.

**Adapter check.** On our spring test, UnCRtainTS scores 28.83 dB / SSIM 0.882 / SAM 8.00 / MAE
0.028, and GLF-CR 28.59 / 0.885 / 7.79 / 0.028. The UnCRtainTS paper (Ebel et al. 2023, Table 4,
all-season SEN12MS-CR test) reports 28.90 / 0.880 / 8.32 / 0.027 and 28.64 / 0.885 / 8.98 / 0.028.
The close match indicates the input conventions in `baselines/run_external.py` are right. (The
31.28 / 31.85 dB in DB-CR Table I come from a different protocol.)

## 1. Main comparison, real clouds (test, 1,500 patches)

| method | PSNR ↑ | SSIM ↑ | MAE ↓ | RMSE ↓ | SAM ↓ | NDVI RMSE ↓ | PSNR cloud ↑ | NDVI RMSE cloud ↓ |
|---|---|---|---|---|---|---|---|---|
| cloudy input | 18.624 | 0.6761 | 0.1247 | 0.1616 | 13.817 | 0.2686 | 17.730 | 0.2867 |
| XGBoost | 25.981 | 0.8028 | 0.0384 | 0.0532 | 9.535 | 0.1669 | 25.699 | 0.1765 |
| **full** (ours, C1+C2+C3) | 28.124 | 0.8664 | 0.0292 | 0.0431 | 8.403 | 0.1530 | 28.033 | 0.1503 |
| matched_baseline (ours, C1–C3 off) | 28.324 | 0.8699 | 0.0286 | 0.0423 | 8.319 | 0.1522 | 28.188 | 0.1500 |
| UnCRtainTS (published weights) | **28.830** | 0.8820 | 0.0280 | **0.0400** | 7.999 | 0.1449 | 28.511 | 0.1468 |
| GLF-CR (published weights) | 28.588 | **0.8867** | **0.0279** | 0.0406 | **7.789** | **0.1286** | **28.600** | **0.1292** |

Paired differences vs `full` (other minus `full`, 95% CI, share of images where the other is better):

| vs full | PSNR | SSIM | SAM | NDVI RMSE |
|---|---|---|---|---|
| matched_baseline | +0.20 [+0.14, +0.27], 52% | +0.0035 [+0.0026, +0.0043] | −0.08 [−0.17, +0.00] | −0.0008 [−0.0027, +0.0011] |
| UnCRtainTS | +0.71 [+0.59, +0.81], 69% | +0.0156 [+0.0139, +0.0173] | −0.40 [−0.54, −0.27] | −0.0080 [−0.0112, −0.0049] |
| GLF-CR | +0.46 [+0.36, +0.57], 55% | +0.0203 [+0.0188, +0.0219] | −0.61 [−0.77, −0.47] | −0.0243 [−0.0274, −0.0214] |
| XGBoost | −2.14 [−2.26, −2.03], 16% | −0.0636 [−0.0663, −0.0608] | +1.13 [+0.97, +1.29] | +0.0139 [+0.0095, +0.0182] |

### 1b. By cloud cover and by scene (real clouds)

From `python scripts/eval_breakdown.py EVAL/test_real_baselines --metrics psnr ndvi_rmse`.
Cloud fraction is the estimated mask's share of the patch.

| cloud fraction | n | input | XGBoost | full | matched_baseline | UnCRtainTS | GLF-CR |
|---|---|---|---|---|---|---|---|
| 0.0–0.1 | 139 | 29.32 | 29.33 | 30.71 | 31.62 | 33.08 | 32.09 |
| 0.1–0.3 | 148 | 26.47 | 27.06 | 29.65 | 30.11 | 30.88 | 31.07 |
| 0.3–0.6 | 196 | 23.05 | 25.92 | 29.34 | 29.67 | 30.55 | 30.23 |
| 0.6–0.9 | 299 | 19.04 | 25.20 | 28.62 | 28.61 | 29.18 | 28.72 |
| 0.9–1.0 | 718 | 13.56 | 25.45 | 26.77 | 26.83 | 26.97 | 26.89 |

Paired differences, run minus `full`:

| | heavy cloud ≥ 0.9 (n = 718): PSNR | NDVI RMSE | partial cloud < 0.9 (n = 782): PSNR | NDVI RMSE |
|---|---|---|---|---|
| matched_baseline | +0.06 [−0.02, +0.14] | +0.0038 [+0.0008, +0.0067] | +0.33 [+0.24, +0.42] | −0.0050 [−0.0071, −0.0029] |
| UnCRtainTS | +0.20 [+0.06, +0.33] | −0.0151 [−0.0199, −0.0105] | +1.17 [+1.01, +1.33] | −0.0015 [−0.0054, +0.0026] |
| GLF-CR | +0.12 [−0.02, +0.28] | −0.0261 [−0.0313, −0.0211] | +0.78 [+0.63, +0.92] | −0.0228 [−0.0261, −0.0193] |
| XGBoost | −1.32 [−1.49, −1.15] | +0.0128 [+0.0052, +0.0204] | −2.90 [−3.04, −2.76] | +0.0149 [+0.0109, +0.0192] |

- **Under near-total cloud (48% of the test set), where only SAR carries information, the four
  deep models are within 0.2 dB.** `full` is level with the matched baseline and GLF-CR (CIs
  include 0) and 0.20 dB behind UnCRtainTS, and has lower NDVI error than the matched baseline.
- **The gap to the published models comes from partly cloudy patches** (0.8–1.2 dB), i.e. from
  using the visible pixels and the date difference between the cloudy and clear acquisitions.
  That is where more training data would be expected to help most.
- Synthetic clouds show the same split: matched_baseline minus `full` is +0.15 [−0.02, +0.33] dB
  at ≥ 0.9 cover (n = 125) and −0.23 [−0.32, −0.15] dB below it (n = 1,375).

PSNR per test scene (the spring test split has only 5 scenes):

| scene | n | mean cloud | input | XGBoost | full | matched_baseline | UnCRtainTS | GLF-CR |
|---|---|---|---|---|---|---|---|---|
| 31 | 300 | 0.69 | 16.59 | 27.33 | 30.75 | 31.24 | 31.11 | 31.33 |
| 44 | 293 | 0.71 | 17.08 | 25.03 | 27.27 | 27.49 | 26.40 | 28.26 |
| 106 | 295 | 0.59 | 24.68 | 28.26 | 29.61 | 30.42 | 31.10 | 30.18 |
| 123 | 282 | 0.57 | 21.25 | 25.28 | 27.35 | 27.27 | 29.13 | 27.98 |
| 140 | 330 | 0.92 | 14.18 | 24.16 | 25.83 | 25.45 | 26.62 | 25.48 |

The ranking changes from scene to scene (on scene 44 both of our models beat UnCRtainTS; on
scene 140, the cloudiest, `full` beats the matched baseline and GLF-CR). Patches from one scene are
correlated, so the per-image bootstrap CIs understate the uncertainty of the overall ranking.

## 2. Main comparison, synthetic clouds (exact masks)

| method | PSNR ↑ | SSIM ↑ | MAE ↓ | SAM ↓ | NDVI RMSE ↓ | PSNR cloud ↑ | NDVI RMSE cloud ↓ |
|---|---|---|---|---|---|---|---|
| cloudy input | 19.510 | 0.6887 | 0.0724 | 8.808 | 0.2258 | 17.208 | 0.2838 |
| **full** | **29.080** | 0.8699 | **0.0251** | **7.491** | **0.1478** | **27.926** | **0.1708** |
| matched_baseline | 28.880 | **0.8761** | 0.0255 | 7.616 | 0.1509 | 27.636 | 0.1718 |

matched_baseline minus full: PSNR −0.20 [−0.28, −0.12]; PSNR cloud −0.29 [−0.37, −0.22];
SSIM +0.0062 [+0.0051, +0.0074]; SAM +0.13 [+0.02, +0.23]; NDVI RMSE +0.0031 [+0.0010, +0.0053].

## 3. Ablations, real clouds (20k steps each)

| run | PSNR ↑ | SSIM ↑ | SAM ↓ | NDVI RMSE ↓ | NDVI RMSE cloud ↓ |
|---|---|---|---|---|---|
| full_20k | 27.839 | 0.8591 | 8.690 | 0.1518 | 0.1578 |
| no_bga (−C1) | 28.203 | 0.8622 | 8.653 | 0.1529 | 0.1497 |
| no_ndvi (−C2) | 27.997 | 0.8606 | 8.624 | 0.1571 | 0.1549 |
| no_unc (−C3) | 27.986 | 0.8638 | 8.426 | 0.1494 | 0.1520 |

Ablation minus full_20k:

| | PSNR | NDVI RMSE | NDVI RMSE cloud |
|---|---|---|---|
| no_bga | +0.36 [+0.29, +0.44] | +0.0011 [−0.0005, +0.0027] | −0.0081 [−0.0118, −0.0046] |
| no_ndvi | +0.16 [+0.11, +0.20] | **+0.0053 [+0.0038, +0.0067]** | −0.0029 [−0.0061, +0.0000] |
| no_unc | +0.15 [+0.09, +0.21] | −0.0024 [−0.0039, −0.0009] | −0.0058 [−0.0083, −0.0036] |

## 4. C5: separate mask stage vs end-to-end (synthetic clouds, 20k steps)

| configuration | PSNR ↑ | SSIM ↑ | SAM ↓ | PSNR cloud ↑ |
|---|---|---|---|---|
| full_20k, exact masks | 28.746 | 0.8625 | 7.952 | 27.556 |
| full_20k, masks from our Stage 1 masker | 28.757 | 0.8642 | 7.948 | 27.582 |
| end_to_end (no mask at all) | 27.658 | 0.7841 | 9.145 | 26.196 |
| full (40k), exact masks | 29.080 | 0.8699 | 7.491 | 27.926 |
| full (40k), masker masks | 29.084 | 0.8713 | 7.485 | 27.947 |

end_to_end minus full_20k: PSNR −1.09 [−1.20, −0.98] (better on 26% of images), SSIM −0.078
[−0.081, −0.076] (better on 3%), PSNR cloud −1.36 [−1.48, −1.24]. Its NDVI RMSE is slightly lower
(−0.0049 [−0.0081, −0.0016]). Its B01 error is almost twice that of the masked model. The masker
(validation IoU 0.964) matches the exact masks.

## 5. C3: uncertainty (real clouds, `full`)

| | NLL | calibration error | Spearman(uncertainty, error) | AUSE |
|---|---|---|---|---|
| raw | −2.209 | 0.085 | 0.431 | 0.00606 |
| temperature-calibrated on val | −2.230 | 0.055 | 0.440 | 0.00593 |

NDVI RMSE on the pixels kept after dropping the most uncertain ones:

| kept | 100% | 90% | 75% | 50% |
|---|---|---|---|---|
| NDVI RMSE | 0.182 | 0.171 | 0.158 | 0.134 |

On synthetic clouds the raw calibration error is 0.037 and Spearman 0.55; the temperature fitted
on real-cloud validation makes it worse (0.060), so calibration does not transfer between cloud
types. Neither the matched baseline nor the UnCRtainTS checkpoint used here outputs uncertainty.

## What the results support

1. **C5 (supported, strongest result).** A separate mask stage beats end-to-end training by
   1.1 dB PSNR and 0.08 SSIM at matched steps, and Stage 1 masks are as good as exact ones.
2. **C3 (supported as a usable signal).** Per-pixel uncertainty ranks errors (Spearman 0.43–0.55)
   and filtering by it lowers NDVI RMSE by 26% at 50% retention. It costs about 0.15 dB PSNR.
3. **C2 (small, target-specific).** The NDVI loss lowers whole-image NDVI RMSE by about 3.5%
   (no_ndvi vs full_20k), at a 0.16 dB PSNR cost; inside clouds the effect is not significant.
4. **C1 (not supported).** Band-group attention lowers PSNR by 0.36 dB and does not lower NIR,
   red-edge or SWIR error, the bands it targets.
5. **Overall.** On real clouds `full` is 0.20 dB below the matched baseline; on synthetic clouds
   it is 0.20 dB above it. Both of our models trail the published models by 0.3–0.7 dB, which
   were trained on far more data; both beat the XGBoost gap-filling baseline by over 2 dB.
   Under near-total cloud the deep models are within 0.2 dB of each other; the gap to the
   published models is in partly cloudy patches.

## Open points

- Training data: 2,434 patches. A larger cache would show whether the gap to the published models
  is data or architecture. Planned, not yet run.
- Seeds: one per run.
- Validation (200 patches) ranked the ablations differently from test; use test numbers only.
- The spring test split covers 5 scenes; CIs are over images, not scenes.
- The Kharif benchmark (C4) is not built yet.
