# dieWolke — Novelty and Positioning

*Last updated: 2026-10-06. Based on `sentinel_cloud_removal_literature_review.md` and `strict_kharif_cloud_removal_prior_art_review.md`. DB-CR and Gupta & Verma were checked against their primary sources.*

## Novelty statement

> A SAR-conditioned diffusion-bridge model for all 13 Sentinel-2 bands, with RGB / red-edge–NIR / SWIR band-group attention, an NDVI-consistency loss, and calibrated per-pixel uncertainty, evaluated on a new Indian Kharif monsoon benchmark built from real monsoon cloud masks, with downstream crop classification compared against classical reconstruction (Gupta & Verma, 2025).

**Wording for the paper (cautious):**

> To our knowledge, no work identified in our review jointly evaluates a SAR-conditioned all-13-band diffusion/bridge reconstruction model with explicit visible / red-edge–NIR / SWIR group processing, a vegetation-index consistency loss, calibrated per-pixel uncertainty, and a real-monsoon-mask Indian Kharif crop benchmark with downstream crop classification.

## Contributions

| # | Contribution | Status in prior art | Evidence we must produce |
|---|---|---|---|
| C1 | **Band-group attention** (separate RGB, red-edge/NIR, SWIR branches) in a SAR-conditioned generative model | Not found. DB-CR uses SAR-vs-optical cross-attention, not spectral groups. | Per-band error (B02–B12) vs. the matched baseline, with the biggest gains expected in NIR / red-edge / SWIR |
| C2 | **NDVI-consistency loss** added to all-band reconstruction | Not found. DB-CR trains with L1 only; CloudBreaker only *evaluates* NDVI after training. | NDVI / EVI RMSE vs. the matched baseline, without a drop in PSNR / SSIM |
| C3 | **Calibrated per-pixel uncertainty** from the bridge model | Not found in diffusion/flow cloud removal. UnCRtainTS has uncertainty but is not generative. | Reliability curves, uncertainty–error correlation; show that filtering high-uncertainty pixels improves NDVI / crop classification |
| C4 | **Indian Kharif monsoon benchmark** (July–October), with real monsoon cloud masks copied onto rare clear Kharif scenes, plus downstream crop classification | Partially done. Gupta & Verma (IIRS/ISRO, 2025) do cloud reconstruction → paddy classification in India, but with classical ML (XGBoost), no SAR, no exact-ground-truth protocol, and no released benchmark. | Released tile IDs, dates, masks, crop labels, split rules; crop-classification accuracy vs. cloudy input, Gupta & Verma-style XGBoost reconstruction, and our model |
| C5 | **Controlled ablation: separate mask stage vs. end-to-end** (supporting finding) | Not found as a clean, matched ablation on SEN12MS-CR / CR-TS | `experiments/mask_ablation.py` results; decides whether Stage 1 stays |

The novelty is the **combination** of these plus the Kharif benchmark. Each part on its own is a modest step.

## Not novel (don't claim)

- SAR-conditioned diffusion bridge for cloud removal: **DB-CR** (Hu, Lohit, Kamilov, Marks, TGRS 2025, arXiv 2504.03607)
- SAR-conditioned flow matching for Sentinel-2: **CloudBreaker** (Ahmed, Nowreen, Rahman 2025, arXiv 2508.03608). It takes SAR only as input and outputs RGB+NIR.
- SAR–optical fusion in general (DSen2-CR, GLF-CR, …)
- Multi-temporal conditioning (UnCRtainTS, CGD-Net, TCM-CR), so **the Stage 3 temporal encoder is postponed**
- Synthetic cloud generation / injection (SatelliteCloudGenerator, SAR-DeCR, …)
- "First Indian cloud reconstruction + crop classification": Gupta & Verma did this first

## Benchmark numbers to position against (SEN12MS-CR, DB-CR Table I)

| Method | PSNR (dB) | SSIM | MAE | SAM (°) |
|---|---:|---:|---:|---:|
| DSen2-CR | 30.29 | 0.878 | 0.034 | 7.008 |
| GLF-CR | 31.85 | 0.896 | 0.020 | 6.722 |
| UnCRtainTS | 31.28 | 0.887 | 0.023 | 6.899 |
| DiffCR | 32.25 | 0.909 | 0.019 | 5.335 |
| **DB-CR** | **33.47** | **0.922** | **0.016** | **4.740** |

**Targets.** The targets in `Plan.pptx` (PSNR > 28 dB, SSIM > 0.88) are below DSen2-CR (2020), so replace them. Realistic goals on a T4 budget:
- Get within about 1 dB PSNR of DB-CR.
- Clearly win on per-band NIR / red-edge / SWIR error, NDVI error, uncertainty calibration, and Kharif crop classification.

**Baselines.** DB-CR has no public code. The fair comparison is our own bridge model with C1–C3 removed (matched baseline), plus GLF-CR and UnCRtainTS, which have public code.

## Required ablations

1. Full model vs. without band-group attention vs. without NDVI loss vs. without uncertainty head
2. Separate mask stage vs. end-to-end (C5)
3. Synthetic-cloud test vs. real-cloud test, to report how much results drop on real clouds
4. India Kharif benchmark: cloudy input vs. classical (XGBoost) reconstruction vs. our model, for crop classification

## Open risks

- **The India gap rests on two AI-assisted searches.** Do a manual Google Scholar check ("cloud removal" + Kharif / paddy / Punjab / ISRO / IIRS) before submission.
- **Gupta & Verma full text was not accessible** (Springer paywall). Confirm their season, region and protocol so our claimed differences are accurate.
- **Unverified citations in the review files.** Entries with placeholder authors ("A. B. S. ?", "Z. ? et al.") or missing metadata must not be cited until checked.
- **Rare clear Kharif scenes.** The benchmark depends on finding enough clear July–October Sentinel-2 scenes over the target districts. Check availability in Google Earth Engine early.
- **Compute.** Diffusion-bridge training on a Colab T4 is tight. Plan on 128×128 crops and few-step sampling (DB-CR reports single-step inference).

## Must-read papers

1. Hu et al. 2025, **DB-CR**: closest architecture, main baseline. https://arxiv.org/abs/2504.03607
2. Gupta & Verma 2025: closest Indian precedent. https://doi.org/10.1007/s41976-025-00254-z
3. Ahmed et al. 2025, **CloudBreaker**: flow-matching comparator, NDVI evaluation. https://arxiv.org/abs/2508.03608
4. Ebel et al. 2023, **UnCRtainTS**: uncertainty baseline, public code. https://arxiv.org/abs/2304.05464
5. Prashnani & Justice 2026: Kharif multi-crop context in Central India. https://doi.org/10.3390/rs18081238
6. Tsardanidis et al. 2024: precedent for synthetic cloud masking plus a downstream task. https://arxiv.org/abs/2403.09554
