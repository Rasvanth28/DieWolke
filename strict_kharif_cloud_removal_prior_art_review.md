# Strict literature check: Kharif cloud removal and SAR-conditioned generative reconstruction

**Scope.** This is a conservative, link-only audit covering 2019–2026. I include only items for which I could verify a working publisher, DOI, arXiv, repository, thesis, or proceedings URL. “Not found” means *not found in the linked material reviewed here*; it is not proof of non-existence. In particular, I do **not** infer that a paper used Kharif, exact clear-image ground truth, or an NDVI loss unless the linked paper says so.

## Executive verdict

**Part A — exact proposed benchmark:** **Partially done, but not as the full combination.** Gupta and Verma (IIRS/ISRO) already evaluate Sentinel-2 cloud reconstruction followed by in-season paddy classification in India, with improved crop-map accuracy; however, their publication does not establish a Kharif July–October benchmark built by transplanting real monsoon masks onto rare contemporaneous clear Kharif targets. [page:31]

**Part B — full model:** **Not found as one published system.** DB-CR is a SAR-conditioned diffusion bridge for all 13 Sentinel-2 bands, but uses MAE—not an NDVI/vegetation-index training loss—and does not publish uncertainty maps. CloudBreaker is a SAR-to-four-band Sentinel-2 conditional flow-matching model that reports NDVI/NDWI *after* reconstruction, not as a stated training loss; it has no band-group branches or calibrated uncertainty. [page:18][page:17]

## Part A: India and South Asia

### Direct cloud reconstruction plus crop task

| Work | Region / season | Sensors and method | Output / target | Ground truth | Downstream task | Link |
|---|---|---|---|---|---|---|
| **Shikha Gupta & Prabhakar Alok Verma (2026)**, “Generating Cloud-Free Sentinel-2 Images for Enhanced In-Season Crop Type Classification: a Case Study,” *Remote Sensing in Earth Systems Sciences*, 9, 9. | Indian case study; the publisher page identifies the authors as Indian Institute of Remote Sensing, ISRO. The available page does not let me verify a July–October Kharif acquisition design or the study district, so I do not label it Kharif. | Sentinel-2 surface reflectance. Cloud detection with RF/SVM/XGBoost; cloud reconstruction with RF/XGBoost; XGBoost crop classification. | Blue, green, red, NIR and red-edge bands; reconstructed multispectral imagery and NDVI time series. | The accessible article page describes masking then ML reconstruction, but does not establish paired, contemporaneous clear-target ground truth or real-mask transplantation. | Paddy map accuracy was 82.53% after reconstruction versus 75.99% from cloud-affected imagery. | [DOI](https://doi.org/10.1007/s41976-025-00254-z) |

**Why this matters:** this is the closest direct precedent for “cloud/gap filling → downstream crop classification” from an Indian institution. Your novelty should therefore not be phrased as the first Indian cloud reconstruction plus crop classification; instead, distinguish the **monsoon-specific paired benchmark design**, **real-cloud-mask transplantation**, **explicit exact-target protocol**, **multi-crop geography**, and **SAR-conditioned generative evaluation**. [page:31]

### Monsoon crop monitoring that avoids reconstruction

| Work | Region / season | Sensors and method | Reconstructs? | Ground truth / task | Link |
|---|---|---|---|---|---|
| **Meghavi Prashnani & Chris Justice (2026)**, “Evaluating SAR-Derived Phenological Metrics for Monsoon (Kharif) Crop Monitoring in Diversified Agricultural Systems: Insights from Central India,” *Remote Sensing*, 18(8), 1238. | Five districts / agroclimatic zones in Madhya Pradesh; 2021 Kharif. Crops: soybean, rice, urad, maize, cotton. | Sentinel-1 VV, VH, VH/VV time series; Savitzky–Golay smoothing; interpretable SAR phenological metrics and Random Forest. | No optical-band or NDVI reconstruction; SAR-only crop monitoring. | Field surveys: 289 fields. Five-class accuracy 48.3%; crop-specific and cross-district results reported. | [DOI](https://doi.org/10.3390/rs18081238) |
| **A. B. S. ? (2020 ISPRS Archives paper)**, “Assessment of SAR-based NDVI estimates over Indian monsoon crops” (title needs verification before manuscript use). | Indian subcontinent southwest monsoon; source states mid-June to mid-October. | Sentinel-1 and Sentinel-2 red/NIR, SAR-to-NDVI relation. | Estimates NDVI from SAR; no cloud-removal image reconstruction established from accessible record. | Reports that SAR-based NDVI estimates reasonably match optical NDVI for grown canopy. | [ISPRS Archives PDF](https://isprs-archives.copernicus.org/articles/XLIII-B3-2020/1379/2020/isprs-archives-XLIII-B3-2020-1379-2020.pdf) |
| **P. Teluguntla et al. (2020)**, “Synergistic Use of Radar and Optical Satellite Data for Improved Monsoon Cropland Mapping in India,” *Remote Sensing*, 12(3), 522. | Five Indian agro-ecological regions; monsoon cropland mapping. | Sentinel-1 plus Sentinel-2 NDVI; Radar Optical Cross Masking (ROM). | No reconstruction: it uses cloud-screened Sentinel-2 and SAR/optical fusion for mapping. | Cropland mapping: reported 90% SAR-only and 93% combined accuracy. | [DOI](https://doi.org/10.3390/rs12030522) |
| **Kharif crop-type mapping of Haryana State using Geospatial Technologies** (Indian conference/technical paper; author metadata not safely recoverable from the indexed PDF excerpt). | Haryana Kharif; paddy, cotton, bajra, sugarcane and others. | Sentinel-1 VV/VH and Sentinel-2 NIR/red/green time series. | No reconstruction; compares SAR and optical crop classification under cloud constraints. | Crop classification; reports Kharif cloud-free optical scarcity. | [Proceedings PDF](https://incaindia.org/images/uploads/Paper%20-%2026.pdf) |
| **“Mapping of Rice Crop using Microwave and Optical Remote Sensing Data”** (Indian thesis/repository record; author metadata not safely recoverable from indexed page). | Punjab and Haryana, May–October 2019 Kharif. | Sentinel-1 dual-pol time series and Sentinel-2 NDVI thresholding. | No reconstruction; uses selected September/October optical scenes because full Sentinel-2 time series was cloud constrained. | Rice-area estimation compared with government-reported area. | [Krishikosh repository](https://krishikosh.egranth.ac.in/items/6ed58e9f-c039-4b22-a54e-1a9147fd2970) |

### SAR–optical/NDVI gap filling outside India

These establish technical precedents for reconstructing an agricultural **index**, but not the Indian Kharif cloud-removal benchmark you propose.

| Work | Region / season | Sensors and method | Output | Ground truth | Downstream task | Link |
|---|---|---|---|---|---|---|
| **Iason Tsardanidis, Alkiviadis Koukos, Vasileios Sitokonstantinou, Thanassis Drivas & Charalampos Kontoes (2024)**, “Cloud gap-filling with deep learning for improved grassland monitoring,” arXiv:2403.09554. | Lithuania, April–September 2020—not South Asia. | Sentinel-1 backscatter/coherence plus cloud-free Sentinel-2 NDVI; CNN-RNN. | NDVI only, at six-day time steps. | Synthetic masking: real cloud-occurrence patterns were bootstrapped and assigned to dense NDVI series; interpolated targets were used under specified conditions. | Mowing-event detection; MAE 0.024 and R-squared 0.92 reported for the gap-filled NDVI task. | [arXiv](https://arxiv.org/abs/2403.09554) |
| **Z. ? et al. (2022)**, “Fusion of optical and SAR images based on deep learning to reconstruct vegetation NDVI time series in cloudy regions.” | China; precise region/author list not safely verified from the linked AGRIS record. | Transformer Temporal-Spatial Model (TTSM), SAR–optical time series. | NDVI time series. | Not verified from accessible record. | Not verified from accessible record. | [AGRIS record](https://agris.fao.org/search/en/providers/122535/records/65defd0963b8185d9caa3110) |
| **Author list not safely verified from search result (2024)**, “A new deep learning-based model for reconstructing high-quality NDVI time series data,” *International Journal of Digital Earth*. | Not India, based on the accessible listing. | SAR and optical fusion for NDVI time-series reconstruction. | NDVI only. | Not verified here. | Not verified here. | [Publisher URL](https://www.tandfonline.com/doi/full/10.1080/17538947.2024.2407941) |

### Dataset and benchmark context

| Dataset / work | What the linked source verifies | India June–October count |
|---|---|---|
| **SEN12MS-CR**, used by Hu et al. | The benchmark contains paired/co-registered Sentinel-1 and Sentinel-2 data across 175 globally distributed ROIs, four 2018 seasons, and 122,218 256-by-256 patch triplets. | **Not reported in the cited DB-CR paper.** Do not claim an India/monsoon scene count without querying metadata/geometries and acquisition dates. [page:18] |
| **CloudBreaker training data**, Ahmed et al. | 900 paired Sentinel-1/Sentinel-2 chips from 18 global flood events, derived from the Cloud to Street–Microsoft Flood and Clouds dataset. | **Not reported.** Nepal and Bangladesh-related case studies appear in the paper, but it is not an Indian agricultural benchmark. [page:17] |

### Part A verdict

**Question: “Has anyone built a Kharif-monson benchmark with exact ground truth and a downstream crop task?” — `Partially`.**

- **Closest direct published precedent:** Gupta & Verma perform Sentinel-2 cloud detection, ML reconstruction, and paddy classification at IIRS/ISRO, and quantify a classification gain. The accessible paper page does **not** verify Kharif July–October, real monsoon-mask transplantation, rare clear Kharif exact targets, SAR conditioning, or a standard released benchmark. [page:31]
- **Closest Kharif agriculture precedent:** Prashnani & Justice are explicitly Kharif, Indian, smallholder-aware, multi-crop, and field-supported, but use Sentinel-1 only and do **not** reconstruct optical bands or NDVI. [page:41]
- **Closest synthetic-mask protocol precedent:** Tsardanidis et al. synthetically mask dense NDVI series using bootstrapped real cloud-occurrence patterns and evaluate a downstream task, but work in Lithuanian grasslands, reconstruct NDVI only, and do not provide an Indian Kharif cloudy/clear optical-image benchmark. [page:36]

## Part B: Model-by-model audit

### CloudBreaker

**Citation.** Saleh Sakib Ahmed, Sara Nowreen & M. Sohel Rahman (2025), “CloudBreaker: Breaking the Cloud Covers of Sentinel-2 Images using Multi-Stage Trained Conditional Flow Matching on Sentinel-1,” arXiv:2508.03608. [arXiv](https://arxiv.org/abs/2508.03608)

| Question | Strict answer |
|---|---|
| Does it use a cloudy optical image? | **No, not as the stated model input.** The mapping is Sentinel-1 to Sentinel-2: separate VQ-VAEs encode 2-channel S1 and 4-channel S2, then a latent conditional flow-matching U-Net translates S1 latent to S2 latent. The paper illustrates application to cloudy optical scenes, but its stated conditional input is Sentinel-1. [page:17] |
| Which bands does it output? | **Four Sentinel-2 channels: RGB plus NIR**, not all 13 bands. NDVI and NDWI are computed afterward from reconstructed channels. [page:17] |
| Vegetation-index term in training loss? | **Not found.** The linked method describes flow-matching interpolation / latent translation and computes NDVI/NDWI only in post-processing and evaluation. That is index evaluation, not an NDVI-loss term. [page:17] |
| Band-specific/group attention? | **Not found.** It separates RGB and NIR only after output for index calculation; the stated translation model is a latent U-Net, not visible/NIR/SWIR group branches or spectral-group attention. [page:17] |
| Per-pixel uncertainty / calibration? | **Not found.** No predictive variance, ensemble uncertainty, calibration curve, or uncertainty–error analysis was identified in the linked manuscript. [page:17] |
| Dataset / region | 900 paired chips from 18 global flood events from Cloud to Street–Microsoft Flood and Clouds; scenes include global cases. It has Nepal Flood and Cyclone Remal (Bangladesh/West Bengal) case studies, not a Kharif agriculture benchmark. [page:17] |
| Reported evaluation | For FM-Cosine at 100 steps: RGB FID 0.7432, RGB SSIM 0.6346, RGB PSNR 17.29 dB, NDVI SSIM 0.6156, NDWI SSIM 0.6874. [page:17] |

### DB-CR

**Citation.** Yuyang Hu, Suhas Lohit, Ulugbek S. Kamilov & Tim K. Marks (2025), “Multimodal Diffusion Bridge with Attention-Based SAR Fusion for Satellite Image Cloud Removal,” arXiv:2504.03607. [arXiv](https://arxiv.org/abs/2504.03607)

| Question | Strict answer |
|---|---|
| Inputs | A **single cloudy 13-channel optical image plus aligned two-channel SAR**; it is not SAR-only. [page:18] |
| Outputs | **All 13 Sentinel-2 multispectral channels.** [page:18] |
| Architecture | Diffusion bridge between cloudy and clear optical states, with optical-restoration and SAR-feature branches plus cross-modal SAR fusion blocks. The attention is **modality cross-attention**, not stated band-group attention for RGB, red-edge/NIR, and SWIR. [page:18] |
| Training loss | MAE / L1 between predicted cloud-free output and target at sampled bridge time. No NDVI/EVI/NDWI/LAI or other vegetation-index loss is stated. [page:18] |
| Uncertainty | **Not found.** A diffusion bridge is stochastic/probabilistic in formulation, but the paper does not publish a calibrated per-pixel predictive uncertainty map or calibration analysis. [page:18] |
| Dataset | SEN12MS-CR: 175 global ROIs, four 2018 seasons; 122,218 patch triplets. [page:18] |
| Code | **No public code link is provided in the linked arXiv manuscript.** This is “not found,” not proof that no private/released code exists. |

#### DB-CR exact results table

The following are **Table I values in the linked arXiv v1**, for DB-CR with NFE=1 on SEN12MS-CR. Metrics are aggregate multi-spectral reconstruction metrics except FID/LPIPS, which the paper computes on RGB. [page:18]

| Method | PSNR dB | SSIM | MAE | SAM degrees | FID | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| DSen2-CR | 30.29 | 0.878 | 0.034 | 7.008 | 80.37 | 0.431 |
| GLF-CR | 31.85 | 0.896 | 0.020 | 6.722 | 87.63 | 0.452 |
| UnCRtainTS | 31.28 | 0.887 | 0.023 | 6.899 | 85.94 | 0.410 |
| DiffCR | 32.25 | 0.909 | 0.019 | 5.335 | 71.55 | 0.398 |
| **DB-CR** | **33.47** | **0.922** | **0.016** | **4.740** | **61.91** | **0.316** |

The explicit comparison baselines are **DSen2-CR, GLF-CR, UnCRtainTS, and DiffCR**. [page:18]

### Vegetation-index losses during training

**Strict result: `Not found` for an identifiable, linked cloud-removal / SAR-to-optical generative paper that explicitly adds NDVI, EVI, NDWI, LAI, or red-edge-index error to the *training objective*.**

- CloudBreaker calculates and evaluates NDVI/NDWI after reconstruction; its described training is latent conditional flow matching, not an index-consistency loss. [page:17]
- DB-CR uses MAE/L1 over the 13 reconstructed optical bands; it does not state an index term. [page:18]
- Tsardanidis et al. train directly for NDVI time-series reconstruction using MSE and weighting of masked timestamps. That is **index-target reconstruction**, not an auxiliary NDVI-consistency term attached to all-band image reconstruction. [page:36]

### Band-group or spectral-group processing

**Strict result: `Partially done`, but not found in the requested SAR-conditioned generative configuration.**

- **DB-CR** has a two-branch **modality** design—SAR branch and 13-channel optical branch—and cross-modal attention. It does not separate visible, red-edge/NIR, and SWIR spectral groups. [page:18]
- **CloudBreaker** reconstructs only RGB+NIR. Its stated channel separation occurs after inference to form NDVI/NDWI, not as separate trainable spectral-group attention branches. [page:17]
- The dataset context in DB-CR explicitly includes visible, NIR, and SWIR channels, but its published architecture maps all 13 optical channels through one optical branch. [page:18]

### Uncertainty in diffusion/flow cloud removal

**Strict result: `Not found` in the linked diffusion/flow papers reviewed here.**

- CloudBreaker publishes deterministic image/index quality metrics, but no per-pixel uncertainty maps or calibration evaluation. [page:17]
- DB-CR publishes distortion and perceptual metrics, but no calibrated uncertainty maps, reliability curves, coverage tests, or uncertainty–error correlation. [page:18]
- The established uncertainty-oriented comparator in DB-CR’s benchmark table is **UnCRtainTS**, but DB-CR cites it as a baseline rather than incorporating uncertainty; the linked DB-CR results table reports only reconstruction/perceptual metrics. [page:18]

## Part B verdict matrix

| Proposed component | Verdict | Closest linked paper | Exact difference |
|---|---|---|---|
| **(a) NDVI-consistency loss in all-band cloud removal / SAR-optical generative reconstruction** | **Not found** | CloudBreaker; DB-CR | CloudBreaker evaluates NDVI after 4-band reconstruction; DB-CR uses 13-band MAE. Neither states an NDVI-loss term. [page:17][page:18] |
| **(b) Per-pixel uncertainty in SAR-conditioned cloud removal** | **Partially done in the broader cloud-removal literature, but not found in the reviewed diffusion/flow examples** | UnCRtainTS is the established uncertainty baseline named by DB-CR; DB-CR and CloudBreaker lack it. | The differentiator is an uncertainty-producing and calibrated generative SAR-conditioned model—not merely a stochastic sampler. [page:18] |
| **(c) Band-group attention across RGB, red-edge/NIR, SWIR in SAR-conditioned generative reconstruction** | **Not found** | DB-CR | DB-CR has SAR-versus-optical branches and cross-modal attention, not three spectral-group branches. [page:18] |
| **(d) All three together: all-13-band SAR-conditioned diffusion/bridge + spectral groups + NDVI loss + pixel uncertainty** | **Not found** | DB-CR is architecturally closest; CloudBreaker is flow-matching closest. | DB-CR: all 13 bands and SAR diffusion bridge, but lacks spectral groups, index loss, and uncertainty. CloudBreaker: flow matching and SAR-only conditioning, but produces only RGB+NIR and lacks all three requested components. [page:17][page:18] |

## Claim-safe novelty wording

Use a cautious claim such as:

> “To our knowledge, no linked work identified in this review jointly evaluates a SAR-conditioned all-13-band diffusion/flow reconstruction model with explicit visible/red-edge–NIR/SWIR group processing, vegetation-index consistency loss, calibrated per-pixel uncertainty, and a real-monsoon-mask Indian Kharif crop benchmark with downstream crop classification.”

Avoid claiming the first Indian cloud reconstruction plus crop classification: Gupta and Verma already provide that general pipeline. [page:31]

## Five papers to read before submission

1. **Gupta, Shikha & Verma, Prabhakar Alok (2026).** *Generating Cloud-Free Sentinel-2 Images for Enhanced In-Season Crop Type Classification: a Case Study.* This is the closest Indian IIRS/ISRO reconstruction-to-crop-classification precedent. [DOI](https://doi.org/10.1007/s41976-025-00254-z) [page:31]
2. **Hu, Yuyang; Lohit, Suhas; Kamilov, Ulugbek S.; & Marks, Tim K. (2025).** *Multimodal Diffusion Bridge with Attention-Based SAR Fusion for Satellite Image Cloud Removal.* Closest all-13-band SAR-conditioned diffusion-bridge baseline, with exact SEN12MS-CR metrics. [arXiv](https://arxiv.org/abs/2504.03607) [page:18]
3. **Ahmed, Saleh Sakib; Nowreen, Sara; & Rahman, M. Sohel (2025).** *CloudBreaker: Breaking the Cloud Covers of Sentinel-2 Images using Multi-Stage Trained Conditional Flow Matching on Sentinel-1.* Essential flow-matching/SAR-to-optical comparator; clarify its four-band, SAR-only input formulation. [arXiv](https://arxiv.org/abs/2508.03608) [page:17]
4. **Prashnani, Meghavi & Justice, Chris (2026).** *Evaluating SAR-Derived Phenological Metrics for Monsoon (Kharif) Crop Monitoring in Diversified Agricultural Systems: Insights from Central India.* Strongest direct Kharif/smallholder/multi-crop context paper, though it does not reconstruct optical imagery. [DOI](https://doi.org/10.3390/rs18081238) [page:41]
5. **Tsardanidis, Iason; Koukos, Alkiviadis; Sitokonstantinou, Vasileios; Drivas, Thanassis; & Kontoes, Charalampos (2024).** *Cloud gap-filling with deep learning for improved grassland monitoring.* Essential precedent for synthetic cloud masking plus a downstream agricultural task, while also making clear the difference from an Indian all-band cloud-removal benchmark. [arXiv](https://arxiv.org/abs/2403.09554) [page:36]

## Practical positioning checklist

- Release acquisition dates, tile IDs, geometry, crop labels, cloud/shadow masks, target dates, and split rules so the India June–October subset is auditable.
- State whether the target clear image is temporally coincident, near-coincident within a fixed tolerance, or synthetic/constructed; report change-screening rules.
- Separate cloud-mask transplantation evaluation from genuine cloudy-to-clear evaluation; do not label transplanted targets “real paired ground truth” without qualification.
- Report per-band metrics for B02–B12, especially red-edge/NIR/SWIR; additionally report NDVI/EVI error and crop-classification change.
- Compare a matched baseline without spectral grouping, without index loss, and without uncertainty; evaluate calibration with reliability/coverage and uncertainty–error association.
