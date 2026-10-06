# Literature Review: Cloud Removal for Sentinel-2 with SAR Fusion (2021–2026)

> **Important note:** This Markdown reproduces the preceding review. Citations and numerical claims should be independently checked against the linked primary papers before using them in a manuscript or proposal.

## 1. Diffusion-based cloud removal

| Paper | Citation & link | Method | Dataset(s) | SAR | Bands | Reported metrics | Code |
|---|---|---|---|---|---|---|---|
| DDPM-CR | Jing, R., Duan, F., Lu, F., Zhang, M., & Zhao, W. (2023). *Denoising Diffusion Probabilistic Feature-Based Network for Cloud Removal in Sentinel-2 Imagery*. **Remote Sensing**, 15(9), 2217. [DOI](https://doi.org/10.3390/rs15092217) | DDPM feature extractor with attention-based cloud-removal head and SAR/optical concatenation | SEN12MS-CR-like custom split | Yes, S1 VV/VH | Reported as five inputs: RGB plus two SAR channels | PSNR 31.77; SSIM 0.903; MAE 0.0229; RMSE 0.0268 | No repo found |
| DiffCR | Zou, X., Li, K., Xing, J., Zhang, Y., Wang, S., Jin, L., & Tao, P. (2024). *DiffCR: A Fast Conditional Diffusion Framework for Cloud Removal From Optical Satellite Images*. **IEEE Transactions on Geoscience and Remote Sensing**, 62, 1–16. [DOI](https://doi.org/10.1109/TGRS.2024.3373920), [arXiv:2308.04417](https://arxiv.org/abs/2308.04417) | Conditional diffusion with data prediction and TCFBlock U-Net | Sen2_MTC_Old, Sen2_MTC_New | No | RGB | Old: PSNR 29.11, SSIM 0.886; New: PSNR 19.15, SSIM 0.671 | [GitHub](https://github.com/XavierJiezou/DiffCR) |
| DB-CR | Hu, Y., Lohit, S., Kamilov, U. S., & Marks, T. K. (2025). *Multimodal Diffusion Bridge with Attention-Based SAR Fusion for Satellite Image Cloud Removal*. **IEEE Transactions on Geoscience and Remote Sensing**, 63, 1–15. [DOI](https://doi.org/10.1109/TGRS.2025.3604654), [arXiv:2504.03607](https://arxiv.org/abs/2504.03607) | Cloudy-to-clear diffusion bridge with dual SAR/optical branches and attention fusion | SEN12MS-CR | Yes | All 13 Sentinel-2 bands | Claims state-of-the-art; verify exact table values in the paper | No repo found |
| DMDiff | Zhang, W., Mei, J., & Wang, Y. (2025). *DMDiff: A Dual-Branch Multimodal Conditional Guided Diffusion Model for Cloud Removal Through SAR-Optical Data Fusion*. **Remote Sensing**, 17(6), 965. [DOI](https://doi.org/10.3390/rs17060965) | Dual-branch SAR/optical conditional diffusion with cross-attention | Airborne, WHU-OPT-SAR, LuojiaSET-OSFCR | Yes | Not Sentinel-2 all-13 benchmark | Not evaluated on SEN12MS-CR/TS | No repo found |
| SAR-DeCR | Wang, M., Hu, S., Song, Y., & Shi, Y. (2025). *SAR-DeCR: Latent Diffusion for SAR-Fused Thick Cloud Removal*. **Remote Sensing**, 17(13), 2241. [DOI](https://doi.org/10.3390/rs17132241) | Coarse reconstruction, SAR fusion using a cloud attention mask, then latent diffusion refinement | SEN12MS-CR plus synthetic cloud augmentation | Yes | All 13 bands | At 90% synthetic cloud: PSNR about 20.92, SSIM 0.553 | No repo found |
| CM-CR | Han, Q., Huang, B., & Li, Y. (2025). *SAR-Conditioned Consistency Model for Effective Cloud Removal in Remote Sensing Images*. **Remote Sensing**, 17(22), 3721. [DOI](https://doi.org/10.3390/rs17223721) | SAR-conditioned score/diffusion teacher distilled to a consistency model | SEN12MS-CR | Yes | RGB | Claims strong/SOTA benchmark performance; verify exact numbers in paper | No repo found |

**Finding:** No verified SAR-conditioned flow-matching or Schrödinger-bridge cloud-removal paper was identified in the preceding review. DB-CR is the identified SAR-conditioned diffusion-bridge work.

## 2. Uncertainty estimation

| Paper | Citation & link | Method | Dataset(s) | SAR | Bands | Reported metrics | Code |
|---|---|---|---|---|---|---|---|
| UnCRtainTS | Ebel, P., Garnot, V. S. F., Schmitt, M., Wegner, J. D., & Zhu, X. X. (2023). *UnCRtainTS: Uncertainty Quantification for Cloud Removal in Optical Satellite Time Series*. **IEEE Transactions on Geoscience and Remote Sensing**, 61, 1–15. [DOI](https://doi.org/10.1109/TGRS.2023.3283920), [arXiv:2304.05464](https://arxiv.org/abs/2304.05464) | Multi-temporal attention fusion with multivariate aleatoric uncertainty; ensembles for epistemic uncertainty | SEN12MS-CR, SEN12MS-CR-TS | Yes, S1 VV/VH | All 13 bands | SEN12MS-CR-TS, L2 ensemble: PSNR 29.01, SSIM 0.898, SAM 5.97° | [GitHub](https://github.com/PatrickTUM/UnCRtainTS) |
| TCM-CR | Shi, X., Zheng, J., Liu, L., Zhou, L., & Bao, X. (2026). *TCM-CR: Multi-Temporal SAR–Optical Cloud Removal with a Reference Image and Gated Bounded Residual*. **Remote Sensing**, 18(15), 2443. [DOI](https://doi.org/10.3390/rs18152443) | Reference-image-guided multi-temporal fusion with gated residuals | SEN12MS-CR-TS | Yes | All 13 bands | Reports a 7.93 dB gain on heavy-cloud cases relative to a reference baseline | No repo found |
| CGD-Net | *CGD-Net: Cloud-gated dual-head reconstruction with SAR-temporal guidance for Sentinel-2 cloud removal*. (2026). **International Journal of Remote Sensing**. [DOI](https://doi.org/10.1080/01431161.2026.2717440) | Cloud-mask-prior, SAR-temporal, dual-head reconstruction | SEN12MS-CR-TS, AllClear | Yes | All 13 bands | PSNR 30.95; SSIM 0.884; SAM 5.35° | No repo found |

**Finding:** UnCRtainTS is the identified method explicitly producing aleatoric uncertainty maps and estimating epistemic uncertainty with ensembles.

## 3. Spectral fidelity beyond RGB

| Paper | Citation & link | Method | Dataset(s) | SAR | Bands | Benchmark metrics | Code |
|---|---|---|---|---|---|---|---|
| SSGAN | *SSGAN: Cloud removal in satellite images using spatiospectral generative adversarial networks*. (2024). **Computers and Electronics in Agriculture**, 226, 109432. [DOI](https://doi.org/10.1016/j.compag.2024.109432) | Spatiospectral GAN using wavelength-based band subgroups | Custom Sentinel-2 | Not reported as SAR fusion | All 13 bands | Not reported on SEN12MS-CR/TS | No repo found |
| RTFormer | *RTFormer: radiative transfer model-coupled transformer for cloud removal in optical remote sensing imagery*. (2025). **GIScience & Remote Sensing**. [DOI](https://doi.org/10.1080/15481603.2025.2584294) | Transformer with spatial/spectral attention and MODTRAN-driven cloud simulation | RTCR, 34 global Sentinel-2 scenes | No | All 13 bands | Not reported on SEN12MS-CR/TS | No repo found |
| Cloud-adaptive loss | *Cloud removal in Sentinel-2 imagery using a deep residual neural network with cloud-adaptive loss*. (2020). **Remote Sensing**, 12(14), 2283. [DOI](https://doi.org/10.3390/rs12142283) | Residual network with cloud-adaptive loss and spectral metrics | Custom Sentinel-2 and SAR data | Yes | All 13 bands | Not reported on SEN12MS-CR/TS | No repo found |

**Finding:** The preceding review did not identify a paper with an explicit NDVI- or EVI-preservation loss for SAR-fused Sentinel-2 cloud removal.

## 4. Explicit masking versus end-to-end

| Paper | Citation & link | Pipeline | Dataset(s) | SAR | Bands | Explicit mask ablation? |
|---|---|---|---|---|---|---|
| CGD-Net | [DOI](https://doi.org/10.1080/01431161.2026.2717440) | Cloud-mask-prior plus SAR-temporal reconstruction | SEN12MS-CR-TS | Yes | 13 | Uses a refined mask, but no clean mask-versus-no-mask ablation identified |
| SAR-DeCR | [DOI](https://doi.org/10.3390/rs17132241) | Coarse reconstruction, mask-guided SAR fusion, latent diffusion | SEN12MS-CR | Yes | 13 | Stage ablations reported; mask effect not isolated cleanly |
| TCM-CR | [DOI](https://doi.org/10.3390/rs18152443) | Reference image and gated residual | SEN12MS-CR-TS | Yes | 13 | No explicit mask ablation identified |

**Finding:** The preceding review found no direct, controlled end-to-end versus mask-then-reconstruct comparison isolating the benefit of an explicit cloud/shadow mask on SEN12MS-CR or SEN12MS-CR-TS.

## 5. Multi-temporal cloud removal

| Paper | Citation & link | Method | Dataset(s) | SAR | Bands | SEN12MS-CR-TS metrics | Code |
|---|---|---|---|---|---|---|---|
| UnCRtainTS | [DOI](https://doi.org/10.1109/TGRS.2023.3283920), [arXiv](https://arxiv.org/abs/2304.05464) | Multi-temporal attention plus uncertainty | SEN12MS-CR-TS | Yes | 13 | 29.01 / 0.898 / 5.97° | [GitHub](https://github.com/PatrickTUM/UnCRtainTS) |
| CGD-Net | [DOI](https://doi.org/10.1080/01431161.2026.2717440) | Multi-temporal SAR and cloud-mask-guided reconstruction | SEN12MS-CR-TS, AllClear | Yes | 13 | 30.95 / 0.884 / 5.35° | No repo found |
| TCM-CR | [DOI](https://doi.org/10.3390/rs18152443) | Multi-temporal reference plus SAR | SEN12MS-CR-TS | Yes | 13 | Absolute aggregate values not recorded in the preceding review | No repo found |
| CloudTran++ | Christopoulos, D., et al. (2024). *CloudTran++: Improved Cloud Removal from Multi-Temporal Satellite Images Using Axial Transformer Networks*. **Remote Sensing**, 17(1), 86. [DOI](https://doi.org/10.3390/rs17010086) | Axial-transformer multi-temporal model | Custom Sentinel-2 L1C/L2A | No | Six bands used: B02, B03, B04, B08, B11, B12 | Custom data: PSNR 52.83/41.27; SSIM 0.9957/0.9750 | [GitHub](https://github.com/DionysisChristopoulos/cloudtran) |

## 6. India and monsoon agriculture

| Paper | Citation & link | Relevance | Dataset(s) | SAR | Downstream task |
|---|---|---|---|---|---|
| SAR phenology for Kharif | *Evaluating SAR-Derived Phenological Metrics for Monsoon (Kharif) Crop Monitoring in Diversified Agricultural Systems: Insights from Central India*. (2026). **Remote Sensing**, 18(8), 1238. [DOI](https://doi.org/10.3390/rs18081238) | Central-India Kharif crop monitoring, but not optical cloud removal | Sentinel-1 time series, Central India | S1 only | Crop classification; reported 48.3% overall accuracy |
| Gap filling for water management | *Evaluating Sentinel-2 gap filling techniques for cloud removal for water resource management*. (2026). **Water**. [DOI](https://doi.org/10.3390/w18040892) | Evaluates gap filling for water-resource use; verify geographic details in the full paper | Custom Sentinel-2 | Not specified in preceding review | Irrigation scheduling |

**Finding:** The preceding review did not identify a study that evaluates SAR-conditioned diffusion cloud removal on Indian or South-Asian monsoon smallholder agriculture with downstream crop classification or yield estimation.

## 7. Synthetic clouds and real-cloud transfer

| Paper | Citation & link | Synthetic-cloud approach | Dataset(s) | SAR | Bands | Real-versus-synthetic evaluation | Code |
|---|---|---|---|---|---|---|---|
| SatelliteCloudGenerator | *SatelliteCloudGenerator: Controllable Cloud and Shadow Synthesis for Multi-Spectral Optical Satellite Images*. (2023). **Remote Sensing**, 15(17), 4138. [DOI](https://doi.org/10.3390/rs15174138) | Procedural cloud and shadow synthesis with controllable morphology, color, thickness, and channel misalignment | Custom Sentinel-2 | No | All 13 bands | Reports that synthetic-training models approach real-cloud performance | [GitHub](https://github.com/strath-ai/SatelliteCloudGenerator) |
| RTFormer | [DOI](https://doi.org/10.1080/15481603.2025.2584294) | MODTRAN-based radiative-transfer cloud simulation | RTCR | No | All 13 bands | Synthetic training plus real-cloud evaluation | No repo found |
| Synthetic cloud injection | *Synthetic cloud injection for deep land cover segmentation*. (2025). [arXiv:2510.03006](https://arxiv.org/abs/2510.03006) | Binary cloud masks and real-cloud radiometry injected into clean Sentinel-2 scenes | Custom Sentinel-2 | No | RGB and normalized-difference indices | Controlled synthetic-cloud evaluation | No repo found |
| SAR-DeCR | [DOI](https://doi.org/10.3390/rs17132241) | SatelliteCloudGenerator-based 30%, 60%, and 90% cloud augmentation | SEN12MS-CR augmented | Yes | 13 | Reports synthetic-cloud robustness experiments | No repo found |

## Reported benchmark leaders

### SEN12MS-CR

| Method | PSNR (dB) | SSIM | SAM (degrees) | Note |
|---|---:|---:|---:|---|
| CM-CR (2025) | Not recorded in the preceding review | Not recorded | Not recorded | Claims state-of-the-art; consult the paper’s results table |
| DB-CR (2025) | Not recorded in the preceding review | Not recorded | Not recorded | Claims state-of-the-art; consult the paper’s results table |
| DDPM-CR (2023) | 31.77 | 0.903 | Not reported | Reported on a SEN12MS-CR-like custom split |
| SAR-DeCR (2025) | Approx. 20.92 at 90% synthetic cloud | 0.553 at 90% synthetic cloud | Not reported | Not directly comparable to standard real-cloud aggregate results |

### SEN12MS-CR-TS

| Method | PSNR (dB) | SSIM | SAM (degrees) | Code |
|---|---:|---:|---:|---|
| CGD-Net (2026) | 30.95 | 0.884 | 5.35 | No repo found |
| UnCRtainTS L2 ensemble (2023) | 29.01 | 0.898 | 5.97 | [GitHub](https://github.com/PatrickTUM/UnCRtainTS) |
| TCM-CR (2026) | Not recorded | Not recorded | Not recorded | No repo found |

> Do not treat this as a definitive leaderboard: published results can differ in cloud fraction, split, normalization, resolution, temporal context, and whether metrics are calculated only inside masks or across full tiles.

## Research gaps

1. **SAR-conditioned diffusion plus uncertainty:** No identified work combines a SAR-conditioned diffusion/bridge model with calibrated aleatoric and epistemic uncertainty maps.
2. **Spectral agricultural objectives:** No identified work combines SAR-conditioned diffusion with band-group attention and explicit NDVI/EVI preservation losses.
3. **Monsoon agricultural validation:** No identified work evaluates that complete stack over Indian/South-Asian Kharif monsoon agriculture and tests effects on crop classification or yield prediction.
4. **Mask utility evidence:** No identified SEN12MS-CR/TS study provides a clean controlled ablation of explicit cloud/shadow masking against an otherwise matched end-to-end reconstruction model.
5. **Diffusion formulation:** No verified SAR-conditioned flow-matching or Schrödinger-bridge cloud-removal paper was identified in the preceding review.
6. **Synthetic-to-real evidence:** No identified study rigorously quantifies the synthetic-to-real domain gap for SAR-conditioned diffusion models on SEN12MS-CR/TS.
7. **Per-band agricultural fidelity:** Per-band reporting, particularly for red-edge, NIR, and SWIR, remains uncommon in diffusion papers; vegetation-index error should be reported alongside PSNR, SSIM, and SAM.

## Closest work to the target combination

| Paper | What it already covers | What is still missing |
|---|---|---|
| UnCRtainTS | SAR, multi-temporal context, all 13 bands, uncertainty maps, SEN12MS-CR and SEN12MS-CR-TS | Diffusion, band-group attention, NDVI/EVI objective, India/Kharif testing, downstream agriculture |
| DB-CR | SAR-conditioned diffusion bridge, attention-based fusion, all 13 bands, SEN12MS-CR | Uncertainty, vegetation-index loss, India/Kharif and downstream agriculture |
| SAR-DeCR | SAR fusion, mask-guided stages, latent diffusion, all 13 bands, synthetic-cloud stress tests | Uncertainty, band-group/VI loss, multi-temporal design, India/Kharif downstream evaluation |
| SSGAN / RTFormer | Spectral grouping or spectral attention; RTFormer also uses physically motivated cloud simulation | SAR diffusion, uncertainty, SEN12MS-CR/TS benchmarking, India/Kharif downstream evaluation |
| Kharif SAR phenology study | Indian monsoon agriculture and crop monitoring | Optical cloud removal, SAR-optical fusion, diffusion, uncertainty, spectral objectives |

## Planned contribution positioning

A defensible combined contribution would be: **a SAR-conditioned multi-temporal diffusion or bridge model for all Sentinel-2 bands, with band-group-aware fusion and vegetation-index-aware losses; calibrated aleatoric and epistemic uncertainty; an explicit mask-versus-end-to-end ablation; synthetic-to-real testing; and evaluation on Indian Kharif agriculture with downstream crop classification and/or yield estimation.**

