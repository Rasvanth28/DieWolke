# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

dieWolke is an early-stage project for cloud removal in optical satellite imagery. It is at the planning and data-exploration phase: there is no model code, build system, test suite, linter, or README yet. `Plan.pptx` holds the project plan and `Architecture.jpeg` holds the pipeline diagram. `.venv/` is a local Python virtualenv (not tracked in the repo's own docs; don't edit it).

`NOVELTY.md` records the agreed research positioning: the contributions, what is already published, baseline numbers, and required ablations. Check it before proposing architecture changes. The two `*_review.md` files are AI-assisted literature searches, and some of their citations are unverified.

## Target architecture (from `Architecture.jpeg`)

Inputs: Sentinel-1 SAR (S1), cloudy Sentinel-2 (S2_cloudy), and clear S2 ground truth (S2_GT, validation only). The pipeline has four stages:

1. **Cloud/shadow masking**: a fine-tuned U-Net predicts a binary cloud mask. The mask is applied to S2_cloudy to give S2_masked.
2. **Reconstruction**: a SAR-conditioned diffusion-bridge backbone takes S2_masked, the mask and S1. A band-group attention module (separate RGB, NIR and SWIR attention branches) feeds an iterative diffusion-bridge refinement. Outputs are the reconstructed S2_recon and a joint aleatoric uncertainty map.
3. **Temporal conditioning (optional)**: a temporal encoder takes before/after S1+S2 pairs and adds contextual features into the refinement stage.
4. **Validation harness**: PSNR/SSIM/RMSE against S2_GT, NDVI RMSE (agricultural task), and crop-classification accuracy on S2_recon vs S2_GT (downstream task).

## Experiments (`experiments/`)

Experiments run on Colab with a GPU. The local machine has no PyTorch, very little free disk and no dataset.

- `mask_ablation.py` tests whether a separate Stage 1 mask helps Stage 2. It compares a single U-Net (S1 + cloudy S2 → clear S2) with a masker + reconstructor pair, plus the reconstructor fed the true mask as an upper bound. Training clouds are real s2cloudless cloud maps from other patches blended onto clear patches. It evaluates on held-out scenes with both synthetic and real clouds and writes `results.md`, `results.json` and sample PNGs to `--out`.
  - Colab: `!pip install -q s2cloudless` then `!python mask_ablation.py --data_root /content/data/sen12ms_cr --out <drive dir>`
  - Pipeline check on random data: `python mask_ablation.py --smoke --out /tmp/smoke`
  - Tif reads are cached as `.npy` memmaps in `--cache`, and trained checkpoints in `--out` are reused unless `--retrain` is passed.

## Dataset (`datasets/`)

The data is SEN12MS-CR (spring ROIs, `ROIs1158_spring_*`). It is expected under `datasets/data/sen12ms_cr/`, which is currently empty. The data is large (the clear-S2 archive is about 20 GB) and is meant to be fetched and extracted on Google Colab, not locally.

- `datasets/VisuzalizeDataset.py` (note the filename typo) is a Colab script, not a locally runnable one. It imports `google.colab`, mounts Drive, and uses `!wget`/`!tar` shell magics. Those magics make it invalid plain Python. It downloads and extracts three archives (`*_s1`, `*_s2_cloudy`, `*_s2`) and plots S1 / cloudy S2 / clear S2 triplets.
- `datasets/visualizeDataset.ipynb` is the notebook version of the same flow. `getDataset.ipynb` is essentially empty (257 bytes).
- Files are laid out as `ROIs1158_spring_s1/s1_<scene>/ROIs1158_spring_s1_<scene>_p<patch>.tif`, with matching `s2` and `s2_cloudy` trees. The string substitution in the visualization script/notebook leaves the `s1_<scene>` folder unchanged, so it builds wrong S2 paths. Match triplets by parsing `(roi, modality, scene, patch)` from the filename instead, as `experiments/mask_ablation.py` does (`NAME_RE`).
- Band conventions used in the code: S2 RGB is bands `[4, 3, 2]` (B4/B3/B2), scaled by `/3000` and clipped to [0, 1]. S1 uses band 1 (VV), clipped to [-25, 0] dB.
- S2 is stored as 13 bands (B1–B12 including B8A and B10), so B4 = index 3 and B8 = index 7 when 0-indexed.
- The script contains a hardcoded FTP credential URL for the TUM data server. Don't copy it into new files or commit more of it.
