# Published baselines: UnCRtainTS and GLF-CR

These run from the authors' pretrained weights. That's inference only, with no training run, so
it replaces training our DSen2-CR re-implementation as the main external comparison.
`run_external.py` feeds them the same patches, cloudy inputs and masks that `evaluate.py` uses,
in the same order. Their scores are then directly comparable with ours.

## UnCRtainTS (Ebel et al., CVPRW 2023, MIT licence)

```
git clone https://github.com/PatrickTUM/UnCRtainTS /opt/UnCRtainTS
pip install fvcore
```
Download the checkpoints from the pCloud link in the UnCRtainTS README. Each folder holds a
`conf.json` and a `model.pth.tar`.

- **Check `conf.json` and pick a folder with `"pretrain": true`.** Those are mono-temporal models
  trained on SEN12MS-CR, the same task as ours. Folders like `diagonal_1` are multi-temporal
  SEN12MS-CR-TS models. The script refuses them unless you pass `--allow_multitemporal`, and even
  then they see one date only, which is not how they were trained.
- **Inputs.** S1 is clipped to [-25, 0] dB for both VV and VH (their convention). S2 is divided by
  10000, or by 2000 if the checkpoint uses `rescale_method=resnet`. Both are handled
  automatically.
- **Uncertainty.** UnCRtainTS predicts a Gaussian variance. It's saved as the Laplace scale with
  the same variance, `b = sqrt(var / 2)`, in `scale_uncrtaints.npy`, so our calibration
  statistics apply to it.

```
python baselines/run_external.py uncrtaints --repo /opt/UnCRtainTS --weights /opt/uncr_ckpt \
    --experiment <folder> --cache /data/cache --split test --mode real --out EVAL/pred_uncrtaints.npy
```

## GLF-CR (Xu et al., ISPRS 2022)

The repository has no licence file, so nothing is copied from it. The script imports it from
your clone.

```
git clone https://github.com/xufangchn/GLF-CR /opt/GLF-CR
pip install timm
```
Download `CR_net.pth` from the Google Drive link in the GLF-CR README.

- **No compilation needed.** GLF-CR's custom CUDA layer was built for PyTorch 1.4.
  `kernelconv2d_torch.py` is a pure-PyTorch version that computes the same per-pixel dynamic
  convolution, and `tests/test_external.py` checks it against the CUDA kernel's formula. Pass
  `--cuda_ext` if you have compiled the original.
- **Tiling.** GLF-CR works on 128 × 128 inputs (its window attention is built for that size). Each
  256 × 256 patch is processed as overlapping 128 × 128 tiles (stride 64) and averaged.
- **Inputs.** Its normalisation is the same as ours: S2 / 10000, and S1 VV/VH clipped at
  -25 / -32.5 dB, mapped to [0, 1].

```
python baselines/run_external.py glfcr --repo /opt/GLF-CR --ckpt /opt/GLF-CR/ckpt/CR_net.pth \
    --cache /data/cache --split test --mode real --out EVAL/pred_glfcr.npy
```

## Score them

```
python -m diewolke.evaluate --runs RUNS/full RUNS/matched_baseline \
    --preds uncrtaints=EVAL/pred_uncrtaints.npy,EVAL/scale_uncrtaints.npy glfcr=EVAL/pred_glfcr.npy \
    --cache /data/cache --split test --mode real --out EVAL/test_real_with_published
```

Use the same `--mode` for `run_external.py` and `evaluate.py`. Scoring on `kharif_synth` works
the same way: point both at the benchmark cache.

## Reporting

- **Training data.** These models were trained by their authors, probably on all four
  SEN12MS-CR seasons and with their own settings. If ours are trained on spring only, the
  published models have seen more data. State this next to their numbers.
- **Validation.** Before trusting the adapters, score UnCRtainTS and GLF-CR on the official test
  split. Their means should land near their published SEN12MS-CR numbers (DB-CR Table I: GLF-CR
  31.85 dB, UnCRtainTS 31.28 dB PSNR). Our metric code and full-patch protocol differ slightly, so
  expect small gaps, not large ones. A large gap means an input-convention mistake.
- **Timing.** On the RTX A4000 this should take roughly 5–20 minutes per model for a few thousand test
  patches. GLF-CR is the slower of the two (9 tiles per patch).
