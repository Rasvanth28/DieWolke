# External baselines (GLF-CR, UnCRtainTS)

Both have public code. They are not re-implemented here. Run them with their own scripts, then
score their outputs with our evaluator so every method gets the same metrics, masks and
bootstrap intervals.

1. Train or download their SEN12MS-CR checkpoints. Use the same seasons and the official split
   (`diewolke/data/sen12mscr.py`, `OFFICIAL_TEST` / `OFFICIAL_VAL`, taken from the UnCRtainTS
   data loader).
   - GLF-CR: https://github.com/xufangchn/GLF-CR
   - UnCRtainTS (mono-temporal SEN12MS-CR mode): https://github.com/PatrickTUM/UnCRtainTS
2. Run them on our test patches in our order. `<cache>/test_keys.json` lists the patch ids
   (`ROIs1158_spring_<scene>_p<patch>`) in cache order.
3. Save one array `pred_<name>.npy` of shape `(N, 13, 256, 256)`: float reflectance in [0, 1],
   or uint16 DN. If the method gives per-pixel variance (UnCRtainTS does), save the Laplace-
   equivalent scale `b = sqrt(var / 2)` as `scale_<name>.npy` with the same shape.
4. Score:

```
python -m diewolke.evaluate --runs RUNS/full --preds glfcr=pred_glfcr.npy \
    uncrtaints=pred_uncrtaints.npy,scale_uncrtaints.npy \
    --cache /content/cache --split test --mode real --out EVAL/test_real_external
```

Only `--mode real` matches what those methods were trained for. For the Kharif splits, run them
on `<benchmark>/kharif_synth_*.npy` in the same way.
