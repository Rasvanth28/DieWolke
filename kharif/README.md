# Indian Kharif monsoon benchmark (C4)

Four scripts, run in order on Colab (or any machine with Earth Engine access). Settings are in
`kharif.yaml`: Earth Engine project, districts, years, thresholds.

| Step | Script | Output |
|---|---|---|
| 1 | `find_scenes.py` | `scenes.csv`: every July–October S2 scene per district and year, with its cloud cover over the district. **Run this first.** It answers whether enough clear Kharif cropland exists (the main feasibility risk in NOVELTY.md). |
| 2 | `export_patches.py` (`--dry_run` first) | 256 x 256 px, 10 m UTM patches: clear cropland patches and cloud-donor patches, each with the nearest S1. Also writes `manifest.jsonl`. |
| 3 | `build_benchmark.py` | `kharif_synth` (real monsoon clouds and shadows transplanted onto clear patches, with exact ground truth) and `kharif_real` (cloudy vs. clear dates of the same cell). Written in the same cache format as SEN12MS-CR, plus `*_meta.json`, which holds the release information: tile IDs, dates, cells, donors and groups. |
| 4 | `crop_classification.py` | Downstream accuracy on all pixels and on cloud pixels: GT vs. cloudy vs. each reconstruction, optionally filtered by uncertainty. |

Reconstruction metrics on the benchmark come from the normal evaluator:
`python -m diewolke.evaluate --runs ... --cache <benchmark_dir> --split kharif_synth --calibrate_on "" --out ...`

## Things to settle before claiming results

- **Labels.** Without `--labels`, the downstream task uses ESA WorldCover 2021 land-cover
  classes. That is a land-cover *proxy*, not crop types, and must be described as such. For real
  crop classification, get a crop-type raster for the same districts and seasons and pass it with
  `build_benchmark.py --labels`. Candidates are a published Kharif crop map, Bhuvan or field data.
  Check its licence before any release.
- **Districts.** The defaults are paddy-dominated examples. Once the Gupta & Verma full text is
  available, align with or deliberately differ from their region and season (see NOVELTY.md).
  GAUL 2015 spellings can differ from current district names, and the script lists close
  matches.
- **Cloud masks** come from s2cloudless (probability > 40 %). **Shadows** are the projected-cloud,
  dark-NIR shadows from the Earth Engine s2cloudless tutorial. Both are estimates. In
  `kharif_synth`, the transplanted mask is exact by construction, but the shadow darkening factor
  (0.6) is a modelling choice.
- **Sentinel-1** is the closest IW acquisition within `s1_max_days`, as a same-day mosaic.
  Patches with more than 1 % missing S1 are dropped.
- **Earth Engine quotas.** `export_patches.py` uses `computePixels` with `workers` parallel
  requests. Lower `workers` if you hit rate limits. Re-running skips patches already downloaded.
