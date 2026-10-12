# M4 - Ink detection (PHerc. 0211, scan 20250821151803)

## Role in the pipeline
Per-pixel ink-confidence for a flattened page. Contract: `ink_probability` (page-shaped, 0-1) + an explicit `validated` flag; anything not validated is shown
with a warning (M6).

## What ran on PHerc. 0211
The community **`ink_9um`** model (the scan type it was trained for: ~9 µm), not this repository's own network:
* Checkpoint `hybrid_3d2d-seed42/step-075000.pth`, sha256 `e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab` (138,360,039 bytes); villa
  `vesuvius` package at commit `a329895e`; official `ink_detection.inference.infer` (overlap 0.5, Hann blend), run with `--no-compile`. The checkpoint
  is not redistributed here. Its embedded config (read when the run was made) lists its training scrolls; PHerc. 0211 is not among them, so there is no
  training/prediction overlap. No model was trained for this entry.
* Driver (ours): `code/scripts/probes/ink9um_on_pages.py` - per page, samples a **17-layer surface volume along the page normal at one voxel
  (9.362 µm) per layer** from the native window, runs the model on it, and writes the output TIFF/zarr.
* **Controls** (the point of this module's test): the same surface shifted **±6 and ±14 voxels along its normal** and each run in **both depth directions**
  (8 controls), against the real surface in both directions. The question asked is whether the output depends on the surface.
* Pages: the three largest *non-empty* pages per window (A contributed one: its other two "large" pages are empty strips) - 10 pages, 4.2-16.7 mm²
  (112.8 mm² in total). Runtime ≈ 45 min on an NVIDIA RTX A5000 (24 GB).

## Results (`results/m4_ink9um/summary.json`, per-window `ink9um_stats_crop_*.json`)
Share of valid pixels with P(ink) > 0.5:

| window | page | area mm² | surface, normal | surface, reversed | controls min / median / max |
|---|---|---|---|---|---|
| A | layer_1 | 9.1 | 0.267 | 0.352 | 0.107 / 0.289 / 0.542 |
| B | layer_1 | 16.7 | 0.013 | 0.029 | 0.010 / 0.044 / 0.107 |
| B | layer_2 | 16.1 | 0.022 | 0.086 | 0.010 / 0.032 / 0.074 |
| B | layer_3 | 15.7 | 0.174 | 0.128 | 0.073 / 0.158 / 0.304 |
| C | layer_1 | 15.0 | 0.217 | 0.455 | 0.070 / 0.272 / 0.537 |
| C | layer_2 | 12.9 | 0.014 | 0.000 | 0.000 / 0.023 / 0.098 |
| C | layer_3 | 12.1 | 0.025 | 0.013 | 0.007 / 0.039 / 0.057 |
| D | layer_1 | 5.8 | 0.081 | 0.062 | 0.051 / 0.063 / 0.163 |
| D | layer_2 | 4.2 | 0.185 | 0.211 | 0.087 / 0.400 / 0.545 |
| D | layer_3 | 5.1 | 0.125 | 0.079 | 0.129 / 0.257 / 0.495 |

* **No specificity to the surface.** The surface value exceeds the largest control in 1 of 20 (page, direction) comparisons (B layer_2 reversed, by 1.2
  percentage points on a page where ≤ 9 % of pixels fire); with 8 controls and no real signal about 2 of 20 would be expected. It lies below the median control in
  15 of 20; on 5 of 10 pages both directions are below the median.
* **No strokes.** High-probability regions are smooth blobs 0.5-1.5 mm across (`figures/m4_ink_detection/<window>/layer_*_ink9um_UNVALIDATED.png`,
  render + normal + reversed + controls side by side). The page with the highest shares (C layer_1) is a featureless bright patch with masked holes and no
  fibre texture: more likely a density effect than text.
* **Deterministic:** B layer_2 (normal) re-run gave a bit-identical output (max abs difference 0 on a 269 x 759 uint8 image).
* **Empty-page pitfall.** A layer_2 and layer_3 (3.2 mm² each) are strips where the CT is exactly zero; their model output is flat zero, which looks like "no
  ink, controls agree". They are listed as `empty` in the summary and excluded. Rank pages by area and you will pick them.

## Interpretation
The output is not evidence of ink, and, with no ground truth and ten small pages, not evidence of its absence elsewhere. The test only asks whether the
output depends on the surface; for this model on these pages it does not. Controls are neighbouring layers of the same papyrus, so they test surface
dependence, not specificity to ink. With sheet spacing ~7-13 voxels the ±6 controls sit between sheets and the ±14 controls near the next sheet.

## Not done / limits
* One checkpoint; other community checkpoints and seeds not run. The repository's own fragment-trained network was **not** run on 0211 (different scan
  type, out of distribution) and is not used for any claim here.
* No ground truth for this scroll, so no precision/recall or true/false-positive examples exist; nothing is "validated".
* The model's 128 x 128 patch covers ~1.2 mm, about half the page height; a large window can produce smooth plausible blobs without ink.
* Only the 10 largest non-empty pages were scored, not the windows.

## Reproduce
```bash
sh scripts/probes/run_ink9um_0211.sh        # 4 windows x 3 pages x 5 shifts x 2 directions
python scripts/probes/ink9um_summary.py data/processed/pherc0211_ink9um --flat-root data/processed/pherc0211_split --scan pherc0211 --voxel-um 9.362 --json summary.json
python scripts/probes/ink9um_figures.py data/processed/pherc0211_ink9um/B --out figures/m4_ink_detection/B
```
Needs the villa `vesuvius` package and the checkpoint (`docs/REPRODUCE.md` §5). The architecture's own M4 code (`code/src/m4_ink_detection/`: U-Net,
candidate extraction, page sampler) is unit-tested; the page sampler is what the driver above uses.
