# PHerc. 0211 (scan 20250821151803): wrap-aware segmentation and flattening of four native windows, and a controlled ink-model test

Team Members: Divesh Adivarekar, Pranav Gugale, Rohan Gorde, Samarth Sangolgi

## 1. Summary

PHerc. 0211 is eligible for the First Letters and Grand Prize tracks, and as of 2026-10-10 the organizers' data browser listed no segments and no
ink predictions for it (we checked; re-check before submitting). We built a six-module pipeline (M1 ingestion → M6 viewer) and ran it on four
native 9.362 µm windows of this scroll. We did **not** find ink or letters and we claim none. What we contribute:

1. **A measured fix for the main failure of connected-component segmentation on this scroll's surface prediction (M2).** In two of four windows
   one connected piece holds 96 % of the labelled surface and a page built from it cuts across wraps. With wrap splitting, no piece holds more than
   9.4 %, the share of mesh vertices lying on predicted surface goes from 45-68 % to 97.6-98.9 %, and the share on cells running across wraps from
   34-76 % to 0-0.7 % (all four windows, every page, repo QA tool; §4.2).
2. **398 flattened pages as tifxyz** (823.8 mm² of valid page area in total; §4.3) plus the 56 baseline pages, each carrying the scan ID,
   source crop, and the pipeline stage that produced it, and passing the tifxyz contract check.
3. **A shifted-surface / depth-reversal control protocol for ink models and its result on 0211 (M4).** The community `ink_9um` checkpoint's output on
   the ten largest usable pages does not depend on the surface: the real surface is at or below the strongest of eight controls in 19 of 20
   comparisons. We also document a pitfall: ranking pages by area selects empty strips where the CT volume is exactly zero.
4. **Open tooling** (resumable bucket mirror, windowed crop and surface-prediction downloaders, level-2 region survey, tifxyz export and audit, a
   headless viewer) with unit tests, and the exact commands that produced every number.

## 2. Why this is a Progress Prize entry and what a reader can reuse
The Progress Prize rewards tools and insights the community can use, shown on real data. The reusable parts are (a) wrap splitting for
surface-prediction-based segmentation, with before/after numbers on a scroll that has no public segments, (b) the audit that makes a page's quality
measurable without ground truth (on-surface share, cross-wrap share, folded cells, contract check), (c) the control protocol that tells you whether an ink
model's output depends on the surface, and (d) a documented, tested pipeline interface between stages. The numbers below say how much each is worth; where
a result is negative we say so in the same place.

## 3. Pipeline (six modules, file-based contracts)

```
 M1 ingest ──▶ M2 segment ──▶ M3 flatten ──▶ M4 ink ──▶ M5 OCR/translate
     │             │              │             │              │
     └─────────────┴──────────────┴─────────────┴──────────────┴──▶ M6 viewer (reads every stage's files)
```
Contract between stages: HDF5 volume (M1) → `labels` (M2) → per-page `page`/`coord_map`/`uv` (M3) → `ink_probability` (M4); tifxyz is the interchange format
for pages. Each stage runs on its own (command line) and the viewer shows any stage's file. On 0211 M1-M4 and M6 produced outputs; M5 did not run (§4.5).

## 4. Results by module (full drafts: `docs/modules/`)

### 4.1 M1 - ingestion (`docs/modules/M1_ingestion.md`)
* Source volume `20250821151803-9.362um-1.2m-113keV-masked.zarr`, level 0 = 19,416 x 7,948 x 7,948 voxels. Levels 2-5 and the organizers' umbilicus and
  surface-prediction levels were mirrored with a resumable downloader; a level-2 survey (1,439 papyrus blocks of 128³) scored planarity per block.
* Four level-0 windows A-D of 256 x 640 x 1536 voxels (z0 y0 x0 = A 9600 3840 4608, B 8320 4096 1024, C 6272 4608 2048, D 11904 1536 4096), each with the
  surface-prediction crop on the same grid (`scale_to_native` = 1). Windows are 84 %, 66 %, 85 % and 100 % non-zero (the rest is the scan's mask).
* No beam-hardening or ring correction was applied: the organizers' volume is already reconstructed and masked, and we did not evaluate our correction
  on it. The only transform is `uint8/255 → float32`.

### 4.2 M2 - segmentation (`docs/modules/M2_segmentation.md`)
Pieces are connected components of the thresholded surface prediction (> 127) cut into 40° wedges about the umbilicus; `split_wraps` then splits
touching wraps by radial runs (link tolerance 0.75 voxel, single-valued per ray; it only splits, never merges). Pieces < 2,000 voxels are dropped.

| window | surface voxels | pieces, plain / split | largest piece, plain / split | raw 26-connected components: largest share |
|---|---|---|---|---|
| A | 51.3 M | 19 / 706 | 96.3 % / 9.4 % | 96.1 % |
| B | 52.6 M | 54 / 1,033 | 24.4 % / 4.2 % | 40.5 % |
| C | 33.4 M | 75 / 727 | 42.8 % / 5.0 % | 67.0 % |
| D | 60.8 M | 28 / 1,452 | 96.1 % / 1.5 % | 97.4 % |

Page quality, every page of both runs, exported as tifxyz at stride 4 and audited with `scripts/tifxyz_qa.py`
(`results/m2_qa_split_vs_plain_summary.json`, vertex-weighted):

| window | pages plain / split | vertices on predicted surface | vertices on cross-wrap cells | vertices on pages < 90 % on surface | median off-surface drift (voxels) |
|---|---|---|---|---|---|
| A | 6 / 44 | 45.2 % → **98.4 %** | 76.3 % → **0.2 %** | 78.2 % → 0.0 % | 4.04 → 0.86 |
| B | 24 / 87 | 67.5 % → **98.9 %** | 35.7 % → **0.2 %** | 47.2 % → 0.4 % | 1.90 → 0.19 |
| C | 17 / 116 | 51.2 % → **97.9 %** | 34.0 % → **0.0 %** | 77.9 % → 1.9 % | 2.78 → 0.55 |
| D | 9 / 151 | 46.0 % → **97.6 %** | 54.1 % → **0.7 %** | 78.9 % → 1.6 % | 3.85 → 0.29 |

Reading it: plain pages sit on the predicted surface for only about half their vertices because they span several wraps (they float between
sheets); split pages are > 97 % on surface. **Caveats:** on-surface share is measured against the same prediction the pieces came from, so part of the gain
is by construction; the cross-wrap share uses the scroll axis and is independent of it. Splitting makes pages smaller (largest page of A: 47.9 k → 6.5 k
vertices) and in window A the split run covers less mesh than the baseline (39.9 k vs 61.9 k vertices) because most of the baseline's area was one page
cutting across wraps. M2 defaults were not re-tuned on 0211 and not swept here.

### 4.3 M3 - flattening and export (`docs/modules/M3_flattening.md`)
Each piece ≥ 20,000 voxels is unwrapped as a plane height field with an edge-length-rescaling relaxation (`patch`), written as a page (`page`,
`coord_map` = page pixel → source voxel, `uv`). 398 pages from the split run, 56 from the baseline.

| window | pages | valid area total (mm²) | largest / median page (mm²) | distortion after: median / 95th pct / max | pages ≤ 0.03 |
|---|---|---|---|---|---|
| A | 44 | 56.1 | 9.1 / 0.90 | 0.012 / 0.029 / 0.033 | 95 % |
| B | 87 | 275.6 | 16.7 / 1.69 | 0.007 / 0.028 / 0.099 | 94 % |
| C | 116 | 288.7 | 15.0 / 1.24 | 0.006 / 0.025 / 0.054 | 97 % |
| D | 151 | 203.3 | 5.8 / 0.98 | 0.014 / 0.051 / 0.073 | 81 % |

For comparison the baseline pages: the page that cuts across wraps in A has distortion 0.48 (and 30 % of its vertices on surface); baseline max distortion is
0.33-0.59 in every window. "Valid area" = valid page pixels x (9.362 µm)². **Distortion** is the mean relative error of grid edge lengths against the 3D
mesh edges along the two grid axes; it is not an area/angle (isometric) measure and does not detect shear. Pages are height-field exports, **not isometric
flattenings**. Two of A's pages (`layer_2`, `layer_3`, 3.2 mm² each) are strips where the CT is exactly zero (window edge): they are in the mesh set
because the surface prediction is non-zero there, but `nonzero_frac` = 0 in `results/m2_m3_stats_per_crop.json` and they were excluded from ink scoring.
Treat any page ranked by area with suspicion until the volume under it has been checked.
Papyrus fibres are visible and run as a crossing grid on the largest pages (`figures/m2_m3_segmentation_flattening/*_largest_flattened_pages.png`).
tifxyz meshes: 454 of 454 pass `check_contract`; `meta.json` records scan, volume, surface-prediction product, crop origin and stage.

### 4.4 M4 - ink detection (`docs/modules/M4_ink_detection.md`)
Model: community `ink_9um` checkpoint `hybrid_3d2d-seed42/step-075000.pth` (sha256 `e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab`),
villa `vesuvius` package at commit `a329895e`, official inference (overlap 0.5, Hann blend). The checkpoint's own config lists its training scrolls;
PHerc. 0211 is not among them, so there is no training/prediction overlap. Input per page: a 17-layer surface volume sampled along the page normal at one
voxel per layer from the native window. Controls: the same surface shifted ±6 and ±14 voxels along its normal, each in both depth directions (8 controls),
plus the real surface in both directions. Metric: share of valid pixels with P(ink) > 0.5. Ten pages scored (A layer_1; B layers 1-3; C layers 1-3; D
layers 1-3; 4.2-16.7 mm²).

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

* The surface exceeds the largest control in **1 of 20** comparisons (B layer_2 reversed, 0.086 vs 0.074); chance with 8 controls would give about 2.
  It is below the median control in 15 of 20; on 5 of 10 pages both directions are below the median.
* The high-probability regions are smooth blobs 0.5-1.5 mm across, not row-aligned strokes; the brightest page (C layer_1) is a featureless patch, more
  likely a density effect than text. One inference was repeated and is bit-identical.
* **Interpretation:** the output does not depend on the surface, so it is not evidence of ink, and (with no ground truth, 10 small pages) not evidence of
  its absence elsewhere in the scroll. The repository's own fragment-trained network was not run on 0211 (different scan type; out of distribution).

### 4.5 M5 - OCR / translation
Not run on 0211: with no ink map that survives the controls there is nothing to transcribe, and the architecture's confidence gate (NFR-6: a
low-confidence reading is never presented as text) is designed to produce exactly this empty result. The module is unit-tested on synthetic data only.

### 4.6 M6 - viewer (`docs/modules/M6_gui.md`)
The viewer's view-model renders segmentation overlays, flattened pages and ink maps from the same files; for an ink map flagged `validated=False` it
carries a visible notice. Three renders from 0211 window B are in `figures/m6_gui/` (generated headless by `code/scripts/pherc0211_m6_render.py`).

## 5. Validation and false-positive safeguards
* **Geometry:** fibre continuity inspected on the largest pages (visible, crossing grid, no obvious sheet jumps), plus quantitative on-surface /
  cross-wrap / folded-cell shares per page and the contract check. This does not replace inspection in VC3D, which we have not done.
* **Ink:** every output is compared with shifted and depth-reversed copies of the same surface; every figure and file carries "UNVALIDATED". No
  letter, row or reading is annotated. No manual annotation was drawn on any image.
* **Training/prediction overlap:** none (no model was trained for this entry; the third-party checkpoint's training scrolls exclude 0211).
* **Randomness:** no stochastic training; inference is deterministic (one page re-run bit-identically). Seeds not applicable.

## 6. Limitations
* Windows are 2.4 mm deep and 0.25 G voxels; this is 4 windows of a ~300 GB volume, chosen once, not a whole-scroll or first-letters region. The
  selection used a planarity survey whose period estimate we found unusable (clipped at the band edge); only planarity was used.
* Pages ≤ 16.7 mm²; the First Letters area (4 cm²) is not reached and the ink test is on small pages.
* Height-field pages: distortion metric is weak; no isometric flattening (LSCM/ARAP on the mesh) exists in the repository yet.
* On-surface share is partly circular (same prediction). M2 defaults not tuned or swept on 0211. No ground-truth segments exist for 0211, so the
  architecture's Dice > 0.90 acceptance gate could not be evaluated.
* One ink model; other community checkpoints not run. Controls are neighbouring layers of the same papyrus: they test dependence on the surface,
  not specificity to ink. Absence here is not absence in the scroll.
* tifxyz meshes not opened in VC3D by us.

## 7. Reproduce
`docs/REPRODUCE.md` lists environment, data download commands and each step. Re-running M2/M3/QA for all four windows takes a few minutes on a
workstation; the ink test about 45 minutes on a 24 GB GPU.


