# PHerc. 0211 - wrap-aware segmentation and flattening of four native 9.4 µm windows, with a controlled ink-model test

**Vesuvius Challenge Progress Prize submission.** Scroll **PHerc. 0211**, scan **`20250821151803`** (9.362 µm, 113 keV, `…-9.362um-1.2m-113keV-masked.zarr`).
This package contains work on this scroll only: no data, meshes, models or results from any other scroll.

**Start here:** [`docs/WRITEUP.md`](docs/WRITEUP.md) (full write-up) → per-module drafts in [`docs/modules/`](docs/modules/).

## Results at a glance (every number is reproduced from files in `results/`)

| Module | What we ran on PHerc. 0211 | Result | Detail |
|---|---|---|---|
| **M1** ingestion | resumable out-of-core mirror of levels 2-5; level-2 region survey (1,439 blocks); four level-0 windows of 256 x 640 x 1536 voxels (2.4 x 6.0 x 14.4 mm) with matching surface-prediction crops | 4 windows, 0.25 G voxels each, surface prediction on the same grid (scale 1) | [M1](docs/modules/M1_ingestion.md) |
| **M2** segmentation | surface prediction → pieces, plain vs. `split_wraps` (wrap splitting), all four windows | largest plain piece holds **96 % / 24 % / 43 % / 96 %** of the labelled surface (A-D); after splitting **≤ 9.4 %**. Pages on predicted surface **45-68 % → 97.6-98.9 %**; cells running across wraps **34-76 % → 0-0.7 %** | [M2](docs/modules/M2_segmentation.md) |
| **M3** flattening | patch unwrap + relaxation of every piece ≥ 20,000 voxels | **398 pages**, 823.8 mm² valid area; median mean-edge-length error 0.009, **90 % of pages ≤ 0.03**; largest page 5.8-16.7 mm² per window; **454 tifxyz meshes** (398 + 56 baseline) pass the contract check | [M3](docs/modules/M3_flattening.md) |
| **M4** ink detection | community `ink_9um` on the 10 largest pages (4.2-16.7 mm²) with 8 shifted / depth-reversed controls each | the real surface is at or below the strongest control in **19 of 20** comparisons; outputs are smooth blobs, not strokes → **no ink evidence** | [M4](docs/modules/M4_ink_detection.md) |
| **M5** OCR / translation | not run | nothing to transcribe; the confidence gate is the correct behaviour (no output) | [M5](docs/modules/M5_ocr_translation.md) |
| **M6** GUI | headless viewer panels rendered from the 0211 outputs (segmentation, flattened page, unvalidated ink view with its warning) | 3 renders in `figures/m6_gui/` | [M6](docs/modules/M6_gui.md) |
| software | unit tests of M1-M6 + tifxyz (synthetic data) | **232 passed, 1 skipped** from inside this package (GUI tests need a display) | `results/pytest_modules_log.txt` |

## Layout

| path | contents |
|---|---|
| `docs/WRITEUP.md` | write-up: problem, approach, per-module results, validation, limitations |
| `docs/modules/M1…M6_*.md` | one draft per module: role, what ran on 0211, numbers, figures, limits, commands |
| `docs/REPRODUCE.md` | environment, exact commands, expected outputs |
| `meshes/pherc0211_split_wraps/` | 398 pages as **tifxyz** (`x/y/z/mask.tif` + `meta.json`; level-0 voxel coordinates of scan `20250821151803`) |
| `meshes/baseline_plain_m2/` | 56 pages from the un-split segmentation (folders and uuids prefixed `baseline_`; the baseline whose pages cut across wraps) |
| `meshes/index.csv` | one row per mesh: window, page, area, distortion, on-surface and cross-wrap shares, whether `ink_9um` scored it |
| `figures/` | per-module figures (each names the mesh/scan it came from) |
| `results/` | all measured numbers as JSON/CSV, test log, per-page QA reports |
| `code/` | the pipeline (`src/`: M1-M6 + tifxyz), the scripts that produced the results, tests |
| `CHECKSUMS.sha256`, `PROVENANCE.json` | file hashes; repository commit, environment |

## For reviewers: a five-minute check
1. `figures/m2_m3_segmentation_flattening/m2_plain_vs_split_wraps_qa.png` - the M2 result in one picture; `baseline_plain_m2_A_layer_1_distortion0.48.png` vs. `split_wraps_A_layer_1.png` shows what a page that cuts across wraps looks like.
2. `figures/m1_ingestion/pherc0211_A_surface_overlay.png` - papyrus layers resolved at 9.4 µm with the official surface prediction overlaid.
3. `figures/m4_ink_detection/B/layer_2_ink9um_UNVALIDATED.png` - model output next to its controls: same blobs everywhere.
4. Open any folder in `meshes/pherc0211_split_wraps/` in VC3D (volume `20250821151803`). **We have not opened them in VC3D ourselves**; they pass this repository's contract check (`results/tifxyz_contract_check.json`: 454 meshes, 0 findings).

## Caveats that matter
* **Pages are small** (largest 5.8-16.7 mm²): the windows are 2.4 mm deep, so no page can be taller than that. The 4 cm² area First Letters needs is not reached, and no letters are claimed.
* **Page flattening is a height-field export, not an isometric flattening**; the distortion number is a mean relative edge-length error on the page's own grid and does not measure shear.
* **No ground truth exists for this scroll**; everything about ink is "does the output depend on the surface", which it does not.
* "On predicted surface" is measured against the same surface prediction the pieces were cut from, so it is partly by construction; the cross-wrap measure uses the scroll axis (umbilicus) and is independent of the prediction.
* Defaults of M2 (`wedge 40°`, `link_tol 0.75`) were fixed during earlier development and **not re-tuned on 0211**.
* Third-party components used as-is: the organizers' surface prediction and umbilicus, and the community `ink_9um` checkpoint (sha256 in the write-up).

## Licence and credits
Scroll data: Vesuvius Challenge open data, CC BY-NC 4.0.
