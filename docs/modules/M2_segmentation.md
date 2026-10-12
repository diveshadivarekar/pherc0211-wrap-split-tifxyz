# M2 - 3D volumetric segmentation (PHerc. 0211, scan 20250821151803)

## Role in the pipeline
Turns the surface prediction into discrete single-sheet pieces without merging adjacent wraps ("layer jumping"). Contract: HDF5 `labels` (int16),
attributes `origin_native_zyx`, `scale_to_native`, `split_wraps`, `wedge_degrees`.

## Method (code: `code/src/m2_segmentation/surface_segments.py`)
1. Threshold the surface prediction at 127 (0-255).
2. Label connected pieces, cut into angular wedges (40°) about the organizers' umbilicus.
3. **`split_wraps`**: along each radial ray occupied voxels are grouped into runs at radial gaps (one run = one sheet crossing); runs in neighbouring
   columns link if their radial centres differ by ≤ `link_tol` (0.75 voxel; steeper ramps are bridges, not sheets); a run thicker than 1.6x the
   median is divided into sub-runs; **single-valued rule**: in each column the runs of a piece are ranked by radius and only equal-rank runs stay linked, so
   a piece has at most one run per ray (a sheet is a height field `r = f(θ, z)`). Splitting only splits or drops junction voxels; it never merges.
4. Pieces under 2,000 voxels are dropped.

## Results (all four windows; `results/m2_m3_stats_per_crop.json`, `results/m2_qa_split_vs_plain_summary.json`)
**Connectivity of the prediction itself.** Raw 26-connected components: A 2,917, B 770, C 2,838, D 3,210; the largest holds 96.1 / 40.5 / 67.0 / 97.4 % of
the surface voxels. Plain M2 (wedges, no wrap splitting):

| window | pieces | largest piece (share of labelled voxels) | after `split_wraps`: pieces | largest piece |
|---|---|---|---|---|
| A | 19 | 49.3 M voxels (96.3 %) | 706 | 0.55 M (9.4 %) |
| B | 54 | 24.4 % | 1,033 | 4.2 % |
| C | 75 | 42.8 % | 727 | 5.0 % |
| D | 28 | 96.1 % | 1,452 | 1.5 % |

**Page-level audit** (every page of both runs → tifxyz stride 4 → `code/scripts/tifxyz_qa.py` with the umbilicus and the prediction; vertex-weighted):

| window | vertices on predicted surface | vertices on cross-wrap cells | vertices on pages < 90 % on surface | median abs. drift (voxels) |
|---|---|---|---|---|
| A plain → split | 45.2 → 98.4 % | 76.3 → 0.2 % | 78.2 → 0.0 % | 4.04 → 0.86 |
| B | 67.5 → 98.9 % | 35.7 → 0.2 % | 47.2 → 0.4 % | 1.90 → 0.19 |
| C | 51.2 → 97.9 % | 34.0 → 0.0 % | 77.9 → 1.9 % | 2.78 → 0.55 |
| D | 46.0 → 97.6 % | 54.1 → 0.7 % | 78.9 → 1.6 % | 3.85 → 0.29 |

Figure: `figures/m2_m3_segmentation_flattening/m2_plain_vs_split_wraps_qa.png`. Per-page reports: `results/qa/report_{plain,split}_{A-D}.json`.
Why plain pages fail: a piece spanning several wraps has no single radius per ray; its page floats between sheets (30 % of the vertices of A's largest
plain page lie on surface; its flattening distortion is 0.48).

## What this does and does not show
* It shows that, on this scroll's official surface prediction, the stock connected-component stage merges sheets in two of four windows and that
  `split_wraps` removes the merged pages by two independent measures (on-surface; cross-wrap against the scroll axis).
* **On-surface is partly by construction** (same prediction). The cross-wrap measure does not depend on the prediction.
* Splitting costs size: pages are smaller (A's largest page: 47.9 k → 6.5 k vertices) and in A total mesh falls (61.9 k → 39.9 k vertices) because most
  baseline area was one cross-wrap page. Wedge size (`--wedge-degrees`) was left at 40° and **not swept on 0211**.
* `link_tol`, `wedge_degrees`, `gap` were fixed during earlier development and **not re-tuned here**.
* No ground-truth segments exist for 0211, so Dice > 0.90 (the architecture's gate) was not measured. No correctness claim beyond the two measures above.
* The prediction is the organizers' (`surface-m7-L0`, th 0.2); this module does not improve it.

## Reproduce
`docs/REPRODUCE.md` §3: `python scripts/pherc0211_tile.py <A|B|C|D> --split-wraps` (plain: omit the flag), then `scripts/export_tifxyz.py` and `scripts/tifxyz_qa.py`.
Unit tests: `code/tests/test_m2_segmentation*.py`, `test_surface_segments.py` (synthetic).
