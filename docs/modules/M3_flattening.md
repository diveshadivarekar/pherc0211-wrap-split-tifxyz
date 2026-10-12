# M3 - Surface flattening and tifxyz export (PHerc. 0211, scan 20250821151803)

## Role in the pipeline
Unfolds each M2 piece into a 2D page and keeps the mapping back to source voxels. Contract per page (HDF5 group `layer_N`): `page` (sampled density),
`coord_map` (page pixel → `z, y, x` in the window frame; NaN outside), `uv`; attributes `distortion_before`, `distortion_after`, `method`.
tifxyz export (`code/scripts/export_tifxyz.py`): `x.tif y.tif z.tif mask.tif meta.json` in **level-0 voxel coordinates of the scan**.

## Method (code: `code/src/m3_flattening/flatten.py`)
* `patch_unwrap`: PCA of the piece's voxels gives an in-plane basis and a normal; the piece is rasterised on a 1-voxel (u, v) grid, each cell's height
  along the normal is the mean of its voxels' offsets (smoothed, holes filled by normalised convolution), and the volume is sampled at that 3D surface
  point (max of three samples across the normal). Then an edge-length-rescaling relaxation (`arap_relax`, 5 iterations).
* **Not an isometric flattening.** The relaxation rescales the grid; it does not change page layout (see "Limits").
* Pieces under 20,000 voxels are skipped.

## Results (`results/m2_m3_stats_per_crop.json`, `meshes/index.csv`)
| window | pages (split) | valid area, total (mm²) | largest / median (mm²) | distortion after: median / 95th / max | pages ≤ 0.03 |
|---|---|---|---|---|---|
| A | 44 | 56.1 | 9.1 / 0.90 | 0.012 / 0.029 / 0.033 | 95 % |
| B | 87 | 275.6 | 16.7 / 1.69 | 0.007 / 0.028 / 0.099 | 94 % |
| C | 116 | 288.7 | 15.0 / 1.24 | 0.006 / 0.025 / 0.054 | 97 % |
| D | 151 | 203.3 | 5.8 / 0.98 | 0.014 / 0.051 / 0.073 | 81 % |
| all | **398** | **823.8** | | median 0.009 | **90 %** |

Baseline (plain M2) pages: 6 / 24 / 17 / 9 pages; distortion after, max 0.48 / 0.59 / 0.33 / 0.51 (`results/m3_plain_baseline_pages.json`).
* **454 tifxyz meshes** (398 split + 56 baseline): `check_contract` 0 findings (`results/tifxyz_contract_check.json`). Each `meta.json` carries `scan_id`,
  `volume`, `surface_prediction`, `crop`, `crop_origin_zyx`, `segmentation`, `flattening` and `uuid` = folder name.
* Fibres: the largest pages show a crossing papyrus-fibre grid at 9.4 µm and continuous texture across the page
  (`figures/m2_m3_segmentation_flattening/pherc0211_{A-D}_largest_flattened_pages.png`, scale bar 1 mm).
* Two of A's pages (`layer_2`, `layer_3`) lie where the CT is exactly zero (window edge): `ct_nonzero_fraction_under_page` = 0 in `meshes/index.csv`.

## Limits
* **Distortion = mean relative error of grid edge lengths vs. the 3D mesh along the two grid axes.** A height-field export keeps this small by
  construction on gentle sheets; it does not measure angle/area distortion or shear. Do not read ≤ 0.03 as "isometric".
* Pages are bounded by the window: 256 voxels (2.4 mm) deep, so no page is taller than 2.4 mm. Largest page 16.7 mm².
* No fold/overlap check beyond `folded_cells` in the QA tool (mean folded fraction 0.0-0.2 % on split pages).
* Meshes were **not opened in VC3D by us**; no external isometric re-flattening was run.

## Reproduce
`docs/REPRODUCE.md` §3-4. Unit tests: `code/tests/test_m3_flattening.py`, `test_tifxyz.py`, `test_export_tifxyz.py`.
