# M1 - Volumetric data ingestion and preprocessing (PHerc. 0211, scan 20250821151803)

## Role in the pipeline
Brings the scan into an out-of-core, chunked form the later stages can stream, and (in the architecture) corrects beam-hardening and ring artefacts.
Output contract: an HDF5 `volume` dataset (z, y, x) plus attributes naming the source (`zarr_path`, level, crop origin, voxel size).

## What ran on PHerc. 0211
| step | tool | detail |
|---|---|---|
| mirror of coarse levels | `code/scripts/download_s3_prefix.py` | resumable, threaded S3 prefix mirror (anonymous, `s3://vesuvius-challenge-open-data/PHerc0211/…`); volume levels 2-5, the organizers' umbilicus, surface-prediction levels 2-5 |
| region survey | `code/scripts/survey_regions.py`, `rank_regions.py` | level 2 (37.4 µm voxels), 128³ blocks → **1,439 papyrus blocks**; structure-tensor planarity per block |
| native windows | `code/scripts/download_native_crop.py` | level-0 crop streamed slab by slab (128-voxel zarr chunks), resumable (`completed_slabs` attribute), HDF5 `volume` |
| surface prediction | `code/scripts/download_surface_prediction.py` | organizers' prediction `…-surface-m7-L0-th0.2.zarr`, same grid, `scale_to_native` = 1 |
| normalisation | `code/scripts/pherc0211_tile.py` | `uint8 / 255 → float32` into `pherc0211_<crop>_corrected.h5` (the file name is historical: **no correction was applied**) |

## Results
Four level-0 windows, 256 x 640 x 1536 voxels each (2.40 x 5.99 x 14.38 mm), uint8, 0.25 G voxels:

| window | origin z, y, x | non-zero fraction | surface voxels (> 127) | level-2 survey blocks overlapped | their planarity percentiles |
|---|---|---|---|---|---|
| A | 9600, 3840, 4608 | 0.84 | 51.3 M | 7 | 0.44-0.90 |
| B | 8320, 4096, 1024 | 0.66 | 52.6 M | 2 | 0.89, 0.98 |
| C | 6272, 4608, 2048 | 0.85 | 33.4 M | 3 | 0.89-0.95 |
| D | 11904, 1536, 4096 | 1.00 | 60.8 M | 6 | 0.11-1.00 |

Source: `results/m1_crops_and_survey.json`. Survey planarity: median 0.43, 90th percentile 0.67 over the 1,439 blocks.
Figures: `figures/m1_ingestion/pherc0211_overview_L3.png` (whole-scroll slice at level 3), `pherc0211_A_surface_overlay.png` (window A, CT and the
official surface prediction on one slice; papyrus layers resolved, prediction follows them).

## Not done / limits
* **Beam-hardening and ring-artefact correction were not applied or evaluated on 0211.** The organizers' volume is reconstructed and masked; the code exists
  (`code/src/m1_ingestion/preprocessing.py`, unit-tested on synthetic phantoms) but has no 0211 result.
* The level-1 mirror (38 GB) was started and not completed; nothing in this package depends on it.
* The survey's sheet-period estimate (FFT of a profile through 37 µm voxels) is clipped at the band edge and measures nothing; only planarity is meaningful.
* Four windows is 0.08 % of the level-0 voxels; nothing here characterises the whole scroll.

## Reproduce
See `docs/REPRODUCE.md` §2.
