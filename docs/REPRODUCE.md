# Reproducing the PHerc. 0211 results

All paths are relative to `code/` (the repository root for these scripts). Scan: `20250821151803`, bucket `s3://vesuvius-challenge-open-data/PHerc0211/`.
The scroll data is **not** included (CC BY-NC 4.0; downloaded from the public bucket, anonymous).

## 1. Environment
Project environment (M1-M3, QA, tests): Python 3.13.3, numpy 1.26.4, scipy 1.17.1, h5py 3.16.0, zarr 2.18.7, tifffile 2026.3.3, scikit-image 0.26.0,
Pillow 12.3.0, s3fs/fsspec 2026.9.0, numcodecs 0.15.1, dask 2026.8.0, torch 2.9.0+cu130, pytest 9.1.1 (`requirements.txt`; versions pinned in `PROVENANCE.json`).
Ink test environment (separate): Python 3.14.4, torch 2.11.0+cu128, numpy 2.5.3, zarr 3.4.1 plus the villa `vesuvius` package at commit `a329895e`.
Hardware used: NVIDIA RTX A5000 24 GB, Windows 11. The M1-M3 steps need no GPU.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\...
python -m pytest -q                                                   # 232 passed, 1 skipped (needs a display; synthetic data), ~70-100 s
```

## 2. M1 - data
```bash
python scripts/download_s3_prefix.py --out data/real_scans/pherc0211 --strip PHerc0211/ \
    --prefix PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr/2 --prefix PHerc0211/representations/umbilicus
python scripts/download_s3_prefix.py --out data/real_scans/pherc0211 --strip PHerc0211/ \
    --prefix PHerc0211/representations/predictions/surfaces/20250821151803-surface-20260413222639-surface-m7-L0-th0.2.zarr/2
python scripts/survey_regions.py --zarr data/real_scans/pherc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr --level 2 --voxel-um 9.362 \
    --out data/processed/pherc0211/survey_L2.csv                      # ~4 min, 1,439 blocks
# four native windows (origin z y x, depth 256, y 640, x 1536) and their surface-prediction crops (scale 1):
#   A 9600 3840 4608   B 8320 4096 1024   C 6272 4608 2048   D 11904 1536 4096
python scripts/download_native_crop.py --out data/real_scans/pherc0211/native_crop_A.h5 --scroll-id PHerc0211 --volume 9.362um --voxel-um 9.362 \
    --z0 9600 --y0 3840 --x0 4608 --depth 256 --ysize 640 --xsize 1536
python scripts/download_surface_prediction.py --out data/real_scans/pherc0211/surface_pred_crop_A.h5 --scroll-id PHerc0211 \
    --name-contains surface-m7-L0 --scale 1 --z0 9600 --y0 3840 --x0 4608 --depth 256 --ysize 640 --xsize 1536
```
(`download_native_crop.py` and `download_surface_prediction.py` take their coordinate arguments in multiples of 128; the surface-prediction level 0 of
this scroll is at native resolution. Repeat for B-D.)

## 3. M2 + M3 (`pherc0211_split` = wrap splitting on; omit `--split-wraps` for the baseline)
```bash
for c in A B C D; do python scripts/pherc0211_tile.py $c --split-wraps --out data/processed/pherc0211_split; done      # ~1 min each
S=scratch; mkdir -p $S/plain                                   # any scratch directory
python scripts/pherc0211_m2_m3_stats.py $S/plain               # per-window piece statistics (plain vs split); writes $S/plain/plain_<crop>.h5 + m23_stats.json
python scripts/pherc0211_m3_plain_baseline.py $S/plain         # M3 on those plain-M2 labels (baseline pages) -> $S/plain/flat/<crop>/, m3_plain.json
```
(These two scripts are the ones that produced `results/m2_m3_stats_per_crop.json` and `results/m3_plain_baseline_pages.json`; the first file is
`m23_stats.json` renamed.)

## 4. tifxyz export and audit (all pages, both variants)
```bash
sh scripts/pherc0211_export_and_qa_all_pages.sh $S              # export_tifxyz.py (stride 4) then tifxyz_qa.py per window and variant; edit PY= at the top for your interpreter
# reports: $S/qa/report_{plain,split}_{A-D}.json
```
Window origins used for `--offset`: A 9600 3840 4608, B 8320 4096 1024, C 6272 4608 2048, D 11904 1536 4096 (z y x); `--voxel-um 9.362`. The package's
meshes additionally have `uuid`, `source.scan_id`, `source.volume`, `source.surface_prediction`, `source.crop`, `source.crop_origin_zyx`,
`source.segmentation`, `source.flattening` set in `meta.json` after export (the contract check was re-run: `results/tifxyz_contract_check.json`).

## 5. M4 - ink test (needs the checkpoint and villa)
```bash
# checkpoint hybrid_3d2d-seed42/step-075000.pth, sha256 e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab; villa vesuvius at commit a329895e
sh scripts/probes/run_ink9um_0211.sh            # edit the interpreter/checkpoint paths at the top; ~45 min on a 24 GB GPU
python scripts/probes/ink9um_summary.py data/processed/pherc0211_ink9um --flat-root data/processed/pherc0211_split --scan pherc0211 --voxel-um 9.362 --json summary.json
python scripts/probes/ink9um_figures.py data/processed/pherc0211_ink9um/B --out figures/m4_ink_detection/B     # likewise A, C, D
```

## 6. M6 renders
```bash
python scripts/pherc0211_m6_render.py figures/m6_gui
```

## Expected values
`results/` holds every number quoted in the write-up. Re-running M2/M3 should reproduce piece and page counts exactly (the stages are deterministic); the
QA shares should match to the printed precision. The ink probabilities are deterministic on the same GPU/software (one page re-run bit-identically).

## Known quirk
`scripts/tifxyz_qa.py` raised a `KeyError` in its roll-up when wrap spacing could not be estimated; this package's copy contains the one-line fix
(`code/CODE_NOTES.md`). Per-page lines were always printed; the JSON report was not written before the fix.
