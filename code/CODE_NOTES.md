# Code notes - how `code/` relates to the source repository

`code/` is the pipeline repository (`virtual-unwrap`) restricted to what produced the PHerc. 0211 results, plus the scripts written for this package.
It runs from `code/` as its root (`python -m pytest`, `python scripts/…`).

## Left out of this copy (not used for 0211)
Modules/scripts that work on other data or other object types: the cells3d sample-scan loader, other-scroll loaders, label-evaluation and
label-rendering scripts, the multi-scroll survey, face probes, and the repository's model weights. No scroll data, meshes or model weights are included.

## Edited in this copy (docstrings, comments and constant defaults only)
`results/code_edits_vs_repo.json` is the machine check (AST comparison with docstrings stripped):

| file | change |
|---|---|
| `src/m2_segmentation/surface_segments.py`, `src/m3_flattening/flatten.py`, `src/tifxyz/qa.py`, `src/m4_ink_detection/run.py` | docstrings only: sentences quoting measurements on other data removed; code identical |
| `scripts/export_tifxyz.py`, `scripts/tifxyz_qa.py`, `scripts/probes/ink9um_on_pages.py` | docstring examples now use 0211 paths; code identical |
| `scripts/download_native_crop.py`, `scripts/download_surface_prediction.py` | docstrings; `--scroll-id` default `PHerc0211`; `--voxel-um` default 9.362 (download_native_crop) |
| `src/m4_ink_detection/candidates.py` | `to_scroll_coordinates(voxel_size_um=…)` default 9.362 (callers pass it explicitly) |

## One bug fix (also applies to the source repository)
`scripts/tifxyz_qa.py::summarise` raised `KeyError: 'wraps_crossed'` when wrap spacing could not be estimated (the field is only set when it can). It now
skips such pages in the median. Per-page output was unaffected; only the JSON roll-up was.

## Written for this package (verbatim copies of what was run)
`scripts/pherc0211_m2_m3_stats.py`, `pherc0211_m3_plain_baseline.py`, `pherc0211_export_and_qa_all_pages.sh`, `pherc0211_m6_render.py`; the 0211 figure
and probe scripts `pherc0211_figures.py`, `probes/ink9um_*.py`, `probes/run_ink9um_0211.sh`.

## Tests
`python -m pytest -q` → 232 passed, 1 skipped (no display; synthetic phantoms/text; no scroll data needed). See `../results/pytest_modules_log.txt`.
