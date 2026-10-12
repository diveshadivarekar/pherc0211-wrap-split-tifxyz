# M6 - Researcher-facing viewer (PHerc. 0211, scan 20250821151803)

## Role in the pipeline
The only user interface: load a scan or any stage's output, run stages, inspect results. Tkinter + Pillow (no GPU dependency); a view-model
(`code/src/m6_gui/viewmodel.py`) decides what is drawn so views can be produced and tested headlessly.

## What ran on PHerc. 0211
`code/scripts/pherc0211_m6_render.py` builds three views of window B from the 0211 files through the same view-model the GUI uses:

| file | view | shows |
|---|---|---|
| `figures/m6_gui/m6_B_segmentation_z128.png` | segmentation | native slice z = 128 (of 256) and the M2 `split_wraps` labels overlaid |
| `figures/m6_gui/m6_B_flattened_layer_5.png` | flattened page | M3 page `layer_5` of window B with its method/distortion caption |
| `figures/m6_gui/m6_B_ink_layer_2_UNVALIDATED.png` | ink | M3 page `layer_2` + the `ink_9um` map, loaded through the `validated=False` path: the notice ("not text") is drawn on the render |

Each render's header names the scroll, scan ID and window. The ink view demonstrates the safeguard: a map flagged `validated=False` carries its notice in the view (drawn on this render), and region boxes are
hidden here so nothing looks like a detection.

## Results / limits
* 3 renders; view-model and GUI logic covered by `code/tests/test_m6_*.py` (synthetic data; part of the 232-test run; the Tk-based ones are display-dependent: across five runs one test skipped ("no display") and one run showed a single intermittent Tk error that did not reproduce).
* **The interactive Tkinter window was not driven on 0211 for this package**; only the headless view-model renders above were produced. Per-phase
  progress/ETA for multi-hundred-GB volumes is not exercised on 0211 (windows are 0.25 GB).
* No usability study; no VC3D integration (the exported tifxyz folders are the bridge: open them in VC3D against volume `20250821151803`).
