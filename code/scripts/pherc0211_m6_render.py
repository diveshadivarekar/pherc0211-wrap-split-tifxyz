"""Render M6 viewer panels (headless, via src.m6_gui.viewmodel) for PHerc. 0211 crop B, scan 20250821151803.

Usage: python scripts/pherc0211_m6_render.py <out_dir>
Inputs: data/processed/pherc0211_split/pherc0211_B/*, data/processed/pherc0211_ink9um/B/layer_2_s+0_normal.tif
"""
import sys, json
from pathlib import Path
import h5py, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, ".")
from src.m6_gui import viewmodel as vm

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
S = Path("data/processed/pherc0211_split")
c = "B"
corr, lab, flat = (S / f"pherc0211_{c}" / f"pherc0211_{c}_{k}.h5" for k in ("corrected", "segmented", "flattened"))
manifest = {"scroll": "PHerc. 0211", "scan_id": "20250821151803", "renders": []}


def save(view, name, header, boxes=True):
    img = vm.compose_png(view, size=(1400, 780), show_boxes=boxes)
    lines = [header] + ([f"NOTICE: {view.notice}"] if view.notice else [])
    canvas = Image.new("RGB", (img.width, img.height + 22 * len(lines) + 8), "white")
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    for i, t in enumerate(lines):
        d.text((8, img.height + 6 + 22 * i), t, fill="#8a1c1c" if t.startswith("NOTICE") else "#1f2933")
    canvas.save(out / name)
    manifest["renders"].append({"file": name, "header": header, "notice": view.notice})
    print("wrote", out / name)


save(vm.build_view("segmentation", {"corrected": corr, "labels": lab}, z=128), "m6_B_segmentation_z128.png",
     "PHerc. 0211 | scan 20250821151803 | crop B | M6 segmentation view, z=128 of 256 (M2 split_wraps labels; boxes hidden)", boxes=False)
save(vm.build_view("flattened", {"flattened": flat}, layer="layer_5"), "m6_B_flattened_layer_5.png",
     "PHerc. 0211 | scan 20250821151803 | crop B | M6 flattened-page view, layer_5 (M3 patch unwrap)")
ink = np.asarray(Image.open("data/processed/pherc0211_ink9um/B/layer_2_s+0_normal.tif")).astype(np.float32)
ink /= 255.0 if ink.max() > 1.5 else 1.0
with h5py.File(flat) as f:
    shape = f["layer_2"]["page"].shape
tmp = out / "_ink_B_layer_2.h5"
with h5py.File(tmp, "w") as g:
    g.attrs["validated"] = False
    g.attrs["notice"] = ("UNVALIDATED ink_9um output (crop B layer_2). Not distinguishable from shifted-surface controls; "
                         "blobs, not strokes. This is not text.")
    full = np.zeros(shape, np.float32)
    h, w = min(shape[0], ink.shape[0]), min(shape[1], ink.shape[1])
    full[:h, :w] = ink[:h, :w]
    g.create_group("layer_2").create_dataset("ink_probability", data=full)
save(vm.build_view("ink", {"flattened": flat, "ink": tmp}, layer="layer_2"), "m6_B_ink_layer_2_UNVALIDATED.png",
     "PHerc. 0211 | scan 20250821151803 | crop B | M6 ink view, layer_2 (validated=False path; region boxes hidden)", boxes=False)
tmp.unlink()
(out / "m6_manifest.json").write_text(json.dumps(manifest, indent=1))
