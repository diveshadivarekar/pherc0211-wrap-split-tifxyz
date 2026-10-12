"""Run the community ``ink_9um`` checkpoint (villa ``vesuvius``) on flattened pages of a native crop, with shifted-surface controls.

Unlike this repository's fragment-trained U-Net (54 keV / 3.24 um, out of distribution on 9 um scans), this feeds the *official*
ink model its own input: a 17-layer surface volume sampled along the page normal at one voxel per layer (9.362 um on 0211).
There is NO ground truth for these scrolls: the output is compared with copies of the same surface shifted along its normal
and with the reversed depth direction, and is only a candidate generator. Run with the ``inkenv`` interpreter (needs torch,
zarr, h5py, scipy, tifffile and the villa ``vesuvius`` package).

Usage:
    inkenv/Scripts/python.exe virtual-unwrap/scripts/probes/ink9um_on_pages.py --crop data/real_scans/pherc0211/native_crop_A.h5 \
        --flat data/processed/pherc0211_split/pherc0211_A/pherc0211_A_flattened.h5 --top 3 \
        --checkpoint ink9um/hybrid_3d2d-seed42/step-075000.pth --out data/processed/pherc0211_ink9um/A
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import tifffile
import zarr
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.m4_ink_detection.scroll_pages import sample_page_stack  # noqa: E402


def shifted(cm: np.ndarray, normals: np.ndarray, shift: float) -> np.ndarray:
    return cm + shift * normals


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crop", required=True)
    p.add_argument("--flat", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--layers", nargs="*")
    p.add_argument("--shifts", type=float, nargs="+", default=[0.0, 6.0, -6.0, 14.0, -14.0])
    p.add_argument("--depth", type=int, default=17)
    p.add_argument("--min-px", type=int, default=128 * 128)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from src.m4_ink_detection.scroll_pages import surface_normals

    result = {"checkpoint": a.checkpoint, "crop": a.crop, "validated": False, "pages": {}}
    with h5py.File(a.flat, "r") as f, h5py.File(a.crop, "r") as c:
        vol = c["volume"]
        counts = {k: int((~np.isnan(f[k]["coord_map"][..., 0])).sum()) for k in f}
        names = a.layers or [k for k, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n >= a.min_px][: a.top]
        for name in names:
            cm = f[name]["coord_map"][...].astype(np.float32)
            valid = ~np.isnan(cm[..., 0])
            nrm = surface_normals(cm)
            page = {"valid_px": int(valid.sum()), "shape": list(valid.shape), "shifts": {}}
            for shift in a.shifts:
                for orient in ("normal", "flip"):
                    key = f"{name}_s{shift:+g}"
                    zpath = out / f"{key}_{orient}.zarr"
                    tif = out / f"{key}_{orient}.tif"
                    if not tif.exists():
                        cm2 = np.where(valid[..., None], cm + shift * nrm, np.nan).astype(np.float32)
                        stack, v2 = sample_page_stack(vol, cm2, n_slices=a.depth, voxel_size_um=None, flip_depth=orient == "flip")
                        arr = zarr.open(str(zpath), mode="w", shape=stack.shape, chunks=(stack.shape[0], 256, 256), dtype="uint8")
                        arr[:] = np.clip(np.rint(stack), 0, 255).astype(np.uint8)
                        subprocess.run([sys.executable, "-m", "vesuvius.ink_detection.inference.infer", str(zpath), a.checkpoint, str(tif),
                                        "--overlap", "0.5", "--blend-mode", "hann", "--batch-size", "4", "--no-compile"], check=True)
                        if orient == "normal" and shift == 0.0:
                            np.save(out / f"{name}_render.npy", stack.mean(0).astype(np.float32))
                    pr = tifffile.imread(tif).astype(np.float32)
                    pr = pr / (255.0 if pr.max() > 1.5 else 1.0)
                    pv = pr[valid] if pr.shape == valid.shape else pr.ravel()
                    page["shifts"].setdefault(f"{shift:+g}", {})[orient] = {
                        "mean": round(float(pv.mean()), 4), "p95": round(float(np.percentile(pv, 95)), 4),
                        "frac_gt_0.5": round(float((pv > 0.5).mean()), 4)}
            result["pages"][name] = page
            print(name, json.dumps(page), flush=True)
    (out / f"ink9um_stats{a.tag}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
