"""Segment and flatten one native PHerc. 0211 crop (scan 20250821151803) with the project's M2 -> M3 pipeline.

Inputs are the crop pair written by scripts/download_native_crop.py and scripts/download_surface_prediction.py. Output is
a segmentation (``*_segmented.h5``) and flattened pages (``*_flattened.h5``); no ink model is run here.

Usage:
    python scripts/pherc0211_tile.py A --out data/processed/pherc0211
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.m2_segmentation.surface_segments import segment_from_surface_prediction  # noqa: E402
from src.m3_flattening.run import run as m3_run  # noqa: E402

ROOT = Path("data/real_scans/pherc0211")
UMBILICUS = ROOT / "representations/umbilicus/20250821151803-umbilicus-20260808112626.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("crop", help="crop letter, e.g. A (native_crop_A.h5 + surface_pred_crop_A.h5)")
    parser.add_argument("--out", default="data/processed/pherc0211")
    parser.add_argument("--wedge-degrees", type=float, default=40.0)
    parser.add_argument("--min-voxels", type=int, default=2000)
    parser.add_argument("--split-wraps", action="store_true")
    args = parser.parse_args()

    name = f"pherc0211_{args.crop}"
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    native, pred = ROOT / f"native_crop_{args.crop}.h5", ROOT / f"surface_pred_crop_{args.crop}.h5"
    t0 = time.time()
    with h5py.File(native, "r") as f:
        origin = tuple(int(f["volume"].attrs[k]) for k in ("z0", "y0", "x0"))
        shape = tuple(int(s) for s in f["volume"].shape)
        tile = out / f"{name}_corrected.h5"
        with h5py.File(tile, "w") as g:
            ds = g.create_dataset("volume", shape=shape, dtype=np.float32, chunks=(16, shape[1], shape[2]))
            for z in range(0, shape[0], 64):
                ds[z : z + 64] = f["volume"][z : z + 64].astype(np.float32) / 255.0
    print(f"[{name}] origin {origin} shape {shape}", flush=True)
    seg = out / f"{name}_segmented.h5"
    s = segment_from_surface_prediction(pred, UMBILICUS, origin, shape, seg, wedge_degrees=args.wedge_degrees,
                                        min_voxels=args.min_voxels, split_wraps_apart=args.split_wraps)
    sizes = sorted(s["voxel_counts"].values(), reverse=True)
    print(f"[{name}] {s['n_segments']} surface pieces; largest {sizes[:8]}; surface fraction {s['surface_fraction']:.3f}", flush=True)
    flat = m3_run(tile, seg, out, min_voxels=args.min_voxels * 10, umbilicus=UMBILICUS)
    print(f"[{name}] flattened -> {flat} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
