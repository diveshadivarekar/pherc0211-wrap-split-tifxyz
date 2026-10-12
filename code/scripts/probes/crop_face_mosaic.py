"""Tile fixed-y (XZ) slices of a native crop to look for ink on sheet faces.

In a region whose sheet normal is close to y, a fixed-y slice (averaged over a few voxels) is a view onto one sheet face.
Prints per-slice contrast stats and writes one PNG mosaic. This is a visual look, not a detector.

Usage: python scripts/probes/crop_face_mosaic.py data/real_scans/pherc0211/native_crop_A.h5 out.png --ys 200 260 320 380 440 500
"""
import argparse
import h5py
import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("crop"); p.add_argument("out")
p.add_argument("--ys", type=int, nargs="+", required=True)
p.add_argument("--avg", type=int, default=3, help="voxels averaged along y")
a = p.parse_args()
tiles = []
with h5py.File(a.crop, "r") as f:
    v = f["volume"]
    for y in a.ys:
        s = np.asarray(v[:, y:y + a.avg, :], dtype=np.float32).mean(axis=1)      # (z, x)
        lo, hi = np.percentile(s, [1, 99.5])
        t = np.clip((s - lo) / (hi - lo + 1e-6), 0, 1)
        print(f"y={y}: mean {s.mean():.1f} std {s.std():.1f} p1 {lo:.0f} p99.5 {hi:.0f}")
        tiles.append((t * 255).astype(np.uint8))
mosaic = np.concatenate([np.pad(t, ((0, 4), (0, 0)), constant_values=255) for t in tiles], axis=0)
Image.fromarray(mosaic).save(a.out)
print(mosaic.shape)
