"""Contact sheet of the largest flattened pages of one or more *_flattened.h5 files (percentile-stretched, 2x).

Usage: python scripts/probes/page_gallery.py out.png --top 4 file1_flattened.h5 [file2 ...]
Prints page size and area (mm^2, assuming a 1-voxel grid at --voxel-um). Visual inspection aid only.
"""
import argparse
import h5py
import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("out"); p.add_argument("files", nargs="+")
p.add_argument("--top", type=int, default=4); p.add_argument("--voxel-um", type=float, default=9.362)
a = p.parse_args()
rows = []
for path in a.files:
    with h5py.File(path, "r") as f:
        items = sorted(((f[k]["page"].shape[0] * f[k]["page"].shape[1], k) for k in f), reverse=True)[: a.top]
        tiles = []
        for n, k in items:
            pg = f[k]["page"][...].astype(np.float32)
            valid = pg > 0
            lo, hi = np.percentile(pg[valid], [1, 99]) if valid.any() else (0, 1)
            t = np.clip((pg - lo) / (hi - lo + 1e-6), 0, 1) * valid
            tiles.append((np.kron(t, np.ones((2, 2))) * 255).astype(np.uint8))
            print(f"{path.split('/')[-1][:14]} {k}: page {pg.shape}, {n * (a.voxel_um / 1000) ** 2:.1f} mm^2, "
                  f"{100 * valid.mean():.0f}% valid, dist {f[k].attrs['distortion_after']:.3f}")
    H = max(t.shape[0] for t in tiles)
    rows.append(np.concatenate([np.pad(t, ((0, H - t.shape[0]), (0, 8)), constant_values=255) for t in tiles], 1))
W = max(r.shape[1] for r in rows)
m = np.concatenate([np.pad(r, ((0, 8), (0, W - r.shape[1])), constant_values=255) for r in rows], 0)
Image.fromarray(m).save(a.out); print(m.shape)
