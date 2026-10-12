"""Rank candidate regions of a scroll volume for flat, regular, well-separated sheets.

For every 128^3 block of one OME-Zarr pyramid level the structure tensor gives the sheet normal and how planar the sheets
are; sampling the CT along that normal and taking the dominant period gives the sheet spacing and how regular it is.
Output is one CSV row per papyrus block (voxel coordinates are at the surveyed level; multiply by 2**level for level 0).

Usage:
    python scripts/survey_regions.py --zarr data/real_scans/pherc0211/volumes/<name>.zarr --level 2 --voxel-um 9.362 \
        --out data/processed/pherc0211/survey_L2.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import gaussian_filter, map_coordinates

BLOCK = 128


def block_metrics(sub: np.ndarray) -> dict | None:
    """Planarity, normal and sheet period for one block (float32, z,y,x); None if it is not papyrus body."""
    if sub.mean() < 40 or (sub > 0).mean() < 0.8:
        return None
    s = gaussian_filter(sub, 1.0)
    gz, gy, gx = np.gradient(s)
    g = np.stack([gz, gy, gx]).reshape(3, -1)
    j = (g @ g.T) / g.shape[1]
    w, v = np.linalg.eigh(j)                      # ascending
    n = v[:, 2]
    planarity = float((w[2] - w[1]) / (w[2] + 1e-9))
    # two in-plane axes
    a = np.cross(n, [1.0, 0.0, 0.0])
    if np.linalg.norm(a) < 1e-3:
        a = np.cross(n, [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    t = np.arange(-40, 40.5, 0.5)
    lat = np.arange(-30, 31, 10.0)
    c = np.array([BLOCK / 2] * 3)
    prof = np.zeros_like(t)
    for u in lat:
        for vv in lat:
            pts = c[:, None] + n[:, None] * t[None, :] + (a * u + b * vv)[:, None]
            prof += map_coordinates(s, pts, order=1, mode="nearest")
    prof /= len(lat) ** 2
    d = prof - prof.mean()
    power = np.abs(np.fft.rfft(d * np.hanning(len(d)))) ** 2
    freq = np.fft.rfftfreq(len(d), d=0.5)         # cycles per voxel
    ok = (freq > 1 / 16) & (freq < 1 / 2)
    if not ok.any() or power[ok].sum() <= 0:
        return None
    k = np.argmax(np.where(ok, power, 0))
    return {
        "planarity": planarity,
        "period_vox": float(1.0 / freq[k]),
        "regularity": float(power[k] / (power[ok].sum() + 1e-9)),
        "contrast": float(d.std() / (prof.mean() + 1e-9)),
        "nz": float(n[0]), "ny": float(n[1]), "nx": float(n[2]),
        "mean": float(sub.mean()),
        # heuristic 8-bit bands: void/noise < 55, papyrus 60-180, saturated debris > 180
        "void_frac": float((sub < 55).mean()),
        "body_frac": float(((sub >= 60) & (sub <= 180)).mean()),
        "bright_frac": float((sub > 180).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zarr", required=True)
    parser.add_argument("--level", default="2")
    parser.add_argument("--voxel-um", type=float, required=True, help="level-0 voxel size")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    arr = zarr.open(args.zarr, mode="r")[args.level]
    scale = 2 ** int(args.level)
    nz, ny, nx = arr.shape
    rows: list[dict] = []
    t0 = time.time()
    for z0 in range(0, nz - BLOCK + 1, BLOCK):
        slab = np.asarray(arr[z0:z0 + BLOCK])
        for y0 in range(0, ny - BLOCK + 1, BLOCK):
            for x0 in range(0, nx - BLOCK + 1, BLOCK):
                m = block_metrics(slab[:, y0:y0 + BLOCK, x0:x0 + BLOCK].astype(np.float32))
                if m is not None:
                    m.update(z0=z0, y0=y0, x0=x0, level=int(args.level),
                             period_um=m["period_vox"] * scale * args.voxel_um)
                    rows.append(m)
        print(f"z {z0}/{nz}: {len(rows)} blocks, {time.time() - t0:.0f}s", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} blocks -> {out}")


if __name__ == "__main__":
    main()
