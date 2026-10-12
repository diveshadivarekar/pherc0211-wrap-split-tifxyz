"""Download a crop of the official surface-prediction volume for a scroll into HDF5.

The Vesuvius open-data bucket holds, per scroll, a volumetric *surface prediction* (a 0-255 score for
"this voxel is on a papyrus surface") under ``representations/predictions/surfaces/``. The
prediction can be coarser than the native scan, so ``--scale`` records the factor and the crop's
offset is stored in the output's attributes (native voxel = prediction voxel * scale).

Usage:
    python scripts/download_surface_prediction.py --out data/real_scans/pherc0211/surface_pred_crop_B.h5 --scroll-id PHerc0211 \
        --name-contains surface-m7-L0 --scale 1 --z0 8320 --y0 4096 --x0 1024 --depth 256 --ysize 640 --xsize 1536
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import s3fs
import zarr

BUCKET = "vesuvius-challenge-open-data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scroll-id", default="PHerc0211")
    parser.add_argument("--z0", type=int, required=True)
    parser.add_argument("--y0", type=int, required=True)
    parser.add_argument("--x0", type=int, required=True)
    parser.add_argument("--depth", type=int, default=512)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--ysize", type=int, default=None, help="y extent if different from --size")
    parser.add_argument("--xsize", type=int, default=None, help="x extent if different from --size")
    parser.add_argument("--name-contains", default=None, help="substring selecting the prediction product (scrolls have several), e.g. m7-L2")
    parser.add_argument("--scale", type=int, default=4, help="native voxels per prediction voxel")
    args = parser.parse_args()

    fs = s3fs.S3FileSystem(anon=True)
    base = f"{BUCKET}/{args.scroll_id}/representations/predictions/surfaces"
    options = [p for p in fs.ls(base) if p.endswith(".zarr")]
    matches = [p for p in options if args.name_contains is None or args.name_contains in p]
    if not matches or (args.name_contains is not None and len(matches) != 1):
        raise SystemExit(f"--name-contains {args.name_contains!r} must match exactly one of: " + ", ".join(p.rsplit('/', 1)[-1] for p in options))
    zarr_path = matches[0]
    arr = zarr.open(s3fs.S3Map(zarr_path, s3=fs), mode="r")["0"]
    ysize, xsize = args.ysize or args.size, args.xsize or args.size
    z1, y1, x1 = args.z0 + args.depth, args.y0 + ysize, args.x0 + xsize
    if z1 > arr.shape[0] or y1 > arr.shape[1] or x1 > arr.shape[2]:
        raise SystemExit(f"crop exceeds the prediction volume {arr.shape}")
    print(f"{zarr_path}\n  z[{args.z0}:{z1}] y[{args.y0}:{y1}] x[{args.x0}:{x1}] of {arr.shape} ({arr.dtype})", flush=True)

    crop = arr[args.z0 : z1, args.y0 : y1, args.x0 : x1]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        ds = f.create_dataset("volume", data=crop, chunks=(32, 128, 128), compression="lzf")
        ds.attrs.update({"zarr_path": zarr_path, "z0": args.z0, "y0": args.y0, "x0": args.x0, "scale_to_native": args.scale})
    print(f"Saved {crop.shape} ({crop.nbytes / 2**20:.0f} MiB, {(crop > 0).mean():.1%} non-zero) -> {out}", flush=True)


if __name__ == "__main__":
    main()
