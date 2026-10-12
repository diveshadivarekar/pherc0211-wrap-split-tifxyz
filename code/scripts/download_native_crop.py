"""Download a native-resolution crop of a Vesuvius Challenge scroll into an HDF5 ``volume``.

The open-data bucket stores each scroll as an OME-Zarr pyramid; level 0 is the native scan
(PHerc0211: 9.362 um voxels, 19416 x 7948 x 7948, ~300 GB compressed), so only a small crop is practical.
The crop is streamed slab by slab (never fully in memory), resumable, and aligned to the zarr's
128-voxel chunks so every read is whole chunks.

Usage:
    python scripts/download_native_crop.py --out data/real_scans/pherc0211/native_crop_B.h5 --scroll-id PHerc0211 --volume 9.362um \
        --voxel-um 9.362 --z0 8320 --y0 4096 --x0 1024 --depth 256 --ysize 640 --xsize 1536
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import s3fs
import zarr

BUCKET = "vesuvius-challenge-open-data"
SLAB = 128  # zarr chunk edge


def open_level0(scroll_id: str, volume: str | None = None):
    """Level 0 of one of the scroll's volumes: ``volume`` is a substring of its folder name (e.g. ``2.400um``); the first one if None."""
    fs = s3fs.S3FileSystem(anon=True)
    options = [p for p in fs.ls(f"s3://{BUCKET}/{scroll_id}/volumes/") if p.endswith(".zarr")]
    matches = [p for p in options if volume is None or volume in p]
    if len(matches) != 1 and volume is not None:
        raise SystemExit(f"--volume {volume!r} must match exactly one of: " + ", ".join(p.rsplit('/', 1)[-1] for p in options))
    path = matches[0]
    return zarr.open(s3fs.S3Map(path, s3=fs), mode="r")["0"], path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scroll-id", default="PHerc0211")
    parser.add_argument("--z0", type=int, required=True)
    parser.add_argument("--y0", type=int, required=True)
    parser.add_argument("--x0", type=int, required=True)
    parser.add_argument("--depth", type=int, default=1024)
    parser.add_argument("--size", type=int, default=2048, help="in-plane edge (y and x)")
    parser.add_argument("--ysize", type=int, default=None, help="y extent if different from --size")
    parser.add_argument("--xsize", type=int, default=None, help="x extent if different from --size")
    parser.add_argument("--volume", default=None, help="substring selecting the volume folder, e.g. 2.400um (needed for scrolls with several volumes)")
    parser.add_argument("--voxel-um", type=float, default=9.362, help="voxel size recorded in the output")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    ysize, xsize = args.ysize or args.size, args.xsize or args.size
    for name, value in (("z0", args.z0), ("y0", args.y0), ("x0", args.x0), ("depth", args.depth), ("ysize", ysize), ("xsize", xsize)):
        if value % SLAB:
            raise SystemExit(f"--{name.replace('_', '-')} must be a multiple of {SLAB} (the zarr chunk size)")

    arr, zarr_path = open_level0(args.scroll_id, args.volume)
    z1, y1, x1 = args.z0 + args.depth, args.y0 + ysize, args.x0 + xsize
    if z1 > arr.shape[0] or y1 > arr.shape[1] or x1 > arr.shape[2]:
        raise SystemExit(f"crop exceeds the scan {arr.shape}")
    shape = (args.depth, ysize, xsize)
    print(f"{zarr_path}\n  crop z[{args.z0}:{z1}] y[{args.y0}:{y1}] x[{args.x0}:{x1}] = {shape}, "
          f"{args.depth * ysize * xsize / 2**30:.2f} GiB ({arr.dtype})", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"scroll_id": args.scroll_id, "zarr_path": zarr_path, "level": 0, "z0": args.z0, "y0": args.y0, "x0": args.x0,
            "voxel_size_um": args.voxel_um}
    with h5py.File(out, "a") as f:
        if "volume" in f and (f["volume"].shape != shape or any(f["volume"].attrs.get(k) != v for k, v in meta.items() if k != "zarr_path")):
            raise SystemExit(f"{out} already holds a different crop; choose another --out")
        ds = f["volume"] if "volume" in f else f.create_dataset("volume", shape=shape, dtype=arr.dtype, chunks=(16, 512, 512))
        ds.attrs.update(meta)
        done = set(int(i) for i in ds.attrs.get("completed_slabs", []))
        todo = [i for i in range(args.depth // SLAB) if i not in done]
        print(f"{len(done)} of {args.depth // SLAB} slabs already done", flush=True)

        def fetch(i: int):
            z = args.z0 + i * SLAB
            return i, arr[z : z + SLAB, args.y0 : y1, args.x0 : x1]

        t0 = time.time()
        with ThreadPoolExecutor(args.workers) as pool:
            for i, slab in pool.map(fetch, todo):
                ds[i * SLAB : (i + 1) * SLAB] = slab
                done.add(i)
                ds.attrs["completed_slabs"] = sorted(done)
                f.flush()
                print(f"  slab {len(done)}/{args.depth // SLAB}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"Download complete -> {out}", flush=True)


if __name__ == "__main__":
    main()
