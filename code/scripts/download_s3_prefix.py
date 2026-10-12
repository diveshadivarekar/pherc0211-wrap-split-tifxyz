"""Mirror parts of the Vesuvius Challenge open-data bucket to disk, resumably.

Objects are copied byte-for-byte (zarr chunks stay zarr chunks), so a mirrored ``*.zarr`` folder is a valid local
OME-Zarr that VC3D or ``zarr.open`` can read. Files already on disk with the right size are skipped; partial files
are written as ``*.part`` and renamed on completion, so the script can be interrupted and re-run.

Usage (PHerc. 0211, scan 20250821151803, coarse pyramid levels + the precomputed representations):
    python scripts/download_s3_prefix.py --out data/real_scans/pherc0211 \
        --prefix PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr/1 \
        --prefix PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr/2 \
        --prefix PHerc0211/representations --prefix PHerc0211/photos

``--prefix`` may also name a single object (e.g. ``.../.zattrs``). Use ``--dry-run`` to only count files and bytes.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import s3fs

BUCKET = "vesuvius-challenge-open-data"


def list_objects(fs: s3fs.S3FileSystem, prefix: str) -> list[tuple[str, int]]:
    """(key, size) for every object under ``prefix`` (or the single object ``prefix``)."""
    full = f"{BUCKET}/{prefix.strip('/')}"
    if fs.isfile(full):
        return [(prefix.strip("/"), fs.info(full)["size"])]
    out: list[tuple[str, int]] = []
    for _, _, files in fs.walk(full, detail=True):
        for f in files.values():
            out.append((f["name"].split("/", 1)[1], f["size"]))
    return out


def fetch(fs: s3fs.S3FileSystem, key: str, size: int, out_root: Path, strip: str, retries: int = 5) -> int:
    dest = out_root / key[len(strip):].lstrip("/") if strip else out_root / key
    if dest.is_file() and dest.stat().st_size == size:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        try:
            fs.get_file(f"{BUCKET}/{key}", str(part))
            if part.stat().st_size != size:
                raise OSError(f"size mismatch for {key}: {part.stat().st_size} != {size}")
            part.replace(dest)
            return size
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="local root; keys keep their path relative to --strip")
    parser.add_argument("--prefix", action="append", required=True, help="bucket prefix or object key (repeatable)")
    parser.add_argument("--strip", default="", help="leading key part to drop (e.g. 'PHerc0211/') - default keeps full keys")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fs = s3fs.S3FileSystem(anon=True)
    out_root = Path(args.out)
    objects: dict[str, int] = {}
    for prefix in args.prefix:
        found = list_objects(fs, prefix)
        print(f"{prefix}: {len(found)} objects, {sum(s for _, s in found) / 1e9:.2f} GB", flush=True)
        objects.update(found)
    total = sum(objects.values())
    todo = [(k, s) for k, s in objects.items()
            if not ((out_root / (k[len(args.strip):].lstrip('/') if args.strip else k)).is_file()
                    and (out_root / (k[len(args.strip):].lstrip('/') if args.strip else k)).stat().st_size == s)]
    print(f"total {len(objects)} objects, {total / 1e9:.2f} GB; {len(todo)} still to fetch "
          f"({sum(s for _, s in todo) / 1e9:.2f} GB)", flush=True)
    if args.dry_run or not todo:
        return

    done_bytes, done_files, t0 = 0, 0, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, fs, k, s, out_root, args.strip) for k, s in todo]
        for fut in as_completed(futures):
            done_bytes += fut.result()
            done_files += 1
            if done_files % 500 == 0 or done_files == len(todo):
                rate = done_bytes / max(time.time() - t0, 1e-9) / 1e6
                print(f"  {done_files}/{len(todo)} files, {done_bytes / 1e9:.2f} GB, {rate:.1f} MB/s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
