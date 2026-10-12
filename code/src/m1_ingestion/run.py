"""M1 pipeline entrypoint: stream a volume from disk, correct it chunk-by-chunk,
write the corrected result back out-of-core.

Usage:
    python -m src.m1_ingestion.run --input data/phantoms/phantom_001.h5 --out data/processed/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py

from src.m1_ingestion.generate_phantom import RAW_DATASET
from src.m1_ingestion.io_utils import VolumeStore, base_stem
from src.m1_ingestion.preprocessing import correct_chunk


def run(input_path: str | Path, out_dir: str | Path, degree: int = 2, median_size: int = 3) -> Path:
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base_stem(input_path)}_corrected.h5"  # an already-corrected input is not suffixed twice
    if out_path.resolve() == input_path.resolve():
        raise ValueError(
            f"{input_path.name} already looks like a corrected volume and the output would overwrite it; "
            "pass the raw volume, or choose a different output directory."
        )

    with h5py.File(input_path, "r") as f:
        if RAW_DATASET not in f:
            raise ValueError(f"{input_path} has no '{RAW_DATASET}' dataset")
        shape = f[RAW_DATASET].shape
        dtype = f[RAW_DATASET].dtype
        chunk_depth = f[RAW_DATASET].chunks[0] if f[RAW_DATASET].chunks else shape[0]

    src = VolumeStore(input_path, mode="r")
    dst = VolumeStore.create(out_path, shape=shape, dtype=dtype, chunk_depth=chunk_depth)

    t0 = time.time()
    n_chunks = 0
    for sl, chunk in src.iter_chunks():
        corrected = correct_chunk(chunk, degree=degree, median_size=median_size)
        dst.write_chunk(sl, corrected)
        n_chunks += 1

    src.close()
    dst.close()
    elapsed = time.time() - t0
    print(
        f"Corrected {shape} volume in {n_chunks} chunks "
        f"(chunk_depth={chunk_depth}) in {elapsed:.2f}s -> {out_path}"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to raw volume (HDF5, 'volume' dataset)")
    parser.add_argument("--out", required=True, help="Output directory for the corrected volume")
    parser.add_argument("--degree", type=int, default=2, help="Beam-hardening polynomial degree")
    parser.add_argument("--median-size", type=int, default=3, help="Ring-artifact median filter size")
    args = parser.parse_args()

    run(args.input, args.out, degree=args.degree, median_size=args.median_size)


if __name__ == "__main__":
    main()
