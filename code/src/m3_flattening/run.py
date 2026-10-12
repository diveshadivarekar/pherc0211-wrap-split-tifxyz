"""M3 pipeline entrypoint: flatten every M2 segment into a 2D page.

Usage:
    python -m src.m3_flattening.run --input data/processed/phantom_001_corrected.h5 \
        --labels data/processed/phantom_001_segmented.h5 --out data/processed/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi

from src.m1_ingestion.io_utils import VolumeStore, available_memory_gb, base_stem
from src.m2_segmentation.segment import label_sizes
from src.m3_flattening.flatten import flatten_layer

# Real-scan segmentations contain thousands of speck-sized segments; a page for
# each is noise, so segments below this many voxels are skipped.
DEFAULT_MIN_VOXELS = 1000
RAM_SAFETY_FRACTION = 0.6


def _label_stats(labels, slab: int = 16) -> tuple[int, np.ndarray, list]:
    """``(n_segments, voxel_counts, bounding_boxes)`` for a label volume, streamed slab by
    slab so the full label array never needs a converted copy (``np.bincount`` on a whole
    int16 volume allocates an int64 temporary four times its size)."""
    n_segments = max((int(np.asarray(labels[z : z + slab]).max()) for z in range(0, labels.shape[0], slab)), default=0)
    sizes = label_sizes(labels, n_segments, slab=slab)
    lows = np.full((n_segments, 3), np.iinfo(np.int64).max, dtype=np.int64)
    highs = np.zeros((n_segments, 3), dtype=np.int64)
    for z in range(0, labels.shape[0], slab):
        for idx, box in enumerate(ndi.find_objects(np.asarray(labels[z : z + slab]), max_label=n_segments)):
            if box is None:
                continue
            start = np.array([box[0].start + z, box[1].start, box[2].start])
            stop = np.array([box[0].stop + z, box[1].stop, box[2].stop])
            lows[idx] = np.minimum(lows[idx], start)
            highs[idx] = np.maximum(highs[idx], stop)
    boxes = [None if sizes[i + 1] == 0 else tuple(slice(int(l), int(h)) for l, h in zip(lows[i], highs[i])) for i in range(n_segments)]
    return n_segments, sizes, boxes


def _tile_axis(umbilicus: str | Path, labels_ds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The scroll axis as control points in the labels' own (tile-local) voxel frame."""
    from src.m2_segmentation.surface_segments import load_umbilicus

    if "origin_native_zyx" not in labels_ds.attrs:
        raise ValueError("the labels file has no 'origin_native_zyx' attribute, so the scroll axis cannot be placed in the tile; re-run M2 segment_from_surface_prediction")
    oz, oy, ox = (float(v) for v in labels_ds.attrs["origin_native_zyx"])
    z, y, x = load_umbilicus(umbilicus)
    return z - oz, y - oy, x - ox


def run(
    input_path: str | Path,
    labels_path: str | Path,
    out_dir: str | Path,
    n_theta: int = 360,
    arap_iters: int = 5,
    min_voxels: int | None = None,
    method: str = "auto",
    umbilicus: str | Path | None = None,
) -> Path:
    """Flatten every segment in ``labels_path``. With ``umbilicus`` (the scroll axis, native-voxel control points as
    for ``surface_segments.load_umbilicus``) single sheets of a rolled scroll are unrolled by their cylindrical height
    field; the labels file must then carry ``origin_native_zyx`` (written by ``segment_from_surface_prediction``)."""
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = VolumeStore(input_path, mode="r")
    labels_file = h5py.File(labels_path, "r")
    labels_ds = labels_file["labels"]

    n_segments, sizes, boxes = _label_stats(labels_ds)
    out_path = out_dir / f"{base_stem(input_path)}_flattened.h5"

    # Pages are cut from each segment's bounding box, so keep both volumes in RAM when they fit
    # (thousands of box reads from chunked HDF5 are slow), otherwise read the boxes from disk.
    need_gb = (store.dataset.size * store.dataset.dtype.itemsize + labels_ds.size * labels_ds.dtype.itemsize) / 1e9
    if need_gb <= RAM_SAFETY_FRACTION * available_memory_gb():
        volume, labels = store.read_all(), labels_ds[...]
    else:
        print(f"Volumes need {need_gb:.1f} GB (more than the RAM budget) - reading each segment's box from disk; this is slower.")
        volume, labels = store.dataset, labels_ds
    if min_voxels is None:  # auto: drop specks on a real scan, but never a real layer of a small volume
        min_voxels = min(DEFAULT_MIN_VOXELS, max(1, int(sizes[1:].max()) // 100)) if n_segments else DEFAULT_MIN_VOXELS
    n_written = 0
    axis = _tile_axis(umbilicus, labels_ds) if umbilicus is not None else None

    with h5py.File(out_path, "w") as f:
        for layer_id in range(1, n_segments + 1):
            if sizes[layer_id] < min_voxels or boxes[layer_id - 1] is None:
                continue
            result = flatten_layer(
                volume, labels, layer_id, n_theta=n_theta, arap_iters=arap_iters, bbox=boxes[layer_id - 1], method=method, axis=axis
            )
            grp = f.create_group(f"layer_{layer_id}")
            for name in ("page", "coord_map", "uv"):
                grp.create_dataset(name, data=result[name], compression="lzf")
            grp.attrs["distortion_before"] = result["distortion_before"]
            grp.attrs["distortion_after"] = result["distortion_after"]
            grp.attrs["method"] = result["method"]
            n_written += 1
            print(
                f"layer {layer_id} [{result['method']}]: page {result['page'].shape}, "
                f"distortion {result['distortion_before']:.4f} -> {result['distortion_after']:.6f}"
            )

    store.close()
    labels_file.close()
    n_skipped = n_segments - n_written
    if n_skipped:
        print(f"Skipped {n_skipped} segments under {min_voxels} voxels.")
    print(f"Flattened {n_written} segments -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Corrected volume HDF5 (M1 output)")
    parser.add_argument("--labels", required=True, help="Segmented labels HDF5 (M2 output)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--n-theta", type=int, default=360, help="Angular resolution of the flattened page")
    parser.add_argument("--arap-iters", type=int, default=5, help="ARAP local/global relaxation iterations")
    parser.add_argument("--min-voxels", type=int, default=None, help="Skip segments smaller than this (auto: up to 1000, scaled to the largest segment)")
    parser.add_argument("--method", choices=["auto", "cylinder", "sheet", "patch"], default="auto", help="Flattening model per segment")
    parser.add_argument("--umbilicus", default=None, help="Scroll axis .json (enables the 'sheet' method for rolled-scroll segments)")
    args = parser.parse_args()

    run(args.input, args.labels, args.out, n_theta=args.n_theta, arap_iters=args.arap_iters, min_voxels=args.min_voxels, method=args.method, umbilicus=args.umbilicus)


if __name__ == "__main__":
    main()
