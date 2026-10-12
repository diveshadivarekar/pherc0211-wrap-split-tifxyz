"""M2 pipeline entrypoint: segment a corrected volume into per-sheet labels.

Automatically picks between two segmentation paths based on the input
volume's size: the simple in-memory one (segment.py) for anything that
comfortably fits in RAM, and the out-of-core one (segment_out_of_core.py)
for anything that doesn't — verified necessary on a real 30 GB scroll crop,
which crashed the in-memory path with an ``ArrayMemoryError`` on a 16 GB
machine. Override the choice with --force-in-memory / --force-out-of-core.

Usage:
    python -m src.m2_segmentation.run --input data/processed/phantom_001_corrected.h5 \
        --out data/processed/ [--ground-truth data/phantoms/phantom_001.h5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.m1_ingestion.io_utils import VolumeStore, available_memory_gb, base_stem
from src.m2_segmentation.segment import FUSION_FRACTION, dice_score, detects_layer_jumping, label_sizes, segment_volume
from src.m2_segmentation.segment_out_of_core import segment_volume_out_of_core

LABELS_DATASET = "labels"
# Measured peak resident memory (11.3 GB for a 3.1 GB real scroll volume: 3.6x) of the in-memory path, fused-sheet splitting included, as a
# multiple of the corrected volume's size. A fixed 4 GB size cut-off ignored both this
# multiple and how much RAM the machine actually has free.
IN_MEMORY_PEAK_FACTOR = 3.7
RAM_SAFETY_FRACTION = 0.7
DEFAULT_MAX_IN_MEMORY_GB = None  # None -> derived from free RAM (see _auto_budget_gb)


def _auto_budget_gb() -> float:
    """Largest volume (GB) the in-memory path may take on this machine right now."""
    return RAM_SAFETY_FRACTION * available_memory_gb() / IN_MEMORY_PEAK_FACTOR


def _volume_nbytes(input_path: Path) -> int:
    with h5py.File(input_path, "r") as f:
        ds = f["volume"]
        return ds.size * ds.dtype.itemsize


def _annotate_fusion(summary: dict, out_of_core: bool = False) -> None:
    """Record how dominant the biggest segment is and warn if it still looks
    fused — a silent one-giant-layer result is the failure this guards."""
    counts = list(summary["voxel_counts"].values())
    frac = (max(counts) / sum(counts)) if counts else 0.0
    summary["largest_segment_fraction"] = frac
    summary["fused_warning"] = frac > FUSION_FRACTION
    if summary["fused_warning"]:
        hint = " The out-of-core path does not split fused sheets; use the in-memory path (--force-in-memory) if RAM allows." if out_of_core else ""
        print(f"WARNING: the largest segment holds {frac:.1%} of the labelled voxels - layers are probably still fused.{hint}")


def run(
    input_path: str | Path,
    out_dir: str | Path,
    ground_truth_path: str | Path | None = None,
    use_cnn_refinement: bool = False,
    max_in_memory_gb: float | None = DEFAULT_MAX_IN_MEMORY_GB,
    force_out_of_core: bool = False,
    force_in_memory: bool = False,
) -> Path:
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = base_stem(input_path)
    out_path = out_dir / f"{stem}_segmented.h5"
    summary_path = out_dir / f"{stem}_segments.json"

    nbytes_gb = _volume_nbytes(input_path) / 1e9
    if max_in_memory_gb is None:
        max_in_memory_gb = _auto_budget_gb()
    use_out_of_core = force_out_of_core or (not force_in_memory and nbytes_gb > max_in_memory_gb)

    if use_out_of_core:
        if use_cnn_refinement:
            print("Note: --cnn-refinement only applies to the in-memory path; ignoring it for this out-of-core run.")
        if ground_truth_path is not None:
            print("Note: Dice-vs-ground-truth validation only runs on the in-memory path (ground truth is always phantom-scale); skipping it here.")

        print(f"Input is {nbytes_gb:.1f} GB (> {max_in_memory_gb:.1f} GB budget) — segmenting out-of-core.")
        result = segment_volume_out_of_core(input_path, out_path)
        summary = result
        if ground_truth_path is not None:
            summary["dice_skipped"] = "Dice validation only runs on the in-memory path"
        if use_cnn_refinement:
            summary["cnn_refinement_skipped"] = "CNN refinement only applies to the in-memory path"
        _annotate_fusion(summary, out_of_core=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Segmented into {summary['n_segments']} components (out-of-core) -> {out_path}")
        return out_path

    store = VolumeStore(input_path, mode="r")
    volume = store.read_all()
    store.close()

    labels = segment_volume(volume, use_cnn_refinement=use_cnn_refinement)
    n_segments = int(labels.max())
    print(f"Labelled {n_segments} segments.")

    chunks = (min(32, labels.shape[0]), labels.shape[1], labels.shape[2])
    with h5py.File(out_path, "w") as f:
        f.create_dataset(LABELS_DATASET, data=labels, chunks=chunks)

    summary: dict = {
        "n_segments": n_segments,
        "voxel_counts": {i: int(c) for i, c in enumerate(label_sizes(labels, n_segments)[1:], start=1)},
    }

    _annotate_fusion(summary)

    if ground_truth_path is not None:
        with h5py.File(ground_truth_path, "r") as f:
            if "layer_id" in f:
                true_labels = f["layer_id"][...]
                mean_dice, mapping = dice_score(labels, true_labels)
                summary["dice_vs_ground_truth"] = mean_dice
                summary["pred_to_true_layer_mapping"] = mapping
                summary["layer_jumping_detected"] = detects_layer_jumping(mapping)

    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Segmented {volume.shape} volume into {n_segments} components -> {out_path}")
    if "dice_vs_ground_truth" in summary:
        print(f"Dice vs ground truth: {summary['dice_vs_ground_truth']:.4f}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Corrected volume HDF5 (M1 output)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--ground-truth", default=None, help="Phantom HDF5 with layer_id for Dice validation (in-memory path only)")
    parser.add_argument("--cnn-refinement", action="store_true", help="Apply the optional learned mask refinement (in-memory path only)")
    parser.add_argument("--max-in-memory-gb", type=float, default=DEFAULT_MAX_IN_MEMORY_GB, help="Volume size threshold for switching to the out-of-core path")
    parser.add_argument("--force-out-of-core", action="store_true", help="Always use the out-of-core path, regardless of size")
    parser.add_argument("--force-in-memory", action="store_true", help="Always use the in-memory path, regardless of size")
    args = parser.parse_args()

    run(
        args.input,
        args.out,
        ground_truth_path=args.ground_truth,
        use_cnn_refinement=args.cnn_refinement,
        max_in_memory_gb=args.max_in_memory_gb,
        force_out_of_core=args.force_out_of_core,
        force_in_memory=args.force_in_memory,
    )


if __name__ == "__main__":
    main()
