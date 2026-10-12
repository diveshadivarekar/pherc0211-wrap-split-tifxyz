"""M2 out-of-core path: segment a volume too large to fit in memory.

``segment.segment_volume`` loads the whole corrected volume into RAM —
fine for the phantom, but it's what crashed on a real 30 GB scroll crop on
a 16 GB machine (``numpy.core._exceptions._ArrayMemoryError: Unable to
allocate 30.4 GiB``, verified). This module is the fix: same algorithm
(Otsu threshold + full-26-connectivity connected components, see
segment.py's docstring for why that's the right classical baseline here),
restructured so no step ever needs the full volume resident at once.

How, concretely:

- The threshold is estimated from a histogram accumulated in two streaming
  passes over the volume (min/max, then binned counts) — see
  :func:`estimate_foreground_threshold_out_of_core`.
- Thresholding and connected-component labelling run on a `dask.array`
  wrapping the HDF5 dataset directly (`dask.array.from_array(h5py_dataset,
  chunks=...)`), via `dask_image.ndmeasure.label` — verified to produce
  byte-identical results to `scipy.ndimage.label` on the phantom (same
  Dice, same component sizes, same layer-jump-free mapping) while never
  materialising more than a chunk at a time.
- Component sizes (for the small-noise-component filter and
  largest-first reordering) come from `dask_image.ndmeasure.area`, and the
  final relabelled output is written back to HDF5 chunk-by-chunk via
  `dask.array.to_hdf5` — not `.compute()`-ed into one array first.

``segment.segment_volume`` (in-memory) stays the default for small inputs
(the phantom, tests) since it's simpler and has one less dependency in the
common case; ``run.py`` picks between the two based on the input file's
size vs. a configurable memory budget.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from src.m1_ingestion.io_utils import VolumeStore
from src.m2_segmentation.segment import (
    FULL_CONNECTIVITY,
    MAX_LABELS,
    MIN_COMPONENT_VOXELS,
    apply_lut,
    label_sizes,
    otsu_threshold_from_histogram,
)

LABELS_DATASET = "labels"


def estimate_foreground_threshold_out_of_core(store: VolumeStore, n_bins: int = 256) -> float:
    """Otsu's method, computed from a histogram built by streaming the
    volume twice (once for the value range, once for binned counts) — never
    holding more than one chunk in memory. Same result as
    :func:`segment.estimate_foreground_threshold` on the same data (it's
    the same Otsu math over the same values, just accumulated differently);
    verified equal on the phantom in tests.
    """
    vmin, vmax = np.inf, -np.inf
    for _, chunk in store.iter_chunks():
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            vmin = min(vmin, float(finite.min()))
            vmax = max(vmax, float(finite.max()))

    if not np.isfinite(vmin) or vmax <= vmin:
        return vmin if np.isfinite(vmin) else 0.0

    bin_edges = np.linspace(vmin, vmax, n_bins + 1)
    hist = np.zeros(n_bins, dtype=np.int64)
    for _, chunk in store.iter_chunks():
        finite = chunk[np.isfinite(chunk)]
        h, _ = np.histogram(finite, bins=bin_edges)
        hist += h

    return otsu_threshold_from_histogram(hist, bin_edges)


def segment_volume_out_of_core(
    input_path: str | Path,
    output_path: str | Path,
    threshold: float | None = None,
    chunk_depth: int | None = None,
    min_component_voxels: int = MIN_COMPONENT_VOXELS,
) -> dict:
    """Out-of-core equivalent of ``segment.segment_volume`` + the HDF5
    writing ``run.py`` does for the in-memory path, combined into one call
    since intermediate results here are dask arrays, not numpy arrays a
    caller could easily hold onto.

    Returns the same kind of summary dict ``run.py`` writes to
    ``<stem>_segments.json`` (``n_segments``, ``voxel_counts``) — no Dice
    comparison here, since ground truth (the phantom's ``layer_id``) is
    always small enough that the in-memory path applies; this path exists
    for real data, which never has ground truth anyway.
    """
    import dask.array as da
    import dask_image.ndmeasure as ndm

    input_path = Path(input_path)
    output_path = Path(output_path)

    src = VolumeStore(input_path, mode="r")
    shape, dtype = src.shape, src.dtype
    depth = chunk_depth or src.chunk_depth

    if threshold is None:
        threshold = estimate_foreground_threshold_out_of_core(src)

    darr = da.from_array(src.dataset, chunks=(depth, shape[1], shape[2]))
    mask = darr > threshold

    labels_da, _ = ndm.label(mask, structure=FULL_CONNECTIVITY)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = (depth, shape[1], shape[2])

    # Compute the (expensive) label graph exactly once, straight to a scratch file. Asking dask
    # for the component count, then the sizes, then the final write re-ran the whole labelling
    # three times; everything after this point streams over the scratch file instead.
    scratch = output_path.with_name(output_path.stem + ".raw_labels.tmp.h5")
    if scratch.exists():
        scratch.unlink()
    da.to_hdf5(str(scratch), "/raw", labels_da, chunks=chunks)

    try:
        with h5py.File(scratch, "r") as raw_file:
            raw = raw_file["/raw"]
            num_components = max((int(raw[z : z + depth].max()) for z in range(0, shape[0], depth)), default=0)

            if output_path.exists():
                output_path.unlink()

            if num_components == 0:
                with h5py.File(output_path, "w") as f:
                    f.create_dataset(LABELS_DATASET, shape=shape, dtype=np.int16, chunks=chunks)
                src.close()
                return {"n_segments": 0, "voxel_counts": {}, "n_dropped_small": 0, "n_dropped_by_cap": 0}

            sizes = label_sizes(raw, num_components, slab=8)[1:]
            keep = np.where(sizes >= min_component_voxels)[0] + 1
            order = keep[np.argsort(-sizes[keep - 1])]  # largest component first
            n_dropped_small = num_components - len(keep)
            n_dropped_by_cap = max(0, len(order) - MAX_LABELS)  # the int16 label volume can't hold more ids
            order = order[:MAX_LABELS]

            remap = np.zeros(num_components + 1, dtype=np.int16)
            remap[order] = np.arange(1, len(order) + 1, dtype=np.int16)

            with h5py.File(output_path, "w") as out_file:
                out = out_file.create_dataset(LABELS_DATASET, shape=shape, dtype=np.int16, chunks=chunks)
                for z in range(0, shape[0], depth):
                    out[z : z + depth] = apply_lut(remap, raw[z : z + depth])
    finally:
        if scratch.exists():
            scratch.unlink()

    voxel_counts = {int(new_id): int(sizes[old_id - 1]) for new_id, old_id in enumerate(order, start=1)}
    src.close()

    return {
        "n_segments": len(order),
        "voxel_counts": voxel_counts,
        "n_dropped_small": int(n_dropped_small),
        "n_dropped_by_cap": int(n_dropped_by_cap),
    }
