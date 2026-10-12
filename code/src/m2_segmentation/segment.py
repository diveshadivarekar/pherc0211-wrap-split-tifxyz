"""M2 — 3D volumetric segmentation: isolate individual fused page/wrap surfaces.

Implements FR-2.1-FR-2.5. The engine here is a classical baseline —
threshold + 3D connected-component labelling — not the FastSAM3D + 3D U-Net
ensemble described in ARCHITECTURE.md, which needs pretrained foundation-model
weights this environment doesn't have. It is deliberately built to the same
*interface* (`segment_volume(volume) -> label_volume`) so a learned model can
be swapped in later without touching M3/M4/M5.

Why connected components can work here at all: M1's phantom (and, for a real
scan, the corrected voxel grid) has genuine background gaps between fused
sheets — that's what "no layer-jumping" means structurally. A 3D
connected-component pass, with full 26-connectivity, groups each physically
distinct sheet into a single component as long as it never touches its
neighbours; it *fails exactly the way the SRS risk register (R1) predicts* if
two sheets touch or fuse, which is the honest, useful failure mode a learned
model is meant to improve on.

An optional lightweight 3D CNN denoises the raw threshold mask before
labelling (removes speckle noise a real scan would add) — a small, real,
GPU-trained model, not a pretrained foundation model, standing in for where
FastSAM3D / a 3D U-Net would plug into this same interface.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

MIN_COMPONENT_VOXELS = 20
FULL_CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)

# A single component holding more than this share of the foreground means the
# sheets are fused (a real wound scroll's wraps touch everywhere) — the
# phantom's separated layers never get near it, so they keep the plain path.
FUSION_FRACTION = 0.5
# Fused-case splitting: re-threshold at rising foreground-intensity quantiles
# until the biggest "core" component is below this share of the mask.
SPLIT_TARGET_FRACTION = 0.05
SPLIT_QUANTILES = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_SEED_VOXELS = 50
# A seed bigger than this share of the mask is itself a fused clump (the tight
# outer wraps of a scroll) — re-threshold inside it at ever-higher quantiles.
MAX_SEED_FRACTION = 0.01
MAX_RESPLIT_ROUNDS = 6
MAX_GROW_ITERATIONS = 40
MAX_LABELS = np.iinfo(np.int16).max


def label_sizes(labels, n_labels: int, slab: int = 8) -> np.ndarray:
    """Voxel count per label id (index 0 = background), accumulated slab by slab.

    ``np.bincount(labels.ravel())`` first casts the whole array to int64, which
    for a 3 GB int32 label volume means a 6 GB temporary; this never holds more
    than one slab. ``labels`` may be an h5py dataset.
    """
    counts = np.zeros(n_labels + 1, dtype=np.int64)
    for z in range(0, labels.shape[0], slab):
        counts += np.bincount(np.asarray(labels[z : z + slab]).ravel(), minlength=n_labels + 1)[: n_labels + 1]
    return counts


def apply_lut(lut: np.ndarray, labels: np.ndarray, slab: int = 8, in_place: bool = False) -> np.ndarray:
    """``lut[labels]`` computed slab by slab (fancy indexing with an int32 index
    array otherwise allocates an int64 copy of the whole volume). ``in_place``
    rewrites ``labels`` itself (``lut`` must have the same dtype) instead of
    allocating a second full-size array."""
    out = labels if in_place else np.empty(labels.shape, dtype=lut.dtype)
    for z in range(0, labels.shape[0], slab):
        out[z : z + slab] = lut[labels[z : z + slab]]
    return out


def otsu_threshold_from_histogram(hist: np.ndarray, bin_edges: np.ndarray) -> float:
    """Otsu's method given a precomputed histogram: pick the threshold that
    best separates background from substrate/ink voxels.

    Factored out from :func:`estimate_foreground_threshold` so the
    out-of-core path (segment_out_of_core.py) can accumulate a histogram
    across chunks — streamed from disk, never the full volume in memory —
    and feed it through the exact same math instead of duplicating it.
    """
    hist = hist.astype(np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    weight_bg = np.cumsum(hist)
    weight_fg = np.cumsum(hist[::-1])[::-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_bg = np.cumsum(hist * bin_centers) / weight_bg
        mean_fg = (np.cumsum((hist * bin_centers)[::-1])[::-1]) / weight_fg

    valid = (weight_bg[:-1] > 0) & (weight_fg[1:] > 0)
    between_class_var = np.zeros_like(hist)
    between_class_var[:-1][valid] = (
        weight_bg[:-1][valid]
        * weight_fg[1:][valid]
        * (mean_bg[:-1][valid] - mean_fg[1:][valid]) ** 2
    )
    best_idx = int(np.argmax(between_class_var))
    return float(bin_centers[best_idx])


def estimate_foreground_threshold(volume: np.ndarray) -> float:
    """Otsu's method on the volume histogram: pick the threshold that best
    separates background from substrate/ink voxels."""
    finite = volume[np.isfinite(volume)]
    hist, bin_edges = np.histogram(finite, bins=256)
    return otsu_threshold_from_histogram(hist, bin_edges)


def _denoise_mask_cnn(volume: np.ndarray, mask: np.ndarray, model_path: str | None = None) -> np.ndarray:
    """Optional learned refinement of the threshold mask.

    Lazy-imports torch so the classical pipeline works with no ML
    dependency installed at all. Trains a tiny 3D CNN on the fly from the
    volume's own high/low-confidence voxels as weak labels (denoising
    autoencoder style) when no pretrained ``model_path`` is given — this is
    a stand-in for a real pretrained segmentation foundation model
    (FastSAM3D / 3D U-Net per ARCHITECTURE.md), not a claim of equivalent
    accuracy.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return mask  # ML refinement is optional; classical mask is still valid.

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class TinyDenoiser3D(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(1, 8, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(8, 8, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(8, 1, 1),
            )

        def forward(self, x):
            return self.net(x)

    model = TinyDenoiser3D().to(device)
    x = torch.from_numpy(volume[np.newaxis, np.newaxis].astype(np.float32)).to(device)
    y = torch.from_numpy(mask[np.newaxis, np.newaxis].astype(np.float32)).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(30):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        refined = torch.sigmoid(model(x)).cpu().numpy()[0, 0] > 0.5
    return refined


def _resplit_oversized_seeds(
    volume: np.ndarray, labels: np.ndarray, max_seed_voxels: int, min_seed_voxels: int
) -> tuple[np.ndarray, int]:
    """Split every seed larger than ``max_seed_voxels`` by re-thresholding
    inside its own bounding box at higher quantiles of its own intensities,
    repeated for a few rounds. Voxels a split drops are re-absorbed later by
    the grow step. Returns the updated labels and how many seeds were split."""
    next_id = int(labels.max()) + 1
    n_split = 0
    for _ in range(MAX_RESPLIT_ROUNDS):
        sizes = label_sizes(labels, next_id - 1)
        oversized = np.where(sizes[1:] > max_seed_voxels)[0] + 1
        if len(oversized) == 0:
            break
        slices = ndi.find_objects(labels, max_label=int(oversized.max()))
        for lid in oversized:
            sl = slices[lid - 1]
            sub = labels[sl]
            own = sub == lid
            vals = volume[sl][own]
            for q in (0.5, 0.65, 0.8, 0.9):
                thr = np.quantile(vals, q)
                parts, n = ndi.label(own & (volume[sl] >= thr), structure=FULL_CONNECTIVITY)
                if n and np.bincount(parts.ravel())[1:].max() <= max_seed_voxels:
                    break
            if n == 0:
                continue
            psizes = np.bincount(parts.ravel())
            good = np.where(psizes[1:] >= min_seed_voxels)[0] + 1
            if len(good) == 0:
                continue
            lut = np.zeros(len(psizes), dtype=labels.dtype)
            lut[good] = np.arange(next_id, next_id + len(good), dtype=labels.dtype)
            next_id += len(good)
            sub[own] = lut[parts[own]]
            n_split += 1
    return labels, n_split


def split_fused_mask(
    volume: np.ndarray,
    mask: np.ndarray,
    min_seed_voxels: int = MIN_SEED_VOXELS,
    target_fraction: float = SPLIT_TARGET_FRACTION,
    max_grow_iterations: int = MAX_GROW_ITERATIONS,
    max_seed_fraction: float = MAX_SEED_FRACTION,
) -> tuple[np.ndarray, dict]:
    """Break a fused foreground mask into separate sheet patches.

    Plain connected components return one giant blob when wraps touch. The
    brightest part of each sheet is usually still separated from its
    neighbours, so: raise the threshold (over quantiles of the foreground
    intensity) until the biggest connected "core" is under
    ``target_fraction`` of the mask, label those cores as seeds, then grow
    the seeds back out into the full mask (constrained 26-neighbour
    dilation, ties go to the larger id). Voxels the growth never reaches stay
    unlabelled.

    Returns ``(int32 labels, info)``; ``info`` records the chosen quantile /
    threshold and how many mask voxels were left unassigned.
    """
    total = int(mask.sum())
    fg_values = volume[mask]
    def rung(q: float):
        thr = float(np.quantile(fg_values, q))
        seeds, n = ndi.label(mask & (volume >= thr), structure=FULL_CONNECTIVITY)
        return thr, seeds, n

    # Only one rung's label volume is alive at a time (each is a full int32 volume); if the
    # next rung turns out empty, the previous one is simply recomputed.
    best_q, seeds, n, sizes, thr = None, None, 0, None, None
    for q in SPLIT_QUANTILES:
        seeds = None
        thr_q, seeds_q, n_q = rung(q)
        if n_q == 0:
            if best_q is not None:
                thr, seeds, n = rung(best_q)
            break
        best_q, thr, seeds, n = q, thr_q, seeds_q, n_q
        sizes = label_sizes(seeds, n)[1:]
        if sizes.max() / total < target_fraction:
            break
    if best_q is None:
        return np.zeros(mask.shape, dtype=np.int32), {"quantile": None, "threshold": None, "unassigned_voxels": total}
    q = best_q
    sizes = label_sizes(seeds, n)[1:]
    keep = np.zeros(n + 1, dtype=np.int32)
    big = np.where(sizes >= min_seed_voxels)[0]
    keep[big + 1] = np.arange(1, len(big) + 1, dtype=np.int32)
    labels = apply_lut(keep, seeds, in_place=True)  # seeds and labels are the same int32 array
    del seeds
    labels, n_resplit = _resplit_oversized_seeds(volume, labels, int(max_seed_fraction * total), min_seed_voxels)

    zs, ys, xs = np.where(mask.any(axis=(1, 2)))[0], np.where(mask.any(axis=(0, 2)))[0], np.where(mask.any(axis=(0, 1)))[0]
    box = (slice(zs[0], zs[-1] + 1), slice(ys[0], ys[-1] + 1), slice(xs[0], xs[-1] + 1))
    sub_labels, sub_mask = labels[box], mask[box]
    # Grow slab by slab (a full-volume maximum_filter would allocate a second label volume's
    # worth of temporaries on every pass); one-plane halos keep neighbours across slab edges.
    nz, slab = sub_labels.shape[0], 16
    for _ in range(max_grow_iterations):
        changed = False
        for z0 in range(0, nz, slab):
            z1 = min(z0 + slab, nz)
            current = sub_labels[z0:z1]
            open_voxels = sub_mask[z0:z1] & (current == 0)
            if not open_voxels.any():
                continue
            lo, hi = max(z0 - 1, 0), min(z1 + 1, nz)
            grown = ndi.maximum_filter(sub_labels[lo:hi], footprint=FULL_CONNECTIVITY)[z0 - lo : z1 - lo]
            fill = open_voxels & (grown > 0)
            if fill.any():
                current[fill] = grown[fill]
                changed = True
        if not changed:
            break
    labels[box] = sub_labels

    info = {
        "quantile": q,
        "threshold": thr,
        "n_seeds": int(len(big)),
        "n_resplit_seeds": n_resplit,
        "unassigned_voxels": int((mask & (labels == 0)).sum()),
    }
    return labels, info


def largest_component_fraction(sizes: np.ndarray) -> float:
    """Share of all labelled voxels held by the biggest component (fusion signal)."""
    return float(sizes.max() / sizes.sum()) if len(sizes) and sizes.sum() else 0.0


def segment_volume(
    volume: np.ndarray,
    threshold: float | None = None,
    use_cnn_refinement: bool = False,
    min_component_voxels: int = MIN_COMPONENT_VOXELS,
    split_fused: bool = True,
) -> np.ndarray:
    """Segment a corrected voxel grid into per-sheet instance labels.

    If one component swallows most of the foreground (fused sheets — the
    normal state of a real scroll), it is split with :func:`split_fused_mask`
    instead of being returned as one giant layer; ``split_fused=False``
    restores the plain connected-component behaviour.

    Returns an int16 label volume, same shape as ``volume``: 0 = background,
    1..K = a distinct fused page/wrap surface, ordered by decreasing size.
    """
    threshold = threshold if threshold is not None else estimate_foreground_threshold(volume)
    mask = volume > threshold

    if use_cnn_refinement:
        mask = _denoise_mask_cnn(volume, mask)

    labels, n_components = ndi.label(mask, structure=FULL_CONNECTIVITY)
    sizes = label_sizes(labels, n_components)[1:]

    if split_fused and n_components and largest_component_fraction(sizes) > FUSION_FRACTION:
        del labels  # the plain-component labels (a full int32 volume) are not needed while splitting
        labels, _ = split_fused_mask(volume, mask)
        n_components = int(labels.max())
        sizes = label_sizes(labels, n_components)[1:]

    keep = np.where(sizes >= min_component_voxels)[0] + 1
    order = keep[np.argsort(-sizes[keep - 1])][:MAX_LABELS]  # largest first; int16 id space caps the count

    remap = np.zeros(n_components + 1, dtype=np.int16)
    for new_id, old_id in enumerate(order, start=1):
        remap[old_id] = new_id

    return apply_lut(remap, labels)


def dice_score(pred_labels: np.ndarray, true_labels: np.ndarray) -> tuple[float, dict[int, int]]:
    """Instance-level Dice: greedily match each predicted segment to the
    ground-truth segment it overlaps most, then average per-segment Dice.

    Returns ``(mean_dice, pred_to_true_mapping)``. A mapping that sends two
    different predicted ids to the same true id is a detected layer-jump /
    over-merge (SRS risk R1); a mapping that leaves a true id unmatched is a
    missed / split layer.
    """
    true_ids = [i for i in np.unique(true_labels) if i != 0]
    pred_ids = [i for i in np.unique(pred_labels) if i != 0]
    if not true_ids:
        return (1.0 if not pred_ids else 0.0), {}

    dices = []
    mapping: dict[int, int] = {}
    for p in pred_ids:
        p_mask = pred_labels == p
        best_dice, best_true = 0.0, None
        for t in true_ids:
            t_mask = true_labels == t
            intersection = np.logical_and(p_mask, t_mask).sum()
            denom = p_mask.sum() + t_mask.sum()
            d = (2.0 * intersection / denom) if denom > 0 else 0.0
            if d > best_dice:
                best_dice, best_true = d, t
        mapping[int(p)] = int(best_true) if best_true is not None else -1
        dices.append(best_dice)

    return float(np.mean(dices)) if dices else 0.0, mapping


def detects_layer_jumping(mapping: dict[int, int]) -> bool:
    """True if two distinct predicted segments were matched to the same
    ground-truth layer — i.e. the segmentation merged two real sheets."""
    matched = [t for t in mapping.values() if t != -1]
    return len(matched) != len(set(matched))
