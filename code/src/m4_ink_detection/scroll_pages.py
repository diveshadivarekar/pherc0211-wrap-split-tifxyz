"""Apply the fragment-trained ink model to flattened scroll pages.

The model reads a *stack* of slices through the papyrus surface (see ``fragments.py``). M3/spiral
fitting give a flat 2D page plus a ``coord_map`` saying which 3D voxel every page pixel came from;
here that surface is thickened back into a stack by sampling the volume at several steps along the
local surface normal, then fed to the network.

Two honest caveats, both reported to the user rather than hidden:

* The fragments were scanned at 3.24 um; a scroll scan is typically 7-8 um. ``voxel_size_um`` lets
  the stack be resampled to the model's resolution (in-plane and in depth), but a model trained on
  one scanner/resolution is **not validated** on another - there is no ink ground truth on the scroll
  to check it against.
* The depth direction of a scroll page (which side is "up") is not known from geometry alone, so
  ``flip_depth`` is exposed instead of guessed.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

FRAGMENT_VOXEL_SIZE_UM = 3.24


def surface_normals(coord_map: np.ndarray, smooth: float = 2.0) -> np.ndarray:
    """Unit normals (H, W, 3) of the surface a page was cut from, from the finite differences of its
    coordinate map. Pixels outside the page get the normal of the nearest valid pixel."""
    valid = ~np.isnan(coord_map[..., 0])
    if not valid.any():
        return np.zeros_like(coord_map)
    w = valid.astype(np.float64)
    filled = np.where(valid[..., None], coord_map, 0.0).astype(np.float64)
    den = ndi.gaussian_filter(w, smooth)
    smoothed = np.stack([ndi.gaussian_filter(filled[..., c] * w, smooth) for c in range(3)], axis=-1)
    smoothed = np.divide(smoothed, den[..., None], out=np.zeros_like(smoothed), where=den[..., None] > 1e-6)
    du = np.gradient(smoothed, axis=1)
    dv = np.gradient(smoothed, axis=0)
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 1e-9)
    good = valid & (norm[..., 0] > 1e-9)
    if not good.any():
        return np.zeros_like(coord_map)
    _, (iy, ix) = ndi.distance_transform_edt(~good, return_indices=True)
    return n[iy, ix].astype(np.float32)


def sample_page_stack(
    volume,
    coord_map: np.ndarray,
    n_slices: int = 16,
    voxel_size_um: float | None = None,
    flip_depth: bool = False,
    pad: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``n_slices`` layers along the surface normal, centred on the page surface.

    ``voxel_size_um`` (the scan's voxel edge) sets the step so the stack spans the same physical depth
    as the training fragments' (``n_slices`` x 3.24 um); ``None`` means one voxel per slice.
    ``volume`` can be an array or an h5py dataset: only the page's bounding box is read.
    Returns ``(stack (K, H, W) float32, valid (H, W) bool)``; pixels outside the page are 0.
    """
    valid = ~np.isnan(coord_map[..., 0])
    h, w = valid.shape
    stack = np.zeros((n_slices, h, w), dtype=np.float32)
    if not valid.any():
        return stack, valid

    step = (FRAGMENT_VOXEL_SIZE_UM / voxel_size_um) if voxel_size_um else 1.0
    offsets = (np.arange(n_slices) - (n_slices - 1) / 2) * step
    if flip_depth:
        offsets = offsets[::-1]

    normals = surface_normals(coord_map)
    pts = coord_map[valid].astype(np.float64)
    nrm = normals[valid].astype(np.float64)
    reach = np.abs(offsets).max() + 1
    lo = np.maximum(np.floor(pts.min(axis=0) - reach - pad), 0).astype(int)
    hi = np.minimum(np.ceil(pts.max(axis=0) + reach + pad) + 1, volume.shape).astype(int)
    crop = np.asarray(volume[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]], dtype=np.float32)
    for k, off in enumerate(offsets):
        coords = (pts + off * nrm - lo).T
        stack[k][valid] = ndi.map_coordinates(crop, coords, order=1, mode="nearest")
    return stack, valid


def standardize_page(stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-page z-score over valid pixels, so a scroll's density scale lines up with the (already
    standardised) fragment inputs the model was trained on."""
    vals = stack[:, valid]
    if vals.size == 0:
        return stack
    mean, std = float(vals.mean()), float(vals.std() + 1e-6)
    out = (stack - mean) / std
    out[:, ~valid] = 0.0
    return out.astype(np.float32)


def predict_page(model, stack: np.ndarray, valid: np.ndarray, voxel_size_um: float | None = None, tile: int = 256, device: str | None = None) -> np.ndarray:
    """Ink probability map (H, W) for one sampled page stack.

    If ``voxel_size_um`` is larger than the fragments' 3.24 um the page is upsampled to the model's
    resolution first and the result resampled back, so the network sees ink strokes at the size it
    learned them.
    """
    from src.m4_ink_detection.infer import predict_stack

    h, w = valid.shape
    x = standardize_page(stack, valid)
    scale = (voxel_size_um / FRAGMENT_VOXEL_SIZE_UM) if voxel_size_um else 1.0
    mask = valid
    if abs(scale - 1.0) > 1e-3:
        x = ndi.zoom(x, (1, scale, scale), order=1)
        mask = ndi.zoom(valid.astype(np.uint8), (scale, scale), order=0) > 0
    # predict_stack treats floats as already normalised (mean 0, std 1)
    prob = predict_stack(model, x, 0.0, 1.0, mask=mask, tile=tile, stride=tile * 3 // 4, device=device)
    if abs(scale - 1.0) > 1e-3:
        prob = ndi.zoom(prob, (h / prob.shape[0], w / prob.shape[1]), order=1)[:h, :w]
    prob[~valid] = 0.0
    return prob.astype(np.float32)


def refine_surface(
    volume,
    coord_map: np.ndarray,
    search: float = 8.0,
    step: float = 1.0,
    depth_sigma: float = 1.5,
    smooth_sigma: float = 3.0,
    papyrus_threshold: float | None = None,
    pad: int = 4,
) -> tuple[np.ndarray, dict]:
    """Snap a page's surface onto the centre of the papyrus sheet it was meant to follow.

    A predicted/segmented surface is only approximately on the sheet, and a flattened page cut from
    it samples air gaps wherever it drifts. For each page pixel this looks along the surface normal
    (+-``search`` voxels), finds the brightest (smoothed) point, and moves the pixel there; the
    per-pixel shifts are smoothed so the surface stays coherent, and pixels with no papyrus within reach
    (peak below ``papyrus_threshold``) are dropped (set to NaN) instead of being fed to the model as
    empty space. Returns ``(refined coord_map, info)``.
    """
    valid = ~np.isnan(coord_map[..., 0])
    info = {"n_pixels": int(valid.sum()), "n_dropped": 0, "mean_abs_shift": 0.0}
    if not valid.any():
        return coord_map.copy(), info

    normals = surface_normals(coord_map)
    pts = coord_map[valid].astype(np.float64)
    nrm = normals[valid].astype(np.float64)
    offsets = np.arange(-search, search + 1e-6, step)
    lo = np.maximum(np.floor(pts.min(axis=0) - search - pad), 0).astype(int)
    hi = np.minimum(np.ceil(pts.max(axis=0) + search + pad) + 1, volume.shape).astype(int)
    crop = np.asarray(volume[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]], dtype=np.float32)

    profile = np.empty((len(offsets), len(pts)), dtype=np.float32)
    for k, off in enumerate(offsets):
        profile[k] = ndi.map_coordinates(crop, (pts + off * nrm - lo).T, order=1, mode="nearest")
    profile = ndi.gaussian_filter1d(profile, depth_sigma / step, axis=0)
    best = profile.argmax(axis=0)
    peak = profile[best, np.arange(len(pts))]
    shift = offsets[best]

    if papyrus_threshold is None:
        from src.m2_segmentation.segment import estimate_foreground_threshold

        sample = crop[:: max(1, crop.shape[0] // 64), :: max(1, crop.shape[1] // 64), :: max(1, crop.shape[2] // 64)]
        papyrus_threshold = estimate_foreground_threshold(sample)
    on_papyrus = peak > papyrus_threshold

    shift_map = np.zeros(valid.shape, dtype=np.float64)
    keep_map = np.zeros(valid.shape, dtype=bool)
    shift_map[valid], keep_map[valid] = shift, on_papyrus
    # smooth shifts over the papyrus pixels only (normalised convolution), so air gaps do not drag them to zero
    w = keep_map.astype(np.float64)
    num = ndi.gaussian_filter(ndi.median_filter(shift_map, size=5) * w, smooth_sigma)
    den = ndi.gaussian_filter(w, smooth_sigma)
    smooth = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)

    refined = np.full_like(coord_map, np.nan, dtype=np.float32)
    use = valid & keep_map
    refined[use] = (coord_map[use].astype(np.float64) + smooth[use][:, None] * normals[use]).astype(np.float32)
    info.update({"n_dropped": int(valid.sum() - use.sum()), "mean_abs_shift": float(np.abs(smooth[use]).mean()) if use.any() else 0.0,
                 "papyrus_threshold": float(papyrus_threshold)})
    return refined, info
