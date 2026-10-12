"""Quality checks for tifxyz surfaces: mesh geometry, cross-wrap detection, and agreement with the volume.

Existing tifxyz auditors look at the mesh alone. A sheet that has *switched* onto a neighbouring wrap
or drifted off the papyrus is a smooth, well-formed mesh, so geometry cannot reveal it - the volume
(and, for a rolled scroll, the scroll axis) can. Three groups of checks:

* :func:`geometry_report` - validity, holes, edge-length vs. the declared ``scale``, cell shear, folded
  quads.
* :func:`radiality` - in a rolled scroll every sheet is, locally, a cylinder wall: its normal points
  along the radius from the scroll axis (the *umbilicus*). Cells whose normal is far from radial are
  cutting *across* wraps - a fold, a ramp from one wrap onto the next, or a hole-filled bridge.
* :func:`surface_support` / :func:`snap_to_prediction` / :func:`ct_centering` - sample the learned
  surface prediction and the raw CT along each vertex normal: how much of the mesh lies on predicted
  surface, how far it is from the nearest one, and (the label-snapping idea from the project's open
  problems) move it there. Because snapping onto a prediction and then scoring against the same
  prediction is circular, :func:`ct_centering` scores against the *raw CT* instead: a correctly placed
  sheet is a density ridge centred on the vertex.

Everything works in level-0 voxels of the scroll volume, ``(z, y, x)`` for volumes and ``(x, y, z)`` for
tifxyz grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from src.tifxyz.io import Surface


# --------------------------------------------------------------------------------------- geometry

def _smooth_nan(a: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing over valid cells only (normalised convolution); ``a`` is ``(H, W[, C])``."""
    w = valid.astype(np.float64)
    den = ndi.gaussian_filter(w, sigma)
    a = np.where(valid.reshape(valid.shape + (1,) * (a.ndim - 2)), a, 0.0)
    if a.ndim == 2:
        a = a[..., None]
    out = np.stack([ndi.gaussian_filter(a[..., c] * w, sigma) for c in range(a.shape[-1])], axis=-1)
    out = np.divide(out, den[..., None], out=np.full_like(out, np.nan), where=den[..., None] > 1e-6)
    return out


def vertex_normals(surface: Surface, smooth: float = 2.0) -> np.ndarray:
    """Unit normals ``(H, W, 3)`` in ``(x, y, z)`` from the smoothed grid (NaN at invalid vertices).

    The sign follows ``d(point)/d(col) x d(point)/d(row)``, so it is consistent across the grid but
    says nothing about which side of the sheet faces the scroll axis.
    """
    valid = surface.valid
    p = _smooth_nan(surface.xyz.astype(np.float64), valid, smooth)
    du = np.gradient(p, axis=1)
    dv = np.gradient(p, axis=0)
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.divide(n, norm, out=np.full_like(n, np.nan), where=norm > 1e-12)
    n[~valid] = np.nan
    return n


def _cells(surface: Surface) -> tuple[np.ndarray, ...]:
    """Per-quad arrays over (rows-1, cols-1): all-corners-valid mask, centre ``(x, y, z)``, unnormalised normal,
    the lengths of the column-wise and row-wise edge vectors, and those two vectors (``du``, ``dv``)."""
    p = surface.xyz.astype(np.float64)
    v = surface.valid
    ok = v[:-1, :-1] & v[:-1, 1:] & v[1:, :-1] & v[1:, 1:]
    a, b, c, d = p[:-1, :-1], p[:-1, 1:], p[1:, :-1], p[1:, 1:]
    du = (b - a + d - c) / 2
    dv = (c - a + d - b) / 2
    return ok, (a + b + c + d) / 4, np.cross(du, dv), np.stack([np.linalg.norm(du, axis=-1), np.linalg.norm(dv, axis=-1)], axis=-1), du, dv


def geometry_report(surface: Surface) -> dict:
    """Mesh-only statistics. Edge ratios are measured edge length divided by the declared spacing ``1 / scale``."""
    valid = surface.valid
    n_valid = int(valid.sum())
    rep: dict = {"grid": list(surface.shape), "valid_vertices": n_valid, "valid_fraction": float(valid.mean()), "scale": list(surface.scale)}
    if n_valid == 0:
        return rep
    comp, n_comp = ndi.label(valid, structure=np.ones((3, 3)))
    sizes = np.bincount(comp.ravel())[1:]
    rep["n_components"] = int(n_comp)
    rep["largest_component_fraction"] = float(sizes.max() / n_valid)
    holes, n_holes = ndi.label(~valid)
    border = np.unique(np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]]))
    enclosed = [i for i in range(1, n_holes + 1) if i not in border]
    rep["n_enclosed_holes"] = len(enclosed)
    rep["enclosed_hole_cells"] = int(sum((holes == i).sum() for i in enclosed)) if enclosed else 0
    rep["area_vx2"] = surface.area_vx2()

    ok, _, n, lens, du, dv = _cells(surface)
    if ok.any():
        sx, sy = surface.scale
        ratio_u = lens[..., 0][ok] * sx
        ratio_v = lens[..., 1][ok] * sy
        both = np.concatenate([ratio_u, ratio_v])
        rep["edge_ratio"] = {
            "p05": float(np.percentile(both, 5)), "p50": float(np.percentile(both, 50)), "p95": float(np.percentile(both, 95)),
            "p99": float(np.percentile(both, 99)), "fraction_over_2x": float((both > 2).mean()),
        }
        cos = np.abs((du * dv).sum(-1)) / np.maximum(lens[..., 0] * lens[..., 1], 1e-12)
        rep["cell_shear_cos_p95"] = float(np.percentile(cos[ok], 95))
        aspect = np.maximum(lens[..., 0], lens[..., 1]) / np.maximum(np.minimum(lens[..., 0], lens[..., 1]), 1e-12)
        rep["cell_aspect_p95"] = float(np.percentile(aspect[ok], 95))
        fold = folded_cells(surface)
        rep["folded_cells"] = int(fold.sum())
        rep["folded_fraction"] = float(fold.sum() / ok.sum())
    return rep


def folded_cells(surface: Surface, window: int = 15) -> np.ndarray:
    """Boolean ``(H-1, W-1)``: quads whose normal points against the median normal of their ``window`` x ``window``
    neighbourhood (a fold-over). The median, not the mean, so a flipped patch smaller than half the window
    cannot drag its own reference with it."""
    ok, _, n, _, _, _ = _cells(surface)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    u = np.divide(n, norm, out=np.zeros_like(n), where=norm > 1e-12)
    # invalid cells would drag the median to zero; fill them with the nearest valid normal first
    _, (iy, ix) = ndi.distance_transform_edt(~ok, return_indices=True)
    filled = u[iy, ix]
    ref = np.stack([ndi.median_filter(filled[..., c], size=window, mode="nearest") for c in range(3)], axis=-1)
    cos = np.einsum("hwc,hwc->hw", u, ref)
    return ok & (cos < 0)


# ------------------------------------------------------------------------------ radiality (umbilicus)

def radiality(surface: Surface, umbilicus: tuple[np.ndarray, np.ndarray, np.ndarray], max_angle_deg: float = 60.0, smooth: float = 2.0) -> tuple[dict, dict]:
    """How radial the sheet is: angle between each vertex normal and the horizontal radius from the scroll axis.

    ``umbilicus`` is ``(z, y, x)`` control-point arrays (``surface_segments.load_umbilicus``). The axis is
    interpolated to each vertex's z and the radius taken in the xy plane, so a strongly tilted scroll
    axis weakens the check. Returns ``(report, maps)``; ``maps["cross_wrap"]`` is a boolean ``(H, W)`` mask.
    """
    from src.m2_segmentation.surface_segments import axis_at

    valid = surface.valid
    n = vertex_normals(surface, smooth)
    pts = surface.xyz.astype(np.float64)
    ay, ax = axis_at(umbilicus, np.where(valid, pts[..., 2], 0.0))
    radial = np.stack([pts[..., 0] - ax, pts[..., 1] - ay, np.zeros(valid.shape)], axis=-1)
    rn = np.linalg.norm(radial, axis=-1, keepdims=True)
    radial = np.divide(radial, rn, out=np.zeros_like(radial), where=rn > 1e-9)
    cos = np.abs(np.einsum("hwc,hwc->hw", np.nan_to_num(n), radial))
    good = valid & np.isfinite(n[..., 0]) & (rn[..., 0] > 1e-9)
    cos[~good] = np.nan
    bad = good & (cos < np.cos(np.deg2rad(max_angle_deg)))
    rep: dict = {"max_angle_deg": max_angle_deg, "evaluated_vertices": int(good.sum())}
    if good.any():
        ang = np.rad2deg(np.arccos(np.clip(cos[good], 0, 1)))
        rep.update({
            "angle_median_deg": float(np.median(ang)), "angle_p95_deg": float(np.percentile(ang, 95)),
            "cross_wrap_fraction": float(bad.sum() / good.sum()),
            "clusters": _clusters(bad, pts),
        })
    return rep, {"cos": cos, "cross_wrap": bad}


def estimate_wrap_spacing(pred: "VolumeRef", umbilicus: tuple[np.ndarray, np.ndarray, np.ndarray], z: float, r_max: float = 1200.0, n_rays: int = 24, threshold: float = 127.0, step: float = 2.0) -> float | None:
    """Median radial distance (voxels) between consecutive predicted sheets, measured on rays cast from the scroll axis at height ``z``.

    Each ray is sampled outward through the surface prediction; every contiguous run above ``threshold``
    is one sheet crossing, and the median gap between run centres over all rays is the wrap spacing.
    Returns ``None`` if the prediction does not cover the rays.
    """
    from src.m2_segmentation.surface_segments import axis_at

    ay, ax = axis_at(umbilicus, np.array([z]))
    radii = np.arange(step, r_max, step)
    gaps = []
    for th in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
        pts = np.stack([ax[0] + radii * np.cos(th), ay[0] + radii * np.sin(th), np.full_like(radii, z)], axis=-1)
        v = pred.sample(pts)
        inside = np.isfinite(v)
        if inside.sum() < 20:
            continue
        above = np.nan_to_num(v) > threshold
        edges = np.diff(np.concatenate([[False], above, [False]]).astype(int))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        centres = radii[((starts + ends - 1) // 2)]
        if len(centres) > 1:
            gaps.extend(np.diff(centres).tolist())
    gaps = [g for g in gaps if g > 4 * step]  # the two edges of one thick predicted sheet are not two wraps
    return float(np.median(gaps)) if gaps else None


def radial_extent(surface: Surface, umbilicus: tuple[np.ndarray, np.ndarray, np.ndarray], spacing: float | None = None, slice_height: float = 16.0, min_vertices: int = 20, reliable_span_deg: float = 90.0) -> dict:
    """How far the mesh runs *across* wraps, measured within horizontal slices.

    In each ``slice_height``-voxel slab of z the 5th-95th percentile spread of vertex distance from the scroll
    axis is taken, and ``radial_spread`` is the median over slabs holding at least ``min_vertices`` vertices.
    A single sheet has a small spread in one slab however much it *leans* in radius from top to bottom - a
    leaning sheet is still one sheet (``radial_spread_total``, the spread over the whole mesh, includes that
    lean and overstates multi-wrap pages).
    With ``spacing`` (see :func:`estimate_wrap_spacing`), ``wraps_crossed`` = ``radial_spread`` / spacing; values
    well above 1 mean the mesh is a plane cutting several wraps.

    **Only trustworthy for pages spanning a small angle.** The scroll cross-section is oval, so one wrap's distance from the
    axis changes along its length: ``wraps_crossed`` grows with the page's angular span, so long pages can read as multi-wrap while still lying on predicted surface. The result
    carries ``angular_span_deg`` (95 % of vertices) and ``wraps_crossed_reliable`` (span <= ``reliable_span_deg``);
    for longer pages use the on-surface and cross-wrap measures instead.
    """
    from src.m2_segmentation.surface_segments import axis_at

    pts = surface.xyz[surface.valid].astype(np.float64)
    ay, ax = axis_at(umbilicus, pts[:, 2])
    r = np.hypot(pts[:, 0] - ax, pts[:, 1] - ay)
    lo, hi = np.percentile(r, [5, 95])
    slab = np.floor((pts[:, 2] - pts[:, 2].min()) / slice_height).astype(np.int64)
    spreads = []
    for k in np.unique(slab):
        rs = r[slab == k]
        if len(rs) >= min_vertices:
            spreads.append(np.percentile(rs, 95) - np.percentile(rs, 5))
    within = float(np.median(spreads)) if spreads else float(hi - lo)
    theta = np.arctan2(pts[:, 1] - ay, pts[:, 0] - ax)
    centre = np.arctan2(np.sin(theta).mean(), np.cos(theta).mean())
    rel = np.degrees((theta - centre + np.pi) % (2 * np.pi) - np.pi)
    span = float(np.percentile(rel, 97.5) - np.percentile(rel, 2.5))
    out = {"radius_p05": float(lo), "radius_p95": float(hi), "radial_spread_total": float(hi - lo), "radial_spread": within, "slabs": len(spreads), "angular_span_deg": span, "wraps_crossed_reliable": bool(span <= reliable_span_deg)}
    if spacing:
        out.update({"wrap_spacing": float(spacing), "wraps_crossed": within / spacing, "wraps_crossed_total": float((hi - lo) / spacing)})
    return out


def _clusters(mask: np.ndarray, pts: np.ndarray, top: int = 8, min_cells: int = 25) -> list[dict]:
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        m = lab[sl] == i
        if m.sum() < min_cells:
            continue
        centre = np.nanmean(pts[sl][m], axis=0)
        out.append({"cells": int(m.sum()), "rows": [int(sl[0].start), int(sl[0].stop)], "cols": [int(sl[1].start), int(sl[1].stop)], "centre_xyz": [round(float(c), 1) for c in centre]})
    return sorted(out, key=lambda c: -c["cells"])[:top]


# --------------------------------------------------------------------------------- volume sampling

@dataclass
class VolumeRef:
    """A 3D array placed in level-0 scroll voxels: voxel ``i`` spans ``origin + i * scale`` (``(z, y, x)``)."""

    data: object  # numpy array or h5py dataset, (Z, Y, X)
    origin_zyx: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0

    def sample(self, pts_xyz: np.ndarray, order: int = 1) -> np.ndarray:
        """Trilinear samples at ``pts_xyz`` (N, 3 in level-0 voxels); NaN outside the volume. Reads only the needed block."""
        pts_xyz = np.asarray(pts_xyz, dtype=np.float64)
        idx = (pts_xyz[:, ::-1] - np.asarray(self.origin_zyx) - self.scale / 2) / self.scale
        shape = np.asarray(self.data.shape)
        inside = np.isfinite(idx).all(axis=1) & (idx >= 0).all(axis=1) & (idx <= shape - 1).all(axis=1)
        out = np.full(len(idx), np.nan, dtype=np.float32)
        if not inside.any():
            return out
        lo = np.floor(idx[inside].min(axis=0)).astype(int)
        hi = np.minimum(np.ceil(idx[inside].max(axis=0)).astype(int) + 2, shape)
        block = np.asarray(self.data[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]], dtype=np.float32)
        out[inside] = ndi.map_coordinates(block, (idx[inside] - lo).T, order=order, mode="nearest")
        return out


def load_volume(path: str | Path) -> VolumeRef:
    """A ``volume`` dataset as written by the download scripts: attrs ``z0/y0/x0`` (the crop's corner, in units
    of its own voxels) and, for downsampled predictions, ``scale_to_native`` (native voxels per voxel; default 1).
    The file stays open for lazy block reads."""
    import h5py

    ds = h5py.File(path, "r")["volume"]
    scale = float(ds.attrs.get("scale_to_native", 1))
    origin = tuple(float(ds.attrs[k]) * scale for k in ("z0", "y0", "x0"))
    return VolumeRef(ds, origin, scale)


load_prediction = load_volume


def _profiles(surface: Surface, vol: VolumeRef, offsets: np.ndarray, indices: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """``(len(indices), len(offsets))`` volume values at ``vertex + offset * normal`` for flat vertex ``indices``."""
    pts = surface.xyz.reshape(-1, 3)[indices].astype(np.float64)
    nrm = normals.reshape(-1, 3)[indices]
    out = np.empty((len(indices), len(offsets)), dtype=np.float32)
    chunk = max(1, 400_000 // len(offsets))
    for s in range(0, len(indices), chunk):
        p = pts[s : s + chunk, None, :] + offsets[None, :, None] * nrm[s : s + chunk, None, :]
        out[s : s + chunk] = vol.sample(p.reshape(-1, 3)).reshape(-1, len(offsets))
    return out


def _pick_vertices(surface: Surface, normals: np.ndarray, max_points: int | None, seed: int = 0) -> np.ndarray:
    ok = np.flatnonzero((surface.valid & np.isfinite(normals[..., 0])).ravel())
    if max_points and len(ok) > max_points:
        ok = np.sort(np.random.default_rng(seed).choice(ok, max_points, replace=False))
    return ok


def _nearest_ridge_offset(prof: np.ndarray, offsets: np.ndarray, threshold: float) -> np.ndarray:
    """Per profile, the offset of the *centre* of the above-``threshold`` run nearest to offset 0 (NaN if none).

    Snapping to the first sample over the threshold would land on the edge of a thick predicted wall; the
    value-weighted centroid of the whole contiguous run is where a person would place the surface.
    """
    above = np.nan_to_num(prof, nan=-np.inf) > threshold
    n = len(prof)
    starts = above & ~np.concatenate([np.zeros((n, 1), bool), above[:, :-1]], axis=1)
    run = np.cumsum(starts, axis=1) * above  # 0 = below threshold, k = k-th run along the profile
    dist = np.where(above, np.abs(offsets)[None, :], np.inf)
    nearest = np.argmin(dist, axis=1)
    found = np.isfinite(dist[np.arange(n), nearest])
    in_run = above & (run == run[np.arange(n), nearest][:, None])
    w = np.where(in_run, np.nan_to_num(prof) - threshold, 0.0)
    total = w.sum(axis=1)
    centroid = np.divide((w * offsets[None, :]).sum(axis=1), total, out=np.full(n, np.nan), where=total > 0)
    return np.where(found, centroid, np.nan)


def surface_support(surface: Surface, pred: VolumeRef, threshold: float = 127.0, search: float = 12.0, step: float | None = None, max_points: int | None = 200_000) -> tuple[dict, dict]:
    """How much of the mesh lies on the predicted surface, and how far each vertex is from the nearest one.

    For each vertex the prediction is sampled along its normal over ``+-search`` voxels. ``on_surface`` is
    the prediction value at the vertex being above ``threshold``; ``drift`` is the signed distance (along the
    normal, voxels) to the centre of the *nearest* above-``threshold`` run, NaN when there is none within ``search``.
    """
    step = step or max(1.0, pred.scale / 2)
    offsets = np.arange(-search, search + step / 2, step)
    normals = vertex_normals(surface)
    idx = _pick_vertices(surface, normals, max_points)
    prof = _profiles(surface, pred, offsets, idx, normals)
    centre = prof[:, len(offsets) // 2]
    drift = _nearest_ridge_offset(prof, offsets, threshold)
    found = np.isfinite(drift)
    known = np.isfinite(centre)
    rep = {
        "threshold": threshold, "search_voxels": search, "vertices_sampled": int(len(idx)), "vertices_in_volume": int(known.sum()),
        "on_surface_fraction": float((centre[known] > threshold).mean()) if known.any() else float("nan"),
        "surface_within_search_fraction": float(found[known].mean()) if known.any() else float("nan"),
        "abs_drift_median": float(np.nanmedian(np.abs(drift[known]))) if found[known].any() else float("nan"),
        "abs_drift_p95": float(np.nanpercentile(np.abs(drift[known]), 95)) if found[known].any() else float("nan"),
    }
    h, w = surface.shape
    support_map, drift_map = np.full(h * w, np.nan, np.float32), np.full(h * w, np.nan, np.float32)
    support_map[idx], drift_map[idx] = centre, drift
    return rep, {"support": support_map.reshape(h, w), "drift": drift_map.reshape(h, w)}


def snap_to_prediction(surface: Surface, pred: VolumeRef, threshold: float = 127.0, search: float = 12.0, sigma: float = 3.0, max_points: int | None = None) -> tuple[Surface, dict]:
    """Move vertices along their normals onto the nearest predicted surface, with the shift field smoothed.

    Vertices with no surface within ``search`` take the smoothed shift of their neighbours (no shift where
    there is no neighbour evidence). The shift is clipped to ``+-search``. Returns the new surface and a
    report of the applied shifts.
    """
    rep, maps = surface_support(surface, pred, threshold, search, max_points=max_points)
    drift = maps["drift"]
    have = np.isfinite(drift)
    smooth = _smooth_nan(np.where(have, drift, 0.0), have, sigma)[..., 0]
    shift = np.where(np.isfinite(smooth), np.clip(smooth, -search, search), 0.0)
    shift[~surface.valid] = 0.0
    normals = vertex_normals(surface)
    xyz = surface.xyz + (shift[..., None] * np.nan_to_num(normals)).astype(np.float32)
    snapped = Surface(xyz=xyz, scale=surface.scale, uuid=surface.uuid, meta=dict(surface.meta))
    moved = np.abs(shift[surface.valid])
    return snapped, {"abs_shift_median": float(np.median(moved)), "abs_shift_p95": float(np.percentile(moved, 95)), "fraction_shifted_over_1vox": float((moved > 1).mean())}


def ct_centering(surface: Surface, ct: VolumeRef, search: float = 30.0, step: float = 1.0, smooth_vox: float = 3.0, centred_within: float = 6.0, max_points: int | None = 100_000) -> dict:
    """Independent (prediction-free) check on placement: is the mesh at a density ridge of the raw CT?

    Along each normal the CT profile is smoothed and its peak found. ``contrast`` is the mean of
    (smoothed CT at the vertex - mean of the profile's outer quarter on each side), in CT units;
    ``centred_fraction`` is the share of vertices whose peak lies within ``centred_within`` voxels, to be
    read against ``centred_chance`` (what a random peak would score). The window must span the sheet
    thickness: at 2.4 um a papyrus sheet is tens of voxels thick, so the defaults look +-30 voxels. Even so the signal is weak (a shifted copy can land on the next wrap). Treat ``contrast`` as a secondary, *paired before/after* indicator, never an accuracy.
    """
    offsets = np.arange(-search, search + step / 2, step)
    normals = vertex_normals(surface)
    idx = _pick_vertices(surface, normals, max_points)
    prof = _profiles(surface, ct, offsets, idx, normals)
    ok = np.isfinite(prof).all(axis=1)
    prof = prof[ok]
    if not len(prof):
        return {"vertices_sampled": 0}
    prof = ndi.gaussian_filter1d(prof, smooth_vox / step, axis=1, mode="nearest")
    peak = offsets[np.argmax(prof, axis=1)]
    c = len(offsets) // 2
    q = max(1, len(offsets) // 4)
    flank = (prof[:, :q].mean(axis=1) + prof[:, -q:].mean(axis=1)) / 2
    return {
        "vertices_sampled": int(len(prof)), "search_voxels": search,
        "centred_fraction": float((np.abs(peak) <= centred_within).mean()),
        "centred_chance": float(min(1.0, centred_within / search)),
        "peak_offset_abs_median": float(np.median(np.abs(peak))),
        "contrast": float((prof[:, c] - flank).mean()),
        "mean_profile": prof.mean(axis=0).round(3).tolist(),
        "offsets": offsets.tolist(),
    }
