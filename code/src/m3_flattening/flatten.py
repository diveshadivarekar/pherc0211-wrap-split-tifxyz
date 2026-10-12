"""M3 — unfold each segmented 3D page/wrap surface into a flat 2D page.

Implements FR-3.1-FR-3.4. Each M2 segment in this phantom is (as a real
rolled scroll is) an approximately cylindrical sheet, so it is *developable*
— zero Gaussian curvature — meaning it can be cut and unrolled into a plane
with essentially no metric distortion. Two steps:

1. Cylindrical unwrap: for every depth slice, fit the segment's centre and
   mean radius, then resample the volume on a regular (angle, depth) grid
   along that fitted radius. This alone is already a near-isometric
   flattening for a true cylinder, and produces the coordinate map
   (FR-3.4: every 2D pixel keeps its source 3D coordinate).
2. ARAP relaxation: treat the unwrapped grid as a mass-spring mesh and run
   local-global As-Rigid-As-Possible relaxation (Sorkine & Alexa 2007) to
   minimise the shear/stretch energy from ARCHITECTURE.md, so a page that
   *isn't* perfectly cylindrical (elliptical cross-section, wobble) still
   gets locally distortion-minimised. On this phantom's true cylinders the
   correction is small — the tests measure and assert exactly that, via the
   mean edge-length distortion metric (FR-3.2/FR-3.3), not a general-purpose
   mesh parameteriser.

This is not the Bullet Physics mass-spring solver named in ARCHITECTURE.md
(no Bullet dependency here); it's a from-scratch NumPy implementation of the
same physical model (a 2D grid mesh relaxed under an ARAP energy) so the
resulting flattening is real, testable output rather than a stub.

Real scroll segments (M2's output on a real scan) are *not* centred
cylinders — they are irregular sheet patches — so a second path,
:func:`patch_unwrap`, fits a plane to the patch (PCA), treats the sheet as a
height field over it, and samples the volume along that surface.
:func:`flatten_layer` picks between the two per segment.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

CYLINDER_MAX_RADIAL_SPREAD = 0.25  # median per-slice std(radius)/mean(radius) below which a segment counts as a ring...
CYLINDER_MIN_ANGULAR_COVERAGE = 0.9  # ...and it must also go (nearly) all the way round, unlike a sheet patch/arc
PATCH_MIN_VOXELS = 8
PATCH_MAX_SIDE = 4096
PATCH_SMOOTH_SIGMA = 2.0


def _segment_center_and_radius(labels: np.ndarray, layer_id: int, depth_idx: int) -> tuple[float, float, float] | None:
    """Fit the (cy, cx, radius) of one segment's ring on a single depth slice."""
    ys, xs = np.where(labels[depth_idx] == layer_id)
    if len(ys) < 8:
        return None
    cy, cx = ys.mean(), xs.mean()
    radius = float(np.mean(np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)))
    return cy, cx, radius


def cylindrical_unwrap(
    volume: np.ndarray, labels: np.ndarray, layer_id: int, n_theta: int = 360
) -> tuple[np.ndarray, np.ndarray]:
    """Unroll one M2 segment into a flat (depth, n_theta) page.

    Returns ``(page, coord_map)``:
    - ``page``: float32 array (depth, n_theta) of resampled density values.
    - ``coord_map``: float32 array (depth, n_theta, 3), the (z, y, x) source
      voxel coordinate each page pixel was sampled from (FR-3.4
      traceability). NaN where the segment has no voxels on that slice.
    """
    depth, h, w = volume.shape
    page = np.zeros((depth, n_theta), dtype=np.float32)
    coord_map = np.full((depth, n_theta, 3), np.nan, dtype=np.float32)
    thetas = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)

    for z in range(depth):
        fit = _segment_center_and_radius(labels, layer_id, z)
        if fit is None:
            continue
        cy, cx, radius = fit
        ys = cy + radius * np.sin(thetas)
        xs = cx + radius * np.cos(thetas)

        sample_coords = np.vstack([ys, xs])
        page[z] = map_coordinates(volume[z], sample_coords, order=1, mode="constant", cval=0.0)

        coord_map[z, :, 0] = z
        coord_map[z, :, 1] = ys
        coord_map[z, :, 2] = xs

    return page, coord_map


def arap_relax(coord_map: np.ndarray, n_iters: int = 5) -> tuple[np.ndarray, float, float]:
    """Local-global ARAP relaxation of the unwrapped grid mesh.

    ``coord_map`` (depth, n_theta, 3) is treated as a regular grid mesh
    embedded in 3D (its *rest* shape); the flattened 2D grid positions
    (row, col) are the *deformed* shape being solved for. Each grid cell's
    best-fit local rotation (from 3D rest edges to the current 2D edges) is
    computed, then a global least-squares step re-solves 2D vertex positions
    to match those rotated rest edges — the standard ARAP local/global loop.

    Returns ``(uv, mean_distortion_before, mean_distortion_after)`` where
    ``uv`` is (depth, n_theta, 2) — the relaxed 2D page coordinates — and
    the distortion metrics are mean relative edge-length error against the
    3D rest mesh (FR-3.2/FR-3.3).

    Known limitation: the cumulative relayout below chains positions with
    ``cumsum``, so a NaN row/column in the *middle* of ``coord_map`` (a
    segment that doesn't span every depth slice or full revolution) would
    propagate NaN to every later position along that axis, not just the gap
    itself. Every layer in this phantom spans full depth by construction
    (see generate_phantom.py), so this never triggers here — but a
    segmentation on messier real data that produces partial-coverage
    segments would need this gap-filled first (e.g. carry the last valid
    direction forward) before relaxing.
    """
    depth, n_theta, _ = coord_map.shape
    valid = ~np.isnan(coord_map[:, :, 0])

    rows, cols = np.mgrid[0:depth, 0:n_theta].astype(np.float64)
    uv = np.stack([rows, cols], axis=-1)  # initial 2D layout: the raw grid itself

    def edge_lengths_3d(axis: int) -> np.ndarray:
        diff = np.diff(coord_map, axis=axis)
        return np.linalg.norm(diff, axis=-1)

    def edge_lengths_2d(pts: np.ndarray, axis: int) -> np.ndarray:
        diff = np.diff(pts, axis=axis)
        return np.linalg.norm(diff, axis=-1)

    def mean_distortion(pts: np.ndarray) -> float:
        errs = []
        for axis in (0, 1):
            len3d = edge_lengths_3d(axis)
            len2d = edge_lengths_2d(pts, axis)
            mask = np.isfinite(len3d) & (len3d > 1e-6)
            if mask.any():
                errs.append(np.abs(len2d[mask] - len3d[mask]) / len3d[mask])
        return float(np.mean(np.concatenate(errs))) if errs else 0.0

    dist_before = mean_distortion(uv)

    # Target 2D edge length per axis = median 3D rest edge length along that
    # axis (a single global scale keeps the local/global loop well-posed for
    # this regular-grid mesh, without a full sparse Poisson solve).
    def _median_edge(axis: int) -> float:
        lengths = edge_lengths_3d(axis)
        lengths = lengths[np.isfinite(lengths) & (lengths > 1e-6)]
        return float(np.median(lengths)) if lengths.size else 1.0

    scale_row = _median_edge(0)
    scale_col = _median_edge(1)

    def _cumulative_relayout(pts: np.ndarray, axis: int, target_scale: float) -> np.ndarray:
        """Re-lay ``pts`` out along ``axis`` at uniform ``target_scale`` spacing,
        keeping each step's *direction* from the current layout (local) but
        chaining positions with a running cumulative sum (global) — the
        local/global ARAP update, vectorised via cumsum instead of a
        column-by-column Python loop."""
        step = np.diff(pts, axis=axis)
        step_norm = np.linalg.norm(step, axis=-1, keepdims=True)
        step_unit = np.divide(step, step_norm, out=np.zeros_like(step), where=step_norm > 1e-9)
        # An edge touching an invalid (NaN) vertex carries no 3D evidence: lay it
        # out along the plain grid axis instead, so a gap neither collapses nor
        # drags every later vertex along the axis out of place.
        edge_valid = np.take(valid, np.arange(1, valid.shape[axis]), axis=axis) & np.take(
            valid, np.arange(0, valid.shape[axis] - 1), axis=axis
        )
        axis_unit = np.zeros(2)
        axis_unit[axis] = 1.0
        step_unit = np.where(edge_valid[..., np.newaxis], step_unit, axis_unit)
        displacement = np.cumsum(step_unit * target_scale, axis=axis)

        anchor = np.take(pts, [0], axis=axis)
        rest = anchor + displacement
        return np.concatenate([anchor, rest], axis=axis)

    for _ in range(n_iters):
        new_uv = _cumulative_relayout(uv, axis=0, target_scale=scale_row)
        new_uv = _cumulative_relayout(new_uv, axis=1, target_scale=scale_col)
        uv = np.where(valid[..., np.newaxis], new_uv, uv)

    dist_after = mean_distortion(uv)
    return uv.astype(np.float32), dist_before, dist_after


def _is_cylindrical(labels: np.ndarray, layer_id: int) -> bool:
    """True if the segment looks like a centred ring on most of its slices
    (the phantom's layers): small per-slice spread of radius about its
    centroid, and angular coverage of (nearly) the full circle — an arc or
    patch can have a small radial spread too, but never goes all the way round."""
    spreads, coverage = [], []
    for z in range(labels.shape[0]):
        ys, xs = np.where(labels[z] == layer_id)
        if len(ys) < 8:
            continue
        dy, dx = ys - ys.mean(), xs - xs.mean()
        r = np.sqrt(dy**2 + dx**2)
        if r.mean() > 1e-6:
            spreads.append(r.std() / r.mean())
            bins = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * 12).astype(int).clip(0, 11)
            coverage.append(len(np.unique(bins)) / 12)
    if not spreads:
        return False
    return float(np.median(spreads)) < CYLINDER_MAX_RADIAL_SPREAD and float(np.median(coverage)) >= CYLINDER_MIN_ANGULAR_COVERAGE


def patch_unwrap(
    volume: np.ndarray, labels: np.ndarray, layer_id: int, origin: tuple[int, int, int] = (0, 0, 0)
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten an irregular sheet patch.

    PCA of the segment's voxels gives an in-plane basis (``e1``, ``e2``) and
    a normal ``e3``. The patch is rasterised on a 1-voxel (u, v) grid; each
    cell's surface height ``w`` is the mean of its voxels' offsets along
    ``e3``, smoothed and hole-filled with normalised convolution, and the
    volume is sampled at that 3D surface point (max of three samples across
    the normal, to cover sheet thickness).

    Returns ``(page, coord_map)`` like :func:`cylindrical_unwrap`: ``page`` is
    (rows=v, cols=u) float32, ``coord_map`` is (rows, cols, 3) float32 holding
    the absolute (z, y, x) source coordinate, NaN outside the patch.
    ``origin`` is the (z, y, x) offset of ``labels``/``volume`` within the
    full volume, so coordinates stay traceable when a bounding-box crop is passed.
    """
    pts = np.argwhere(labels == layer_id).astype(np.float64)
    if len(pts) < PATCH_MIN_VOXELS:
        return np.zeros((1, 1), np.float32), np.full((1, 1, 3), np.nan, np.float32)

    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    e1, e2, e3 = vt[0], vt[1], vt[2]
    rel = pts - centre
    u, v, w = rel @ e1, rel @ e2, rel @ e3

    u0, v0 = np.floor(u.min()), np.floor(v.min())
    cols = int(np.floor(u.max() - u0)) + 1
    rows = int(np.floor(v.max() - v0)) + 1
    step = max(rows, cols) / PATCH_MAX_SIDE if max(rows, cols) > PATCH_MAX_SIDE else 1.0  # coarsen rather than build an enormous grid
    cols, rows = int(cols / step) + 1, int(rows / step) + 1
    ci = ((u - u0) / step).astype(int).clip(0, cols - 1)
    ri = ((v - v0) / step).astype(int).clip(0, rows - 1)

    flat = ri * cols + ci
    count = np.bincount(flat, minlength=rows * cols).astype(np.float64).reshape(rows, cols)
    wsum = np.bincount(flat, weights=w, minlength=rows * cols).reshape(rows, cols)

    support = count > 0
    # normalised convolution: smooth heights over occupied cells only, which also fills small holes
    num = ndi.gaussian_filter(wsum, PATCH_SMOOTH_SIGMA)
    den = ndi.gaussian_filter(count, PATCH_SMOOTH_SIGMA)
    with np.errstate(invalid="ignore", divide="ignore"):
        height = num / den
    valid = ndi.binary_closing(support, structure=np.ones((3, 3)), iterations=2) & (den > 1e-3)

    rr, cc = np.nonzero(valid)
    uu = u0 + cc * step
    vv = v0 + rr * step
    ww = height[rr, cc]
    pos = centre + uu[:, None] * e1 + vv[:, None] * e2 + ww[:, None] * e3  # crop-local (z, y, x)

    page = np.zeros((rows, cols), dtype=np.float32)
    coord_map = np.full((rows, cols, 3), np.nan, dtype=np.float32)
    samples = [map_coordinates(volume, (pos + k * e3).T, order=1, mode="nearest") for k in (-1.0, 0.0, 1.0)]
    page[rr, cc] = np.max(samples, axis=0)
    coord_map[rr, cc] = pos + np.asarray(origin, dtype=np.float64)
    return page, coord_map


def _grid_normals(coord_map: np.ndarray) -> np.ndarray:
    """Unit normals ``(H, W, 3)`` of a page's coordinate grid (finite differences); NaN where the page is empty."""
    valid = ~np.isnan(coord_map[..., 0])
    filled = np.where(valid[..., None], coord_map, 0.0).astype(np.float64)
    w = valid.astype(np.float64)
    den = ndi.gaussian_filter(w, 1.5)
    sm = np.stack([ndi.gaussian_filter(filled[..., c] * w, 1.5) for c in range(3)], axis=-1)
    sm = np.divide(sm, den[..., None], out=np.zeros_like(sm), where=den[..., None] > 1e-6)
    n = np.cross(np.gradient(sm, axis=1), np.gradient(sm, axis=0))
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.divide(n, norm, out=np.full_like(n, np.nan), where=norm > 1e-9)
    n[~valid] = np.nan
    return n


def sheet_unwrap(
    volume: np.ndarray,
    labels: np.ndarray,
    layer_id: int,
    axis: tuple[np.ndarray, np.ndarray, np.ndarray],
    origin: tuple[int, int, int] = (0, 0, 0),
    cell: float = 1.0,
    smooth_sigma: float = 3.0,
    min_support: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Unroll one single-sheet segment of a rolled scroll by its cylindrical height field ``r = f(theta, z)``.

    Unlike :func:`patch_unwrap` (a plane height field) this parametrisation needs no plane to be fitted, and it
    is exact for any single-valued sheet however far it wraps around the axis. Neither can represent a piece in
    which one ray from the axis crosses the sheet more than once (two sheets, or a sheet folded over itself):
    there both put points in the air between the layers. This is the cause of "floating" pages and is fixed upstream by ``split_wraps(single_valued=True)``.

    1. every voxel gives ``(theta, z, r)`` about the axis (``axis`` = ``(z, y, x)`` control-point arrays in the
       frame of ``volume``/``labels`` *before* any bounding-box crop; ``origin`` is the crop's offset in that frame);
    2. voxels are binned on a fine ``(z, theta)`` grid, the mean ``r`` per bin is smoothed over occupied bins
       (normalised convolution) and kept only where support is at least ``min_support`` - small gaps are filled,
       large ones are left empty rather than bridged;
    3. each row is re-sampled at ``cell`` voxels of arc length at that row's radius, about a common central meridian,
       so a sheet that leans in radius is not stretched wider where it is further out.

    Trade-off against :func:`patch_unwrap` (same segments): no vertices floating off the
    sheet and fewer cross-wrap cells, but a more sheared grid.
    Constant-theta columns do not remove that shear: it comes from the sheet's radius changing along both directions,
    and only a proper isometric flattening of the surface would. It is therefore opt-in (``method="sheet"``).

    Returns ``(page, coord_map)`` like :func:`patch_unwrap`: ``coord_map`` is ``(rows = z, cols = arc, 3)`` float32
    holding the ``(z, y, x)`` coordinate in the frame of ``volume`` (NaN outside the sheet); ``page`` is the density
    at the surface (max of three samples across the normal, to cover sheet thickness).
    """
    pts = np.argwhere(labels == layer_id).astype(np.float64)
    if len(pts) < PATCH_MIN_VOXELS:
        return np.zeros((1, 1), np.float32), np.full((1, 1, 3), np.nan, np.float32)
    az, ay, ax = axis
    oz, oy, ox = origin
    z_t = pts[:, 0] + oz
    dy = pts[:, 1] + oy - np.interp(z_t, az, ay)
    dx = pts[:, 2] + ox - np.interp(z_t, az, ax)
    r, theta = np.hypot(dy, dx), np.arctan2(dy, dx)
    centre = np.arctan2(np.sin(theta).mean(), np.cos(theta).mean())
    rel = (theta - centre + np.pi) % (2 * np.pi) - np.pi  # angle from the piece's central meridian

    r_hi = float(r.max())
    d_theta = cell / r_hi  # fine bins: at most one voxel of arc anywhere on the sheet
    t0 = rel.min()
    n_t = int(np.ceil((rel.max() - t0) / d_theta)) + 1
    z0 = int(pts[:, 0].min())
    n_z = int(pts[:, 0].max()) - z0 + 1
    ti = np.minimum(((rel - t0) / d_theta).astype(np.int64), n_t - 1)
    zi = (pts[:, 0] - z0).astype(np.int64)
    flat = zi * n_t + ti
    cnt = np.bincount(flat, minlength=n_z * n_t).astype(np.float64).reshape(n_z, n_t)
    rsum = np.bincount(flat, weights=r, minlength=n_z * n_t).reshape(n_z, n_t)
    occ = (cnt > 0).astype(np.float64)
    den = ndi.gaussian_filter(occ, smooth_sigma)
    num = ndi.gaussian_filter(np.divide(rsum, cnt, out=np.zeros_like(rsum), where=cnt > 0) * occ, smooth_sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_s = num / den
    ok = den >= min_support * ndi.gaussian_filter(np.ones_like(occ), smooth_sigma)  # relative to a fully occupied neighbourhood
    ok &= den > 1e-6

    # per-row radius (smoothed over z) sets the arc step of that row
    row_r = np.array([np.median(r_s[j][ok[j]]) if ok[j].sum() >= 2 else np.nan for j in range(n_z)])
    good_rows = np.isfinite(row_r)
    if good_rows.sum() < 2:
        return np.zeros((1, 1), np.float32), np.full((1, 1, 3), np.nan, np.float32)
    row_r = np.interp(np.arange(n_z), np.flatnonzero(good_rows), row_r[good_rows])
    row_r = ndi.gaussian_filter1d(row_r, 8.0, mode="nearest")
    half = int(np.ceil(max(abs(t0), abs(t0 + (n_t - 1) * d_theta)) * row_r.max() / cell))
    ks = np.arange(-half, half + 1)
    theta_f = t0 + np.arange(n_t) * d_theta

    rr = np.full((n_z, len(ks)), np.nan)
    for j in range(n_z):
        if ok[j].sum() < 2:
            continue
        th = ks * cell / row_r[j]
        inside = (th >= theta_f[ok[j]][0]) & (th <= theta_f[ok[j]][-1])
        if not inside.any():
            continue
        val = np.interp(th, theta_f[ok[j]], r_s[j][ok[j]])
        kept = np.interp(th, theta_f, ok[j].astype(np.float64)) > 0.95  # do not bridge gaps between occupied runs
        rr[j] = np.where(inside & kept, val, np.nan)
    cols_used = np.isfinite(rr).any(axis=0)
    if not cols_used.any():
        return np.zeros((1, 1), np.float32), np.full((1, 1, 3), np.nan, np.float32)
    lo_c, hi_c = np.flatnonzero(cols_used)[[0, -1]]
    rr, ks = rr[:, lo_c : hi_c + 1], ks[lo_c : hi_c + 1]

    zz = (z0 + np.arange(n_z))[:, None] * np.ones((1, len(ks)))
    ang = centre + ks[None, :] * cell / row_r[:, None]
    zt = zz + oz
    cy, cx = np.interp(zt, az, ay), np.interp(zt, az, ax)
    coord_map = np.stack([zz, cy + rr * np.sin(ang) - oy, cx + rr * np.cos(ang) - ox], axis=-1).astype(np.float32)
    coord_map[~np.isfinite(rr)] = np.nan

    page = np.zeros(rr.shape, dtype=np.float32)
    valid = ~np.isnan(coord_map[..., 0])
    nrm = _grid_normals(coord_map)
    base = coord_map[valid].astype(np.float64)
    nv = nrm[valid]
    good = np.isfinite(nv).all(axis=1)
    samples = [ndi.map_coordinates(volume, (base + k * np.nan_to_num(nv)).T, order=1, mode="nearest") for k in (-1.0, 0.0, 1.0)]
    vals = np.max(samples, axis=0)
    vals[~good] = samples[1][~good]
    page[valid] = vals
    return page, coord_map


def flatten_layer(
    volume: np.ndarray,
    labels: np.ndarray,
    layer_id: int,
    n_theta: int = 360,
    arap_iters: int = 5,
    bbox: tuple[slice, slice, slice] | None = None,
    method: str = "auto",
    axis: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Full M3 pipeline for one segment: unwrap then ARAP-relax.

    ``method`` is ``"cylinder"`` (a ring about its own centre), ``"sheet"`` (a single sheet of a rolled scroll
    by its height field about the scroll axis, :func:`sheet_unwrap`; needs ``axis``; opt-in), ``"patch"`` (a plane
    height field, :func:`patch_unwrap`) or ``"auto"`` (rings get the cylindrical unwrap, everything else the
    patch unwrap).
    ``axis`` is the scroll axis as ``(z, y, x)`` control-point arrays in the frame of ``volume``/``labels``
    before the ``bbox`` crop.
    ``bbox`` restricts work to the segment's bounding box (from
    ``scipy.ndimage.find_objects``) — essential when there are thousands of
    segments in a big volume; coordinates in the result stay absolute.

    Returns a dict with ``page``, ``coord_map``, ``uv`` (relaxed 2D layout),
    ``distortion_before``, ``distortion_after``, ``method``.
    """
    if method not in ("auto", "cylinder", "sheet", "patch"):
        raise ValueError(f"method must be 'auto', 'cylinder', 'sheet' or 'patch', got {method!r}")
    if method == "sheet" and axis is None:
        raise ValueError("method 'sheet' needs the scroll axis (axis=(z, y, x) control points)")
    origin = (0, 0, 0)
    if bbox is not None:
        volume, labels = volume[bbox], labels[bbox]
        origin = tuple(int(sl.start or 0) for sl in bbox)

    if method == "auto":
        method = "cylinder" if _is_cylindrical(labels, layer_id) else "patch"

    if method == "cylinder":
        page, coord_map = cylindrical_unwrap(volume, labels, layer_id, n_theta=n_theta)
        coord_map = coord_map + np.asarray(origin, dtype=np.float32)  # NaNs stay NaN
    elif method == "sheet":
        page, coord_map = sheet_unwrap(volume, labels, layer_id, axis, origin=origin)
        coord_map = coord_map + np.asarray(origin, dtype=np.float32)  # NaNs stay NaN
    else:
        page, coord_map = patch_unwrap(volume, labels, layer_id, origin=origin)

    uv, dist_before, dist_after = arap_relax(coord_map, n_iters=arap_iters)
    return {
        "page": page,
        "coord_map": coord_map,
        "uv": uv,
        "distortion_before": dist_before,
        "distortion_after": dist_after,
        "method": method,
    }
