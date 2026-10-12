"""Segments from the official surface-prediction volume, instead of thresholding the scan.

M2's classical segmentation has to find sheets in a dense, touching stack of papyrus. The
Vesuvius Challenge publishes a learned *surface prediction* for each scroll (a 0-255 score per voxel
for "this is on a papyrus surface", thin and well separated even where the wraps touch). Labelling
its connected components gives clean sheets, which are far better input for flattening and ink
detection than a threshold on the raw density.

Two extra steps make the sheets flatten well:

* A rolled scroll's surface is one long spiral, but M3 flattens a roughly planar patch best.
  Each connected sheet is therefore cut into angular wedges around the scroll axis (the
  "umbilicus", interpolated along z from the published control points), and each connected piece
  of a wedge becomes one segment.
* The prediction is coarser than the scan (``scale`` native voxels per prediction voxel), so the
  labels are written back out at the scan's resolution (nearest-neighbour), aligned to the tile
  that the pages will be sampled from.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as cc_label

LABELS_DATASET = "labels"


def load_umbilicus(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Control points of the scroll axis as sorted ``(z, y, x)`` arrays in native voxels."""
    pts = json.loads(Path(path).read_text())["control_points"]
    order = np.argsort([p["z"] for p in pts])
    z = np.array([pts[i]["z"] for i in order], dtype=np.float64)
    return z, np.array([pts[i]["y"] for i in order], dtype=np.float64), np.array([pts[i]["x"] for i in order], dtype=np.float64)


def axis_at(umbilicus: tuple[np.ndarray, np.ndarray, np.ndarray], z_native: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scroll axis (y, x) in native voxels at the given native z values (linear interpolation)."""
    z, y, x = umbilicus
    return np.interp(z_native, z, y), np.interp(z_native, z, x)


def wedge_index(y: np.ndarray, x: np.ndarray, axis_y: np.ndarray, axis_x: np.ndarray, wedge_degrees: float) -> np.ndarray:
    """Which angular wedge (0, 1, ...) around the axis each point falls in."""
    theta = np.arctan2(y - axis_y, x - axis_x) % (2 * np.pi)
    return np.floor(theta / np.deg2rad(wedge_degrees)).astype(np.int64)


def split_wraps(
    pieces: np.ndarray,
    z_native: np.ndarray,
    y_native: np.ndarray,
    x_native: np.ndarray,
    umbilicus,
    theta_degrees: float | None = None,
    gap: float | None = None,
    link_tol: float | None = None,
    fat_factor: float = 1.6,
    max_sheets: int = 4,
    min_voxels: int = 200,
    single_valued: bool = True,
) -> np.ndarray:
    """Cut every piece into single sheets along the radius from the scroll axis.

    Where two wraps touch, the surface prediction is one 3D-connected blob, so a piece can hold several
    wraps (a plane fitted through it then crosses all of them). In cylindrical terms a sheet is a height
    field ``r = f(theta, z)``, so:

    1. along each radial ray (one angular bin x one-voxel-tall column) the occupied voxels split
       into *runs* wherever the radial gap exceeds ``gap`` - each run is one sheet crossing;
    2. runs in neighbouring columns are linked when their radial centres differ by at most ``link_tol``
       (a sheet is continuous in ``r``), and the linked runs form one sheet;
    3. a run thicker than ``fat_factor`` x the median radial extent is wraps in contact (two touching
       sheets form one double-thick run, whose centre sits between them and would link to both). It is
       divided radially into ``round(extent / median)`` equal sub-runs, one per sheet; a run thicker than
       ``max_sheets`` sheets is not a sheet at all (a radial fold) and is cut (label 0).

    ``gap`` defaults to 2.5 prediction voxels and ``link_tol`` to 0.75 (``scale`` native voxels per prediction
    voxel, read from the coordinate spacing): a sheet's radius may move by at most ``link_tol`` between
    neighbouring columns, i.e. a ramp steeper than about 37 degrees is a bridge, not a sheet. These defaults were fixed during earlier development of the pipeline and were not re-tuned for each scroll; sweep ``link_tol`` when the scale or resolution changes. The angular bin ``theta_degrees`` defaults to the arc of one prediction
    voxel at the 5th-percentile radius, so neighbouring bins are never empty at the inner edge. Returns labels like :func:`label_surface_pieces`: 1..N, largest first;
    cutting can only split pieces or remove junction voxels, never merge.

    Linked runs can still form a piece that has two sheets in one ray (two sheets joined by a gentle path
    elsewhere), which no height field ``r = f(theta, z)`` can represent and which makes a page float between its
    sheets. ``single_valued=True`` (the default) removes that:
    the runs of a piece in each column are ranked by radius and only runs of equal rank stay linked, so every
    piece has at most one run per column - at the price of cutting a sheet wherever a neighbouring sheet
    starts or ends beside it (more, smaller pieces).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    iz, iy, ix = np.nonzero(pieces)
    if not len(iz):
        return np.zeros(pieces.shape, dtype=np.int32)
    scale = float(np.median(np.diff(z_native))) if len(z_native) > 1 else 1.0
    gap = gap if gap is not None else 2.5 * scale
    link_tol = link_tol if link_tol is not None else 0.75 * scale

    piece = pieces[iz, iy, ix].astype(np.int64)
    axis_y, axis_x = axis_at(umbilicus, z_native[iz])
    dy, dx = y_native[iy] - axis_y, x_native[ix] - axis_x
    r = np.hypot(dy, dx)
    if theta_degrees is None:
        theta_degrees = float(np.degrees(scale / max(np.percentile(r, 5), scale)))
    b = np.floor((np.degrees(np.arctan2(dy, dx)) % 360.0) / theta_degrees).astype(np.int64)
    order = np.lexsort((r, b, iz, piece))  # by piece, z, angle bin, then radius
    piece, iz_s, b, r = piece[order], iz[order], b[order], r[order]

    new_col = np.r_[True, (piece[1:] != piece[:-1]) | (iz_s[1:] != iz_s[:-1]) | (b[1:] != b[:-1])]
    new_run = new_col | np.r_[True, (r[1:] - r[:-1]) > gap]
    run_of = np.cumsum(new_run) - 1
    starts = np.flatnonzero(new_run)
    extent0 = np.maximum.reduceat(r, starts) - np.minimum.reduceat(r, starts)
    typical = max(float(np.median(extent0)), scale)
    n_parts = np.where(extent0 > fat_factor * typical, np.rint(extent0 / typical).astype(np.int64), 1).clip(1, None)
    junk = n_parts > max_sheets  # too thick to be one or a few sheets: a radial fold or a blob
    # divide thick runs into one sub-run per sheet, by radial position (sub-runs stay ordered by radius)
    run_lo = np.minimum.reduceat(r, starts)
    sub = np.floor((r - run_lo[run_of]) / np.maximum(extent0[run_of], 1e-9) * n_parts[run_of]).astype(np.int64)
    sub = np.minimum(sub, n_parts[run_of] - 1)
    sub_key = run_of * (max_sheets + 1) + sub
    new_sub = np.r_[True, sub_key[1:] != sub_key[:-1]]
    run_of = np.cumsum(new_sub) - 1
    starts = np.flatnonzero(new_sub)
    n_runs = len(starts)
    r_lo, r_hi = np.minimum.reduceat(r, starts), np.maximum.reduceat(r, starts)
    centre = (r_lo + r_hi) / 2
    cut = junk[np.cumsum(new_run)[starts] - 1]  # per sub-run: did its parent run turn out to be junk?

    nz, nb = int(iz.max()) + 3, int(np.ceil(360.0 / theta_degrees)) + 3
    run_piece, run_z, run_b = piece[starts], iz_s[starts] + 1, b[starts] + 1  # +1 keeps shifted keys non-negative
    code = (run_piece * nz + run_z) * nb + run_b  # non-decreasing: runs are already sorted by (piece, z, angle bin)
    src_all, dst_all = [], []
    for dz, db in ((0, 1), (1, -1), (1, 0), (1, 1)):  # half of the 8 neighbours; the graph is undirected
        target = (run_piece * nz + run_z + dz) * nb + run_b + db
        lo, hi = np.searchsorted(code, target, "left"), np.searchsorted(code, target, "right")
        count = hi - lo
        src = np.repeat(np.arange(n_runs), count)
        dst = np.repeat(lo, count) + (np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count))
        keep = (np.abs(centre[src] - centre[dst]) <= link_tol) & ~cut[src] & ~cut[dst]
        src_all.append(src[keep]), dst_all.append(dst[keep])
    src, dst = np.concatenate(src_all), np.concatenate(dst_all)
    graph = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n_runs, n_runs))
    _, comp = connected_components(graph, directed=False)
    if single_valued:
        # rank of each run among the runs of its component in the same column (ordered by radius)
        key = comp.astype(np.int64) * (int(code.max()) + 1) + code
        order_r = np.lexsort((centre, key))
        sk = key[order_r]
        first = np.r_[True, sk[1:] != sk[:-1]]
        group_start = np.maximum.accumulate(np.where(first, np.arange(n_runs), 0))
        rank = np.empty(n_runs, dtype=np.int64)
        rank[order_r] = np.arange(n_runs) - group_start
        same = rank[src] == rank[dst]
        graph = coo_matrix((np.ones(int(same.sum()), dtype=np.int8), (src[same], dst[same])), shape=(n_runs, n_runs))
        _, comp = connected_components(graph, directed=False)

    voxel_comp = comp[run_of] + 1
    voxel_comp[cut[run_of]] = 0
    sizes = np.bincount(voxel_comp)
    sizes[0] = 0
    kept = np.flatnonzero(sizes >= min_voxels)
    ranked = kept[np.argsort(-sizes[kept])]
    lut = np.zeros(len(sizes), dtype=np.int32)
    lut[ranked] = np.arange(1, len(ranked) + 1, dtype=np.int32)
    out = np.zeros(pieces.shape, dtype=np.int32)
    out[iz[order], iy[order], ix[order]] = lut[voxel_comp]
    return out


def label_surface_pieces(
    mask: np.ndarray,
    z_native: np.ndarray,
    y_native: np.ndarray,
    x_native: np.ndarray,
    umbilicus,
    wedge_degrees: float = 40.0,
    min_voxels: int = 200,
    split_wraps_apart: bool | dict = False,
) -> np.ndarray:
    """Label each connected piece of each (sheet, wedge); ids run 1..N, largest first.

    ``mask`` is the binary surface prediction; ``z/y/x_native`` give the native-voxel coordinate of
    each prediction voxel along its own axis (1D arrays), used to find the scroll axis and angle.
    ``split_wraps_apart`` additionally cuts every piece into single sheets (see :func:`split_wraps`),
    for scans where touching wraps make the prediction one connected blob; pass a dict to override that
    function's tolerances (``{"link_tol": 20.0}``).
    """
    sheets, n_sheets = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n_sheets == 0:
        return np.zeros(mask.shape, dtype=np.int32)
    zz, yy, xx = np.nonzero(sheets)
    axis_y, axis_x = axis_at(umbilicus, z_native[zz])
    wedge = wedge_index(y_native[yy], x_native[xx], axis_y, axis_x, wedge_degrees)
    n_wedges = int(np.ceil(360.0 / wedge_degrees))
    key = np.zeros(mask.shape, dtype=np.int64)
    key[zz, yy, xx] = sheets[zz, yy, xx].astype(np.int64) * n_wedges + wedge + 1  # unique per (sheet, wedge)

    pieces = cc_label(key, background=0, connectivity=3)  # connected voxels sharing one key
    sizes = np.bincount(pieces.ravel())
    sizes[0] = 0
    keep = np.where(sizes >= min_voxels)[0]
    order = keep[np.argsort(-sizes[keep])]
    lut = np.zeros(len(sizes), dtype=np.int32)
    lut[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    labelled = lut[pieces]
    if split_wraps_apart:
        options = split_wraps_apart if isinstance(split_wraps_apart, dict) else {}
        return split_wraps(labelled, z_native, y_native, x_native, umbilicus, **{"min_voxels": min_voxels, **options})
    return labelled


def segment_from_surface_prediction(
    prediction_path: str | Path,
    umbilicus_path: str | Path,
    tile_origin_native: tuple[int, int, int],
    tile_shape_native: tuple[int, int, int],
    out_path: str | Path,
    threshold: int = 127,
    wedge_degrees: float = 40.0,
    min_voxels: int = 200,
    split_wraps_apart: bool | dict = False,
) -> dict:
    """Write an M2-style ``labels`` file for one tile of the scan from the surface prediction.

    ``tile_origin_native`` is the tile's corner in the scroll's native voxel coordinates, and must
    (like the tile shape) be a multiple of the prediction's ``scale_to_native``.
    """
    with h5py.File(prediction_path, "r") as f:
        ds = f["volume"]
        scale = int(ds.attrs["scale_to_native"])
        pred_origin = np.array([ds.attrs["z0"], ds.attrs["y0"], ds.attrs["x0"]], dtype=np.int64)
        origin = np.array(tile_origin_native, dtype=np.int64)
        shape = np.array(tile_shape_native, dtype=np.int64)
        if (origin % scale).any() or (shape % scale).any():
            raise ValueError(f"tile origin/shape must be multiples of the prediction scale ({scale})")
        lo = origin // scale - pred_origin
        hi = lo + shape // scale
        if (lo < 0).any() or (hi > ds.shape).any():
            raise ValueError("the tile lies outside the downloaded surface-prediction crop")
        mask = ds[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] > threshold

    coarse = np.arange(3)
    native_coords = [(origin[a] // scale + np.arange(mask.shape[a])) * scale + scale / 2 for a in coarse]  # voxel centres
    umbilicus = load_umbilicus(umbilicus_path)
    pieces = label_surface_pieces(mask, native_coords[0], native_coords[1], native_coords[2], umbilicus, wedge_degrees, min_voxels, split_wraps_apart)
    n = int(pieces.max())
    dtype = np.int16 if n < np.iinfo(np.int16).max else np.int32

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    depth = int(shape[0])
    with h5py.File(out_path, "w") as f:
        out = f.create_dataset(LABELS_DATASET, shape=tuple(int(s) for s in shape), dtype=dtype, chunks=(min(16, depth), int(shape[1]), int(shape[2])))
        out.attrs.update({"source": "surface prediction", "scale_to_native": scale, "wedge_degrees": wedge_degrees, "split_wraps": bool(split_wraps_apart), "origin_native_zyx": [int(v) for v in origin]})
        step = scale * 4  # native slices written per pass
        for z in range(0, depth, step):
            zc = slice(z // scale, (z + step) // scale)
            block = np.repeat(np.repeat(np.repeat(pieces[zc], scale, 0), scale, 1), scale, 2)
            out[z : z + block.shape[0]] = block.astype(dtype)

    counts = np.bincount(pieces.ravel(), minlength=n + 1)[1:] * scale**3
    return {"n_segments": n, "voxel_counts": {i + 1: int(c) for i, c in enumerate(counts)}, "surface_fraction": float(mask.mean())}
