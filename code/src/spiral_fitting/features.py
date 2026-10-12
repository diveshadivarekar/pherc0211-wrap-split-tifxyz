"""Feature extraction: sparse surface paths from a thresholded volume.

Implements the paper's key diagnosis and fix for M2's failure on real data
(see this package's __init__.py and segment.py's docstring): "naive use of
3D connected components yields CCs with many sheets combined, since their
segmentation masks touch. Instead, we extract 2D slices from the volume,
and find (2D) CCs within these slices only... this yields CCs that only
very rarely include multiple sheets."

Scoped down from the paper: only "surface paths" (from generic
density/ink threshold), not "fiber paths" (which need a separately-trained
fiber-orientation model this project doesn't have). Each surface path is
one connected, skeletonized curve within a single z-slice — i.e. the
cross-sectional outline of one winding, in one slice, traced out as a 3D
point sequence with an approximately-radial normal and a rough
winding-index estimate for the losses in losses.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, shortest_path
from skimage.morphology import skeletonize

MIN_PATH_PIXELS = 6


@dataclass
class SurfacePath:
    """One extracted winding-outline curve within a single z-slice.

    ``points`` is (N, 3) float32 in (z, y, x) volume-index order.
    ``normals`` is (N, 3) float32, the approximate radially-outward
    direction at each point (from the slice's fitted center) — a
    reasonable stand-in for the paper's Sobel-filter-derived normals given
    this project has no separate fiber/orientation prediction model.
    ``winding_index`` is a rough ordinal (0 = innermost path in this slice,
    1 = next, ...), used as a *relative* signal by the winding-number loss,
    not an absolute one.
    """

    points: np.ndarray
    normals: np.ndarray
    winding_index: int
    slice_z: int


def _longest_path_in_skeleton(skeleton_yx: np.ndarray) -> list[tuple[int, int]] | None:
    """Longest shortest-path through a 2D skeleton mask (the standard
    two-BFS "tree diameter" approach, which also works well enough on the
    rare small cycle a skeleton can have — worst case it just doesn't find
    the *global* longest path, still a valid long path).

    Runs on a scipy sparse pixel-adjacency graph. (A networkx graph built in
    pure Python took minutes per slice on a real 1971x1971 scroll slice; this
    is vectorised and orders of magnitude faster.)
    """
    ys, xs = np.nonzero(skeleton_yx)
    n = len(ys)
    if n < MIN_PATH_PIXELS:
        return None

    index = -np.ones(skeleton_yx.shape, dtype=np.int64)
    index[ys, xs] = np.arange(n)
    rows, cols = [], []
    h, w = skeleton_yx.shape
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):  # each undirected edge once
        ny, nx_ = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < h) & (nx_ >= 0) & (nx_ < w)
        nbr = np.full(n, -1, dtype=np.int64)
        nbr[ok] = index[ny[ok], nx_[ok]]
        hit = nbr >= 0
        rows.append(np.arange(n)[hit])
        cols.append(nbr[hit])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()
    graph = graph + graph.T

    def farthest(start: int) -> tuple[int, np.ndarray]:
        dist = shortest_path(graph, method="D", unweighted=True, indices=start)
        dist[~np.isfinite(dist)] = -1
        return int(np.argmax(dist)), dist

    a, _ = farthest(0)
    b, dist_a = farthest(a)
    if a == b or dist_a[b] + 1 < MIN_PATH_PIXELS:
        return None

    _, predecessors = breadth_first_order(graph, a, directed=False, return_predecessors=True)
    path, node = [], b
    while node != a and node >= 0:
        path.append(node)
        node = predecessors[node]
    path.append(a)
    path.reverse()
    return [(int(ys[i]), int(xs[i])) for i in path] if len(path) >= MIN_PATH_PIXELS else None


def _fit_slice_center(mask_yx: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask_yx)
    if len(ys) == 0:
        return None
    return float(ys.mean()), float(xs.mean())


@dataclass
class WindingPair:
    """Two 3D points, on the same normal ray, believed to be exactly one
    winding apart — the evidence :func:`losses.winding_number_loss` uses.

    Why this exists as its own structure rather than just ranking same-slice
    paths by radius (an earlier, simpler approach this replaced): a genuine
    spiral cross-section is topologically *one* connected curve per slice
    (that's the point of it being a spiral, not concentric rings) — so
    there is usually only one :class:`SurfacePath` per slice, and
    same-slice path-radius-ranking never has a second path to pair against.
    Casting a ray outward from a point on the path, along its normal, and
    recording the next foreground pixel hit works regardless of whether
    that next winding is a separate path or (as is typical for real spiral
    data) further along the very same path — matching the paper's own
    ray-casting approach in spirit, simplified to 2D-per-slice rather than
    a full 3D graph-voting scheme.
    """

    inner_point: np.ndarray  # (3,) (z, y, x)
    outer_point: np.ndarray  # (3,) (z, y, x)


def _block_mean(plane: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return plane
    h, w = (plane.shape[0] // factor) * factor, (plane.shape[1] // factor) * factor
    return plane[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def estimate_initial_omega(winding_pairs: list[WindingPair]) -> float | None:
    """Closed-form initial omega estimate from winding pairs — no gradient
    descent needed. For an Archimedean spiral (radius = theta/omega), the
    radial gap between two points one winding apart, *at the same angle*,
    is exactly ``2*pi/omega`` regardless of radius (each full revolution
    adds theta by 2*pi, so radius by exactly 2*pi/omega). Each
    :class:`WindingPair` is approximately that (a point and its
    radially-outward neighbour), so the median inter-pair distance gives a
    direct estimate — verified on a synthetic spiral phantom to land within
    ~1% of the true omega, on data an iterative fit alone took thousands of
    steps to approach only roughly.

    This is deliberately an *undeformed* estimate (no diffeomorphism
    involved yet) — good enough to seed :func:`fit.fit_spiral`'s joint
    optimisation much closer to the answer than a blind guess, which is
    what actually matters: the joint fit was overshooting/drifting slowly
    from a poor starting point, not failing outright.
    """
    if not winding_pairs:
        return None
    gaps = [float(np.linalg.norm(p.outer_point[1:] - p.inner_point[1:])) for p in winding_pairs]
    median_gap = float(np.median(gaps))
    if median_gap < 1e-6:
        return None
    return 2 * np.pi / median_gap


def estimate_winding_spacing(
    slice_yx: np.ndarray,
    center_yx: tuple[float, float],
    n_theta: int = 180,
    detrend_sigma: float = 8.0,
    min_lag: int = 3,
    min_prominence: float = 0.05,
) -> float | None:
    """Radial spacing between successive windings, in pixels of ``slice_yx``,
    from the phase-invariant radial autocorrelation: resample the slice onto
    (theta, r), high-pass each radial profile, autocorrelate along r, average
    over theta (which makes it independent of where each winding sits at a
    given angle), and take the first clear peak. Returns ``None`` when there
    is no clear periodicity — e.g. on a real scroll whose wraps are
    irregularly spaced — so the caller can fall back to another estimate
    instead of trusting a spurious one.
    """
    from scipy.signal import find_peaks

    cy, cx = center_yx
    h, w = slice_yx.shape
    max_r = int(min(cy, cx, h - 1 - cy, w - 1 - cx))
    if max_r < 4 * min_lag:
        return None
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    r = np.arange(max_r)
    yy = cy + r[None, :] * np.sin(theta)[:, None]
    xx = cx + r[None, :] * np.cos(theta)[:, None]
    polar = ndi.map_coordinates(slice_yx.astype(np.float32), [yy, xx], order=1)
    polar = polar - ndi.gaussian_filter1d(polar, detrend_sigma, axis=1)

    n = polar.shape[1]
    spectrum = np.fft.rfft(polar, n=2 * n, axis=1)
    ac = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, :n].mean(axis=0)
    if ac[0] <= 1e-12:
        return None
    ac = ac / ac[0]

    limit = max(min_lag + 2, n // 2)  # beyond half the radius the estimate is mostly padding
    peaks, props = find_peaks(ac[min_lag:limit], prominence=min_prominence)
    if len(peaks) == 0:
        return None
    return float(peaks[0] + min_lag)


def _cast_ray_to_next_winding(
    mask: np.ndarray,
    center_yx: tuple[float, float],
    point_yx: np.ndarray,
    dead_zone: float,
    max_search: float,
) -> np.ndarray | None:
    cy, cx = center_yx
    py, px = point_yx
    dy, dx = py - cy, px - cx
    dist = float(np.hypot(dy, dx))
    if dist < 1e-6:
        return None
    uy, ux = dy / dist, dx / dist

    h, w = mask.shape
    for step in np.arange(dead_zone, max_search, 1.0):
        y = int(round(py + uy * step))
        x = int(round(px + ux * step))
        if 0 <= y < h and 0 <= x < w and mask[y, x]:
            return np.array([y, x], dtype=np.float32)
    return None


def extract_surface_paths(
    volume: np.ndarray,
    threshold: float,
    z_stride: int = 1,
    min_component_pixels: int = MIN_PATH_PIXELS,
    winding_search_radius: float = 30.0,
    winding_dead_zone: float = 3.0,
    winding_samples_per_path: int = 12,
    downsample: int = 1,
    max_components_per_slice: int = 200,
) -> tuple[list[SurfacePath], list[WindingPair]]:
    """Extract sparse evidence for the spiral fit: :class:`SurfacePath`\\ s
    (via 2D connected components + skeletonization + longest-path — never
    3D connected components, which is exactly what merges touching real
    sheets, see module docstring) and :class:`WindingPair`\\ s (via radial
    ray-casting from sampled path points, see that class's docstring for
    why this replaced same-slice path ranking).

    ``z_stride`` > 1 skips slices for speed on large volumes; paths from
    the sampled slices are enough evidence for the global spiral fit,
    which doesn't need every single slice.

    ``downsample`` > 1 block-averages each slice before thresholding and
    skeletonizing (coordinates are scaled back, so everything downstream still
    sees full-resolution volume coordinates; ``winding_search_radius`` /
    ``winding_dead_zone`` / ``min_component_pixels`` are in downsampled
    pixels). ``max_components_per_slice`` keeps only the biggest components
    of a slice — a real scan's speckle otherwise yields hundreds of useless
    fragments per slice.
    """
    depth = volume.shape[0]
    paths: list[SurfacePath] = []
    winding_pairs: list[WindingPair] = []
    rng = np.random.default_rng(0)

    scale = float(downsample)
    for z in range(0, depth, z_stride):
        plane = _block_mean(volume[z], downsample)
        mask = plane > threshold
        if not mask.any():
            continue

        center = _fit_slice_center(mask)
        if center is None:
            continue
        cy, cx = center

        labels, n_components = ndi.label(mask, structure=np.ones((3, 3)))
        sizes = np.bincount(labels.ravel(), minlength=n_components + 1)
        boxes = ndi.find_objects(labels)
        comp_order = [c for c in np.argsort(-sizes[1:]) + 1 if sizes[c] >= min_component_pixels][:max_components_per_slice]
        for comp_id in comp_order:
            box = boxes[comp_id - 1]
            pad = tuple(slice(max(sl.start - 1, 0), sl.stop + 1) for sl in box)
            comp_mask = labels[pad] == comp_id
            skeleton = skeletonize(comp_mask)
            pixel_path = _longest_path_in_skeleton(skeleton)
            if pixel_path is None:
                continue

            pts_yx = np.array(pixel_path, dtype=np.float32) + np.array([pad[0].start, pad[1].start], dtype=np.float32)
            radii = np.sqrt((pts_yx[:, 0] - cy) ** 2 + (pts_yx[:, 1] - cx) ** 2)
            mean_radius = float(radii.mean())

            normals_yx = pts_yx - np.array([cy, cx], dtype=np.float32)
            norms = np.linalg.norm(normals_yx, axis=1, keepdims=True)
            normals_yx = np.divide(normals_yx, norms, out=np.zeros_like(normals_yx), where=norms > 1e-6)

            points_zyx = np.concatenate([np.full((len(pts_yx), 1), z, dtype=np.float32), pts_yx * scale], axis=1)
            normals_zyx = np.concatenate([np.zeros((len(pts_yx), 1), dtype=np.float32), normals_yx], axis=1)
            paths.append(SurfacePath(points=points_zyx, normals=normals_zyx, winding_index=0, slice_z=z))

            n_samples = min(winding_samples_per_path, len(pts_yx))
            sample_idx = rng.choice(len(pts_yx), size=n_samples, replace=False)
            for i in sample_idx:
                outer_yx = _cast_ray_to_next_winding(mask, (cy, cx), pts_yx[i], winding_dead_zone, winding_search_radius)
                if outer_yx is not None:
                    winding_pairs.append(
                        WindingPair(
                            inner_point=np.array([z, pts_yx[i, 0] * scale, pts_yx[i, 1] * scale], dtype=np.float32),
                            outer_point=np.array([z, outer_yx[0] * scale, outer_yx[1] * scale], dtype=np.float32),
                        )
                    )

        # Rank same-slice paths by radius too (still useful winding_index
        # metadata on the rarer occasions a slice does split into multiple
        # components — a break, or genuinely-separated windings).
        this_slice = [p for p in paths if p.slice_z == z]
        this_slice.sort(key=lambda p: float(np.linalg.norm(p.points[:, 1:] - np.array([cy, cx]), axis=-1).mean()))
        for idx, p in enumerate(this_slice):
            p.winding_index = idx

    return paths, winding_pairs
