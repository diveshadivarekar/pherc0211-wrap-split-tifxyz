"""Flag candidate ink regions on a page's ink-probability map, with their 3D location in the scan.

These are *candidates for a human to look at*, not detections: the ink model was trained on
fragments and has not been validated on the scroll. A region qualifies if it is big enough to be
more than speckle, confidently scored, and away from the page border (where the sampling and the
model's tile edges are least reliable). Each candidate carries its position in the scan so it can be
opened in a viewer and checked by eye.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def find_candidates(
    prob: np.ndarray,
    coord_map: np.ndarray,
    threshold: float = 0.5,
    min_pixels: int = 300,
    min_mean: float = 0.6,
    edge_margin: int = 25,
    max_per_page: int = 10,
) -> list[dict]:
    """Candidate regions on one page, best first.

    ``coord_map`` (H, W, 3) is the page's (z, y, x) scan coordinate per pixel (NaN outside the page).
    """
    valid = ~np.isnan(coord_map[..., 0])
    if not valid.any():
        return []
    interior = ndi.distance_transform_edt(np.pad(valid, 1))[1:-1, 1:-1] > edge_margin  # the array border counts as a page edge too
    hot = prob > threshold
    labels, n = ndi.label(hot & interior, structure=np.ones((3, 3)))
    full_labels, _ = ndi.label(hot, structure=np.ones((3, 3)))
    full_sizes = np.bincount(full_labels.ravel())
    out = []
    for i, sl in enumerate(ndi.find_objects(labels), start=1):
        region = labels[sl] == i
        area = int(region.sum())
        if area < min_pixels:
            continue
        ys0, xs0 = np.nonzero(region)
        whole = full_sizes[full_labels[ys0[0] + sl[0].start, xs0[0] + sl[1].start]]
        if area < 0.7 * whole:  # most of this blob lies in the unreliable border zone: edge artefact, not a candidate
            continue
        p = prob[sl][region]
        if p.mean() < min_mean:
            continue
        ys, xs = np.nonzero(region)
        ys, xs = ys + sl[0].start, xs + sl[1].start
        # elongation from the second moments: strokes are long and thin, blobs are round
        cov = np.cov(np.stack([ys, xs]).astype(np.float64)) if area > 2 else np.eye(2)
        eig = np.sort(np.linalg.eigvalsh(cov))
        elongation = float(np.sqrt(eig[1] / max(eig[0], 1e-9)))
        centre = np.nanmean(coord_map[ys, xs], axis=0)
        out.append(
            {
                "bbox": [int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)],  # x0, y0, x1, y1 on the page
                "area_px": area,
                "mean_prob": float(p.mean()),
                "max_prob": float(p.max()),
                "elongation": elongation,
                "solidity": float(area / ((sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start))),
                "centre_page_yx": [float(ys.mean()), float(xs.mean())],
                "centre_zyx": [float(c) for c in centre],
                "score": float(p.mean() * np.log(area)),
            }
        )
    out.sort(key=lambda c: -c["score"])
    return out[:max_per_page]


def to_scroll_coordinates(candidate: dict, tile_offset_zyx: tuple[int, int, int], voxel_size_um: float = 9.362) -> dict:
    """Add the candidate's position in the full scan's native voxel coordinates (tile offset + local centre)."""
    c = candidate["centre_zyx"]
    native = [float(c[a] + tile_offset_zyx[a]) for a in range(3)]
    return {**candidate, "centre_native_zyx": native, "centre_native_um": [v * voxel_size_um for v in native]}
