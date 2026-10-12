"""M1 artifact correction: beam-hardening (cupping) and ring-artifact removal.

Implements FR-1.2 / FR-1.3 from the SRS. Each correction operates on one
in-memory chunk at a time (see :mod:`io_utils`), so the full pipeline in
:mod:`run` never needs the whole volume resident in memory (NFR-2).
"""

from __future__ import annotations

import numpy as np


def correct_beam_hardening(
    chunk: np.ndarray, degree: int = 2, n_bins: int = 64, opening_size: int = 9
) -> np.ndarray:
    """Iterative polynomial cupping correction.

    Fits a low-degree polynomial in normalised radius (from the slice centre)
    to the radial *baseline* intensity profile, then subtracts the fitted
    curve's deviation from the profile's outer-edge value. This flattens the
    radial "cupping" bias introduced by beam-hardening.

    Real page/layer structure only ever adds density on top of the
    substrate — it never subtracts — so a plain radial mean is biased
    upward wherever a layer happens to sit and cannot be fit directly. A
    greyscale morphological opening (erosion then dilation) of the radial
    mean profile strips bumps narrower than ``opening_size`` bins (i.e. any
    single layer) while preserving the true smooth cupping floor that
    every radius shares, giving the polynomial fit a clean baseline to
    match instead of the raw, structure-contaminated profile.

    Operates per depth-slice so it works unmodified on a partial chunk.
    """
    from scipy.ndimage import grey_opening

    corrected = np.empty_like(chunk)
    depth, h, w = chunk.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = (r / r.max()).ravel()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(r_norm, bin_edges) - 1, 0, n_bins - 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    counts = np.bincount(bin_idx, minlength=n_bins)
    populated = counts > 0
    centers_p = bin_centers[populated]

    for i in range(depth):
        sl = chunk[i]
        flat = sl.ravel().astype(np.float64)

        sums = np.bincount(bin_idx, weights=flat, minlength=n_bins)
        radial_mean = np.divide(sums, counts, out=np.zeros(n_bins), where=populated)
        values_p = radial_mean[populated]
        baseline_profile = grey_opening(values_p, size=min(opening_size, len(values_p)))

        coeffs = np.polyfit(centers_p, baseline_profile, deg=degree)
        fitted_profile = np.polyval(coeffs, r_norm)
        baseline = np.polyval(coeffs, 1.0)  # value at the outer edge, treated as artifact-free
        bias = (fitted_profile - baseline).reshape(h, w)

        corrected[i] = sl - bias

    return corrected.astype(chunk.dtype)


def correct_ring_artifacts(chunk: np.ndarray, median_size: int = 3) -> np.ndarray:
    """Polar-coordinate ring-artifact suppression.

    Rings are additive, angle-independent artifacts that are narrow in
    radius, whereas real page/layer structure varies more smoothly with
    radius. So: compute the mean intensity at each integer radius (the
    angle-independent radial profile), median-filter it to get the smooth
    "structure" trend, and treat the residual (profile minus its smoothed
    version) as the ring artifact — narrow radial spikes survive a median
    filter's high-frequency rejection, broad structure does not.
    """
    from scipy.ndimage import median_filter

    corrected = np.empty_like(chunk)
    depth, h, w = chunk.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    max_r = int(np.ceil(r.max())) + 1
    r_idx = np.clip(np.round(r).astype(int), 0, max_r - 1).ravel()
    counts = np.bincount(r_idx, minlength=max_r)
    populated = counts > 0

    for i in range(depth):
        sl = chunk[i]
        flat = sl.ravel().astype(np.float64)

        sums = np.bincount(r_idx, weights=flat, minlength=max_r)
        radial_profile = np.divide(sums, counts, out=np.zeros(max_r), where=populated)
        radial_profile_smoothed = median_filter(radial_profile, size=median_size, mode="nearest")
        ring_bias_per_radius = radial_profile - radial_profile_smoothed
        ring_bias_per_radius[~populated] = 0.0

        bias_map = ring_bias_per_radius[r_idx].reshape(h, w)
        corrected[i] = sl - bias_map

    return corrected.astype(chunk.dtype)


def correct_chunk(chunk: np.ndarray, degree: int = 2, median_size: int = 3) -> np.ndarray:
    """Run the full M1 correction pipeline (beam-hardening then ring removal) on one chunk."""
    chunk = correct_beam_hardening(chunk, degree=degree)
    chunk = correct_ring_artifacts(chunk, median_size=median_size)
    return chunk
