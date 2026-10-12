"""Extract a flattened page from a fitted spiral model — the paper's Sec. 5.4.

Because the canonical spiral's own (theta, z) coordinates already form a
valid 2D parameterisation (unlike M3's per-layer independent unwrap, no
separate ARAP/flattening step is needed here — the diffeomorphism fit
*is* the flattening). Output uses the same ``page`` + ``coord_map``
convention as M3's flatten.py, so M4/M5 can consume either's output
unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import map_coordinates

from src.spiral_fitting.canonical import spiral_point
from src.spiral_fitting.fit import SpiralFitModel


def extract_flattened_page(
    model: SpiralFitModel,
    volume: np.ndarray,
    n_winding: int,
    n_theta: int = 256,
    z_samples: int | None = None,
    device: str | None = None,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``volume`` along the ``n_winding``-th turn of the fitted
    spiral. Returns ``(page, coord_map)`` matching M3's flatten.py
    convention: ``page`` is (z_samples, n_theta) float32; ``coord_map`` is
    (z_samples, n_theta, 3) float32 giving each page pixel's source (z, y,
    x) volume coordinate (FR-3.4 traceability), NaN where out of bounds.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    depth = volume.shape[0]
    z_samples = z_samples or depth

    theta = torch.linspace(
        n_winding * 2 * torch.pi + 1e-3, (n_winding + 1) * 2 * torch.pi, n_theta, device=device
    )
    z = torch.linspace(0, depth - 1, z_samples, device=device)
    theta_grid, z_grid = torch.meshgrid(theta, z, indexing="ij")  # (n_theta, z_samples)
    theta_flat = theta_grid.reshape(-1)
    z_flat = z_grid.reshape(-1)

    with torch.no_grad():
        canonical_xyz = spiral_point(theta_flat, z_flat, model.omega)  # (N, 3), (x, y, z)

        scan_xyz = torch.empty_like(canonical_xyz)
        for start in range(0, len(canonical_xyz), batch_size):
            end = start + batch_size
            scan_xyz[start:end] = model.diffeo(canonical_xyz[start:end])

    scan_xyz_np = scan_xyz.cpu().numpy()
    # (x, y, z) -> (z, y, x) volume-index order.
    coords_zyx = scan_xyz_np[:, [2, 1, 0]].reshape(n_theta, z_samples, 3).transpose(1, 0, 2)

    in_bounds = (
        (coords_zyx[..., 0] >= 0) & (coords_zyx[..., 0] < volume.shape[0])
        & (coords_zyx[..., 1] >= 0) & (coords_zyx[..., 1] < volume.shape[1])
        & (coords_zyx[..., 2] >= 0) & (coords_zyx[..., 2] < volume.shape[2])
    )

    sample_coords = coords_zyx.reshape(-1, 3).T  # (3, N)
    page_flat = map_coordinates(volume, sample_coords, order=1, mode="constant", cval=0.0)
    page = page_flat.reshape(z_samples, n_theta).astype(np.float32)

    coord_map = coords_zyx.astype(np.float32)
    coord_map[~in_bounds] = np.nan
    page[~in_bounds] = 0.0

    return page, coord_map
