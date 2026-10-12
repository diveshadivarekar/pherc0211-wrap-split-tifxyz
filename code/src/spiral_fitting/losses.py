"""Losses fitting the spiral+diffeomorphism model to extracted surface paths.

Implements the core losses from the paper's Sec. 5.3 that don't depend on
fiber predictions (this project has none): L_normal, L_radius, L_windings,
L_distance (phased in after an initial "find the overall structure" phase,
exactly as the paper does), plus L_stretch and L_center regularizers.
Fiber-orientation losses (L_horizontal/L_vertical) are omitted — the
paper's own ablation shows removing them has "minimal effect on metrics"
besides some in-plane distortion of the flattened result, which is an
acceptable trade for not needing a fiber-prediction model.
"""

from __future__ import annotations

import torch

from src.spiral_fitting.canonical import nearest_winding_offset, point_to_winding_radius, winding_spacing
from src.spiral_fitting.diffeomorphism import FlowFieldDiffeomorphism


def normal_loss(
    diffeo: FlowFieldDiffeomorphism, points: torch.Tensor, normals: torch.Tensor, omega: torch.Tensor, eps: float = 1.0
) -> torch.Tensor:
    """Encourages the fitted surface's local radial direction (in
    canonical space) to match the extracted normal at each point, after
    mapping both into canonical space — Eq. 3.
    """
    canonical = diffeo.inverse(points)
    canonical_plus_n = diffeo.inverse(points + eps * normals)
    canonical_radial_dir = canonical_plus_n - canonical

    xy = canonical[:, :2]
    radial_unit = xy / (torch.linalg.norm(xy, dim=-1, keepdim=True) + 1e-8)
    radial_unit_3d = torch.cat([radial_unit, torch.zeros_like(radial_unit[:, :1])], dim=-1)

    cos_sim = torch.nn.functional.cosine_similarity(radial_unit_3d, canonical_radial_dir, dim=-1)
    return (1 - cos_sim).mean()


def radius_consistency_loss(diffeo: FlowFieldDiffeomorphism, path_points: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Points on the same extracted path should map to (near-)constant
    spiral-adjusted radius in canonical space — Eq. 4. ``path_points`` is
    (N, 3), all points from one path.
    """
    canonical = diffeo.inverse(path_points)
    r = point_to_winding_radius(canonical[:, :2], omega)
    return (r - r.mean()).abs().mean()


def winding_number_loss(
    diffeo: FlowFieldDiffeomorphism,
    inner_points: torch.Tensor,
    outer_points: torch.Tensor,
    expected_winding_gap: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    """Each (inner, outer) point pair (matched 1:1 by row, from
    :func:`features.extract_surface_paths`'s radial ray-casting — see
    ``WindingPair``) is believed to be ``expected_winding_gap`` windings
    apart; their canonical-space spiral-adjusted radii should differ by
    that many winding-spacings. Computed per-pair then averaged (not
    averaged-then-differenced), so batching many independent pairs together
    doesn't conflate one pair's inner radius with another's outer radius.
    """
    r_inner = point_to_winding_radius(diffeo.inverse(inner_points)[:, :2], omega)
    r_outer = point_to_winding_radius(diffeo.inverse(outer_points)[:, :2], omega)
    expected_gap = expected_winding_gap * winding_spacing(omega)
    return ((r_outer - r_inner) - expected_gap).abs().mean()


def distance_loss(diffeo: FlowFieldDiffeomorphism, path_points: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Snaps each path exactly onto its nearest spiral winding — Eq. 5.
    Only enabled in the second half of optimisation (see fit.py), after
    the overall structure is found, matching the paper's phasing."""
    canonical = diffeo.inverse(path_points)
    r = point_to_winding_radius(canonical[:, :2], omega)
    r_star = r - nearest_winding_offset(r, omega)
    return (r - r_star).abs().mean()


def stretch_loss(diffeo: FlowFieldDiffeomorphism, points: torch.Tensor, tangents: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Discourages the deformation from stretching the surface tangentially
    (it should only bend, matching a real papyrus sheet's near-inextensibility)."""
    a = diffeo(points)
    b = diffeo(points + delta * tangents)
    return (torch.linalg.norm(b - a, dim=-1) - delta).abs().mean()


def center_loss(diffeo: FlowFieldDiffeomorphism, center_point: torch.Tensor) -> torch.Tensor:
    """Keeps the scroll's centerline anchored near its original position,
    preventing the whole fit from drifting."""
    return torch.linalg.norm(diffeo(center_point.unsqueeze(0))[0] - center_point)
