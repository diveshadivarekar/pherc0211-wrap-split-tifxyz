"""Canonical (pre-deformation) Archimedean spiral, per the paper's Eq. 1.

The idealized, undistorted scroll: a rectangular sheet rolled into a
cylinder, represented in cross-section as an Archimedean spiral (radius
proportional to angle) extruded along z. One free parameter, ``omega``,
controls how tightly it winds.
"""

from __future__ import annotations

import torch


def spiral_point(theta: torch.Tensor, z: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """s(theta, z; omega) -> (x, y, z), per Eq. 1. theta > 0 required (the
    spiral starts at the origin and winds outward)."""
    r = theta / omega
    x = r * torch.cos(theta)
    y = -r * torch.sin(theta)
    return torch.stack([x, y, z], dim=-1)


def point_to_winding_radius(xy: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """A "spiral-adjusted radius" that is constant for points on the same
    winding, and jumps by exactly one winding-spacing between adjacent
    windings — the building block for the radius-consistency and
    distance-to-nearest-winding losses (paper Eqs. 4/6).

    Derivation (self-consistent with this module's spiral convention, Eq. 1
    using radius = theta/omega — noted because the paper's Eq. 4 as
    transcribed multiplies by omega rather than dividing, which isn't
    dimensionally consistent with its own Eq. 1; this is the corrected
    form): for a point exactly on the spiral at some *accumulated* angle
    theta = theta_principal + 2*pi*k (k = winding index, theta_principal in
    (-pi, pi] the angle you actually measure from x/y), its true radius is
    theta/omega = theta_principal/omega + 2*pi*k/omega. Subtracting
    theta_principal/omega from the measured radius therefore leaves
    2*pi*k/omega — a value that's ~0 for winding 0, and changes by exactly
    the winding spacing (2*pi/omega) for each winding further out.
    """
    rho = torch.linalg.norm(xy, dim=-1)
    theta_principal = torch.arctan2(-xy[..., 1], xy[..., 0])
    return rho - theta_principal / omega


def winding_spacing(omega: torch.Tensor) -> torch.Tensor:
    return 2 * torch.pi / omega


def nearest_winding_offset(r: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """How far ``r`` (from :func:`point_to_winding_radius`) sits from the
    *nearest* multiple of the winding spacing (0, +-spacing, +-2*spacing,
    ...), wrapping correctly at the boundary between two windings. Used by
    the distance loss to snap a path onto its nearest winding.
    """
    spacing = winding_spacing(omega)
    m = torch.remainder(r, spacing)
    return torch.where(m < spacing / 2, m, m - spacing)
