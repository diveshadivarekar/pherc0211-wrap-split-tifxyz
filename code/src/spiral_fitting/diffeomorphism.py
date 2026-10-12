"""Flow-field diffeomorphism: T_flow from the paper's Sec. 5.2/Appendix A.2.

Scoped-down from the paper's full transform (T_affine ∘ T_flow ∘ T_gap) to
just this one stage — the paper's own ablation (Table 2) shows the
flow-field alone already gets close to the full model (slightly *better*
MRWD/ChD, somewhat worse WJF), and it's "in principle... sufficiently
expressive without the other stages" per the paper's own text. Adding the
per-slice affine and gap-scaling stages back is future work, not a
correctness gap in what's here.

A diffeomorphism (smooth, invertible, topology-preserving map) is
constructed as the flow induced by integrating a smooth velocity field: a
point's transformed position is where it ends up after "flowing" through
the velocity field for one unit of time. Reversing that flow (negating the
velocity field) gives the exact inverse — which is why a coarse velocity
field is a convenient way to *parameterize* the space of diffeomorphisms,
rather than fitting an arbitrary invertible function directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowFieldDiffeomorphism(nn.Module):
    """A diffeomorphism parameterized by a coarse, trilinearly-interpolated
    3D velocity field over a bounding box, integrated via explicit Euler
    steps (the paper found this more stable than scaling-and-squaring for
    this problem).

    ``grid_shape`` is (Dz, Dy, Dx) — deliberately coarse (a handful of
    points per axis is enough to capture smooth, large-scale distortion;
    this is *not* meant to model per-voxel detail, matching the paper's
    own "fine" flow grid being 48x lower resolution than the raw scan).
    """

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        bounds_min: tuple[float, float, float],
        bounds_max: tuple[float, float, float],
        n_euler_steps: int = 16,
        canonical_origin: tuple[float, float, float] | None = None,
        mirror: bool = False,
    ):
        super().__init__()
        self.grid_shape = grid_shape
        self.n_euler_steps = n_euler_steps
        self.register_buffer("bounds_min", torch.tensor(bounds_min, dtype=torch.float32))
        self.register_buffer("bounds_max", torch.tensor(bounds_max, dtype=torch.float32))
        # The canonical spiral is centred on the origin of its own (x, y) plane, but a scroll's
        # axis sits somewhere inside the scan. ``canonical_origin`` is where that axis is in scan
        # coordinates (default: none, i.e. a plain identity-at-init flow); without it every
        # canonical point lands in the volume's corner and sampled pages come out blank.
        # ``mirror`` flips the canonical y axis for a scroll wound the other way round.
        if canonical_origin is None:
            canonical_origin = (0.0, 0.0, 0.0)
        self.register_buffer("canonical_origin", torch.tensor(canonical_origin, dtype=torch.float32))
        self.register_buffer("canonical_sign", torch.tensor([1.0, -1.0 if mirror else 1.0, 1.0]))

        # Velocity grid: (1, 3, Dz, Dy, Dx) — channel dim holds (vx, vy, vz),
        # initialised near zero so the diffeomorphism starts as ~identity.
        self.velocity = nn.Parameter(torch.randn(1, 3, *grid_shape) * 1e-3)

    def _sample_velocity(self, points: torch.Tensor) -> torch.Tensor:
        """points: (N, 3) in (x, y, z) world coordinates. Returns (N, 3)
        velocity vectors via trilinear interpolation of the grid."""
        normalized = 2 * (points - self.bounds_min) / (self.bounds_max - self.bounds_min) - 1  # -> [-1, 1]
        # grid_sample expects normalized coords in (x, y, z) order matching
        # the *last* 3 dims of the 5D input (Dz, Dy, Dx) reversed, i.e. (x,y,z).
        grid = normalized.view(1, 1, 1, -1, 3)
        sampled = F.grid_sample(self.velocity, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.view(3, -1).transpose(0, 1)  # (N, 3)

    def _integrate(self, points: torch.Tensor, velocity_sign: float) -> torch.Tensor:
        dt = 1.0 / self.n_euler_steps
        x = points
        for _ in range(self.n_euler_steps):
            v = self._sample_velocity(x)
            x = x + velocity_sign * dt * v
        return x

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Canonical space -> scan space (T_S->V): integrate forward."""
        return self._integrate(points * self.canonical_sign + self.canonical_origin, velocity_sign=1.0)

    def inverse(self, points: torch.Tensor) -> torch.Tensor:
        """Scan space -> canonical space (T_V->S): integrate the *negated*
        field, which is the exact inverse of a flow-induced diffeomorphism."""
        return (self._integrate(points, velocity_sign=-1.0) - self.canonical_origin) * self.canonical_sign
