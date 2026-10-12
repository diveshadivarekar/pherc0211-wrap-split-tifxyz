"""ASTRA-style synthetic CT phantom generator (Phase 1, R7 mitigation).

Produces a synthetic rolled-scroll volume with known ground-truth geometry
and ink placement, plus a corrupted "raw scan" version carrying the same
artifacts a real micro-CT reconstruction exhibits: beam-hardening cupping
and polar ring artifacts. This lets M1's correction routines (and later
modules) be validated against ground truth even without access to a real
scanner or the full ASTRA Toolbox.

Usage:
    python -m src.m1_ingestion.generate_phantom --config configs/phantom_default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml

GROUND_TRUTH_DATASET = "ground_truth"
RAW_DATASET = "volume"
LAYER_ID_DATASET = "layer_id"
INK_MASK_DATASET = "ink_mask"


def _rolled_scroll_ground_truth(
    shape: tuple[int, int, int],
    n_layers: int,
    substrate_density: float,
    ink_density: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a clean density volume: concentric rolled sheets with embedded ink.

    Axis 0 = depth along the scroll's roll axis, axes 1/2 = cross-section.
    Concentric rings model fused, rolled papyrus layers; short high-density
    arcs on a subset of rings model ink strokes.

    Returns ``(density, layer_id, ink_mask)``:

    - ``density``: the float density volume used to build the raw scan.
    - ``layer_id``: int16 volume, 0 = background, 1..n_layers = which sheet
      a voxel belongs to. Ground truth for M2 segmentation (Dice, no
      layer-jumping checks) — never available from a real scan.
    - ``ink_mask``: bool volume, True where ink was actually placed. Ground
      truth for M4 ink detection.
    """
    depth, h, w = shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    theta = np.arctan2(yy - cy, xx - cx)

    max_r = min(cy, cx) - 2
    layer_radii = np.linspace(max_r * 0.15, max_r, n_layers)
    # Kept strictly thinner than the layer spacing so distinct sheets stay
    # separated by real background gaps (a solid merged disk would defeat the
    # point of modelling *individual* fused layers) — but still several
    # voxels wide, wider than the sub-voxel ring artifacts injected later, so
    # radial-frequency-based correction can tell "real layer" from "ring
    # artifact" apart (see preprocessing.correct_ring_artifacts).
    spacing = (layer_radii[1] - layer_radii[0]) if n_layers > 1 else max_r
    layer_thickness = float(np.clip(spacing * 0.4, 1.5, spacing * 0.8))

    cross_section = np.zeros((h, w), dtype=np.float32)
    layer_id_2d = np.zeros((h, w), dtype=np.int16)
    for idx, radius in enumerate(layer_radii, start=1):
        band = np.abs(r - radius) <= layer_thickness / 2
        cross_section[band] = substrate_density
        layer_id_2d[band] = idx

    volume = np.repeat(cross_section[np.newaxis, :, :], depth, axis=0).copy()
    layer_id = np.repeat(layer_id_2d[np.newaxis, :, :], depth, axis=0).copy()
    ink_mask = np.zeros(shape, dtype=bool)

    # Scatter short ink arcs across a random subset of (layer, depth-slab) pairs.
    n_ink_strokes = max(4, n_layers)
    for _ in range(n_ink_strokes):
        radius = layer_radii[rng.integers(0, n_layers)]
        center_theta = rng.uniform(-np.pi, np.pi)
        arc_width = rng.uniform(0.15, 0.4)
        d0 = rng.integers(0, max(1, depth - depth // 6))
        d1 = min(depth, d0 + max(1, depth // 8))
        angular_mask = np.abs(np.angle(np.exp(1j * (theta - center_theta)))) <= arc_width
        radial_mask = np.abs(r - radius) <= layer_thickness / 2
        stroke = angular_mask & radial_mask
        volume[d0:d1, stroke] = ink_density
        ink_mask[d0:d1, stroke] = True

    return volume, layer_id, ink_mask


def _apply_beam_hardening(volume: np.ndarray, strength: float) -> np.ndarray:
    """Radial cupping artifact: apparent density falsely rises toward the centre,
    approximating the polynomial attenuation-hardening effect of Beer-Lambert
    absorption through increasing material thickness."""
    depth, h, w = volume.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / r.max()
    cupping = strength * (1.0 - r_norm) ** 2
    return volume + cupping[np.newaxis, :, :]


def _apply_ring_artifacts(
    volume: np.ndarray, strength: float, n_rings: int, rng: np.random.Generator
) -> np.ndarray:
    """Additive ring artifacts: detector miscalibration produces artifacts that
    are constant in angle and vary only with radius, in polar coordinates."""
    depth, h, w = volume.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    ring_radii = rng.uniform(0, r.max(), size=n_rings)
    ring_amps = rng.uniform(-strength, strength, size=n_rings)
    # Sub-voxel width: real sensor-miscalibration rings are much narrower in
    # radius than a manuscript layer band, which is what lets a radial
    # frequency filter separate the two (see preprocessing.correct_ring_artifacts).
    ring_widths = rng.uniform(0.3, 0.5, size=n_rings)

    pattern = np.zeros((h, w), dtype=np.float32)
    for radius, amp, width in zip(ring_radii, ring_amps, ring_widths):
        pattern += amp * np.exp(-((r - radius) ** 2) / (2 * width**2))

    return volume + pattern[np.newaxis, :, :]


def generate_phantom(cfg: dict, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_corrupted_volume, ground_truth_volume) as float32 arrays.

    Kept as the stable, minimal M1 interface (see tests/test_m1_ingestion.py).
    For the M2-M4 ground-truth labels (layer id, ink mask) use
    :func:`generate_phantom_full`.
    """
    raw, ground_truth, _, _ = generate_phantom_full(cfg, rng=rng)
    return raw, ground_truth


def generate_phantom_full(
    cfg: dict, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (raw, ground_truth, layer_id, ink_mask) as arrays.

    ``layer_id`` (int16) and ``ink_mask`` (bool) are the ground truth later
    modules validate against and are never available for a real scan.
    """
    rng = rng or np.random.default_rng(cfg.get("seed", 0))
    shape = tuple(cfg["shape"])

    ground_truth, layer_id, ink_mask = _rolled_scroll_ground_truth(
        shape=shape,
        n_layers=cfg.get("n_layers", 12),
        substrate_density=cfg.get("substrate_density", 0.3),
        ink_density=cfg.get("ink_density", 0.9),
        rng=rng,
    )

    raw = ground_truth.copy()
    raw = _apply_beam_hardening(raw, strength=cfg.get("beam_hardening_strength", 0.25))
    raw = _apply_ring_artifacts(
        raw,
        strength=cfg.get("ring_artifact_strength", 0.08),
        n_rings=cfg.get("n_rings", 6),
        rng=rng,
    )
    noise_sigma = cfg.get("noise_sigma", 0.02)
    if noise_sigma > 0:
        raw = raw + rng.normal(0, noise_sigma, size=raw.shape).astype(np.float32)

    return raw.astype(np.float32), ground_truth.astype(np.float32), layer_id, ink_mask


def save_phantom(
    path: str | Path,
    raw: np.ndarray,
    ground_truth: np.ndarray,
    cfg: dict,
    layer_id: np.ndarray | None = None,
    ink_mask: np.ndarray | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk_depth = min(cfg.get("chunk_depth", 32), raw.shape[0])
    chunks = (chunk_depth, raw.shape[1], raw.shape[2])
    with h5py.File(path, "w") as f:
        ds = f.create_dataset(RAW_DATASET, data=raw, chunks=chunks)
        f.create_dataset(GROUND_TRUTH_DATASET, data=ground_truth, chunks=chunks)
        if layer_id is not None:
            f.create_dataset(LAYER_ID_DATASET, data=layer_id, chunks=chunks)
        if ink_mask is not None:
            f.create_dataset(INK_MASK_DATASET, data=ink_mask, chunks=chunks)
        for k, v in cfg.items():
            ds.attrs[k] = v


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to phantom config YAML")
    parser.add_argument("--out", default=None, help="Override output path from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_path = args.out or cfg["output_path"]

    raw, ground_truth, layer_id, ink_mask = generate_phantom_full(cfg)
    save_phantom(out_path, raw, ground_truth, cfg, layer_id=layer_id, ink_mask=ink_mask)
    print(f"Phantom written to {out_path}  shape={raw.shape}  dtype={raw.dtype}")


if __name__ == "__main__":
    main()
