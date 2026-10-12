"""Spiral-fitting pipeline entrypoint: a global-model alternative to M2+M3.

Operates directly on the M1-corrected volume — it does *not* consume M2's
segmentation output, since the whole point (see this package's __init__.py)
is to replace per-layer classical segmentation with one global fit. Output
uses the same page/coord_map HDF5 convention as M3's flatten.py, so M4/M5
can consume it unchanged.

Usage:
    python -m src.spiral_fitting.run --input data/processed/phantom_001_corrected.h5 \
        --out data/processed/ --initial-omega 0.3 --n-windings 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.m1_ingestion.io_utils import VolumeStore, base_stem
from src.m2_segmentation.segment import estimate_foreground_threshold
from src.spiral_fitting.features import (
    _block_mean,
    estimate_initial_omega,
    estimate_winding_spacing,
    extract_surface_paths,
)
from src.spiral_fitting.fit import FitConfig, fit_spiral
from src.spiral_fitting.mesh import extract_flattened_page


TARGET_SLICE_SIDE = 512  # feature extraction works on slices block-averaged to about this many pixels per side
TARGET_N_SLICES = 20
MIN_VALID_FRACTION = 0.05
MIN_LOSS_IMPROVEMENT = 0.1
MIN_ON_SHEET_FRACTION = 0.2


def _slice_center_and_spacing(volume: np.ndarray, threshold: float, downsample: int) -> tuple[tuple[float, float], float | None]:
    """Axis position and winding spacing (both in downsampled pixels) from the middle slice."""
    plane = _block_mean(volume[volume.shape[0] // 2], downsample)
    ys, xs = np.nonzero(plane > threshold)
    center = (float(ys.mean()), float(xs.mean()))
    return center, estimate_winding_spacing(plane, center)


def _page_quality(page: np.ndarray, coord_map: np.ndarray, threshold: float) -> dict:
    valid = ~np.isnan(coord_map[..., 0])
    n_valid = int(valid.sum())
    return {
        "valid_fraction": float(valid.mean()),
        "page_max": float(page.max()),
        "on_sheet_fraction": float((page[valid] > threshold).mean()) if n_valid else 0.0,
    }


def run(
    input_path: str | Path,
    out_dir: str | Path,
    initial_omega: float | None = None,
    n_windings: int = 4,
    threshold: float | None = None,
    z_stride: int | None = None,
    n_theta: int = 256,
    config: FitConfig | None = None,
    downsample: int | None = None,
) -> Path:
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = config or FitConfig()

    store = VolumeStore(input_path, mode="r")
    volume = store.read_all()
    store.close()

    if threshold is None:
        threshold = estimate_foreground_threshold(volume)
    print(f"Threshold: {threshold:.4f}", flush=True)

    # Real-size slices are far too large to skeletonize at full resolution (and a real scan's
    # fine detail isn't what a global spiral fit needs), so work on block-averaged slices
    # and a sample of slices; the phantom-sized inputs stay at full resolution.
    if downsample is None:
        downsample = max(1, round(max(volume.shape[1:]) / TARGET_SLICE_SIDE))
    if z_stride is None:
        z_stride = max(1, volume.shape[0] // TARGET_N_SLICES) if downsample > 1 else 2
    print(f"Feature extraction: downsample x{downsample}, every {z_stride}th slice", flush=True)

    _, spacing_px = _slice_center_and_spacing(volume, threshold, downsample)
    extract_kwargs = {}
    if spacing_px is not None:
        # look for the next winding where one is expected rather than at a fixed, arbitrary distance
        extract_kwargs = {"winding_dead_zone": 0.5 * spacing_px, "winding_search_radius": 1.6 * spacing_px}
        print(f"Radial winding spacing: {spacing_px:.1f} px (downsampled) = {spacing_px * downsample:.1f} px", flush=True)
        if initial_omega is None:
            initial_omega = 2 * np.pi / (spacing_px * downsample)
            print(f"Initial omega from radial spacing: {initial_omega:.4f}", flush=True)
    else:
        print("No clear radial periodicity (irregular winding spacing) - falling back to winding-pair estimate.", flush=True)

    print("Extracting surface paths (per-slice connected components)...", flush=True)
    paths, winding_pairs = extract_surface_paths(
        volume,
        threshold,
        z_stride=z_stride,
        downsample=downsample,
        min_component_pixels=max(6, 40 // downsample),
        **extract_kwargs,
    )
    print(f"  {len(paths)} surface paths, {len(winding_pairs)} winding pairs", flush=True)

    if initial_omega is None:
        initial_omega = estimate_initial_omega(winding_pairs)
        if initial_omega is None:
            raise ValueError("Couldn't auto-estimate an initial omega (no winding pairs found) — pass --initial-omega explicitly.")
        print(f"Auto-estimated initial omega: {initial_omega:.4f}", flush=True)

    print(f"Fitting spiral + diffeomorphism ({config.n_iterations} iterations)...", flush=True)
    model, history = fit_spiral(paths, winding_pairs, volume.shape, initial_omega, config=config)
    print(f"Final omega: {model.omega.item():.4f}  (started at {initial_omega:.4f})", flush=True)

    stem = base_stem(input_path)
    out_path = out_dir / f"{stem}_spiral_flattened.h5"
    windings_quality = []
    with h5py.File(out_path, "w") as f:
        for winding in range(n_windings):
            page, coord_map = extract_flattened_page(model, volume, winding, n_theta=n_theta)
            quality = {"winding": winding, **_page_quality(page, coord_map, threshold)}
            windings_quality.append(quality)
            grp = f.create_group(f"layer_{winding + 1}")
            grp.create_dataset("page", data=page)
            grp.create_dataset("coord_map", data=coord_map)
            print(
                f"  winding {winding}: page {page.shape}, valid {quality['valid_fraction']:.0%}, "
                f"on-sheet {quality['on_sheet_fraction']:.0%}, max={quality['page_max']:.4f}",
                flush=True,
            )

    losses = np.asarray(history["loss"])
    k = max(1, len(losses) // 10)
    loss_improvement = float(1 - losses[-k:].mean() / losses[:k].mean()) if losses[:k].mean() > 0 else 0.0
    best_on_sheet = max(q["on_sheet_fraction"] for q in windings_quality)
    warnings = []
    if loss_improvement < MIN_LOSS_IMPROVEMENT:
        warnings.append(f"fit loss improved only {loss_improvement:.0%} - the spiral model did not settle on this scan")
    if best_on_sheet < MIN_ON_SHEET_FRACTION:
        warnings.append(f"best flattened page has only {best_on_sheet:.0%} of pixels on sheet material - the fitted surface does not follow the papyrus")

    summary = {
        "initial_omega": float(initial_omega),
        "final_omega": model.omega.item(),
        "handedness": float(model.diffeo.canonical_sign[1].item()),
        "n_surface_paths": len(paths),
        "n_winding_pairs": len(winding_pairs),
        "n_iterations": config.n_iterations,
        "final_loss": history["loss"][-1],
        "loss_improvement": loss_improvement,
        "downsample": downsample,
        "winding_spacing_px": spacing_px * downsample if spacing_px is not None else None,
        "windings": windings_quality,
        "fit_ok": not warnings,
        "warnings": warnings,
    }
    summary_path = out_dir / f"{stem}_spiral_fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    for w in warnings:
        print(f"WARNING: {w}", flush=True)

    # A fit that produces nothing but blank pages is a failure, not a result: say so loudly
    # instead of letting downstream stages (ink detection, OCR) run on empty pages.
    if all(q["valid_fraction"] < MIN_VALID_FRACTION or q["page_max"] <= 1e-6 for q in windings_quality):
        raise ValueError(
            "The spiral fit produced only blank pages (the fitted surface lies outside the volume or on empty space). "
            "Check the foreground threshold and --initial-omega."
        )

    print(f"Flattened output -> {out_path}", flush=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Corrected volume HDF5 (M1 output)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--initial-omega", type=float, default=None, help="Starting guess for winding tightness (auto-estimated from winding pairs if omitted)")
    parser.add_argument("--n-windings", type=int, default=4, help="Number of windings to extract flattened pages for")
    parser.add_argument("--threshold", type=float, default=None, help="Foreground threshold (auto-estimated via Otsu if omitted)")
    parser.add_argument("--z-stride", type=int, default=None, help="Slice sampling stride for feature extraction (auto if omitted)")
    parser.add_argument("--downsample", type=int, default=None, help="Block-average factor for feature extraction (auto if omitted)")
    parser.add_argument("--n-theta", type=int, default=256, help="Angular resolution of each flattened page")
    parser.add_argument("--n-iterations", type=int, default=None, help="Override FitConfig's default iteration count")
    args = parser.parse_args()

    config = FitConfig()
    if args.n_iterations is not None:
        config.n_iterations = args.n_iterations

    run(
        args.input,
        args.out,
        initial_omega=args.initial_omega,
        n_windings=args.n_windings,
        threshold=args.threshold,
        z_stride=args.z_stride,
        downsample=args.downsample,
        n_theta=args.n_theta,
        config=config,
    )


if __name__ == "__main__":
    main()
