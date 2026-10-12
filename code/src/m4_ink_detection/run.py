"""M4 pipeline entrypoint: detect ink on every M3-flattened page.

Trains a small 3D CNN on the phantom's own ground-truth ink mask (a stand-in
for a pretrained TimeSformer/3D-CNN ensemble — see ink_detection.py), then
predicts a per-pixel ink-confidence map for each flattened page.

Usage:
    python -m src.m4_ink_detection.run --flattened data/processed/phantom_001_flattened.h5 \
        --phantom data/phantoms/phantom_001.h5 --out data/processed/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from src.m4_ink_detection.ink_detection import (
    DEFAULT_WINDOW,
    classical_ink_probability,
    estimate_reference_densities,
    predict_ink_map,
    sample_training_windows,
    train_ink_model,
)


DEFAULT_MAX_PAGES = 100


def run_fragment_model(
    flattened_path: Path,
    volume_path: str | Path,
    out_dir: Path,
    model_path: str | Path,
    voxel_size_um: float | None = None,
    max_pages: int = DEFAULT_MAX_PAGES,
    flip_depth: bool = False,
    refine: bool = True,
    refine_search: float = 8.0,
) -> Path:
    """Ink maps from the U-Net trained on the official Vesuvius fragments (see train_fragments.py).

    Only the ``max_pages`` largest pages are processed (each needs a stack sampled from the volume
    and a tiled forward pass); the rest are left out of the output file. ``refine_search`` is how far (voxels) along the normal the refine step
    looks for the papyrus ridge.
    """
    from src.m4_ink_detection.scroll_pages import predict_page, refine_surface, sample_page_stack
    from src.m4_ink_detection.train_fragments import load_model

    model, meta = load_model(model_path)
    names = sorted(_layer_names(flattened_path), key=_layer_sort_key)
    with h5py.File(flattened_path, "r") as f:
        areas = {n: int((~np.isnan(f[n]["coord_map"][:, :, 0])).sum()) for n in names}
    chosen = sorted(sorted(names, key=lambda n: -areas[n])[:max_pages], key=_layer_sort_key)

    held_out = (meta.get("metrics") or {}).get("f_beta")
    domain = "the scan's voxel size is unknown, so no resolution matching was applied" if not voxel_size_um else f"resampled from {voxel_size_um:g} um to the fragments' 3.24 um"
    notice = (
        f"Ink map from a U-Net trained on the official Vesuvius fragments (held-out fragment F0.5 {held_out:.2f}); " if held_out is not None else "Ink map from a U-Net trained on the official Vesuvius fragments; "
    ) + f"{domain}. It has NOT been validated on this scan (no ink ground truth exists for it), so treat the regions as candidates."
    print(f"Fragment-model ink detection on {len(chosen)} of {len(names)} pages. {notice}", flush=True)

    out_path = out_dir / f"{flattened_path.stem.replace('_flattened', '')}_ink.h5"
    with h5py.File(volume_path, "r") as vf, h5py.File(flattened_path, "r") as fin, h5py.File(out_path, "w") as fout:
        volume = vf["volume"]
        fout.attrs.update(
            {"mode": "fragment_unet", "ground_truth_available": False, "validated": False, "notice": notice,
             "model": str(model_path), "voxel_size_um": voxel_size_um or 0.0, "held_out_f_beta": held_out if held_out is not None else -1.0}
        )
        for i, name in enumerate(chosen, 1):
            coord_map = fin[name]["coord_map"][...]
            dropped = ""
            if refine:  # snap onto the papyrus and drop air-gap pixels before the model sees them
                coord_map, info = refine_surface(volume, coord_map, search=refine_search)
                dropped = f", refined (mean shift {info['mean_abs_shift']:.1f} vox, {info['n_dropped']}/{info['n_pixels']} air pixels dropped)"
            stack, valid = sample_page_stack(volume, coord_map, meta["n_slices"], voxel_size_um, flip_depth)
            prob = predict_page(model, stack, valid, voxel_size_um)
            grp = fout.create_group(name)
            grp.create_dataset("ink_probability", data=prob, compression="lzf")
            grp.create_dataset("page", data=stack[len(stack) // 2], compression="lzf")  # what the model's middle slice saw
            grp.create_dataset("coord_map", data=coord_map, compression="lzf")
            print(f"{name} ({i}/{len(chosen)}): ink probability map {prob.shape}, mean={prob.mean():.4f}, max={prob.max():.4f}{dropped}", flush=True)
    print(f"Ink detection complete -> {out_path}", flush=True)
    return out_path


def _layer_sort_key(name: str) -> tuple[str, int]:
    prefix, _, num = name.rpartition("_")
    return (prefix, int(num)) if num.isdigit() else (name, -1)  # layer_2 before layer_10


def _layer_names(flattened_path: Path) -> list[str]:
    with h5py.File(flattened_path, "r") as f:
        return list(f.keys())


def _calibrate_from_pages(flattened_path: Path, layer_names: list[str], max_samples: int = 2_000_000) -> tuple[float, float]:
    """Pool a random sample of valid page pixels across all layers and derive
    substrate/ink reference levels from their distribution."""
    rng = np.random.default_rng(0)
    chunks = []
    with h5py.File(flattened_path, "r") as f:
        for name in layer_names:
            valid = ~np.isnan(f[name]["coord_map"][:, :, 0])
            vals = f[name]["page"][...][valid]
            if len(vals) > 5000:
                vals = rng.choice(vals, size=5000, replace=False)
            chunks.append(vals)
    pooled = np.concatenate(chunks) if chunks else np.empty(0)
    if len(pooled) > max_samples:
        pooled = rng.choice(pooled, size=max_samples, replace=False)
    return estimate_reference_densities(pooled)


def run(
    flattened_path: str | Path,
    phantom_path: str | Path,
    out_dir: str | Path,
    window: tuple[int, int, int] = DEFAULT_WINDOW,
    epochs: int = 15,
    use_cnn: bool = True,
    model_path: str | Path | None = None,
    voxel_size_um: float | None = None,
    max_pages: int = DEFAULT_MAX_PAGES,
    flip_depth: bool = False,
    refine: bool = True,
) -> Path:
    flattened_path = Path(flattened_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if model_path:
        return run_fragment_model(flattened_path, phantom_path, out_dir, model_path, voxel_size_um, max_pages, flip_depth, refine)

    with h5py.File(phantom_path, "r") as f:
        has_ground_truth = "ink_mask" in f
        substrate_attr = f["volume"].attrs.get("substrate_density")
        ink_attr = f["volume"].attrs.get("ink_density")
        # Only the CNN path needs the whole volume (and only when there is an ink
        # mask to learn from) — a real scan's baseline just reads the flattened pages.
        if use_cnn and has_ground_truth:
            volume = f["volume"][...]
            ink_mask = f["ink_mask"][...]
        else:
            volume = ink_mask = None

    model = None
    if volume is not None:
        substrate_density = float(substrate_attr) if substrate_attr is not None else 0.3
        ink_density = float(ink_attr) if ink_attr is not None else 0.9
        try:
            foreground_mask = volume > (substrate_density / 2)
            train_w, train_y = sample_training_windows(
                volume, ink_mask, foreground_mask, n_positive=400, n_negative=400, window=window
            )
            if len(train_w) > 0:
                model, history = train_ink_model(train_w, train_y, epochs=epochs)
                print(f"Trained ink-detection CNN: final train loss {history['train_loss'][-1]:.4f}")
        except ImportError:
            print("torch not installed; falling back to classical ink-probability baseline")

    mode = "cnn" if model is not None else "classical"
    layer_names = sorted(_layer_names(flattened_path), key=_layer_sort_key)

    calibrated_from = "phantom attributes"
    if model is None and (substrate_attr is None or ink_attr is None):
        substrate_density, ink_density = _calibrate_from_pages(flattened_path, layer_names)
        calibrated_from = "scan statistics (median / 99th percentile of valid page pixels)"
    elif model is None:
        substrate_density, ink_density = float(substrate_attr), float(ink_attr)

    if mode == "classical" and not has_ground_truth:
        print(
            "WARNING: no ink ground truth in the input, so no detector was trained. The output is a "
            f"density-contrast heuristic (substrate={substrate_density:.3f}, ink={ink_density:.3f}, calibrated from {calibrated_from}) "
            "- brighter-than-typical material, NOT a validated ink detection."
        )

    out_path = out_dir / f"{flattened_path.stem.replace('_flattened', '')}_ink.h5"
    with h5py.File(flattened_path, "r") as fin, h5py.File(out_path, "w") as fout:
        fout.attrs["mode"] = mode
        fout.attrs["ground_truth_available"] = has_ground_truth
        fout.attrs["validated"] = bool(model is not None)
        if mode == "classical":
            fout.attrs["substrate_density"] = substrate_density
            fout.attrs["ink_density"] = ink_density
            fout.attrs["calibrated_from"] = calibrated_from
        for layer_name in layer_names:
            coord_map = fin[layer_name]["coord_map"][...]
            page = fin[layer_name]["page"][...]

            if model is not None:
                ink_prob = predict_ink_map(model, volume, coord_map, window=window)
            else:
                ink_prob = classical_ink_probability(page, substrate_density, ink_density)
                ink_prob[np.isnan(coord_map[:, :, 0])] = 0.0  # pixels outside the segment carry no evidence

            grp = fout.create_group(layer_name)
            grp.create_dataset("ink_probability", data=ink_prob, compression="lzf")
            print(f"{layer_name}: ink probability map {ink_prob.shape}, mean={ink_prob.mean():.4f}, max={ink_prob.max():.4f}")

    print(f"Ink detection complete -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flattened", required=True, help="Flattened pages HDF5 (M3 output)")
    parser.add_argument("--phantom", required=True, help="Phantom HDF5 with volume + ink_mask ground truth")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--no-cnn", action="store_true", help="Force the classical baseline, skip CNN training")
    parser.add_argument("--model", default=None, help="Fragment-trained U-Net checkpoint (train_fragments.py); --phantom must then be the volume the pages were cut from")
    parser.add_argument("--voxel-size-um", type=float, default=None, help="Scan voxel size, to resample pages to the model's 3.24 um")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help="Largest pages to process with --model")
    parser.add_argument("--flip-depth", action="store_true", help="Reverse the slice order (which side of the page is up)")
    parser.add_argument("--no-refine", action="store_true", help="Skip snapping pages onto the papyrus before inference")
    parser.add_argument("--refine-search", type=float, default=8.0, help="Refine step: search +- this many voxels along the normal for the papyrus ridge")
    args = parser.parse_args()

    run(
        args.flattened, args.phantom, args.out, epochs=args.epochs, use_cnn=not args.no_cnn,
        model_path=args.model, voxel_size_um=args.voxel_size_um, max_pages=args.max_pages, flip_depth=args.flip_depth, refine=not args.no_refine, refine_search=args.refine_search,
    )


if __name__ == "__main__":
    main()
