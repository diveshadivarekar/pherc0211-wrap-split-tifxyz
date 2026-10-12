"""Audit tifxyz surfaces against the volume: geometry, cross-wrap (radiality), surface-prediction support, CT centering.

Works on any tifxyz directory (or a folder of them), not just this project's. Checks that need extra
inputs run only when given: ``--umbilicus`` enables the radiality check, ``--prediction`` the surface
support / drift check (and ``--snap-out`` writes label-snapped copies), ``--ct`` the independent CT
centering score. All coordinates are level-0 scroll voxels, as in the tifxyz files.

Usage:
    python scripts/tifxyz_qa.py data/processed/tifxyz/pherc0211_B \
        --umbilicus data/real_scans/pherc0211/representations/umbilicus/20250821151803-umbilicus-20260808112626.json \
        --prediction data/real_scans/pherc0211/surface_pred_crop_B.h5 --report data/processed/tifxyz/qa_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import tifxyz as tx  # noqa: E402
from src.m2_segmentation.surface_segments import load_umbilicus  # noqa: E402


def find_surfaces(root: Path) -> list[Path]:
    if (root / "meta.json").exists():
        return [root]
    return sorted((p.parent for p in root.glob("*/meta.json")), key=lambda p: (len(p.name), p.name))


def _ramp(a: np.ndarray, lo: float, hi: float, colours: np.ndarray) -> np.ndarray:
    t = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    pos = np.linspace(0, 1, len(colours))
    return np.stack([np.interp(t, pos, colours[:, c]) for c in range(3)], axis=-1)


def overlay_png(path: Path, surface: tx.Surface, maps: dict, cell: int = 4) -> None:
    """Panels of per-vertex maps (invalid vertices grey): prediction support, signed drift, cross-wrap and folds."""
    h, w = surface.shape
    grey = np.array([60, 60, 60], np.float64)
    panels = []
    seq = np.array([[20, 20, 90], [40, 120, 200], [250, 230, 90]], np.float64)  # low -> high support
    div = np.array([[30, 60, 200], [235, 235, 235], [200, 40, 30]], np.float64)
    if "support" in maps:
        s = np.nan_to_num(maps["support"])
        panels.append((_ramp(s, 0, 255, seq), np.isfinite(maps["support"])))
        d = maps["drift"]
        panels.append((_ramp(np.nan_to_num(d), -12, 12, div), np.isfinite(d)))
    if "cross_wrap" in maps:
        img = np.where(maps["cross_wrap"][..., None], [220, 50, 40], [90, 170, 90]).astype(np.float64)
        panels.append((img, surface.valid))
    if "folded" in maps:
        f = np.zeros((h, w), bool)
        f[:-1, :-1] = maps["folded"]
        panels.append((np.where(f[..., None], [220, 50, 40], [90, 170, 90]).astype(np.float64), surface.valid))
    if not panels:
        return
    tiles = []
    for rgb, known in panels:
        rgb = np.where(known[..., None], rgb, grey)
        tiles.append(np.kron(rgb, np.ones((cell, cell, 1))).astype(np.uint8))
    gap = np.full((tiles[0].shape[0], 6, 3), 255, np.uint8)
    row = tiles[0]
    for t in tiles[1:]:
        row = np.concatenate([row, gap, t], axis=1)
    Image.fromarray(row).save(path)


def audit_one(path: Path, args: argparse.Namespace, umb, pred, ct, png_dir: Path | None, snap_dir: Path | None) -> dict:
    surf = tx.read_tifxyz(path)
    rep: dict = {"name": path.name, "contract": tx.check_contract(path), "geometry": tx.geometry_report(surf)}
    maps: dict = {}
    if not surf.valid.any():
        return rep
    maps["folded"] = tx.folded_cells(surf)
    if umb is not None:
        rep["radiality"], m = tx.radiality(surf, umb, args.max_angle)
        maps["cross_wrap"] = m["cross_wrap"]
        spacing = None
        if pred is not None:
            spacing = tx.estimate_wrap_spacing(pred, umb, float(np.median(surf.xyz[surf.valid][:, 2])))
        rep["radial_extent"] = tx.radial_extent(surf, umb, spacing)
    if pred is not None:
        rep["support"], m = tx.surface_support(surf, pred, args.threshold, args.search)
        maps.update(support=m["support"], drift=m["drift"])
    if ct is not None:
        rep["ct_centering"] = tx.ct_centering(surf, ct)
    if pred is not None and snap_dir is not None:
        snapped, move = tx.snap_to_prediction(surf, pred, args.threshold, args.search)
        rep["snap"] = {"applied": move, "support_after": tx.surface_support(snapped, pred, args.threshold, args.search)[0]}
        if ct is not None:
            rep["snap"]["ct_centering_after"] = tx.ct_centering(snapped, ct)
        tx.write_tifxyz(snap_dir / path.name, snapped, voxel_size_um=surf.meta.get("voxelsize"), overwrite=True)
    if png_dir is not None:
        overlay_png(png_dir / f"{path.name}.png", surf, maps)
    return rep


def snap_paired(reports: list[dict]) -> dict:
    """Per-page before/after means (same weighting on both sides), split by how many wraps the page spans."""
    def stats(rs: list[dict]) -> dict:
        if not rs:
            return {"pages": 0}
        d = {"pages": len(rs),
             "on_surface_before": float(np.mean([r["support"]["on_surface_fraction"] for r in rs])),
             "on_surface_after": float(np.mean([r["snap"]["support_after"]["on_surface_fraction"] for r in rs])),
             "abs_shift_median_voxels": float(np.median([r["snap"]["applied"]["abs_shift_median"] for r in rs]))}
        both = [r for r in rs if "ct_centering" in r and "ct_centering_after" in r["snap"] and r["ct_centering"].get("vertices_sampled") and r["snap"]["ct_centering_after"].get("vertices_sampled")]
        if both:
            before = np.array([r["ct_centering"]["contrast"] for r in both])
            after = np.array([r["snap"]["ct_centering_after"]["contrast"] for r in both])
            d.update({"ct_contrast_before": float(before.mean()), "ct_contrast_after": float(after.mean()), "pages_ct_contrast_improved": int((after > before).sum()), "pages_with_ct": len(both)})
        return d

    snapped = [r for r in reports if "snap" in r and "support" in r]
    wraps = lambda r: r.get("radial_extent", {}).get("wraps_crossed")  # noqa: E731
    return {
        "all": stats(snapped),
        "spanning_2_wraps_or_fewer": stats([r for r in snapped if wraps(r) is not None and wraps(r) <= 2]),
        "spanning_over_2_wraps": stats([r for r in snapped if wraps(r) is not None and wraps(r) > 2]),
    }


def summarise(reports: list[dict]) -> dict:
    """Vertex-weighted roll-up of the per-surface numbers."""
    def wavg(section: str, key: str, weight: str) -> float | None:
        pairs = [(r[section][key], r[section][weight]) for r in reports if section in r and weight in r[section] and key in r[section] and np.isfinite(r[section][key]) and r[section][weight] > 0]
        return float(sum(v * w for v, w in pairs) / sum(w for _, w in pairs)) if pairs else None

    out: dict = {"surfaces": len(reports)}
    out["contract_findings"] = sum(len(r["contract"]) for r in reports)
    out["folded_fraction_mean"] = float(np.mean([r["geometry"].get("folded_fraction", 0.0) for r in reports]))
    out["cross_wrap_fraction"] = wavg("radiality", "cross_wrap_fraction", "evaluated_vertices")
    crossed = [r["radial_extent"]["wraps_crossed"] for r in reports if r.get("radial_extent", {}).get("wraps_crossed_reliable") and "wraps_crossed" in r["radial_extent"]]
    n_long = sum(1 for r in reports if "wraps_crossed" in r.get("radial_extent", {}) and not r["radial_extent"].get("wraps_crossed_reliable", True))
    if crossed:
        out["wraps_crossed_median"] = float(np.median(crossed))
        out["surfaces_spanning_over_2_wraps"] = int(sum(c > 2 for c in crossed))
    if n_long:
        out["surfaces_too_long_for_wraps_crossed"] = n_long  # angular span > 90 deg: use on-surface / cross-wrap instead
    out["on_surface_fraction"] = wavg("support", "on_surface_fraction", "vertices_in_volume")
    out["abs_drift_median_voxels"] = wavg("support", "abs_drift_median", "vertices_in_volume")
    out["ct_centred_fraction"] = wavg("ct_centering", "centred_fraction", "vertices_sampled")
    out["ct_contrast"] = wavg("ct_centering", "contrast", "vertices_sampled")
    if any("snap" in r for r in reports):
        out["snap_paired"] = snap_paired(reports)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("surfaces", type=Path, help="a tifxyz directory, or a folder of them")
    p.add_argument("--umbilicus", type=Path, help="scroll-axis control points .json (enables the radiality check)")
    p.add_argument("--prediction", type=Path, help="surface-prediction crop .h5 (enables the support / drift check)")
    p.add_argument("--ct", type=Path, help="native CT crop .h5 (enables the independent CT centering score)")
    p.add_argument("--snap-out", type=Path, help="write label-snapped copies here (needs --prediction)")
    p.add_argument("--report", type=Path, help="write the full JSON report here")
    p.add_argument("--png-dir", type=Path, help="write per-surface diagnostic PNGs here")
    p.add_argument("--threshold", type=float, default=127.0, help="surface-prediction threshold (0-255)")
    p.add_argument("--search", type=float, default=12.0, help="normal-direction search radius, voxels")
    p.add_argument("--max-angle", type=float, default=60.0, help="radiality: flag cells whose normal is further than this from radial")
    p.add_argument("--limit", type=int, help="audit only the first N surfaces")
    args = p.parse_args()

    paths = find_surfaces(args.surfaces)[: args.limit]
    if not paths:
        sys.exit(f"no tifxyz surfaces under {args.surfaces}")
    umb = load_umbilicus(args.umbilicus) if args.umbilicus else None
    pred = tx.load_volume(args.prediction) if args.prediction else None
    ct = tx.load_volume(args.ct) if args.ct else None
    for d in (args.png_dir, args.snap_out):
        if d:
            d.mkdir(parents=True, exist_ok=True)
    reports = []
    for i, path in enumerate(paths, 1):
        r = audit_one(path, args, umb, pred, ct, args.png_dir, args.snap_out if pred is not None else None)
        reports.append(r)
        bits = [f"{r['geometry'].get('valid_vertices', 0)} vtx"]
        if "radiality" in r and "cross_wrap_fraction" in r["radiality"]:
            bits.append(f"cross-wrap {r['radiality']['cross_wrap_fraction']:.1%}")
        if "wraps_crossed" in r.get("radial_extent", {}):
            re_ = r["radial_extent"]
            bits.append(f"spans {re_['wraps_crossed']:.1f} wraps" + ("" if re_.get("wraps_crossed_reliable", True) else f" (unreliable: {re_['angular_span_deg']:.0f} deg of arc)"))
        if "support" in r:
            bits.append(f"on-surface {r['support']['on_surface_fraction']:.1%}, drift {r['support']['abs_drift_median']:.1f}vx")
        if "ct_centering" in r and r["ct_centering"].get("vertices_sampled"):
            bits.append(f"CT-centred {r['ct_centering']['centred_fraction']:.1%}")
        print(f"[{i}/{len(paths)}] {path.name}: " + ", ".join(bits), flush=True)
    summary = summarise(reports)
    print(json.dumps(summary, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"summary": summary, "surfaces": reports}, indent=2))


if __name__ == "__main__":
    main()
