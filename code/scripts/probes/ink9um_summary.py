"""Aggregate ``ink9um_on_pages.py`` output over crops: per page, share of valid pixels with P(ink) > 0.5 on the surface vs its shifted copies.

Reads the per-page ``*_s<shift>_<orient>.tif`` files directly (the per-crop JSON is only written at the end of a crop). The number that
matters is whether the surface stands out from its controls; there is no ground truth here.

Usage:
    python scripts/probes/ink9um_summary.py data/processed/pherc0211_ink9um --flat-root data/processed/pherc0211_split \
        --scan pherc0211 --voxel-um 9.362 --json docs/figures/pherc0211_ink9um/summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import tifffile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--flat-root", type=Path, required=True)
    ap.add_argument("--scan", default="pherc0211")
    ap.add_argument("--voxel-um", type=float, default=9.362)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    rows = []
    for crop_dir in sorted(p for p in a.root.iterdir() if p.is_dir()):
        flat = a.flat_root / f"{a.scan}_{crop_dir.name}" / f"{a.scan}_{crop_dir.name}_flattened.h5"
        if not flat.exists():
            continue
        names = sorted({re.match(r"(layer_\d+)_s", p.name).group(1) for p in crop_dir.glob("layer_*_s*.tif")})
        with h5py.File(flat, "r") as f:
            for name in names:
                valid = ~np.isnan(f[name]["coord_map"][..., 0])
                # pages whose volume is exactly zero (masked / padding) carry no CT signal: list them, do not score them
                nonzero = float((f[name]["page"][...][valid] > 0).mean()) if valid.any() else 0.0
                if nonzero < 0.5:
                    rows.append({"crop": crop_dir.name, "page": name, "valid_px": int(valid.sum()),
                                 "area_mm2": round(valid.sum() * (a.voxel_um / 1000) ** 2, 2), "empty": True,
                                 "nonzero_fraction": round(nonzero, 3)})
                    continue
                res: dict[str, dict[str, float]] = {}
                for tif in crop_dir.glob(f"{name}_s*_*.tif"):
                    m = re.match(rf"{name}_s([+-]\d+)_(normal|flip)\.tif", tif.name)
                    if not m:
                        continue
                    pr = tifffile.imread(tif).astype(np.float32)
                    pr /= 255.0 if pr.max() > 1.5 else 1.0
                    res.setdefault(m.group(1), {})[m.group(2)] = float((pr[valid] > 0.5).mean())
                if "+0" not in res or len(res["+0"]) < 2 or not any(len(d) == 2 for s, d in res.items() if s != "+0"):
                    continue
                ctl = [v for s, d in res.items() if s != "+0" for v in d.values()]
                surf = list(res["+0"].values())
                rows.append({"crop": crop_dir.name, "page": name, "valid_px": int(valid.sum()),
                             "area_mm2": round(valid.sum() * (a.voxel_um / 1000) ** 2, 2),
                             "surface_normal": round(res["+0"]["normal"], 3), "surface_flip": round(res["+0"]["flip"], 3),
                             "controls_min": round(min(ctl), 3) if ctl else None,
                             "controls_median": round(float(np.median(ctl)), 3) if ctl else None,
                             "controls_max": round(max(ctl), 3) if ctl else None,
                             "n_controls": len(ctl),
                             "surface_within_control_range": bool(ctl and min(ctl) <= max(surf) and min(surf) <= max(ctl))})
    print(f"{'crop':>4} {'page':>9} {'mm2':>6} {'surf N':>7} {'surf F':>7} {'ctl min':>8} {'ctl med':>8} {'ctl max':>8} {'n':>3}")
    for r in rows:
        if r.get("empty"):
            print(f"{r['crop']:>4} {r['page']:>9} {r['area_mm2']:6.1f}   EMPTY: volume is zero under the page (nonzero {r['nonzero_fraction']})")
            continue
        print(f"{r['crop']:>4} {r['page']:>9} {r['area_mm2']:6.1f} {r['surface_normal']:7.3f} {r['surface_flip']:7.3f} "
              f"{r['controls_min']:8.3f} {r['controls_median']:8.3f} {r['controls_max']:8.3f} {r['n_controls']:3d}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
