"""Regenerate the PHerc. 0211 feasibility figures programmatically (scan 20250821151803).

Every figure is cut from the CT volume / surface prediction / flattened pages on disk; nothing is drawn onto the
papyrus by hand. Each PNG has a header naming the scroll, scan ID, voxel size and source file, and a scale bar.
These document feasibility work (no ink was found); they are NOT qualifying First Letters evidence (no 4 cm^2
region, no submitted tifxyz mesh, 1 mm bars because the pages are narrower than 1 cm).

Usage:
    python scripts/pherc0211_figures.py --out docs/figures/pherc0211
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import zarr
from PIL import Image, ImageDraw

SCAN_ID = "20250821151803"
VOXEL_UM = 9.362
ROOT = Path("data/real_scans/pherc0211")
SPLIT = Path("data/processed/pherc0211_split")


def stretch(a: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    v = a[valid] if valid is not None and valid.any() else a.ravel()
    lo, hi = np.percentile(v, [1, 99])
    t = np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
    if valid is not None:
        t = t * valid
    return (t * 255).astype(np.uint8)


def frame(img: np.ndarray | Image.Image, header: str, bar_mm: float, px_per_mm: float) -> Image.Image:
    """Add a header line and a labelled scale bar under the image."""
    im = Image.fromarray(img).convert("RGB") if isinstance(img, np.ndarray) else img.convert("RGB")
    bar_px = int(round(bar_mm * px_per_mm))
    head, foot = 26, 30
    out = Image.new("RGB", (max(im.width, bar_px + 120, 8 * len(header)), im.height + head + foot), "white")
    out.paste(im, (0, head))
    d = ImageDraw.Draw(out)
    d.text((4, 6), header, fill="black")
    y = head + im.height + 14
    d.rectangle([6, y, 6 + bar_px, y + 5], fill="black")
    d.text((14 + bar_px, y - 3), f"{bar_mm:g} mm", fill="black")
    return out


def fig_overview(out: Path) -> dict:
    vol = zarr.open(glob.glob(str(ROOT / "volumes/*.zarr"))[0], mode="r")["3"]
    z = vol.shape[0] // 2
    sl = np.asarray(vol[z])
    ys, xs = np.nonzero(sl > 20)
    sl = sl[max(ys.min() - 20, 0): ys.max() + 20, max(xs.min() - 20, 0): xs.max() + 20]
    um = VOXEL_UM * 8
    frame(stretch(sl), f"PHerc. 0211 | scan {SCAN_ID} | {VOXEL_UM} um 113 keV | level 3 ({um:.0f} um) z={z} (level-3 voxels)",
          10.0, 1000.0 / um).save(out)
    return {"file": out.name, "source": "volumes/*.zarr level 3", "z_level3": z, "bar_mm": 10}


def fig_overlay(crop: str, out: Path) -> dict:
    with h5py.File(ROOT / f"native_crop_{crop}.h5", "r") as f, h5py.File(ROOT / f"surface_pred_crop_{crop}.h5", "r") as g:
        z = f["volume"].shape[0] // 2
        ct, pr = np.asarray(f["volume"][z, 100:400, 300:800]), np.asarray(g["volume"][z, 100:400, 300:800])
    up = lambda a: np.kron(a, np.ones((3, 3), dtype=a.dtype))  # noqa: E731
    base = np.stack([stretch(up(ct))] * 3, -1).astype(np.float32)
    ov = base.copy()
    m = up(pr) > 127
    ov[m] = 0.5 * ov[m] + 0.5 * np.array([255, 0, 0])
    side = np.concatenate([base, np.full((base.shape[0], 8, 3), 255, np.float32), ov], 1).astype(np.uint8)
    frame(side, f"PHerc. 0211 | scan {SCAN_ID} | crop {crop} native xy slice (left CT, right + surface prediction >127, red) | 3x",
          1.0, 1000.0 / VOXEL_UM * 3).save(out)
    return {"file": out.name, "source": f"native_crop_{crop}.h5 + surface_pred_crop_{crop}.h5", "z_in_crop": z, "bar_mm": 1}


def fig_pages(crop: str, out: Path, top: int = 4) -> dict:
    path = SPLIT / f"pherc0211_{crop}" / f"pherc0211_{crop}_flattened.h5"
    tiles, notes = [], []
    with h5py.File(path, "r") as f:
        items = sorted(((f[k]["page"].shape[0] * f[k]["page"].shape[1], k) for k in f), reverse=True)[:top]
        for n, k in items:
            pg = f[k]["page"][...]
            valid = pg > 0
            t = Image.fromarray(stretch(pg, valid)).resize((pg.shape[1] * 2, pg.shape[0] * 2), Image.NEAREST)
            d = ImageDraw.Draw(t)
            d.text((3, 3), f"{k} {n * (VOXEL_UM / 1000) ** 2:.1f} mm2", fill=255)
            tiles.append(np.asarray(t))
            notes.append({"layer": k, "page_shape": list(pg.shape), "area_mm2": round(n * (VOXEL_UM / 1000) ** 2, 1),
                          "valid_fraction": round(float(valid.mean()), 2),
                          "distortion_after": round(float(f[k].attrs["distortion_after"]), 4)})
    H = max(t.shape[0] for t in tiles)
    row = np.concatenate([np.pad(t, ((0, H - t.shape[0]), (0, 8)), constant_values=255) for t in tiles], 1)
    frame(row, f"PHerc. 0211 | scan {SCAN_ID} | crop {crop} | largest flattened pages, M2 --split-wraps -> M3 | 2x | no ink seen",
          1.0, 1000.0 / VOXEL_UM * 2).save(out)
    return {"file": out.name, "source": str(path).replace("\\", "/"), "pages": notes, "bar_mm": 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="docs/figures/pherc0211")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scroll": "PHerc. 0211", "scan_id": SCAN_ID, "voxel_um": VOXEL_UM, "kev": 113,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "status": "feasibility figures; no ink found; not qualifying submission evidence",
        "figures": [],
    }
    manifest["figures"].append(fig_overview(out / "pherc0211_overview_L3.png"))
    manifest["figures"].append(fig_overlay("A", out / "pherc0211_A_surface_overlay.png"))
    for c in "ABCD":
        manifest["figures"].append(fig_pages(c, out / f"pherc0211_{c}_flattened_pages.png"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest['figures'])} figures + manifest.json -> {out}")


if __name__ == "__main__":
    main()
