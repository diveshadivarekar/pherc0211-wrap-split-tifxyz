"""Figures for ``ink9um_on_pages.py`` output: fibre-visible render, ink probability (normal / reversed depth), and shifted controls.

Every figure is titled UNVALIDATED: there is no ground truth for these scrolls.

Usage:
    python scripts/probes/ink9um_figures.py data/processed/pherc0211_ink9um/A --out docs/figures/pherc0211_ink9um/A
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402


def load(p: Path) -> np.ndarray:
    a = tifffile.imread(p).astype(np.float32)
    return a / (255.0 if a.max() > 1.5 else 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("indir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--voxel-um", type=float, default=9.362)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    stats = json.loads(next(a.indir.glob("ink9um_stats*.json")).read_text())
    for name, page in stats["pages"].items():
        render = np.load(a.indir / f"{name}_render.npy")
        shifts = sorted(page["shifts"], key=float)
        cols = 2 + len(shifts) * 2
        fig, ax = plt.subplots(2, max(3, len(shifts)), figsize=(3.2 * max(3, len(shifts)), 6.4))
        ax = np.atleast_2d(ax)
        for r in ax:
            for x in r:
                x.axis("off")
        lo, hi = np.percentile(render[render > 0], [1, 99]) if (render > 0).any() else (0, 1)
        ax[0, 0].imshow(render, cmap="gray", vmin=lo, vmax=hi)
        ax[0, 0].set_title(f"{name} render (mean of 17 layers)\n{render.shape[1] * a.voxel_um / 1000:.1f} x "
                           f"{render.shape[0] * a.voxel_um / 1000:.1f} mm", fontsize=8)
        for i, orient in enumerate(("normal", "flip")):
            ax[0, 1 + i].imshow(load(a.indir / f"{name}_s+0_{orient}.tif"), cmap="magma", vmin=0, vmax=1)
            ax[0, 1 + i].set_title(f"ink_9um P(ink), depth {orient}\nUNVALIDATED", fontsize=8)
        for j, s in enumerate(shifts):
            if s == "+0":
                continue
            ax[1, j].imshow(load(a.indir / f"{name}_s{s}_normal.tif"), cmap="magma", vmin=0, vmax=1)
            st = page["shifts"][s]["normal"]
            ax[1, j].set_title(f"control shift {s} vox\nfrac>0.5 {st['frac_gt_0.5']:.2f}", fontsize=8)
        fig.suptitle(f"{stats['crop']}  ({name})  -  no ground truth, not evidence of letters", fontsize=9)
        fig.tight_layout()
        fig.savefig(a.out / f"{name}_ink9um_UNVALIDATED.png", dpi=110)
        plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
