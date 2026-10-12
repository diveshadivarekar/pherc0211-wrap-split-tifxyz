"""Rank the blocks written by scripts/survey_regions.py and print diverse level-0 crop windows.

A window is a (depth x ysize x xsize) native box aligned to the zarr's 128-voxel chunks. Each block is scored on planarity,
regularity and sheet period; windows are picked greedily from the best blocks, keeping at least ``--min-sep`` level-0
voxels between picks so the candidates do not all sit in one place.

Usage:
    python scripts/rank_regions.py --csv data/processed/pherc0211/survey_L2.csv --top 6
"""

from __future__ import annotations

import argparse
import csv

import numpy as np


def pct_rank(x: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1] (ties broken by order)."""
    return np.argsort(np.argsort(x)) / max(len(x) - 1, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--min-period-um", type=float, default=60.0, help="ignore blocks whose sheet period is below this")
    parser.add_argument("--min-body", type=float, default=0.0, help="min fraction of papyrus-band voxels (needs a survey with body_frac)")
    parser.add_argument("--max-bright", type=float, default=1.0, help="max fraction of saturated debris voxels (needs bright_frac)")
    parser.add_argument("--min-sep", type=int, default=2048, help="min level-0 distance between picks")
    parser.add_argument("--depth", type=int, default=256)
    parser.add_argument("--ysize", type=int, default=640)
    parser.add_argument("--xsize", type=int, default=1536)
    args = parser.parse_args()

    with open(args.csv, newline="") as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    col = lambda name, rs=rows: np.array([r[name] for r in rs])  # noqa: E731
    print(f"{len(rows)} papyrus blocks; period um: median {np.median(col('period_um')):.0f}, "
          f"planarity median {np.median(col('planarity')):.2f}, regularity median {np.median(col('regularity')):.2f}")
    rows = [r for r in rows if r["period_um"] >= args.min_period_um]
    if rows and "body_frac" in rows[0]:
        rows = [r for r in rows if r["body_frac"] >= args.min_body and r["bright_frac"] <= args.max_bright]
        print(f"{len(rows)} blocks pass body >= {args.min_body} and bright <= {args.max_bright}")
    score = pct_rank(col("planarity", rows)) + pct_rank(col("regularity", rows)) + 0.5 * pct_rank(col("period_um", rows))
    picks: list[tuple[float, dict, np.ndarray]] = []
    for i in np.argsort(-score):
        r = rows[i]
        s = 2 ** int(r["level"])
        centre = np.array([(r["z0"] + 64) * s, (r["y0"] + 64) * s, (r["x0"] + 64) * s])
        if all(np.linalg.norm(centre - c) >= args.min_sep for _, _, c in picks):
            picks.append((float(score[i]), r, centre))
        if len(picks) >= args.top:
            break
    for i, (sc, r, c) in enumerate(picks):
        z0 = int(round((c[0] - args.depth / 2) / 128) * 128)
        y0 = int(round((c[1] - args.ysize / 2) / 128) * 128)
        x0 = int(round((c[2] - args.xsize / 2) / 128) * 128)
        print(f"[{i}] score {sc:.2f} planarity {r['planarity']:.2f} regularity {r['regularity']:.2f} "
              f"body {r.get('body_frac', float('nan')):.2f} bright {r.get('bright_frac', float('nan')):.3f}  normal(z,y,x)=({r['nz']:+.2f},{r['ny']:+.2f},{r['nx']:+.2f})  "
              f"--z0 {z0} --y0 {y0} --x0 {x0} --depth {args.depth} --ysize {args.ysize} --xsize {args.xsize}")


if __name__ == "__main__":
    main()
