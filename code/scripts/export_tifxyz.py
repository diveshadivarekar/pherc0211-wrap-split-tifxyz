"""Export the flattened pages of an M3 run as tifxyz surfaces that open in VC3D.

Each ``layer_N`` group of the flattened ``.h5`` holds a ``coord_map`` (page pixel -> source voxel). It is
written as ``<out>/layer_N/{x,y,z}.tif + mask.tif + meta.json`` in level-0 scroll voxels, so ``--offset``
must be the tile's origin in the *whole* scroll volume (z y x). An ``index.json`` lists what was written.

Usage:
    python scripts/export_tifxyz.py data/processed/pherc0211_split/pherc0211_B/pherc0211_B_flattened.h5 \
        --out data/processed/tifxyz/pherc0211_B --offset 8320 4096 1024 --voxel-um 9.362 --stride 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tifxyz import check_contract, from_coord_map, write_tifxyz  # noqa: E402


def export(flattened: Path, out: Path, offset: tuple[float, float, float], stride: int, voxel_um: float | None, min_valid: int, overwrite: bool) -> list[dict]:
    index = []
    with h5py.File(flattened, "r") as f:
        for name in sorted(f, key=lambda n: (len(n), n)):
            cm = f[name]["coord_map"][...]
            if np.isfinite(cm[..., 0]).sum() < min_valid:
                continue
            meta = {"source": {"pipeline": "virtual-unwrap M3", "flattened": flattened.name, "page": name, "method": f[name].attrs.get("method", "")}}
            surf = from_coord_map(cm, offset_zyx=offset, stride=stride, uuid=name, meta=meta)
            if not surf.valid.any():
                continue
            path = write_tifxyz(out / name, surf, voxel_size_um=voxel_um, overwrite=overwrite)
            problems = check_contract(path)
            index.append({"page": name, "path": str(path), "grid": list(surf.shape), "scale": list(surf.scale), "valid_vertices": int(surf.valid.sum()), "findings": problems})
            print(f"{name}: grid {surf.shape[0]}x{surf.shape[1]}, scale {surf.scale[0]:.3f}, {int(surf.valid.sum())} valid vertices" + (f"  [{len(problems)} finding(s)]" if problems else ""), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(index, indent=2))
    return index


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("flattened", type=Path, help="M3 output .h5 (groups with a coord_map)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--offset", type=float, nargs=3, metavar=("Z", "Y", "X"), default=(0.0, 0.0, 0.0), help="origin of the tile the pages were cut from, in level-0 scroll voxels")
    p.add_argument("--stride", type=int, default=1, help="keep every n-th row/column (4 gives a ~4 voxel grid)")
    p.add_argument("--voxel-um", type=float, default=None, help="scan voxel size, for area_cm2 in meta.json")
    p.add_argument("--min-valid", type=int, default=1000, help="skip pages with fewer valid vertices")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    idx = export(a.flattened, a.out, tuple(a.offset), a.stride, a.voxel_um, a.min_valid, a.overwrite)
    print(f"wrote {len(idx)} tifxyz surfaces to {a.out}")


if __name__ == "__main__":
    main()
