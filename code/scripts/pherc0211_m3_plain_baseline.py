import sys, json, time
from pathlib import Path
import h5py, numpy as np
sys.path.insert(0, ".")
from src.m3_flattening.run import run as m3_run
sp = Path(sys.argv[1]); UMB = Path("data/real_scans/pherc0211/representations/umbilicus/20250821151803-umbilicus-20260808112626.json")
res = {}
for c in "ABCD":
    t0 = time.time()
    tile = Path(f"data/processed/pherc0211_split/pherc0211_{c}/pherc0211_{c}_corrected.h5")
    out = sp/"flat"/c; out.mkdir(parents=True, exist_ok=True)
    flat = m3_run(tile, sp/f"plain_{c}.h5", out, min_voxels=20000, umbilicus=UMB)
    pages = []
    with h5py.File(flat) as f:
        for k in f:
            g = f[k]; cm = g["coord_map"][...]; valid = ~np.isnan(cm[...,0]); pg = g["page"][...]
            pages.append({"layer": k, "shape": list(pg.shape), "valid_px": int(valid.sum()), "area_mm2": float(valid.sum()*(9.362e-3)**2),
                          "dist_before": float(g.attrs["distortion_before"]), "dist_after": float(g.attrs["distortion_after"]), "method": str(g.attrs["method"])})
    res[c] = {"pages": pages, "secs": time.time()-t0}
    da = [p["dist_after"] for p in pages]
    print(c, len(pages), "median %.4f max %.4f"%(np.median(da), max(da)), "secs %.0f"%(time.time()-t0), flush=True)
json.dump(res, open(sp/"m3_plain.json","w"), indent=1)
