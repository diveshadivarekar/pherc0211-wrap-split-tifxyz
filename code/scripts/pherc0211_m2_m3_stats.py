import sys, json, time
from pathlib import Path
import h5py, numpy as np
from scipy import ndimage
sys.path.insert(0, ".")
from src.m2_segmentation.surface_segments import segment_from_surface_prediction
ROOT = Path("data/real_scans/pherc0211")
UMB = ROOT / "representations/umbilicus/20250821151803-umbilicus-20260808112626.json"
VOX = 9.362e-3  # mm
out = {}
for c in "ABCD":
    t0 = time.time()
    r = {}
    with h5py.File(ROOT/f"native_crop_{c}.h5") as f:
        v = f["volume"]; a = v.attrs
        r["shape_zyx"] = list(v.shape); r["origin_zyx"] = [int(a["z0"]), int(a["y0"]), int(a["x0"])]
        r["dtype"] = str(v.dtype)
        # intensity stats via z-subsample
        s = v[::8]
        r["nonzero_frac"] = float((s > 0).mean()); r["mean_nonzero"] = float(s[s > 0].mean()) if (s>0).any() else 0
    with h5py.File(ROOT/f"surface_pred_crop_{c}.h5") as f:
        mask = f["volume"][...] > 127
    r["surface_voxels"] = int(mask.sum())
    lab, n = ndimage.label(mask, structure=np.ones((3,3,3)))
    cnt = np.bincount(lab.ravel())[1:]
    cnt = np.sort(cnt)[::-1]
    r["cc26_components"] = int(n); r["cc26_largest"] = int(cnt[0]); r["cc26_largest_share"] = float(cnt[0]/mask.sum())
    r["cc26_top5"] = [int(x) for x in cnt[:5]]
    del lab, mask
    # plain M2 (wedge 40, no wrap split) to scratch
    plain = Path(sys.argv[1])/f"plain_{c}.h5"
    shape = tuple(r["shape_zyx"]); origin = tuple(r["origin_zyx"])
    s = segment_from_surface_prediction(ROOT/f"surface_pred_crop_{c}.h5", UMB, origin, shape, plain, wedge_degrees=40.0, min_voxels=2000, split_wraps_apart=False)
    sizes = np.array(sorted(s["voxel_counts"].values(), reverse=True))
    r["plain_m2"] = {"n_pieces": int(s["n_segments"]), "largest": int(sizes[0]), "largest_share_of_labelled": float(sizes[0]/sizes.sum()), "top5": [int(x) for x in sizes[:5]], "surface_fraction": float(s["surface_fraction"])}
    # split M2 from the existing output
    with h5py.File(f"data/processed/pherc0211_split/pherc0211_{c}/pherc0211_{c}_segmented.h5") as f:
        L = f["labels"][...]
    bc = np.bincount(L.ravel())[1:]; bc = bc[bc>0]; bc = np.sort(bc)[::-1]
    r["split_m2"] = {"n_pieces": int(len(bc)), "largest": int(bc[0]), "largest_share_of_labelled": float(bc[0]/bc.sum()), "top5": [int(x) for x in bc[:5]], "labelled_voxels": int(bc.sum())}
    del L
    # M3 pages
    pages = []
    with h5py.File(f"data/processed/pherc0211_split/pherc0211_{c}/pherc0211_{c}_flattened.h5") as f:
        for k in f:
            g = f[k]; cm = g["coord_map"][...]; valid = ~np.isnan(cm[...,0])
            pg = g["page"][...]
            pages.append({"layer": k, "shape": list(pg.shape), "valid_px": int(valid.sum()), "area_mm2": float(valid.sum()*VOX**2),
                          "nonzero_frac": float((pg[valid] > 0).mean()) if valid.any() else 0.0,
                          "dist_before": float(g.attrs["distortion_before"]), "dist_after": float(g.attrs["distortion_after"]), "method": str(g.attrs["method"])})
    r["pages"] = pages
    r["secs"] = time.time()-t0
    out[c] = r
    print(c, json.dumps({k: v for k, v in r.items() if k != "pages"}), flush=True)
json.dump(out, open(Path(sys.argv[1])/"m23_stats.json","w"), indent=1)
