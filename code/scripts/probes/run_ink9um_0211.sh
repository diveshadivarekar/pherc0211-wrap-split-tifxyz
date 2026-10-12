#!/bin/sh
# Official ink_9um on the largest pages of each PHerc 0211 crop, with shifted-surface controls (inkenv interpreter).
for C in A B C D; do
  ../inkenv/Scripts/python.exe scripts/probes/ink9um_on_pages.py \
    --crop data/real_scans/pherc0211/native_crop_$C.h5 \
    --flat data/processed/pherc0211_split/pherc0211_$C/pherc0211_${C}_flattened.h5 --top 3 \
    --checkpoint ../ink9um/hybrid_3d2d-seed42/step-075000.pth --out data/processed/pherc0211_ink9um/$C 2>&1 | grep -v -E "^\s+[a-z_]+:" 
done
