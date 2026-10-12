#!/bin/sh
# Export every M3 page (split-wraps run and plain-M2 run) of each 0211 crop as tifxyz and audit it with the repo's QA tool.
SP="$1"
PY=".venv/Scripts/python.exe"
UMB="data/real_scans/pherc0211/representations/umbilicus/20250821151803-umbilicus-20260808112626.json"
ORIG_A="9600 3840 4608"; ORIG_B="8320 4096 1024"; ORIG_C="6272 4608 2048"; ORIG_D="11904 1536 4096"
for C in A B C D; do
  eval "O=\$ORIG_$C"
  for V in split plain; do
    if [ "$V" = split ]; then F="data/processed/pherc0211_split/pherc0211_$C/pherc0211_${C}_flattened.h5"; else F="$SP/plain/flat/$C/pherc0211_${C}_flattened.h5"; fi
    $PY scripts/export_tifxyz.py "$F" --out "$SP/qa/$V/$C" --offset $O --stride 4 --voxel-um 9.362 --overwrite > "$SP/qa/export_${V}_$C.log" 2>&1
    $PY scripts/tifxyz_qa.py "$SP/qa/$V/$C" --umbilicus "$UMB" --prediction "data/real_scans/pherc0211/surface_pred_crop_$C.h5" --report "$SP/qa/report_${V}_$C.json" > "$SP/qa/qa_${V}_$C.log" 2>&1
    echo "done $V $C"
  done
done
