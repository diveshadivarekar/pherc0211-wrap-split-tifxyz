"""Tests for exporting flattened M3 pages as tifxyz."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

from src import tifxyz as tx

spec = importlib.util.spec_from_file_location("export_tifxyz", Path(__file__).resolve().parent.parent / "scripts" / "export_tifxyz.py")
export_tifxyz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_tifxyz)


def _flattened(path, n_pages=2):
    yy, xx = np.mgrid[:60, :80].astype(np.float32)
    with h5py.File(path, "w") as f:
        for i in range(n_pages):
            cm = np.stack([yy - 10, np.full_like(yy, 20.0 + i), xx], axis=-1)  # tile-local: z reaches below 0
            cm[:, 70:] = np.nan
            g = f.create_group(f"layer_{i + 1}")
            g.create_dataset("coord_map", data=cm)
            g.attrs["method"] = "patch"
    return path


def test_every_page_becomes_a_loadable_surface_in_absolute_voxels(tmp_path):
    flat = _flattened(tmp_path / "flat.h5")
    index = export_tifxyz.export(flat, tmp_path / "out", offset=(5000, 100, 200), stride=2, voxel_um=2.4, min_valid=10, overwrite=False)
    assert [i["page"] for i in index] == ["layer_1", "layer_2"] and all(i["findings"] == [] for i in index)

    s = tx.read_tifxyz(tmp_path / "out" / "layer_2")
    assert s.meta["source"]["page"] == "layer_2" and s.meta["source"]["pipeline"] == "virtual-unwrap M3"
    assert s.xyz[s.valid][:, 2].min() >= 4990 and s.scale == (0.5, 0.5) or np.allclose(s.scale, (0.5, 0.5), rtol=0.05)
    assert np.allclose(s.xyz[s.valid][:, 1], 100 + 21.0)  # y: tile-local 21 + offset 100
    assert json.loads((tmp_path / "out" / "index.json").read_text())[0]["valid_vertices"] == int(tx.read_tifxyz(tmp_path / "out" / "layer_1").valid.sum())


def test_pages_with_too_few_valid_vertices_are_skipped(tmp_path):
    flat = _flattened(tmp_path / "flat.h5")
    assert export_tifxyz.export(flat, tmp_path / "out", (5000, 0, 0), 1, None, min_valid=10**9, overwrite=False) == []
