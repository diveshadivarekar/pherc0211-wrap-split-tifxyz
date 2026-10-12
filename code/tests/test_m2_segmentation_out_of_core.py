"""Tests for M2's out-of-core segmentation path.

Verifies the out-of-core algorithm (segment_out_of_core.py) produces
identical results to the in-memory one (segment.py) on data small enough
to run both ways, plus run.py's size-based routing between them. Doesn't
attempt to construct an actual multi-GB test fixture — correctness is what
these check; the real memory-safety property was verified manually against
the 30 GB scroll crop that motivated this (see commit history / README).
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from src.m1_ingestion.generate_phantom import generate_phantom_full
from src.m1_ingestion.io_utils import VolumeStore
from src.m1_ingestion.preprocessing import correct_chunk
from src.m2_segmentation.segment import FULL_CONNECTIVITY, detects_layer_jumping, dice_score, segment_volume
from src.m2_segmentation.segment_out_of_core import (
    estimate_foreground_threshold_out_of_core,
    segment_volume_out_of_core,
)

dask = pytest.importorskip("dask", reason="out-of-core M2 tests require dask + dask-image")
pytest.importorskip("dask_image", reason="out-of-core M2 tests require dask + dask-image")

CFG = {
    "seed": 0,
    "shape": [16, 96, 96],
    "n_layers": 6,
    "substrate_density": 0.3,
    "ink_density": 0.9,
    "beam_hardening_strength": 0.25,
    "ring_artifact_strength": 0.08,
    "n_rings": 4,
    "noise_sigma": 0.01,
    "chunk_depth": 8,
}


@pytest.fixture(scope="module")
def corrected_and_truth():
    raw, ground_truth, layer_id, ink_mask = generate_phantom_full(CFG)
    corrected = correct_chunk(raw)
    return corrected, layer_id


@pytest.fixture
def corrected_h5_path(tmp_path, corrected_and_truth):
    corrected, _ = corrected_and_truth
    path = tmp_path / "corrected.h5"
    store = VolumeStore.create(path, shape=corrected.shape, dtype=np.float32, chunk_depth=8)
    store.write_chunk(slice(0, corrected.shape[0]), corrected)
    store.close()
    return path


class TestThresholdMatchesInMemory:
    def test_out_of_core_threshold_matches_in_memory(self, corrected_and_truth, corrected_h5_path):
        from src.m2_segmentation.segment import estimate_foreground_threshold

        corrected, _ = corrected_and_truth
        in_memory_threshold = estimate_foreground_threshold(corrected)

        store = VolumeStore(corrected_h5_path, mode="r")
        ooc_threshold = estimate_foreground_threshold_out_of_core(store)
        store.close()

        # Different histogram binning (fixed 256 bins over the full range
        # both ways, but accumulated differently) can shift the estimate by
        # a small amount; it should still land very close.
        assert ooc_threshold == pytest.approx(in_memory_threshold, abs=1e-2)


class TestSegmentVolumeOutOfCore:
    def test_matches_in_memory_exactly(self, corrected_and_truth, corrected_h5_path, tmp_path):
        corrected, layer_id = corrected_and_truth
        labels_in_memory = segment_volume(corrected)

        out_path = tmp_path / "labels.h5"
        summary = segment_volume_out_of_core(corrected_h5_path, out_path)

        with h5py.File(out_path, "r") as f:
            labels_ooc = f["labels"][...]

        np.testing.assert_array_equal(labels_ooc, labels_in_memory)
        assert summary["n_segments"] == int(labels_in_memory.max())

    def test_dice_and_no_layer_jumping_on_out_of_core_output(self, corrected_and_truth, corrected_h5_path, tmp_path):
        _, layer_id = corrected_and_truth
        out_path = tmp_path / "labels.h5"
        segment_volume_out_of_core(corrected_h5_path, out_path)

        with h5py.File(out_path, "r") as f:
            labels_ooc = f["labels"][...]

        mean_dice, mapping = dice_score(labels_ooc, layer_id)
        assert mean_dice >= 0.90
        assert not detects_layer_jumping(mapping)

    def test_voxel_counts_match_actual_label_counts(self, corrected_h5_path, tmp_path):
        out_path = tmp_path / "labels.h5"
        summary = segment_volume_out_of_core(corrected_h5_path, out_path)

        with h5py.File(out_path, "r") as f:
            labels_ooc = f["labels"][...]

        for label_id, expected_count in summary["voxel_counts"].items():
            assert int((labels_ooc == label_id).sum()) == expected_count

    def test_handles_no_foreground_gracefully(self, tmp_path):
        """An all-background volume should produce zero segments, not crash."""
        path = tmp_path / "empty.h5"
        store = VolumeStore.create(path, shape=(8, 16, 16), dtype=np.float32, chunk_depth=4)
        store.write_chunk(slice(0, 8), np.zeros((8, 16, 16), dtype=np.float32))
        store.close()

        out_path = tmp_path / "labels.h5"
        summary = segment_volume_out_of_core(path, out_path, threshold=0.5)

        assert summary["n_segments"] == 0
        with h5py.File(out_path, "r") as f:
            labels = f["labels"][...]
        assert labels.shape == (8, 16, 16)
        assert (labels == 0).all()

    def test_respects_explicit_threshold(self, corrected_h5_path, tmp_path):
        out_path_auto = tmp_path / "auto.h5"
        out_path_fixed = tmp_path / "fixed.h5"

        summary_auto = segment_volume_out_of_core(corrected_h5_path, out_path_auto)
        summary_fixed = segment_volume_out_of_core(corrected_h5_path, out_path_fixed, threshold=0.9)

        # A much higher threshold should find fewer (or zero) foreground voxels.
        assert summary_fixed["n_segments"] <= summary_auto["n_segments"]


class TestRunRouting:
    def _write_corrected(self, tmp_path, shape=(16, 96, 96)):
        raw, ground_truth, layer_id, ink_mask = generate_phantom_full({**CFG, "shape": list(shape)})
        corrected = correct_chunk(raw)
        path = tmp_path / "phantom_corrected.h5"
        store = VolumeStore.create(path, shape=corrected.shape, dtype=np.float32, chunk_depth=8)
        store.write_chunk(slice(0, corrected.shape[0]), corrected)
        store.close()
        return path, corrected, layer_id

    def test_small_volume_uses_in_memory_path_by_default(self, tmp_path, monkeypatch):
        from src.m2_segmentation import run as run_module

        path, _, _ = self._write_corrected(tmp_path)
        called = {"in_memory": False, "out_of_core": False}

        original_segment_volume = run_module.segment_volume
        original_ooc = run_module.segment_volume_out_of_core

        def spy_in_memory(*a, **kw):
            called["in_memory"] = True
            return original_segment_volume(*a, **kw)

        def spy_ooc(*a, **kw):
            called["out_of_core"] = True
            return original_ooc(*a, **kw)

        monkeypatch.setattr(run_module, "segment_volume", spy_in_memory)
        monkeypatch.setattr(run_module, "segment_volume_out_of_core", spy_ooc)

        run_module.run(path, tmp_path / "out", max_in_memory_gb=4.0)

        assert called["in_memory"] is True
        assert called["out_of_core"] is False

    def test_small_volume_with_tiny_budget_uses_out_of_core_path(self, tmp_path, monkeypatch):
        """Forcing the size threshold below the actual (tiny) volume size
        should route to the out-of-core path even though the volume itself
        is phantom-scale — this is how the routing logic is tested without
        needing an actual multi-GB fixture."""
        from src.m2_segmentation import run as run_module

        path, _, _ = self._write_corrected(tmp_path)
        called = {"in_memory": False, "out_of_core": False}

        original_segment_volume = run_module.segment_volume
        original_ooc = run_module.segment_volume_out_of_core

        def spy_in_memory(*a, **kw):
            called["in_memory"] = True
            return original_segment_volume(*a, **kw)

        def spy_ooc(*a, **kw):
            called["out_of_core"] = True
            return original_ooc(*a, **kw)

        monkeypatch.setattr(run_module, "segment_volume", spy_in_memory)
        monkeypatch.setattr(run_module, "segment_volume_out_of_core", spy_ooc)

        run_module.run(path, tmp_path / "out", max_in_memory_gb=1e-9)

        assert called["out_of_core"] is True
        assert called["in_memory"] is False

    def test_force_out_of_core_overrides_size_check(self, tmp_path):
        from src.m2_segmentation.run import run

        path, _, _ = self._write_corrected(tmp_path)
        out_path = run(path, tmp_path / "out", force_out_of_core=True)
        assert out_path.exists()

        import json

        summary = json.loads((tmp_path / "out" / "phantom_segments.json").read_text())
        assert "n_segments" in summary
        assert "dice_vs_ground_truth" not in summary  # out-of-core path skips Dice

    def test_out_of_core_output_is_readable_and_correct(self, tmp_path):
        from src.m2_segmentation.run import run

        path, corrected, layer_id = self._write_corrected(tmp_path)
        out_path = run(path, tmp_path / "out", force_out_of_core=True)

        with h5py.File(out_path, "r") as f:
            labels = f["labels"][...]

        reference = segment_volume(corrected)
        np.testing.assert_array_equal(labels, reference)


class TestD8D9Hardening:
    @staticmethod
    def _write_corrected(tmp_path, name="phantom_corrected.h5"):
        raw, _, layer_id, _ = generate_phantom_full(CFG)
        corrected = correct_chunk(raw)
        path = tmp_path / name
        with h5py.File(path, "w") as f:
            f.create_dataset("volume", data=corrected, chunks=(8, 96, 96))
        return path

    def test_base_stem_only_strips_trailing_suffix(self):
        from src.m1_ingestion.io_utils import base_stem

        assert base_stem("a/scan_corrected.h5") == "scan"
        assert base_stem("a/scan_corrected_corrected.h5") == "scan"
        assert base_stem("a/my_corrected_scan_corrected.h5") == "my_corrected_scan"
        assert base_stem("a/scan.h5") == "scan"

    def test_m1_does_not_double_suffix_or_overwrite_its_input(self, tmp_path):
        from src.m1_ingestion.run import run as m1_run

        corrected = self._write_corrected(tmp_path)
        raw_named = tmp_path / "other_corrected.h5"
        raw_named.write_bytes(corrected.read_bytes())
        with h5py.File(raw_named, "a") as f:
            f.move("volume", "volume_tmp")
        with pytest.raises(ValueError):  # no 'volume' dataset
            m1_run(raw_named, tmp_path / "o")

        raw = tmp_path / "scan.h5"
        with h5py.File(raw, "w") as f:
            f.create_dataset("volume", data=np.random.default_rng(0).random((8, 16, 16), dtype=np.float32), chunks=(4, 16, 16))
        first = m1_run(raw, tmp_path / "o")
        assert first.name == "scan_corrected.h5"
        again = m1_run(first, tmp_path / "o2")  # re-correcting an already-corrected file
        assert again.name == "scan_corrected.h5"
        with pytest.raises(ValueError, match="overwrite"):
            m1_run(first, first.parent)

    def test_label_sizes_and_apply_lut_match_numpy(self):
        from src.m2_segmentation.segment import apply_lut, label_sizes

        labels = np.random.default_rng(0).integers(0, 7, size=(21, 9, 9)).astype(np.int32)
        np.testing.assert_array_equal(label_sizes(labels, 6, slab=4), np.bincount(labels.ravel(), minlength=7))
        lut = np.arange(7, dtype=np.int16)[::-1].copy()
        np.testing.assert_array_equal(apply_lut(lut, labels, slab=5), lut[labels])

    def test_out_of_core_caps_labels_cleans_scratch_and_reports(self, tmp_path, monkeypatch):
        import src.m2_segmentation.segment_out_of_core as ooc

        path = self._write_corrected(tmp_path)
        monkeypatch.setattr(ooc, "MAX_LABELS", 3)
        out = tmp_path / "out" / "seg.h5"
        summary = ooc.segment_volume_out_of_core(path, out)
        assert summary["n_segments"] == 3 and summary["n_dropped_by_cap"] >= 1
        with h5py.File(out, "r") as f:
            assert int(f["labels"][...].max()) == 3
        assert not list(out.parent.glob("*.tmp.h5"))

    def test_run_records_skipped_validation_and_auto_budget_routes_out_of_core(self, tmp_path, monkeypatch):
        import json

        from src.m2_segmentation import run as run_module

        path = self._write_corrected(tmp_path)
        monkeypatch.setattr(run_module, "available_memory_gb", lambda: 1e-6)  # pretend the machine is out of RAM
        run_module.run(path, tmp_path / "out", ground_truth_path=path)
        summary = json.loads((tmp_path / "out" / "phantom_segments.json").read_text())
        assert "dice_skipped" in summary and "dice_vs_ground_truth" not in summary
