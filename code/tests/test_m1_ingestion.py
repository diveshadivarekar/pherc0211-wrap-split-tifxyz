"""Tests for M1: phantom generation, out-of-core I/O, and artifact correction.

Covers SRS Test Plan T1 (out-of-core loading correctness) at PoC scale, plus
correctness checks for the beam-hardening and ring-artifact corrections
against the phantom's known ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.m1_ingestion.generate_phantom import generate_phantom, save_phantom
from src.m1_ingestion.io_utils import VolumeStore
from src.m1_ingestion.preprocessing import correct_beam_hardening, correct_chunk, correct_ring_artifacts

SMALL_CFG = {
    "seed": 0,
    # Deliberately not tiny: the correction algorithms rely on individual
    # fused layers spanning several radius bins with real background gaps
    # between them (see generate_phantom.py) — too small a canvas merges
    # layers together or makes them indistinguishable from ring artifacts.
    "shape": [8, 128, 128],
    "n_layers": 8,
    "substrate_density": 0.3,
    "ink_density": 0.9,
    "beam_hardening_strength": 0.25,
    "ring_artifact_strength": 0.08,
    "n_rings": 6,
    "noise_sigma": 0.0,  # deterministic for correction-accuracy assertions
    "chunk_depth": 4,
}


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


class TestPhantomGeneration:
    def test_shape_and_dtype(self):
        raw, gt = generate_phantom(SMALL_CFG)
        assert raw.shape == tuple(SMALL_CFG["shape"])
        assert gt.shape == tuple(SMALL_CFG["shape"])
        assert raw.dtype == np.float32
        assert gt.dtype == np.float32

    def test_deterministic_with_seed(self):
        raw1, gt1 = generate_phantom(SMALL_CFG)
        raw2, gt2 = generate_phantom(SMALL_CFG)
        np.testing.assert_array_equal(raw1, raw2)
        np.testing.assert_array_equal(gt1, gt2)

    def test_ground_truth_has_substrate_and_ink_values(self):
        _, gt = generate_phantom(SMALL_CFG)
        values = np.unique(gt)
        assert 0.0 in values
        assert SMALL_CFG["substrate_density"] in values
        assert SMALL_CFG["ink_density"] in values

    def test_raw_differs_from_ground_truth(self):
        """The corrupted volume must actually carry the injected artifacts."""
        raw, gt = generate_phantom(SMALL_CFG)
        assert not np.allclose(raw, gt)


class TestVolumeStore:
    def test_create_write_read_roundtrip(self, tmp_path):
        path = tmp_path / "vol.h5"
        shape = (8, 16, 16)
        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

        store = VolumeStore.create(path, shape=shape, dtype=np.float32, chunk_depth=4)
        for start in range(0, shape[0], 4):
            sl = slice(start, start + 4)
            store.write_chunk(sl, data[sl])
        store.close()

        readback = VolumeStore(path, mode="r")
        np.testing.assert_array_equal(readback.read_all(), data)
        assert readback.shape == shape
        readback.close()

    def test_iter_chunks_covers_full_volume_out_of_core(self, tmp_path):
        """T1: chunked iteration must reconstruct the full volume without
        ever materialising it all at once."""
        path = tmp_path / "phantom.h5"
        raw, gt = generate_phantom(SMALL_CFG)
        save_phantom(path, raw, gt, SMALL_CFG)

        store = VolumeStore(path, mode="r")
        reassembled = np.zeros(store.shape, dtype=store.dtype)
        max_chunk_rows_seen = 0
        for sl, chunk in store.iter_chunks():
            max_chunk_rows_seen = max(max_chunk_rows_seen, chunk.shape[0])
            reassembled[sl] = chunk
        store.close()

        np.testing.assert_array_equal(reassembled, raw)
        assert max_chunk_rows_seen <= SMALL_CFG["chunk_depth"]


class TestArtifactCorrection:
    def test_beam_hardening_correction_reduces_cupping_error(self):
        raw, gt = generate_phantom(SMALL_CFG)
        corrected = correct_beam_hardening(raw)
        assert _rmse(corrected, gt) < _rmse(raw, gt)

    def test_ring_artifact_correction_reduces_ring_error(self):
        raw, gt = generate_phantom(SMALL_CFG)
        # Isolate ring artifacts by starting from ground truth + rings only,
        # so this test doesn't also depend on beam-hardening correction.
        from src.m1_ingestion.generate_phantom import _apply_ring_artifacts

        rng = np.random.default_rng(SMALL_CFG["seed"])
        ringed = _apply_ring_artifacts(
            gt.copy(), strength=SMALL_CFG["ring_artifact_strength"], n_rings=SMALL_CFG["n_rings"], rng=rng
        )
        corrected = correct_ring_artifacts(ringed)
        assert _rmse(corrected, gt) < _rmse(ringed, gt)

    def test_full_correction_pipeline_reduces_total_error(self):
        raw, gt = generate_phantom(SMALL_CFG)
        corrected = correct_chunk(raw)
        assert _rmse(corrected, gt) < _rmse(raw, gt)

    def test_correction_preserves_shape_and_dtype(self):
        raw, _ = generate_phantom(SMALL_CFG)
        corrected = correct_chunk(raw)
        assert corrected.shape == raw.shape
        assert corrected.dtype == raw.dtype

    def test_correction_works_on_partial_chunk(self):
        """Correction must be safe to call on a sub-volume, since the real
        pipeline runs it chunk-by-chunk (out-of-core), not on the full volume."""
        raw, _ = generate_phantom(SMALL_CFG)
        chunk = raw[:4]
        corrected = correct_chunk(chunk)
        assert corrected.shape == chunk.shape


class TestPipelineRun:
    def test_run_produces_corrected_output_closer_to_ground_truth(self, tmp_path):
        from src.m1_ingestion.run import run

        raw, gt = generate_phantom(SMALL_CFG)
        input_path = tmp_path / "phantom.h5"
        save_phantom(input_path, raw, gt, SMALL_CFG)

        out_dir = tmp_path / "processed"
        out_path = run(input_path, out_dir)

        assert out_path.exists()
        store = VolumeStore(out_path, mode="r")
        corrected = store.read_all()
        store.close()

        assert corrected.shape == gt.shape
        assert _rmse(corrected, gt) < _rmse(raw, gt)
