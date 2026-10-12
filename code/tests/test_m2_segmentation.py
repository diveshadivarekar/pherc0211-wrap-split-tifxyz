"""Tests for M2: connected-component segmentation and its Dice/layer-jump metrics.

Covers SRS Test Plan T3 (Dice > 0.90 vs annotated cross-sections) and T4
(zero cross-layer merges) at PoC scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.m1_ingestion.generate_phantom import generate_phantom_full
from src.m1_ingestion.preprocessing import correct_chunk
from src.m2_segmentation.segment import (
    detects_layer_jumping,
    dice_score,
    estimate_foreground_threshold,
    segment_volume,
)

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
    return corrected, ground_truth, layer_id, ink_mask


class TestThreshold:
    def test_threshold_separates_background_from_substrate(self, corrected_and_truth):
        corrected, ground_truth, layer_id, _ = corrected_and_truth
        t = estimate_foreground_threshold(corrected)
        # Background voxels (layer_id == 0) should mostly fall below threshold,
        # substrate/ink voxels (layer_id > 0) mostly above it.
        bg_above = (corrected[layer_id == 0] > t).mean()
        fg_above = (corrected[layer_id != 0] > t).mean()
        assert fg_above > bg_above


class TestSegmentVolume:
    def test_recovers_expected_number_of_layers(self, corrected_and_truth):
        corrected, _, layer_id, _ = corrected_and_truth
        labels = segment_volume(corrected)
        assert labels.max() == layer_id.max()

    def test_background_stays_unlabelled(self, corrected_and_truth):
        corrected, _, layer_id, _ = corrected_and_truth
        labels = segment_volume(corrected)
        # Voxels far from any true layer must not be picked up as a segment.
        far_background = layer_id == 0
        assert (labels[far_background] == 0).mean() > 0.95

    def test_shape_and_dtype(self, corrected_and_truth):
        corrected, *_ = corrected_and_truth
        labels = segment_volume(corrected)
        assert labels.shape == corrected.shape
        assert labels.dtype == np.int16


class TestDiceAndLayerJumping:
    def test_dice_above_090_gate(self, corrected_and_truth):
        """NFR-3 / T3: Dice >= 0.90 on continuous sheet tracking."""
        corrected, _, layer_id, _ = corrected_and_truth
        labels = segment_volume(corrected)
        mean_dice, _ = dice_score(labels, layer_id)
        assert mean_dice >= 0.90

    def test_no_layer_jumping(self, corrected_and_truth):
        """T4: zero cross-layer merges over the test stack."""
        corrected, _, layer_id, _ = corrected_and_truth
        labels = segment_volume(corrected)
        _, mapping = dice_score(labels, layer_id)
        assert not detects_layer_jumping(mapping)

    def test_dice_score_self_identity(self):
        labels = np.zeros((10, 10, 10), dtype=np.int16)
        labels[2:5, 2:5, 2:5] = 1
        labels[6:9, 6:9, 6:9] = 2
        mean_dice, mapping = dice_score(labels, labels)
        assert mean_dice == pytest.approx(1.0)
        assert mapping == {1: 1, 2: 2}

    def test_layer_jumping_detected_when_two_preds_match_one_truth(self):
        mapping = {1: 5, 2: 5, 3: 6}
        assert detects_layer_jumping(mapping)

    def test_no_layer_jumping_when_mapping_is_one_to_one(self):
        mapping = {1: 5, 2: 6, 3: 7}
        assert not detects_layer_jumping(mapping)


class TestPipelineRun:
    def test_run_end_to_end(self, tmp_path):
        from src.m1_ingestion.generate_phantom import save_phantom
        from src.m2_segmentation.run import run

        raw, ground_truth, layer_id, ink_mask = generate_phantom_full(CFG)
        corrected = correct_chunk(raw)

        phantom_path = tmp_path / "phantom.h5"
        save_phantom(phantom_path, raw, ground_truth, CFG, layer_id=layer_id, ink_mask=ink_mask)

        import h5py

        corrected_path = tmp_path / "phantom_corrected.h5"
        with h5py.File(corrected_path, "w") as f:
            f.create_dataset("volume", data=corrected)

        out_path = run(corrected_path, tmp_path, ground_truth_path=phantom_path)
        assert out_path.exists()

        summary_path = tmp_path / "phantom_segments.json"
        assert summary_path.exists()
        import json

        summary = json.loads(summary_path.read_text())
        assert summary["dice_vs_ground_truth"] >= 0.90
        assert summary["layer_jumping_detected"] is False


class TestFusedSheets:
    """Real scroll wraps touch; plain connected components then returns one giant layer."""

    @staticmethod
    def _fused_pair():
        # two bright planar sheets joined by a dim bridge (thick, so it survives thresholding)
        vol = np.zeros((12, 40, 40), dtype=np.float32)
        vol[:, 8:11, 4:36] = 1.0
        vol[:, 27:30, 4:36] = 1.0
        vol[:, 11:27, 18:22] = 0.45
        return vol

    def test_plain_components_merge_fused_sheets(self):
        labels = segment_volume(self._fused_pair(), threshold=0.3, split_fused=False)
        assert labels.max() == 1

    def test_split_separates_fused_sheets(self):
        vol = self._fused_pair()
        labels = segment_volume(vol, threshold=0.3)
        top, bottom = labels[6, 9, 20], labels[6, 28, 20]
        assert top > 0 and bottom > 0 and top != bottom
        assert (labels > 0).sum() >= 0.95 * (vol > 0.3).sum()

    def test_separated_layers_untouched_by_split(self, corrected_and_truth):
        corrected = corrected_and_truth[0]
        assert np.array_equal(segment_volume(corrected), segment_volume(corrected, split_fused=False))
