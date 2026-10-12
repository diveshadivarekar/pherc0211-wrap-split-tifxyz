"""Tests for M4: ink-window sampling, the classical baseline, and the CNN.

Covers SRS Test Plan T6 (ink map separates ink from substrate above
threshold). The CNN-specific tests are skipped if torch isn't installed —
the classical baseline has no such dependency and is always exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.m1_ingestion.generate_phantom import generate_phantom_full
from src.m1_ingestion.preprocessing import correct_chunk
from src.m2_segmentation.segment import segment_volume
from src.m3_flattening.flatten import flatten_layer
from src.m4_ink_detection.ink_detection import (
    classical_ink_probability,
    extract_window,
    sample_training_windows,
)

torch = pytest.importorskip("torch", reason="M4 CNN tests require torch")

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
def phantom_data():
    raw, ground_truth, layer_id, ink_mask = generate_phantom_full(CFG)
    corrected = correct_chunk(raw)
    return corrected, ink_mask


class TestExtractWindow:
    def test_center_voxel_matches_source(self, phantom_data):
        volume, _ = phantom_data
        center = (8, 48, 48)
        window = extract_window(volume, center, (5, 5, 5))
        assert window.shape == (5, 5, 5)
        assert window[2, 2, 2] == pytest.approx(volume[center])

    def test_border_center_is_zero_padded_not_crashing(self, phantom_data):
        volume, _ = phantom_data
        window = extract_window(volume, (0, 0, 0), (7, 7, 7))
        assert window.shape == (7, 7, 7)
        assert window[0, 0, 0] == 0.0  # padded region


class TestClassicalBaseline:
    def test_probability_in_unit_range(self, phantom_data):
        volume, _ = phantom_data
        prob = classical_ink_probability(volume, CFG["substrate_density"], CFG["ink_density"])
        assert prob.min() >= 0.0
        assert prob.max() <= 1.0

    def test_ink_voxels_score_higher_than_substrate(self, phantom_data):
        volume, ink_mask = phantom_data
        prob = classical_ink_probability(volume, CFG["substrate_density"], CFG["ink_density"])
        assert prob[ink_mask].mean() > prob[~ink_mask].mean()


class TestSampleTrainingWindows:
    def test_balanced_and_correctly_labelled(self, phantom_data):
        volume, ink_mask = phantom_data
        foreground = volume > (CFG["substrate_density"] / 2)
        windows, labels = sample_training_windows(
            volume, ink_mask, foreground, n_positive=50, n_negative=50, window=(5, 5, 5)
        )
        assert windows.shape[1:] == (5, 5, 5)
        assert set(np.unique(labels)) <= {0.0, 1.0}
        assert (labels == 1.0).sum() > 0
        assert (labels == 0.0).sum() > 0


class TestTrainAndPredict:
    def test_trained_model_separates_ink_from_substrate(self, phantom_data):
        """T6: the ink map must separate ink from substrate above threshold."""
        from src.m4_ink_detection.ink_detection import predict_ink_map, train_ink_model

        volume, ink_mask = phantom_data
        foreground = volume > (CFG["substrate_density"] / 2)
        windows, labels = sample_training_windows(
            volume, ink_mask, foreground, n_positive=300, n_negative=300, window=(5, 5, 5)
        )
        model, history = train_ink_model(windows, labels, epochs=10)
        assert history["train_loss"][-1] < history["train_loss"][0]

        # Build a tiny synthetic coord_map covering a handful of known ink and
        # known substrate voxels, and check the model separates them.
        ink_coords = np.argwhere(ink_mask)[:20]
        sub_coords = np.argwhere(foreground & ~ink_mask)[:20]
        all_coords = np.concatenate([ink_coords, sub_coords], axis=0).astype(np.float32)
        coord_map = all_coords.reshape(1, -1, 3)

        prob_map = predict_ink_map(model, volume, coord_map, window=(5, 5, 5))
        ink_probs = prob_map[0, : len(ink_coords)]
        sub_probs = prob_map[0, len(ink_coords) :]
        assert ink_probs.mean() > sub_probs.mean()


class TestFullPipelineIntegration:
    def test_flatten_then_detect_ink(self, phantom_data):
        from src.m4_ink_detection.ink_detection import predict_ink_map, train_ink_model

        volume, ink_mask = phantom_data
        labels = segment_volume(volume)
        foreground = volume > (CFG["substrate_density"] / 2)

        # Find a layer that actually carries ink.
        inked_layer = None
        for lid in range(1, int(labels.max()) + 1):
            if ink_mask[labels == lid].sum() > 5:
                inked_layer = lid
                break
        if inked_layer is None:
            pytest.skip("no inked layer in this phantom config")

        result = flatten_layer(volume, labels, inked_layer, n_theta=180, arap_iters=3)

        windows, train_labels = sample_training_windows(
            volume, ink_mask, foreground, n_positive=300, n_negative=300, window=(5, 5, 5)
        )
        model, _ = train_ink_model(windows, train_labels, epochs=10)
        ink_prob_map = predict_ink_map(model, volume, result["coord_map"], window=(5, 5, 5))

        assert ink_prob_map.shape == result["page"].shape
        assert ink_prob_map.min() >= 0.0
        assert ink_prob_map.max() <= 1.0


class TestRealScanMode:
    """A real scan has no ink ground truth and its own density scale."""

    @staticmethod
    def _write_flattened(path, layers):
        import h5py

        with h5py.File(path, "w") as f:
            for name, (page, valid) in layers.items():
                grp = f.create_group(name)
                grp.create_dataset("page", data=page)
                coords = np.zeros(page.shape + (3,), dtype=np.float32)
                coords[~valid] = np.nan
                grp.create_dataset("coord_map", data=coords)

    @staticmethod
    def _write_volume(path):
        import h5py

        with h5py.File(path, "w") as f:  # no ink_mask, no density attrs, negative-valued like a normalised scan
            f.create_dataset("volume", data=np.random.default_rng(0).normal(-0.05, 0.1, (4, 8, 8)).astype(np.float32))

    def test_reference_densities_follow_the_data(self):
        from src.m4_ink_detection.ink_detection import estimate_reference_densities

        vals = np.random.default_rng(0).normal(-0.05, 0.1, 100000)
        substrate, ink = estimate_reference_densities(vals)
        assert abs(substrate + 0.05) < 0.01 and ink > substrate + 0.1

    def test_scan_run_calibrates_and_flags_unvalidated(self, tmp_path):
        import h5py

        from src.m4_ink_detection.run import run

        rng = np.random.default_rng(1)
        layers = {}
        for i in (1, 2, 10):
            page = rng.normal(-0.05, 0.1, (20, 30)).astype(np.float32)
            valid = np.ones(page.shape, dtype=bool)
            valid[:, :5] = False
            layers[f"layer_{i}"] = (page, valid)
        self._write_flattened(tmp_path / "scan_flattened.h5", layers)
        self._write_volume(tmp_path / "scan_corrected.h5")

        out = run(tmp_path / "scan_flattened.h5", tmp_path / "scan_corrected.h5", tmp_path)
        with h5py.File(out, "r") as f:
            assert f.attrs["mode"] == "classical"
            assert not f.attrs["ground_truth_available"] and not f.attrs["validated"]
            assert f.attrs["substrate_density"] < 0.1  # not the phantom's 0.3 default
            assert set(f.keys()) == {"layer_1", "layer_2", "layer_10"}
            prob = f["layer_1"]["ink_probability"][...]
            assert (prob[:, :5] == 0).all()  # outside the segment: no evidence
            assert 0.0 < (prob > 0.5).mean() < 0.1  # a calibrated map is selective, not all-0 or all-1
