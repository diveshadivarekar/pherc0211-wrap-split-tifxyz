"""Tests for M6: pipeline orchestration and preview-image conversion.

The orchestrator (business logic, no widgets) is tested directly. The
Tkinter app itself needs a real display; its widget-construction is smoke
tested if one is available and skipped otherwise (e.g. headless CI), since
Tk's own event loop isn't something worth re-testing here.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from src.m1_ingestion.generate_phantom import generate_phantom_full, save_phantom
from src.m6_gui.orchestrator import PipelineOrchestrator, friendly_error, load_before_after_slice

CFG = {
    "seed": 0,
    "shape": [8, 64, 64],
    "n_layers": 4,
    "substrate_density": 0.3,
    "ink_density": 0.9,
    "beam_hardening_strength": 0.25,
    "ring_artifact_strength": 0.08,
    "n_rings": 3,
    "noise_sigma": 0.01,
    "chunk_depth": 4,
}


@pytest.fixture
def phantom_file(tmp_path):
    raw, ground_truth, layer_id, ink_mask = generate_phantom_full(CFG)
    path = tmp_path / "phantom.h5"
    save_phantom(path, raw, ground_truth, CFG, layer_id=layer_id, ink_mask=ink_mask)
    return path


class TestFriendlyError:
    def test_file_not_found_gets_friendly_message(self):
        msg = friendly_error(FileNotFoundError("data/missing.h5"))
        assert "Couldn't find" in msg

    def test_unknown_exception_gets_generic_message(self):
        msg = friendly_error(RuntimeError("boom"))
        assert "Something went wrong" in msg
        assert "boom" in msg


class TestOrchestratorRunAll:
    def test_run_all_completes_m1_through_m4(self, tmp_path, phantom_file):
        orch = PipelineOrchestrator(tmp_path / "work")
        log = []
        results = orch.run_all(phantom_file, callback=lambda phase, msg: log.append((phase, msg)))

        by_phase = {r.name.split()[0]: r for r in results}
        assert by_phase["M1"].ok
        assert by_phase["M2"].ok
        assert by_phase["M3"].ok
        # M4/M5 need torch; if unavailable they fail gracefully rather than crash the run.
        assert len(log) > 0

    def test_segmentation_refuses_to_run_before_ingestion(self, tmp_path):
        orch = PipelineOrchestrator(tmp_path / "work")
        result = orch.run_segmentation()
        assert not result.ok
        assert "ingestion" in result.message.lower()

    def test_flattening_refuses_to_run_before_segmentation(self, tmp_path):
        orch = PipelineOrchestrator(tmp_path / "work")
        result = orch.run_flattening()
        assert not result.ok
        assert "segmentation" in result.message.lower()

    def test_ingestion_failure_reports_friendly_message_not_traceback(self, tmp_path):
        orch = PipelineOrchestrator(tmp_path / "work")
        result = orch.run_ingestion(tmp_path / "does_not_exist.h5")
        assert not result.ok
        assert "Traceback" not in result.message


class TestSpiralFittingEngine:
    """The spiral-fitting engine (src/spiral_fitting/) as an M2+M3
    alternative, wired through the orchestrator. Torch-gated like M4/M5."""

    def test_refuses_to_run_before_ingestion(self, tmp_path):
        orch = PipelineOrchestrator(tmp_path / "work")
        result = orch.run_spiral_fitting(initial_omega=0.5)
        assert not result.ok
        assert "ingestion" in result.message.lower()

    def test_flattened_result_is_none_when_nothing_ran(self, tmp_path):
        orch = PipelineOrchestrator(tmp_path / "work")
        assert orch.flattened_result() is None

    def test_flattened_result_prefers_spiral_when_both_present(self, tmp_path, phantom_file):
        pytest.importorskip("torch", reason="spiral fitting requires torch")
        orch = PipelineOrchestrator(tmp_path / "work")
        orch.run_ingestion(phantom_file)
        orch.run_segmentation()
        m3 = orch.run_flattening()
        assert m3.ok
        spiral = orch.run_spiral_fitting(initial_omega=0.3, n_windings=1, n_iterations=5)
        assert spiral.ok
        assert orch.flattened_result() is spiral

    def test_flattened_result_falls_back_to_m3_when_spiral_not_run(self, tmp_path, phantom_file):
        orch = PipelineOrchestrator(tmp_path / "work")
        orch.run_ingestion(phantom_file)
        orch.run_segmentation()
        m3 = orch.run_flattening()
        assert m3.ok
        assert orch.flattened_result() is m3

    def test_run_all_rejects_unknown_engine(self, tmp_path, phantom_file):
        orch = PipelineOrchestrator(tmp_path / "work")
        with pytest.raises(ValueError, match="engine"):
            orch.run_all(phantom_file, engine="not-a-real-engine")

    def test_run_all_spiral_engine_auto_estimates_omega_when_not_given(self, tmp_path, phantom_file):
        """spiral_initial_omega=None (the default) auto-estimates from
        winding pairs rather than requiring the caller to guess — this
        used to raise ValueError; that's no longer the API."""
        pytest.importorskip("torch", reason="spiral fitting requires torch")
        orch = PipelineOrchestrator(tmp_path / "work")
        results = orch.run_all(phantom_file, engine="spiral", spiral_n_windings=1, spiral_n_iterations=5)
        by_phase = {r.name.split()[0]: r for r in results}
        assert by_phase["M1"].ok
        assert orch.results["spiral_fitting"].ok
        assert orch.results["spiral_fitting"].details.get("initial_omega") is not None

    def test_run_all_spiral_engine_end_to_end(self, tmp_path, phantom_file):
        pytest.importorskip("torch", reason="spiral fitting requires torch")
        orch = PipelineOrchestrator(tmp_path / "work")
        results = orch.run_all(
            phantom_file, engine="spiral", spiral_initial_omega=0.3, spiral_n_windings=1, spiral_n_iterations=5
        )

        by_phase = {r.name.split()[0]: r for r in results}
        assert by_phase["M1"].ok
        assert "m2" not in orch.results  # classical path never ran
        assert orch.results["spiral_fitting"].ok
        # M4 must have consumed spiral fitting's output, not M3's (M3 never ran).
        assert orch.flattened_result() is orch.results["spiral_fitting"]

    def test_m4_consumes_spiral_output_when_m3_never_ran(self, tmp_path, phantom_file):
        pytest.importorskip("torch", reason="spiral fitting requires torch")
        orch = PipelineOrchestrator(tmp_path / "work")
        orch.run_ingestion(phantom_file)
        spiral = orch.run_spiral_fitting(initial_omega=0.3, n_windings=1, n_iterations=5)
        assert spiral.ok

        m4 = orch.run_ink_detection(phantom_file)
        assert m4.ok


class TestPreviewLoading:
    def test_load_before_after_slice_before_correction(self, phantom_file):
        raw, corrected = load_before_after_slice(phantom_file, None, slice_idx=0)
        assert raw.shape == (CFG["shape"][1], CFG["shape"][2])
        assert corrected is None

    def test_load_before_after_slice_after_correction(self, tmp_path, phantom_file):
        orch = PipelineOrchestrator(tmp_path / "work")
        result = orch.run_ingestion(phantom_file)
        assert result.ok
        raw, corrected = load_before_after_slice(phantom_file, result.output_path, slice_idx=0)
        assert corrected is not None
        assert corrected.shape == raw.shape


class TestArrayToPhotoImage:
    def test_conversion_produces_expected_size(self):
        pytest.importorskip("tkinter")
        try:
            import tkinter as tk
        except Exception:
            pytest.skip("Tk not available")
        try:
            root = tk.Tk()
            root.withdraw()
        except Exception:
            pytest.skip("no display available")

        from src.m6_gui.app import PREVIEW_SIZE, array_to_photoimage

        arr = np.random.default_rng(0).normal(size=(20, 30)).astype(np.float32)
        img = array_to_photoimage(arr)
        assert img.width() == PREVIEW_SIZE[0]
        assert img.height() == PREVIEW_SIZE[1]
        root.destroy()

    def test_labels_to_photoimage_handles_flat_input(self):
        try:
            import tkinter as tk
        except Exception:
            pytest.skip("Tk not available")
        try:
            root = tk.Tk()
            root.withdraw()
        except Exception:
            pytest.skip("no display available")

        from src.m6_gui.app import labels_to_photoimage

        labels = np.zeros((10, 10), dtype=np.int16)
        labels[2:5, 2:5] = 1
        img = labels_to_photoimage(labels)
        assert img.width() > 0 and img.height() > 0
        root.destroy()
