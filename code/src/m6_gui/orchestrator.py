"""M6 — pipeline orchestration logic, kept separate from the Tkinter widgets.

Implements the non-UI half of FR-6.1-FR-6.4: run M1-M5 phase by phase,
report per-phase status/progress through a callback (so the GUI can render
it, and so this is testable without a display), and load before/after
preview arrays for each phase. Errors are translated to short, jargon-free
messages here (NFR: no jargon-only failure states for non-technical users)
so the GUI layer never has to interpret a raw traceback.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from src.m1_ingestion.io_utils import base_stem

PhaseCallback = Callable[[str, str], None]  # (phase_name, status_message)

_JARGON_FRIENDLY_ERRORS = {
    FileNotFoundError: "Couldn't find that file. Check the path and try again.",
    ValueError: "That file doesn't look like a valid scan for this step.",
    KeyError: "That file is missing data this step needs — try re-running the previous phase.",
}


def friendly_error(exc: Exception) -> str:
    for exc_type, message in _JARGON_FRIENDLY_ERRORS.items():
        if isinstance(exc, exc_type):
            return f"{message} ({exc})"
    return f"Something went wrong in this step: {exc}"


@dataclass
class PhaseResult:
    name: str
    ok: bool
    output_path: Path | None = None
    message: str = ""
    details: dict = field(default_factory=dict)


class PipelineOrchestrator:
    """Runs M1-M5 against one working directory, tracking each phase's
    output path so the next phase (and the GUI's preview panel) can find it.
    """

    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, PhaseResult] = {}

    def set_work_dir(self, work_dir: str | Path) -> None:
        """Change where later phases write; results already produced stay where they are."""
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def register_input(self, key: str, path: str | Path, name: str | None = None) -> PhaseResult:
        """Treat an existing file as the finished output of phase ``key`` (m1 = corrected
        volume, m2 = labels, m3 = flattened pages, m4 = ink maps) so any module can be run
        on its own, without running the phases before it."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        result = PhaseResult(name or key, True, path, "Provided by user")
        self.results[key] = result
        return result

    def _input_for(self, key: str, override: str | Path | None) -> Path | None:
        """An explicitly chosen input wins (and is remembered); otherwise the previous phase's output."""
        if override:
            return self.register_input(key, override).output_path
        existing = self.results.get(key)
        return existing.output_path if existing and existing.ok else None

    def _report(self, callback: PhaseCallback | None, phase: str, message: str) -> None:
        if callback:
            callback(phase, message)

    def run_ingestion(self, phantom_path: str | Path, callback: PhaseCallback | None = None) -> PhaseResult:
        from src.m1_ingestion.run import run as m1_run

        phase = "M1 Ingestion & Preprocessing"
        try:
            self._report(callback, phase, "Correcting artifacts...")
            out_path = m1_run(phantom_path, self.work_dir)
            result = PhaseResult(phase, True, out_path, "Done")
        except Exception as exc:  # noqa: BLE001 - deliberately broad: this is the UI error boundary
            result = PhaseResult(phase, False, None, friendly_error(exc), {"traceback": traceback.format_exc()})
        self.results["m1"] = result
        self._report(callback, phase, result.message)
        return result

    def run_segmentation(
        self,
        ground_truth_path: str | Path | None = None,
        callback: PhaseCallback | None = None,
        input_path: str | Path | None = None,
    ) -> PhaseResult:
        from src.m1_ingestion.io_utils import base_stem
        from src.m2_segmentation.run import run as m2_run

        phase = "M2 Segmentation"
        corrected = self._input_for("m1", input_path)
        if corrected is None:
            result = PhaseResult(phase, False, None, "Run ingestion first, or choose a corrected volume as this module's input.")
            self.results["m2"] = result
            return result
        try:
            self._report(callback, phase, "Separating fused layers...")
            out_path = m2_run(corrected, self.work_dir, ground_truth_path=ground_truth_path)
            summary_path = self.work_dir / f"{base_stem(corrected)}_segments.json"
            details = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            result = PhaseResult(phase, True, out_path, "Done", details)
        except Exception as exc:
            result = PhaseResult(phase, False, None, friendly_error(exc), {"traceback": traceback.format_exc()})
        self.results["m2"] = result
        self._report(callback, phase, result.message)
        return result

    def run_flattening(
        self,
        callback: PhaseCallback | None = None,
        corrected_path: str | Path | None = None,
        labels_path: str | Path | None = None,
        min_voxels: int | None = None,
        method: str = "auto",
    ) -> PhaseResult:
        from src.m3_flattening.run import run as m3_run

        phase = "M3 Flattening"
        corrected = self._input_for("m1", corrected_path)
        labels = self._input_for("m2", labels_path)
        if corrected is None or labels is None:
            first, missing = ("segmentation", "segment labels") if labels is None else ("ingestion", "a corrected volume")
            result = PhaseResult(phase, False, None, f"Run {first} first, or choose {missing} as this module's input.")
            self.results["m3"] = result
            return result
        try:
            self._report(callback, phase, "Unrolling pages...")
            out_path = m3_run(corrected, labels, self.work_dir, min_voxels=min_voxels, method=method)
            result = PhaseResult(phase, True, out_path, "Done")
        except Exception as exc:
            result = PhaseResult(phase, False, None, friendly_error(exc), {"traceback": traceback.format_exc()})
        self.results["m3"] = result
        self._report(callback, phase, result.message)
        return result

    def run_spiral_fitting(
        self,
        initial_omega: float | None = None,
        n_windings: int = 4,
        n_iterations: int | None = None,
        callback: PhaseCallback | None = None,
        input_path: str | Path | None = None,
    ) -> PhaseResult:
        """Global-model alternative to M2 (segmentation) + M3 (flattening)
        combined — see src/spiral_fitting/__init__.py. Operates directly on
        M1's corrected volume; doesn't need (or produce) M2's per-layer
        labels. Its output uses the same page/coord_map HDF5 convention as
        M3's, so it's stored under a separate "spiral_fitting" key but
        :func:`flattened_result` lets M4 consume whichever engine ran.

        ``initial_omega=None`` (default) auto-estimates a starting value
        from extracted winding pairs (see features.estimate_initial_omega)
        instead of requiring the caller to guess — verified this alone
        lands within ~1% of ground truth on a synthetic spiral phantom.
        """
        from src.spiral_fitting.fit import FitConfig
        from src.spiral_fitting.run import run as spiral_run

        phase = "Spiral Fitting (M2+M3 alternative)"
        corrected = self._input_for("m1", input_path)
        if corrected is None:
            result = PhaseResult(phase, False, None, "Run ingestion first, or choose a corrected volume as this module's input.")
            self.results["spiral_fitting"] = result
            return result
        try:
            self._report(callback, phase, "Extracting surface paths and fitting spiral...")
            config = FitConfig()
            if n_iterations is not None:
                config.n_iterations = n_iterations
            out_path = spiral_run(
                corrected,
                self.work_dir,
                initial_omega=initial_omega,
                n_windings=n_windings,
                config=config,
            )
            summary_path = self.work_dir / f"{base_stem(corrected)}_spiral_fit_summary.json"
            details = {}
            if summary_path.exists():
                import json

                details = json.loads(summary_path.read_text())
            result = PhaseResult(phase, True, out_path, "Done", details)
        except Exception as exc:
            result = PhaseResult(phase, False, None, friendly_error(exc), {"traceback": traceback.format_exc()})
        self.results["spiral_fitting"] = result
        self._report(callback, phase, result.message)
        return result

    def flattened_result(self) -> PhaseResult | None:
        """Whichever of M3 (classical) or spiral-fitting last produced a
        usable flattened-page output — both write the same page/coord_map
        HDF5 convention, so M4 doesn't need to know which one ran. Prefers
        spiral-fitting if both are present and the spiral-fitting run is
        newer (i.e. the user explicitly re-ran it after M3)."""
        chosen = self.results.get("flattened_input")
        if chosen and chosen.ok:
            return chosen
        m3 = self.results.get("m3")
        spiral = self.results.get("spiral_fitting")
        if spiral and spiral.ok:
            return spiral
        if m3 and m3.ok:
            return m3
        return None

    def run_ink_detection(
        self,
        phantom_path: str | Path,
        callback: PhaseCallback | None = None,
        flattened_path: str | Path | None = None,
        epochs: int = 15,
        model_path: str | Path | None = None,
        voxel_size_um: float | None = None,
        max_pages: int = 100,
    ) -> PhaseResult:
        from src.m4_ink_detection.run import run as m4_run

        phase = "M4 Ink Detection"
        if flattened_path:
            self.register_input("flattened_input", flattened_path)
        flattened = self.flattened_result()
        if flattened is None:
            result = PhaseResult(phase, False, None, "Run flattening (classical M3, or spiral fitting) first.")
            self.results["m4"] = result
            return result
        try:
            self._report(callback, phase, "Training ink-detection model...")
            out_path = m4_run(
                flattened.output_path, phantom_path, self.work_dir, epochs=epochs,
                model_path=model_path, voxel_size_um=voxel_size_um, max_pages=max_pages,
            )
            result = PhaseResult(phase, True, out_path, "Done")
        except Exception as exc:
            result = PhaseResult(phase, False, None, friendly_error(exc), {"traceback": traceback.format_exc()})
        self.results["m4"] = result
        self._report(callback, phase, result.message)
        return result

    def run_ocr_translation(
        self, callback: PhaseCallback | None = None, ink_path: str | Path | None = None, epochs: int = 100
    ) -> PhaseResult:
        from src.m5_ocr_translation.run import run as m5_run

        phase = "M5 OCR & Translation"
        try:
            self._report(callback, phase, "Recognising and translating text...")
            if ink_path:
                self.register_input("m4", ink_path)
            m4 = self.results.get("m4")
            ink_file = m4.output_path if m4 and m4.ok else None
            out_path = m5_run(self.work_dir, ink_path=ink_file, n_train=100, n_eval=15, epochs=epochs)
            result = PhaseResult(phase, True, out_path, "Done")
        except Exception as exc:
            result = PhaseResult(phase, False, None, friendly_error(exc), {"traceback": traceback.format_exc()})
        self.results["m5"] = result
        self._report(callback, phase, result.message)
        return result

    def run_all(
        self,
        phantom_path: str | Path,
        callback: PhaseCallback | None = None,
        engine: str = "classical",
        spiral_initial_omega: float | None = None,
        spiral_n_windings: int = 4,
        spiral_n_iterations: int | None = None,
        skip_ingestion: bool = False,
    ) -> list[PhaseResult]:
        """``engine="classical"`` (default) runs M2+M3; ``engine="spiral"``
        runs spiral fitting instead. ``spiral_initial_omega=None`` (default)
        auto-estimates a starting value rather than requiring a guess."""
        if engine not in ("classical", "spiral"):
            raise ValueError(f"engine must be 'classical' or 'spiral', got {engine!r}")

        if skip_ingestion:  # the scan is already a corrected volume
            results = [self.register_input("m1", phantom_path, "M1 Ingestion & Preprocessing")]
        else:
            results = [self.run_ingestion(phantom_path, callback)]
        if engine == "classical":
            if results[-1].ok:
                results.append(self.run_segmentation(phantom_path, callback))
            if results[-1].ok:
                results.append(self.run_flattening(callback))
        else:
            if results[-1].ok:
                results.append(
                    self.run_spiral_fitting(
                        spiral_initial_omega, spiral_n_windings, n_iterations=spiral_n_iterations, callback=callback
                    )
                )
        if results[-1].ok:
            results.append(self.run_ink_detection(phantom_path, callback))
        results.append(self.run_ocr_translation(callback))
        return results


def middle_slice_index(path: str | Path, dataset: str = "volume") -> int:
    """Pick the middle depth slice of a volume for previews.

    Slice 0 looks fine on the synthetic phantom (every slice is identical),
    but real scans (src/m1_ingestion/real_scan.py) are genuinely non-uniform
    along depth, and the first slice is often a near-empty/out-of-focus
    edge of the stack — the middle slice is a much more representative
    default preview.
    """
    with h5py.File(path, "r") as f:
        return f[dataset].shape[0] // 2


def load_before_after_slice(phantom_path: str | Path, corrected_path: str | Path | None, slice_idx: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """M1 preview: (raw slice, corrected slice or None)."""
    if slice_idx is None:
        slice_idx = middle_slice_index(phantom_path)
    with h5py.File(phantom_path, "r") as f:
        raw = f["volume"][slice_idx]
    corrected = None
    if corrected_path is not None and Path(corrected_path).exists():
        with h5py.File(corrected_path, "r") as f:
            corrected = f["volume"][slice_idx]
    return raw, corrected


def load_segmentation_overlay(corrected_path: str | Path, labels_path: str | Path, slice_idx: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """M2 preview: (volume slice, label slice) for overlay rendering."""
    if slice_idx is None:
        slice_idx = middle_slice_index(corrected_path)
    with h5py.File(corrected_path, "r") as f:
        volume = f["volume"][slice_idx]
    with h5py.File(labels_path, "r") as f:
        labels = f["labels"][slice_idx]
    return volume, labels


def load_flattened_page(flattened_path: str | Path, layer_id: int = 1) -> np.ndarray:
    with h5py.File(flattened_path, "r") as f:
        return f[f"layer_{layer_id}"]["page"][...]


def load_ink_map(ink_path: str | Path, layer_id: int = 1) -> np.ndarray:
    with h5py.File(ink_path, "r") as f:
        return f[f"layer_{layer_id}"]["ink_probability"][...]
