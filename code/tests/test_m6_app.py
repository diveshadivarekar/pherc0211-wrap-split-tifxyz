"""Functional tests of the Tk window itself (skipped when no display is available)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.m1_ingestion.generate_phantom import generate_phantom_full, save_phantom

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
def app(tmp_path):
    try:
        import tkinter as tk

        probe = tk.Tk()
        probe.destroy()
    except Exception:
        pytest.skip("no display available")
    from src.m6_gui.app import App

    window = App()
    window.geometry("1100x700")
    window.update()
    yield window
    window.destroy()


@pytest.fixture
def phantom(tmp_path):
    raw, gt, layer_id, ink = generate_phantom_full(CFG)
    path = tmp_path / "phantom.h5"
    save_phantom(path, raw, gt, CFG, layer_id=layer_id, ink_mask=ink)
    return path


def wait(app, cond, timeout=120.0):
    end = time.time() + timeout
    while time.time() < end:
        app.update()
        time.sleep(0.02)
        if cond():
            return True
    return False


def test_opening_a_scan_shows_it_with_a_slice_slider_and_box(app, phantom):
    app.v_scan.set(str(phantom))
    app._after_scan_set(Path(phantom))
    assert wait(app, lambda: app._view is not None, 30)
    assert [p.title for p in app._view.panels][0].startswith("Raw")
    assert app._view.depth == 8 and app.slice_scale.winfo_ismapped()
    assert app._view.panels[0].boxes  # scroll extent


def test_modules_run_individually_on_chosen_inputs(app, phantom, tmp_path):
    app.v_out.set(str(tmp_path / "out"))
    app.v_scan.set(str(phantom))
    app._on_run_m1()
    assert wait(app, lambda: not app._busy and app.orchestrator and "m1" in app.orchestrator.results)
    corrected = app.orchestrator.results["m1"].output_path
    assert app.v_corrected.get() == str(corrected)  # M1's output fills M2's input

    # a fresh session: M2 and M3 run on files chosen by hand, with no earlier phase run
    app.orchestrator = None
    app.v_labels.set("")
    app._on_run_m2()
    assert wait(app, lambda: not app._busy and app.orchestrator and "m2" in app.orchestrator.results)
    assert app.orchestrator.results["m2"].ok and app.v_labels.get()
    app.orchestrator = None
    app._on_run_m3()
    assert wait(app, lambda: not app._busy and app.orchestrator and "m3" in app.orchestrator.results)
    assert app.orchestrator.results["m3"].ok and app.v_flat.get().endswith("_flattened.h5")
    assert wait(app, lambda: app._view is not None and "flattened page" in app._view.panels[0].title)
    assert app._view.layers


def test_missing_or_bad_inputs_show_inline_errors_not_dialogs(app, tmp_path):
    app._on_run_m3()
    assert "Choose a file" in app._cards["m3"].error.cget("text")
    app.v_corrected.set(str(tmp_path / "nope.h5"))
    app._on_run_m2()
    assert "not found" in app._cards["m2"].error.cget("text")
    app.v_labels.set("x")
    app.v_corrected.set("")
    app.omega_var.set("-1")
    app.engine_var.set("spiral")
    app._on_engine_changed()
    assert app._spiral_options() is None and "positive" in app._cards["spiral_fitting"].error.cget("text")
    app.m4_epochs.set("abc")
    assert app._int_option("m4", app.m4_epochs, "Training epochs", 1) is None


def test_controls_are_locked_while_running(app, phantom, tmp_path):
    app.v_out.set(str(tmp_path / "out"))
    app.v_scan.set(str(phantom))
    app._on_run_m1()
    assert app._busy and str(app.run_all_btn.cget("state")) == "disabled"
    assert wait(app, lambda: not app._busy)
    assert str(app.run_all_btn.cget("state")) == "normal"


def test_engine_switch_swaps_the_layer_modules(app):
    assert app.classical_group.winfo_manager() and not app.spiral_group.winfo_manager()
    app.engine_var.set("spiral")
    app._on_engine_changed()
    app.update()
    assert app.spiral_group.winfo_manager() and not app.classical_group.winfo_manager()


@pytest.mark.parametrize("size", ["900x560", "1400x900", "1000x700"])
def test_layout_stays_inside_the_window_at_any_size(app, phantom, size):
    app.v_scan.set(str(phantom))
    app._after_scan_set(Path(phantom))
    assert wait(app, lambda: app._view is not None, 30)
    app.geometry(size)
    app.update()
    time.sleep(0.2)
    app.update()
    w, h = app.winfo_width(), app.winfo_height()
    canvas = app.canvas
    for p in app._placed:
        assert 0 <= p.x and p.x + p.w <= canvas.winfo_width() + 1 and p.y + p.h <= canvas.winfo_height() + 1
    for widget in (app.canvas, app.notebook, app.run_all_btn):
        assert widget.winfo_rootx() + widget.winfo_width() <= app.winfo_rootx() + w + 2
        assert widget.winfo_rooty() + widget.winfo_height() <= app.winfo_rooty() + h + 2


def test_long_paths_are_elided_not_overflowing(app):
    app.v_scan.set("C:/" + "very_long_folder_name/" * 12 + "scan_corrected.h5")
    app.update()
    app._refresh_scan_label()
    app.update()
    shown = app.scan_label.cget("text")
    assert "…" in shown and shown.endswith("scan_corrected.h5")
