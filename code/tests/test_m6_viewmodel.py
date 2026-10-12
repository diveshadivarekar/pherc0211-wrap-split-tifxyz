"""Tests for the GUI's display logic (no Tk needed): bounding boxes, layout, hit-testing, view building."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from src.m6_gui import viewmodel as vm


class TestText:
    def test_natural_sort(self):
        names = ["layer_10", "layer_2", "layer_1"]
        assert sorted(names, key=vm.natural_key) == ["layer_1", "layer_2", "layer_10"]

    def test_middle_truncate_keeps_both_ends_and_fits(self):
        text = "D:/very/long/folder/structure/scan_corrected.h5"
        short = vm.middle_truncate(text, 20)
        assert len(short) == 20 and short.startswith("D:/") and short.endswith("ed.h5") and "…" in short
        assert vm.middle_truncate("short", 20) == "short"


class TestBoxes:
    def test_segment_boxes_are_tight_and_sorted_by_size(self):
        lab = np.zeros((100, 100), dtype=np.int16)
        lab[10:30, 20:60] = 1  # 800 px
        lab[50:60, 50:55] = 2  # 50 px, under the minimum
        lab[70:95, 5:45] = 3  # 1000 px
        boxes = vm.segment_boxes(lab, min_pixels=100)
        assert [b.label for b in boxes] == ["3", "1"]
        b1 = boxes[1]
        assert (b1.x0, b1.y0, b1.x1, b1.y1) == (20, 10, 60, 30)

    def test_segment_boxes_cap_and_empty(self):
        assert vm.segment_boxes(np.zeros((10, 10), dtype=np.int16)) == []
        lab = np.zeros((60, 60), dtype=np.int16)
        for i in range(1, 10):
            lab[i * 5, i * 5 : i * 5 + 5] = i
        assert len(vm.segment_boxes(lab, min_pixels=1, max_boxes=4)) == 4

    def test_ink_boxes_flag_low_confidence(self):
        ink = np.zeros((60, 60), dtype=np.float32)
        ink[5:15, 5:15] = 0.9
        ink[30:45, 30:45] = 0.55
        boxes = vm.ink_boxes(ink, min_pixels=20)
        colours = {(b.x0, b.y0): b.color for b in boxes}
        assert colours[(5, 5)] == vm.INK_OK_COLOR and colours[(30, 30)] == vm.INK_LOW_COLOR
        assert "LOW" in next(b.info for b in boxes if (b.x0, b.y0) == (30, 30))

    def test_foreground_and_page_extent_boxes(self):
        img = np.zeros((50, 80), dtype=np.float32)
        img[10:30, 20:70] = 1.0
        box = vm.foreground_box(img)
        assert (box.x0, box.y0, box.x1, box.y1) == (20, 10, 70, 30)
        coords = np.full((20, 30, 3), np.nan, dtype=np.float32)
        coords[5:15, 4:20] = 1.0
        ext = vm.page_extent_box(coords)
        assert (ext.x0, ext.y0, ext.x1, ext.y1) == (4, 5, 20, 15)
        assert vm.page_extent_box(np.full((4, 4, 3), np.nan)) is None


class TestLayout:
    def test_panels_never_overflow_the_canvas(self):
        for shapes in ([(200, 300)], [(1971, 1971)] * 2, [(295, 692)] * 2, [(10, 10)] * 3):
            for w, h in [(300, 200), (1000, 600), (150, 900), (900, 150), (60, 60)]:
                for p in vm.fit_layout(shapes, w, h):
                    assert p.x >= 0 and p.y >= vm.TITLE_HEIGHT - 1 and p.x + p.w <= w + 1e-6 and p.y + p.h <= h + 1e-6, (shapes, w, h, p)

    def test_wide_pages_stack_and_squares_sit_side_by_side(self):
        wide = vm.fit_layout([(100, 800)] * 2, 700, 500)
        assert wide[0].y != wide[1].y and wide[0].x == wide[1].x
        square = vm.fit_layout([(400, 400)] * 2, 900, 450)
        assert square[0].x != square[1].x and square[0].y == square[1].y

    def test_common_scale_and_degenerate_sizes(self):
        placed = vm.fit_layout([(100, 100), (200, 200)], 800, 400)
        assert placed[0].scale == placed[1].scale
        assert vm.fit_layout([(10, 10)], 5, 5) == [] and vm.fit_layout([], 500, 500) == []

    def test_hit_test_maps_back_to_image_pixels(self):
        placed = vm.fit_layout([(100, 200)], 600, 400)
        p = placed[0]
        idx, ix, iy = vm.hit_test(placed, p.x + 20 * p.scale, p.y + 10 * p.scale)
        assert idx == 0 and ix == pytest.approx(20) and iy == pytest.approx(10)
        assert vm.hit_test(placed, -5, -5) is None

    def test_box_at_prefers_the_innermost(self):
        panel = vm.Panel("p", np.zeros((50, 50)), boxes=[vm.Box(0, 0, 40, 40, label="outer"), vm.Box(10, 10, 20, 20, label="inner")])
        assert vm.box_at(panel, 15, 15).label == "inner" and vm.box_at(panel, 30, 30).label == "outer" and vm.box_at(panel, 45, 45) is None

    def test_readout_reports_segment_and_value(self):
        lab = np.zeros((10, 10), dtype=np.int16)
        lab[2:5, 2:5] = 7
        seg = vm.Panel("Segments", lab.astype(np.float32), kind="rgb", rgb=np.zeros((10, 10, 3)), values=lab)
        assert "segment 7" in vm.readout(seg, 3.2, 3.9) and "background" in vm.readout(seg, 8, 8)
        assert "value 0.500" in vm.readout(vm.Panel("Scan", np.full((4, 4), 0.5)), 1, 1)
        assert vm.readout(seg, 50, 50) == ""


def _write_files(tmp_path):
    rng = np.random.default_rng(0)
    vol = rng.random((6, 40, 50)).astype(np.float32)
    vol[:, 10:30, 15:40] += 1.0
    labels = np.zeros((6, 40, 50), dtype=np.int16)
    labels[:, 10:20, 15:40] = 1
    labels[:, 22:38, 15:35] = 2
    paths = {}
    for name, ds, data in (("corrected", "volume", vol), ("labels", "labels", labels)):
        paths[name] = tmp_path / f"{name}.h5"
        with h5py.File(paths[name], "w") as f:
            f.create_dataset(ds, data=data)
    paths["scan"] = paths["corrected"]
    paths["flattened"], paths["ink"] = tmp_path / "flat.h5", tmp_path / "ink.h5"
    with h5py.File(paths["flattened"], "w") as f:
        for i in (1, 2, 10):
            g = f.create_group(f"layer_{i}")
            g.create_dataset("page", data=rng.random((20, 30)).astype(np.float32))
            coords = np.full((20, 30, 3), np.nan, dtype=np.float32)
            coords[2:18, 3:27] = 1.0
            g.create_dataset("coord_map", data=coords)
            g.attrs["distortion_after"] = 0.04
            g.attrs["method"] = "patch"
    with h5py.File(paths["ink"], "w") as f:
        f.attrs["validated"] = False
        for i in (1, 2, 10):
            m = np.zeros((20, 30), dtype=np.float32)
            m[5:12, 5:14] = 0.9
            f.create_group(f"layer_{i}").create_dataset("ink_probability", data=m)
    return paths


class TestBuildView:
    def test_scan_view_has_depth_and_extent_box(self, tmp_path):
        view = vm.build_view("scan", _write_files(tmp_path), z=2)
        assert view.depth == 6 and view.panels[0].boxes and view.panels[0].boxes[0].label == "scroll extent"

    def test_segmentation_view_overlays_and_boxes(self, tmp_path):
        view = vm.build_view("segmentation", _write_files(tmp_path), z=3)
        overlay = view.panels[1]
        assert overlay.rgb.shape == (40, 50, 3) and {b.label for b in overlay.boxes} == {"1", "2"}
        assert overlay.values.max() == 2

    def test_flattened_and_ink_views_list_layers_in_natural_order(self, tmp_path):
        paths = _write_files(tmp_path)
        flat = vm.build_view("flattened", paths, layer="layer_2")
        assert flat.layers == ["layer_1", "layer_2", "layer_10"] and "layer_2" in flat.panels[0].title
        ink = vm.build_view("ink", paths, layer="layer_10")
        assert len(ink.panels) == 2 and ink.panels[1].boxes and ink.notice and "unvalidated" in ink.notice

    def test_unknown_layer_falls_back_to_first(self, tmp_path):
        assert "layer_1" in vm.build_view("flattened", _write_files(tmp_path), layer="nope").panels[0].title

    def test_missing_inputs_give_actionable_errors(self):
        for kind in ("scan", "segmentation", "flattened", "ink"):
            with pytest.raises(ValueError):
                vm.build_view(kind, {})

    def test_compose_png_renders_panels_and_boxes(self, tmp_path):
        img = vm.compose_png(vm.build_view("segmentation", _write_files(tmp_path), z=3), (600, 360))
        assert img.size == (600, 360) and len(img.getcolors(maxcolors=100000)) > 5
