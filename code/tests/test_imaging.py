"""Tests for src/imaging.py: shared preview rendering used by the GUI and
scripts/export_previews.py."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.imaging import (
    apply_hot_colormap,
    labels_to_rgb,
    normalize_to_uint8,
    save_side_by_side,
    to_display_image,
)


class TestNormalizeToUint8:
    def test_output_range(self):
        arr = np.random.default_rng(0).normal(size=(20, 20)).astype(np.float32)
        out = normalize_to_uint8(arr)
        assert out.dtype == np.uint8
        assert out.min() >= 0 and out.max() <= 255

    def test_handles_nan(self):
        arr = np.array([[1.0, np.nan], [3.0, 4.0]])
        out = normalize_to_uint8(arr)
        assert np.isfinite(out).all()

    def test_constant_array_does_not_crash(self):
        arr = np.full((5, 5), 3.0)
        out = normalize_to_uint8(arr)
        assert out.shape == (5, 5)

    def test_outliers_dont_wash_out_contrast(self):
        """A handful of extreme outliers (well under 1% of pixels) shouldn't
        collapse the rest of the image's own contrast to near-zero — that's
        the whole point of percentile stretch over min/max stretch. Uses a
        smoothly varying background (not a flat constant) so there's real
        contrast to preserve, and a large-enough array that the
        99th-percentile cutoff has room to exclude the outliers (on a tiny
        array, 1% is less than one pixel and can't exclude anything)."""
        background = np.linspace(0.3, 0.7, 50 * 50).reshape(50, 50)
        arr = background.copy()
        arr[0, :5] = 1000.0  # 5 outliers out of 2500 pixels = 0.2%

        out = normalize_to_uint8(arr)
        out_without_outlier_row = out[1:]
        # The background's own spread (dim to bright) should still show up
        # as real contrast, not be crushed toward black by the outlier.
        assert out_without_outlier_row.max() > 200
        assert out_without_outlier_row.min() < 50


class TestColormaps:
    def test_hot_colormap_shape_and_range(self):
        gray = np.linspace(0, 255, 100).astype(np.uint8).reshape(10, 10)
        rgb = apply_hot_colormap(gray)
        assert rgb.shape == (10, 10, 3)
        assert rgb.dtype == np.uint8

    def test_hot_colormap_black_at_zero_white_at_max(self):
        gray = np.array([[0, 255]], dtype=np.uint8)
        rgb = apply_hot_colormap(gray)
        np.testing.assert_array_equal(rgb[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(rgb[0, 1], [255, 255, 255])

    def test_labels_to_rgb_background_is_black(self):
        labels = np.array([[0, 1], [2, 0]])
        rgb = labels_to_rgb(labels)
        np.testing.assert_array_equal(rgb[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(rgb[1, 1], [0, 0, 0])
        assert not np.array_equal(rgb[0, 1], [0, 0, 0])

    def test_labels_to_rgb_stable_across_calls(self):
        labels = np.array([[1, 2, 3]])
        rgb1 = labels_to_rgb(labels)
        rgb2 = labels_to_rgb(labels)
        np.testing.assert_array_equal(rgb1, rgb2)


class TestToDisplayImage:
    def test_grayscale_output(self):
        arr = np.random.default_rng(0).normal(size=(20, 30))
        img = to_display_image(arr)
        assert img.mode == "L"
        assert img.size == (30, 20)

    def test_resize_uses_lanczos_not_nearest(self):
        arr = np.random.default_rng(0).normal(size=(8, 8))
        img = to_display_image(arr, size=(80, 80))
        assert img.size == (80, 80)
        # A LANCZOS-upscaled image of noise should have more distinct pixel
        # values than a NEAREST-upscaled one (which just repeats blocks).
        nearest = Image.fromarray(normalize_to_uint8(arr), mode="L").resize((80, 80), Image.NEAREST)
        assert len(np.unique(np.array(img))) >= len(np.unique(np.array(nearest)))

    def test_labels_mode_produces_rgb(self):
        labels = np.array([[0, 1], [2, 3]])
        img = to_display_image(labels, is_labels=True)
        assert img.mode == "RGB"

    def test_hot_cmap_produces_rgb(self):
        arr = np.random.default_rng(0).uniform(size=(10, 10))
        img = to_display_image(arr, cmap="hot")
        assert img.mode == "RGB"


class TestSaveSideBySide:
    def test_writes_combined_image(self, tmp_path):
        left = np.random.default_rng(0).normal(size=(10, 10))
        right = np.random.default_rng(1).normal(size=(10, 10))
        dest = tmp_path / "combo.png"
        save_side_by_side(left, right, str(dest), size=(50, 50))

        assert dest.exists()
        img = Image.open(dest)
        assert img.size == (50 * 2 + 12, 50)  # default gap=12

    def test_supports_mixed_modes(self, tmp_path):
        vol_slice = np.random.default_rng(0).normal(size=(10, 10))
        labels = np.zeros((10, 10), dtype=np.int16)
        labels[2:5, 2:5] = 1
        dest = tmp_path / "combo.png"
        save_side_by_side(vol_slice, labels, str(dest), size=(40, 40), right_is_labels=True)
        assert dest.exists()
