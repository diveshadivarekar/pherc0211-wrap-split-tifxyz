"""Tests for segmenting from the official surface prediction."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from src.m2_segmentation import surface_segments as ss


def _umbilicus(tmp_path, y=32.0, x=32.0):
    path = tmp_path / "umb.json"
    path.write_text(json.dumps({"control_points": [{"x": x, "y": y, "z": z, "score": 100} for z in (0, 100)]}))
    return path


def _ring_prediction(tmp_path, scale=2, size=64, depth=12):
    """Two concentric thin ring 'sheets' (a stand-in for predicted surfaces) in a (depth, size, size) volume."""
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(yy - size / 2, xx - size / 2)
    pred = np.zeros((depth, size, size), dtype=np.uint8)
    pred[:, (r > 10) & (r < 11.5)] = 255
    pred[:, (r > 20) & (r < 21.5)] = 255
    path = tmp_path / "pred.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("volume", data=pred).attrs.update({"z0": 0, "y0": 0, "x0": 0, "scale_to_native": scale})
    return path, pred


class TestWedges:
    def test_axis_interpolates_between_control_points(self, tmp_path):
        path = tmp_path / "u.json"
        path.write_text(json.dumps({"control_points": [{"x": 0, "y": 10, "z": 0}, {"x": 100, "y": 30, "z": 100}]}))
        y, x = ss.axis_at(ss.load_umbilicus(path), np.array([0.0, 50.0, 100.0, 200.0]))
        assert list(y) == [10, 20, 30, 30] and list(x) == [0, 50, 100, 100]

    def test_wedge_index_covers_the_circle(self):
        angles = np.deg2rad(np.array([5, 50, 95, 185, 275, 355]))
        w = ss.wedge_index(np.sin(angles), np.cos(angles), np.zeros(6), np.zeros(6), 90)
        assert list(w) == [0, 0, 1, 2, 3, 3]


class TestLabelPieces:
    def test_rings_are_cut_into_wedges_that_stay_connected(self, tmp_path):
        _, pred = _ring_prediction(tmp_path)
        z = np.arange(12) * 2.0
        coords = np.arange(64) * 2.0
        labels = ss.label_surface_pieces(pred > 127, z, coords, coords, ss.load_umbilicus(_umbilicus(tmp_path, 64.0, 64.0)), wedge_degrees=90, min_voxels=20)
        assert labels.max() == 8  # 2 rings x 4 quadrants
        sizes = np.bincount(labels.ravel())[1:]
        assert (np.diff(sizes) <= 0).all()  # largest first
        # each piece sits in one wedge: its voxels share a quadrant about the axis
        yy, xx = np.nonzero(labels[0])
        piece = labels[0][yy, xx]
        quad = ((yy >= 32).astype(int) * 2 + (xx >= 32).astype(int))
        for p in np.unique(piece):
            assert len(np.unique(quad[piece == p])) <= 2  # allowing a piece to straddle a boundary pixel

    def test_small_specks_are_dropped_and_empty_input_is_safe(self, tmp_path):
        mask = np.zeros((6, 20, 20), bool)
        mask[2, 5, 5] = True
        c = np.arange(20.0)
        assert ss.label_surface_pieces(mask, np.arange(6.0), c, c, ss.load_umbilicus(_umbilicus(tmp_path)), min_voxels=5).max() == 0
        assert ss.label_surface_pieces(np.zeros((3, 4, 4), bool), np.arange(3.0), np.arange(4.0), np.arange(4.0), ss.load_umbilicus(_umbilicus(tmp_path))).max() == 0


SCALE = 2.0  # native voxels per prediction voxel in the synthetic volumes below
# Rasterised toy rings jitter by about a voxel in radius from column to column, more than a real (smooth, thick)
# prediction does; the default link tolerance (0.75 voxel) was tuned on real data, so the toys loosen it.
SPLIT = {"link_tol": 1.5 * SCALE}


def _coords(n):
    return np.arange(n) * SCALE


def _rings(size=200, depth=6, radii=(40, 70), thick=3.0, bridge=True):
    """Concentric rings of realistic sheet thickness (3 voxels) 30 voxels apart, optionally joined at angle 0 by a radial bridge."""
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(yy - size / 2 + 0.5, xx - size / 2 + 0.5)
    pred = np.zeros((depth, size, size), bool)
    for r0 in radii:
        pred[:, (r >= r0) & (r < r0 + thick)] = True
    if bridge:
        pred[:, size // 2 - 1 : size // 2 + 1, size // 2 + radii[0] : size // 2 + radii[-1] + 1] = True
    return pred


class TestSplitWraps:
    SIZE = 200

    def _label(self, tmp_path, pred, split, wedge=360.0):
        n = pred.shape[1]
        c = _coords(n)
        umb = ss.load_umbilicus(_umbilicus(tmp_path, n * SCALE / 2, n * SCALE / 2))
        return ss.label_surface_pieces(pred, np.arange(pred.shape[0]) * SCALE, c, c, umb, wedge_degrees=wedge, min_voxels=50, split_wraps_apart=SPLIT if split else False)

    def test_a_bridge_merges_two_wraps_until_they_are_split(self, tmp_path):
        pred = _rings()
        assert self._label(tmp_path, pred, split=False).max() == 1
        split = self._label(tmp_path, pred, split=True)
        assert split.max() == 2
        # each label is one ring (voxels at one radius); the bridge between them is cut
        yy, xx = np.mgrid[: self.SIZE, : self.SIZE]
        r = np.hypot(yy - self.SIZE / 2 + 0.5, xx - self.SIZE / 2 + 0.5)
        for k in (1, 2):
            rk = r[split[0] == k]
            assert rk.max() - rk.min() < 5.0
        assert split[0][(r > 46) & (r < 64)].max() == 0

    def test_a_single_ring_is_not_fragmented(self, tmp_path):
        pred = _rings(radii=(40,), bridge=False)
        assert self._label(tmp_path, pred, split=False).max() == 1 and self._label(tmp_path, pred, split=True).max() == 1

    def test_a_spiral_sheet_keeps_its_wedge_pieces(self, tmp_path):
        size, depth = 240, 6
        yy, xx = np.mgrid[:size, :size]
        dy, dx = yy - size / 2 + 0.5, xx - size / 2 + 0.5
        r, th = np.hypot(dy, dx), np.arctan2(dy, dx) % (2 * np.pi)
        sheet = np.zeros((size, size), bool)
        for turn in range(3):  # Archimedean spiral, 30 voxels per turn, sheet 3 voxels thick
            sheet |= np.abs(r - (40 + 30 * (th / (2 * np.pi) + turn))) < 1.5
        pred = np.repeat(sheet[None], depth, axis=0)
        c = _coords(size)
        args = (pred, np.arange(depth) * SCALE, c, c, ss.load_umbilicus(_umbilicus(tmp_path, size * SCALE / 2, size * SCALE / 2)))
        plain = ss.label_surface_pieces(*args, wedge_degrees=90, min_voxels=50)
        split = ss.label_surface_pieces(*args, wedge_degrees=90, min_voxels=50, split_wraps_apart=SPLIT)
        assert split.max() == plain.max()
        assert abs(int((split > 0).sum()) - int((plain > 0).sum())) <= 0.03 * int((plain > 0).sum())  # only seam voxels may be lost

    def test_empty_input(self, tmp_path):
        assert self._label(tmp_path, np.zeros((4, 64, 64), bool), split=True).max() == 0

    def test_single_valued_pieces_have_one_sheet_per_ray(self, tmp_path):
        # Two wraps joined by a gentle ramp (slope below the link tolerance) are one connected piece in which a
        # ray from the axis crosses the sheet three times - no height field r = f(theta) can describe it, and a page
        # built from it floats between the layers. single_valued must leave at most one crossing per ray per piece.
        size, depth = 240, 4
        yy, xx = np.mgrid[:size, :size]
        dy, dx = yy - size / 2 + 0.5, xx - size / 2 + 0.5
        r, th = np.hypot(dy, dx), np.degrees(np.arctan2(dy, dx)) % 360
        sheet = (np.abs(r - 40) < 1.5) | (np.abs(r - 65) < 1.5)  # two full wraps
        sheet |= (np.abs(r - (40 + 25 * th / 180.0)) < 1.5) & (th < 180)  # a half-turn ramp from the inner wrap onto the outer one (gentle enough to link)
        pred = np.repeat(sheet[None], depth, axis=0)
        c = _coords(size)
        umb = ss.load_umbilicus(_umbilicus(tmp_path, size * SCALE / 2, size * SCALE / 2))
        args = (pred, np.arange(depth) * SCALE, c, c, umb)
        multi = ss.label_surface_pieces(*args, wedge_degrees=360, min_voxels=50, split_wraps_apart={**SPLIT, "single_valued": False})
        single = ss.label_surface_pieces(*args, wedge_degrees=360, min_voxels=50, split_wraps_apart=SPLIT)

        def max_crossings(labels):
            iy, ix = np.nonzero(labels[0])
            ang = np.floor(np.degrees(np.arctan2(iy - size / 2 + 0.5, ix - size / 2 + 0.5)) % 360).astype(int)
            rad, lab = np.hypot(iy - size / 2 + 0.5, ix - size / 2 + 0.5), labels[0][iy, ix]
            worst = 0
            for k in np.unique(lab):
                for b in np.unique(ang[lab == k]):
                    rb = np.sort(rad[(lab == k) & (ang == b)])
                    worst = max(worst, 1 + int((np.diff(rb) > 5).sum()))
            return worst

        assert max_crossings(multi) >= 2 and max_crossings(single) == 1
        assert int((single > 0).sum()) > 0.8 * int((multi > 0).sum())  # cutting the ramp must not eat the wraps


class TestSegmentFromPrediction:
    def test_writes_labels_at_native_resolution_aligned_to_the_tile(self, tmp_path):
        pred_path, pred = _ring_prediction(tmp_path, scale=2)
        out = tmp_path / "seg.h5"
        summary = ss.segment_from_surface_prediction(pred_path, _umbilicus(tmp_path, 64.0, 64.0), (0, 0, 0), (24, 128, 128), out, wedge_degrees=90, min_voxels=20)
        with h5py.File(out, "r") as f:
            labels = f["labels"][...]
        assert labels.shape == (24, 128, 128) and summary["n_segments"] == 8
        # labels are the prediction upsampled 2x: foreground exactly where the prediction is
        assert np.array_equal(labels[::2, ::2, ::2] > 0, pred > 127)
        assert sum(summary["voxel_counts"].values()) == int((labels > 0).sum())

    def test_sub_tile_offset_selects_the_matching_part(self, tmp_path):
        pred_path, pred = _ring_prediction(tmp_path, scale=2)
        out = tmp_path / "seg.h5"
        ss.segment_from_surface_prediction(pred_path, _umbilicus(tmp_path, 64.0, 64.0), (8, 20, 20), (8, 60, 60), out, min_voxels=5)
        with h5py.File(out, "r") as f:
            labels = f["labels"][...]
        assert np.array_equal(labels[::2, ::2, ::2] > 0, pred[4:8, 10:40, 10:40] > 127)

    def test_misaligned_or_outside_tiles_are_rejected(self, tmp_path):
        pred_path, _ = _ring_prediction(tmp_path, scale=2)
        umb = _umbilicus(tmp_path)
        with pytest.raises(ValueError, match="multiples"):
            ss.segment_from_surface_prediction(pred_path, umb, (1, 0, 0), (8, 8, 8), tmp_path / "a.h5")
        with pytest.raises(ValueError, match="outside"):
            ss.segment_from_surface_prediction(pred_path, umb, (0, 0, 0), (400, 8, 8), tmp_path / "a.h5")
