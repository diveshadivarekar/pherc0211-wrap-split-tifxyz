"""Tests for M3: cylindrical unwrap + ARAP relaxation.

Covers SRS Test Plan T5 (mean edge-length distortion within tolerance) and
FR-3.4 (3D<->2D coordinate traceability) at PoC scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.m1_ingestion.generate_phantom import generate_phantom_full
from src.m1_ingestion.preprocessing import correct_chunk
from src.m2_segmentation.segment import segment_volume
from src.m3_flattening.flatten import arap_relax, cylindrical_unwrap, flatten_layer

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
N_THETA = 180


@pytest.fixture(scope="module")
def segmented():
    raw, ground_truth, layer_id, ink_mask = generate_phantom_full(CFG)
    corrected = correct_chunk(raw)
    labels = segment_volume(corrected)
    return corrected, labels, ink_mask


class TestCylindricalUnwrap:
    def test_page_and_coord_map_shapes(self, segmented):
        corrected, labels, _ = segmented
        page, coord_map = cylindrical_unwrap(corrected, labels, layer_id=1, n_theta=N_THETA)
        assert page.shape == (corrected.shape[0], N_THETA)
        assert coord_map.shape == (corrected.shape[0], N_THETA, 3)

    def test_coord_map_indices_are_within_volume_bounds(self, segmented):
        corrected, labels, _ = segmented
        _, coord_map = cylindrical_unwrap(corrected, labels, layer_id=1, n_theta=N_THETA)
        valid = ~np.isnan(coord_map[..., 0])
        z, y, x = coord_map[valid, 0], coord_map[valid, 1], coord_map[valid, 2]
        assert (z >= 0).all() and (z < corrected.shape[0]).all()
        assert (y >= -1).all() and (y < corrected.shape[1] + 1).all()
        assert (x >= -1).all() and (x < corrected.shape[2] + 1).all()

    def test_every_depth_slice_is_covered(self, segmented):
        """A segment spanning the full depth must produce a valid coord_map row per slice."""
        corrected, labels, _ = segmented
        _, coord_map = cylindrical_unwrap(corrected, labels, layer_id=1, n_theta=N_THETA)
        covered = ~np.isnan(coord_map[:, 0, 0])
        assert covered.all()

    def test_ink_visible_in_flattened_page(self, segmented):
        """Sanity: a layer known (from ink_mask ground truth) to carry ink
        should show elevated density somewhere in its flattened page."""
        corrected, labels, ink_mask = segmented
        for layer_id in range(1, int(labels.max()) + 1):
            if ink_mask[labels == layer_id].sum() > 0:
                page, _ = cylindrical_unwrap(corrected, labels, layer_id, n_theta=N_THETA)
                assert page.max() > page.mean() + page.std()
                return
        pytest.skip("no inked layer in this phantom config")


class TestArapRelax:
    def test_distortion_decreases(self, segmented):
        corrected, labels, _ = segmented
        _, coord_map = cylindrical_unwrap(corrected, labels, layer_id=1, n_theta=N_THETA)
        _, before, after = arap_relax(coord_map, n_iters=5)
        assert after < before

    def test_distortion_near_zero_for_true_cylinder(self, segmented):
        """T5: this phantom's segments are true cylinders (developable
        surfaces), so ARAP-relaxed edge-length distortion should be tiny."""
        corrected, labels, _ = segmented
        _, coord_map = cylindrical_unwrap(corrected, labels, layer_id=1, n_theta=N_THETA)
        _, _, after = arap_relax(coord_map, n_iters=5)
        assert after < 0.01

    def test_uv_shape(self, segmented):
        corrected, labels, _ = segmented
        _, coord_map = cylindrical_unwrap(corrected, labels, layer_id=1, n_theta=N_THETA)
        uv, _, _ = arap_relax(coord_map, n_iters=5)
        assert uv.shape == (coord_map.shape[0], coord_map.shape[1], 2)


class TestFlattenLayer:
    def test_returns_expected_keys(self, segmented):
        corrected, labels, _ = segmented
        result = flatten_layer(corrected, labels, layer_id=1, n_theta=N_THETA, arap_iters=5)
        assert set(result.keys()) == {"page", "coord_map", "uv", "distortion_before", "distortion_after", "method"}

    def test_all_segments_flatten_without_error(self, segmented):
        corrected, labels, _ = segmented
        for layer_id in range(1, int(labels.max()) + 1):
            result = flatten_layer(corrected, labels, layer_id, n_theta=N_THETA, arap_iters=3)
            assert result["page"].shape[0] == corrected.shape[0]
            assert result["distortion_after"] <= result["distortion_before"] + 1e-9


class TestPipelineRun:
    def test_run_end_to_end(self, tmp_path):
        import h5py

        from src.m3_flattening.run import run

        raw, ground_truth, layer_id, ink_mask = generate_phantom_full(CFG)
        corrected = correct_chunk(raw)
        labels = segment_volume(corrected)

        corrected_path = tmp_path / "phantom_corrected.h5"
        with h5py.File(corrected_path, "w") as f:
            f.create_dataset("volume", data=corrected)

        labels_path = tmp_path / "phantom_segmented.h5"
        with h5py.File(labels_path, "w") as f:
            f.create_dataset("labels", data=labels)

        out_path = run(corrected_path, labels_path, tmp_path, n_theta=N_THETA, arap_iters=3)
        assert out_path.exists()

        with h5py.File(out_path, "r") as f:
            assert len(f.keys()) == int(labels.max())
            for grp_name in f.keys():
                grp = f[grp_name]
                assert "page" in grp and "coord_map" in grp and "uv" in grp
                assert grp.attrs["distortion_after"] <= grp.attrs["distortion_before"] + 1e-9


class TestPatchUnwrap:
    """Real-scan segments are irregular sheet patches, not centred rings."""

    @staticmethod
    def _tilted_sheet():
        # a gently curved sheet, tilted off every axis, carrying a stripe pattern along one in-plane direction
        shape = (40, 80, 80)
        vol = np.zeros(shape, dtype=np.float32)
        labels = np.zeros(shape, dtype=np.int16)
        for y in range(10, 70):
            for x in range(10, 70):
                z = int(round(8 + 0.35 * y + 0.1 * x + 3 * np.sin(x / 15.0)))
                if 1 <= z < shape[0] - 1:
                    labels[z, y, x] = 1
                    vol[z, y, x] = 0.4 + 0.5 * ((x // 6) % 2)
        return vol, labels

    def test_picks_patch_for_a_non_ring(self):
        vol, labels = self._tilted_sheet()
        assert flatten_layer(vol, labels, 1)["method"] == "patch"

    def test_page_is_valid_and_traceable(self):
        vol, labels = self._tilted_sheet()
        result = flatten_layer(vol, labels, 1)
        coords = result["coord_map"]
        valid = ~np.isnan(coords[..., 0])
        assert valid.mean() > 0.5
        # every valid pixel maps back to a point within ~1.5 voxels of a segment voxel
        from scipy.ndimage import distance_transform_edt

        dist = distance_transform_edt(labels != 1)
        pts = np.round(coords[valid]).astype(int)
        assert (dist[pts[:, 0], pts[:, 1], pts[:, 2]] < 1.5).mean() > 0.95

    def test_stripes_survive_flattening(self):
        vol, labels = self._tilted_sheet()
        page = flatten_layer(vol, labels, 1)["page"]
        # the flattened page keeps the on/off stripe contrast instead of blurring it away
        assert page.max() > 0.7 and (page[page > 0].std() > 0.1)

    def test_distortion_is_low_and_finite(self):
        vol, labels = self._tilted_sheet()
        result = flatten_layer(vol, labels, 1)
        assert np.isfinite(result["distortion_after"]) and result["distortion_after"] < 0.15

    def test_bbox_crop_gives_same_absolute_coordinates(self):
        from scipy import ndimage as ndi

        vol, labels = self._tilted_sheet()
        full = flatten_layer(vol, labels, 1, method="patch")
        box = ndi.find_objects(labels)[0]
        cropped = flatten_layer(vol, labels, 1, bbox=box, method="patch")
        a, b = full["coord_map"], cropped["coord_map"]
        assert np.nanmax(np.abs(np.nanmean(a, axis=(0, 1)) - np.nanmean(b, axis=(0, 1)))) < 1.0

    def test_tiny_segment_does_not_crash(self):
        vol = np.zeros((6, 6, 6), dtype=np.float32)
        labels = np.zeros((6, 6, 6), dtype=np.int16)
        labels[2, 2, 2] = 1
        result = flatten_layer(vol, labels, 1, method="patch")
        assert np.isnan(result["coord_map"]).all()

    def test_arap_survives_a_gap_in_the_middle(self):
        coord_map = np.stack(np.meshgrid(np.arange(6.0), np.arange(8.0), np.zeros(1), indexing="ij"), axis=-1)[:, :, 0, :]
        coord_map = np.concatenate([coord_map, np.zeros((6, 8, 1))], axis=-1)[..., :3].astype(np.float32)
        coord_map[2:4, :, :] = np.nan
        uv, before, after = arap_relax(coord_map, n_iters=3)
        assert np.isfinite(uv).all() and np.isfinite(after)


class TestMemoryAwareRun:
    def _files(self, tmp_path):
        import h5py

        raw, _, _, _ = generate_phantom_full(CFG)
        corrected = correct_chunk(raw)
        labels = segment_volume(corrected)
        with h5py.File(tmp_path / "scan_corrected.h5", "w") as f:
            f.create_dataset("volume", data=corrected)
        with h5py.File(tmp_path / "scan_segmented.h5", "w") as f:
            f.create_dataset("labels", data=labels, chunks=(8, 96, 96))
        return labels

    def test_label_stats_match_numpy(self, tmp_path):
        import h5py
        from scipy import ndimage as ndi

        from src.m3_flattening.run import _label_stats

        labels = self._files(tmp_path)
        with h5py.File(tmp_path / "scan_segmented.h5", "r") as f:
            n, sizes, boxes = _label_stats(f["labels"], slab=5)
        assert n == labels.max()
        np.testing.assert_array_equal(sizes, np.bincount(labels.ravel(), minlength=n + 1))
        assert boxes == [tuple(b) for b in ndi.find_objects(labels)]

    def test_disk_backed_path_gives_same_pages_as_in_memory(self, tmp_path, monkeypatch):
        import h5py

        import src.m3_flattening.run as m3run

        self._files(tmp_path)
        in_mem = m3run.run(tmp_path / "scan_corrected.h5", tmp_path / "scan_segmented.h5", tmp_path / "a", n_theta=N_THETA, arap_iters=2)
        monkeypatch.setattr(m3run, "available_memory_gb", lambda: 0.0)  # force reading boxes from disk
        on_disk = m3run.run(tmp_path / "scan_corrected.h5", tmp_path / "scan_segmented.h5", tmp_path / "b", n_theta=N_THETA, arap_iters=2)
        with h5py.File(in_mem, "r") as a, h5py.File(on_disk, "r") as b:
            assert list(a.keys()) == list(b.keys()) and a.keys()
            for k in a:
                np.testing.assert_allclose(a[k]["page"][...], b[k]["page"][...], atol=1e-6)


class TestSheetUnwrap:
    """Rolled-scroll sheets: the cylindrical height-field builder vs the plane builder."""

    @staticmethod
    def _arc(radius=40.0, half_angle_deg=75.0, depth=24, size=140, thick=3.0):
        yy, xx = np.mgrid[:size, :size]
        dy, dx = yy - size / 2, xx - size / 2
        r, th = np.hypot(dy, dx), np.degrees(np.arctan2(dy, dx))
        wall = (np.abs(r - radius) < thick / 2) & (np.abs(th) < half_angle_deg)
        labels = np.repeat(wall[None], depth, axis=0).astype(np.int32)
        vol = np.repeat(np.where(wall, 200.0, 20.0)[None], depth, axis=0).astype(np.float32)
        axis = (np.array([0.0, depth]), np.full(2, size / 2), np.full(2, size / 2))  # scroll axis along z through the centre
        return vol, labels, axis, size

    def test_a_strongly_curved_piece_stays_on_the_wall(self):
        vol, labels, axis, size = self._arc()
        res = flatten_layer(vol, labels, 1, method="sheet", axis=axis)
        cm = res["coord_map"]
        v = ~np.isnan(cm[..., 0])
        r = np.hypot(cm[..., 1] - size / 2, cm[..., 2] - size / 2)[v]
        assert abs(np.median(r) - 40.0) < 0.6 and np.percentile(np.abs(r - 40.0), 99) < 2.0
        # spacing between neighbouring page pixels is about one voxel along both directions
        step_c = np.linalg.norm(np.diff(cm, axis=1), axis=-1)
        step_r = np.linalg.norm(np.diff(cm, axis=0), axis=-1)
        assert abs(np.nanmedian(step_c) - 1.0) < 0.1 and abs(np.nanmedian(step_r) - 1.0) < 0.1
        assert res["method"] == "sheet" and res["page"][v].mean() > 150  # the page samples the bright wall

    def test_sheet_needs_the_axis_and_auto_does_not_pick_it(self):
        vol, labels, axis, _ = self._arc()
        with pytest.raises(ValueError, match="axis"):
            flatten_layer(vol, labels, 1, method="sheet")
        assert flatten_layer(vol, labels, 1, method="auto", axis=axis)["method"] in ("patch", "cylinder")
