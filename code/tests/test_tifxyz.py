"""Tests for tifxyz writing/reading and the surface QA checks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src import tifxyz as tx
from src.tifxyz.qa import VolumeRef, vertex_normals

AX, AY = 100.0, 100.0  # scroll axis position in the synthetic volume (x, y)


def cylinder_coord_map(r=40.0, rows=40, cols=120, z0=10.0, theta0=0.0, ramp=None):
    """An M3-style coord_map (z, y, x) of a cylinder wall around the axis, 1-voxel grid spacing.

    ``ramp`` = (theta_start, theta_end, dr) lifts the radius by ``dr`` over that angular range and
    keeps it lifted afterwards - a sheet that has switched onto the next wrap.
    """
    th = theta0 + np.arange(cols) / r
    zz = z0 + np.arange(rows, dtype=np.float64)
    theta, z = np.meshgrid(th, zz)
    rad = np.full_like(theta, r)
    if ramp is not None:
        a, b, dr = ramp
        rad = rad + dr * np.clip((theta - a) / (b - a), 0, 1)
    return np.stack([z, AY + rad * np.sin(theta), AX + rad * np.cos(theta)], axis=-1).astype(np.float32)


UMBILICUS = (np.array([0.0, 200.0]), np.array([AY, AY]), np.array([AX, AX]))


class TestRoundTrip:
    def test_writes_the_documented_layout_and_reads_it_back(self, tmp_path):
        cm = cylinder_coord_map()
        cm[:5, :8] = np.nan  # a corner outside the page
        s = tx.from_coord_map(cm, offset_zyx=(1000, 0, 0), uuid="abc")
        out = tx.write_tifxyz(tmp_path / "seg", s, voxel_size_um=2.399)

        assert {p.name for p in out.iterdir()} == {"x.tif", "y.tif", "z.tif", "mask.tif", "meta.json"}
        meta = json.loads((out / "meta.json").read_text())
        assert meta["format"] == "tifxyz" and meta["uuid"] == "abc" and meta["type"] == "seg"
        assert meta["scale"] == pytest.approx([1.0, 1.0], rel=0.02)  # 1-voxel grid -> 1 cell per voxel, never the step
        assert meta["bbox"][0][2] >= 1010 and meta["area_vx2"] > 0 and meta["area_cm2"] == pytest.approx(meta["area_vx2"] * (2.399e-4) ** 2)

        back = tx.read_tifxyz(out)
        assert back.shape == s.shape and np.array_equal(back.valid, s.valid)
        assert np.allclose(back.xyz[s.valid], s.xyz[s.valid])
        assert np.allclose(tx.to_coord_map(s)[s.valid], cm[s.valid] + np.array([1000, 0, 0], np.float32))

    def test_invalid_vertices_are_written_as_the_sentinel_and_z_is_positive_for_valid_ones(self, tmp_path):
        import tifffile

        cm = cylinder_coord_map()
        cm[0, 0] = np.nan
        out = tx.write_tifxyz(tmp_path / "s", tx.from_coord_map(cm, offset_zyx=(500, 0, 0)))
        xyz = np.stack([tifffile.imread(out / f"{c}.tif") for c in "xyz"], axis=-1)
        assert xyz.dtype == np.float32 and (xyz[0, 0] == -1).all()
        assert (xyz[1:, 1:, 2] > 0).all()

    def test_refuses_to_clobber_and_rejects_nonpositive_z(self, tmp_path):
        s = tx.from_coord_map(cylinder_coord_map(), offset_zyx=(100, 0, 0))
        tx.write_tifxyz(tmp_path / "s", s)
        with pytest.raises(FileExistsError):
            tx.write_tifxyz(tmp_path / "s", s)
        tx.write_tifxyz(tmp_path / "s", s, overwrite=True)
        low = cylinder_coord_map(z0=-5.0)  # tile-local coordinates reaching below z = 0
        with pytest.raises(ValueError, match="offset_zyx"):
            tx.from_coord_map(low)

    def test_stride_coarsens_the_grid_and_the_scale_follows(self):
        s = tx.from_coord_map(cylinder_coord_map(), offset_zyx=(100, 0, 0), stride=4)
        assert s.shape == (10, 30) and s.scale[0] == pytest.approx(0.25, rel=0.03) and s.scale[1] == pytest.approx(0.25, rel=0.03)

    def test_reader_honours_an_integer_multiple_mask(self, tmp_path):
        import tifffile

        out = tx.write_tifxyz(tmp_path / "s", tx.from_coord_map(cylinder_coord_map(), offset_zyx=(100, 0, 0)))
        mask = np.full((80, 240), 255, np.uint8)
        mask[:2, :2] = 0  # kills grid vertex (0, 0) only
        tifffile.imwrite(out / "mask.tif", mask)
        back = tx.read_tifxyz(out)
        assert not back.valid[0, 0] and back.valid[0, 1] and back.valid[1, 0]


class TestContract:
    def _write(self, tmp_path, **meta_override):
        out = tx.write_tifxyz(tmp_path / "s", tx.from_coord_map(cylinder_coord_map(), offset_zyx=(100, 0, 0)))
        meta = json.loads((out / "meta.json").read_text())
        meta.update(meta_override)
        (out / "meta.json").write_text(json.dumps(meta))
        return out

    def test_clean_surface_has_no_findings(self, tmp_path):
        assert tx.check_contract(self._write(tmp_path)) == []

    def test_writing_the_step_instead_of_the_scale_is_caught(self, tmp_path):
        codes = {i["code"] for i in tx.check_contract(self._write(tmp_path, scale=[20, 20]))}
        assert {"scale-is-step", "scale-mismatch"} <= codes

    def test_missing_file_and_wrong_format(self, tmp_path):
        out = self._write(tmp_path, format="other")
        assert {i["code"] for i in tx.check_contract(out)} == {"format"}
        (out / "z.tif").unlink()
        assert tx.check_contract(out)[0]["code"] == "missing-file"


class TestGeometry:
    def test_clean_cylinder(self):
        rep = tx.geometry_report(tx.from_coord_map(cylinder_coord_map(), offset_zyx=(100, 0, 0)))
        assert rep["n_components"] == 1 and rep["n_enclosed_holes"] == 0 and rep["folded_cells"] == 0
        assert rep["edge_ratio"]["p50"] == pytest.approx(1.0, rel=0.02) and rep["cell_shear_cos_p95"] < 0.05

    def test_holes_and_components(self):
        cm = cylinder_coord_map()
        cm[15:20, 40:50] = np.nan  # enclosed hole
        cm[:, 90:95] = np.nan      # cuts the page in two
        rep = tx.geometry_report(tx.from_coord_map(cm, offset_zyx=(100, 0, 0)))
        assert rep["n_enclosed_holes"] == 1 and rep["enclosed_hole_cells"] == 50 and rep["n_components"] == 2

    def test_a_flipped_patch_is_a_fold(self):
        cm = cylinder_coord_map()
        patch = cm[10:20, 40:60].copy()
        cm[10:20, 40:60] = patch[:, ::-1]  # mirror the patch in place: its orientation reverses
        s = tx.from_coord_map(cm, offset_zyx=(100, 0, 0), scale=(1.0, 1.0))
        assert tx.geometry_report(s)["folded_cells"] > 50


class TestRadiality:
    def test_cylinder_wall_is_radial(self):
        s = tx.from_coord_map(cylinder_coord_map(), offset_zyx=(100, 0, 0))
        rep, maps = tx.radiality(s, UMBILICUS)
        assert rep["cross_wrap_fraction"] < 0.01 and rep["angle_median_deg"] < 5
        assert maps["cross_wrap"].shape == s.shape

    def test_a_plane_containing_the_axis_cuts_across_wraps(self):
        zz, rr = np.meshgrid(np.arange(40.0) + 10, np.arange(30.0) + 20)
        cm = np.stack([zz, np.full_like(zz, AY) + 0.0, AX + rr], axis=-1).astype(np.float32)  # y = const, spans radius
        rep, _ = tx.radiality(tx.from_coord_map(cm, offset_zyx=(100, 0, 0)), UMBILICUS)
        assert rep["cross_wrap_fraction"] > 0.95

    def test_a_ramp_onto_the_next_wrap_is_localised(self):
        ramp = (0.9, 1.1, 14.0)  # a steep (~35 deg off the cylinder wall) switch to r + 14 between theta 0.9 and 1.1 rad
        cm = cylinder_coord_map(r=40, cols=160, ramp=ramp)
        rep, maps = tx.radiality(tx.from_coord_map(cm, offset_zyx=(100, 0, 0)), UMBILICUS, max_angle_deg=25)
        cols = np.flatnonzero(maps["cross_wrap"].any(axis=0))
        assert 0 < rep["cross_wrap_fraction"] < 0.35
        th = np.arange(160) / 40.0
        assert th[cols].min() > 0.7 and th[cols].max() < 1.4  # flagged only around the ramp, not on the flat parts


class TestWrapsCrossed:
    def test_a_plane_across_wraps_spans_several_a_single_wall_spans_one_or_less(self):
        # three concentric walls, 10 voxels apart
        pred = VolumeRef(np.maximum.reduce([prediction_cylinder(r=r) for r in (30.0, 40.0, 50.0)]))
        spacing = tx.estimate_wrap_spacing(pred, UMBILICUS, z=120.0, r_max=80.0)
        assert spacing == pytest.approx(10.0, abs=1.5)

        wall = tx.from_coord_map(cylinder_coord_map(r=40.0, cols=60), offset_zyx=(100, 0, 0))
        zz, rr = np.meshgrid(np.arange(40.0) + 10, np.arange(30.0) + 25)
        plane = tx.from_coord_map(np.stack([zz, np.full_like(zz, AY), AX + rr], axis=-1).astype(np.float32), offset_zyx=(100, 0, 0))
        assert tx.radial_extent(wall, UMBILICUS, spacing)["wraps_crossed"] < 0.5
        assert tx.radial_extent(plane, UMBILICUS, spacing)["wraps_crossed"] > 2.0


def test_a_sheet_that_leans_in_radius_is_still_one_wrap_per_slice():
    # a single sheet whose radius grows 30 voxels over its height: total spread is large, per-slice spread is not
    rows, cols = 120, 40
    zz, cc = np.meshgrid(np.arange(rows, dtype=np.float64) + 10, np.arange(cols, dtype=np.float64))
    r = 30.0 + 0.25 * (zz - 10)
    th = cc / 40.0
    cm = np.stack([zz, AY + r * np.sin(th), AX + r * np.cos(th)], axis=-1).astype(np.float32)
    ext = tx.radial_extent(tx.from_coord_map(cm, offset_zyx=(100, 0, 0)), UMBILICUS, spacing=10.0)
    assert ext["wraps_crossed_total"] > 2.5 and ext["wraps_crossed"] < 0.6


def test_wraps_crossed_is_flagged_unreliable_for_long_pages():
    short = tx.from_coord_map(cylinder_coord_map(r=40, cols=60), offset_zyx=(100, 0, 0))            # about 85 degrees
    long = tx.from_coord_map(cylinder_coord_map(r=40, cols=200), offset_zyx=(100, 0, 0))            # about 285 degrees
    a, b = tx.radial_extent(short, UMBILICUS, spacing=10.0), tx.radial_extent(long, UMBILICUS, spacing=10.0)
    assert 60 < a["angular_span_deg"] < 90 and a["wraps_crossed_reliable"]
    assert b["angular_span_deg"] > 200 and not b["wraps_crossed_reliable"]


def prediction_cylinder(r=40.0, half_width=1.5, shape=(200, 200, 200), value=255.0):
    """Volume with a thin cylinder wall at radius ``r`` around the axis (voxel centres at i + 0.5)."""
    zz, yy, xx = (np.arange(n) + 0.5 for n in shape)
    rad = np.hypot(yy[None, :, None] - AY, xx[None, None, :] - AX)
    wall = np.exp(-0.5 * ((rad - r) / (half_width / 1.5)) ** 2)
    return (wall * value).astype(np.float32) * np.ones((shape[0], 1, 1), np.float32)


class TestVolumeAgreement:
    def test_a_drifted_mesh_is_detected_and_snapped_onto_the_prediction(self):
        pred = VolumeRef(prediction_cylinder(r=40.0))
        drifted = tx.from_coord_map(cylinder_coord_map(r=44.0, cols=150), offset_zyx=(100, 0, 0))
        before, maps = tx.surface_support(drifted, pred, search=8)
        assert before["on_surface_fraction"] < 0.05 and before["abs_drift_median"] == pytest.approx(4.0, abs=1.0)
        assert np.isfinite(maps["drift"]).any()

        snapped, move = tx.snap_to_prediction(drifted, pred, search=8)
        after, _ = tx.surface_support(snapped, pred, search=8)
        assert after["on_surface_fraction"] > 0.9 and move["abs_shift_median"] == pytest.approx(4.0, abs=1.0)
        r = np.hypot(snapped.xyz[..., 1] - AY, snapped.xyz[..., 0] - AX)[snapped.valid]
        assert np.median(r) == pytest.approx(40.0, abs=0.8)

    def test_ct_centering_improves_when_the_mesh_moves_onto_the_ridge(self):
        ct = VolumeRef(prediction_cylinder(r=40.0, half_width=2.5, value=200.0) + 30.0)
        drifted = tx.from_coord_map(cylinder_coord_map(r=43.0, cols=150), offset_zyx=(100, 0, 0))
        pred = VolumeRef(prediction_cylinder(r=40.0))
        snapped, _ = tx.snap_to_prediction(drifted, pred, search=8)
        kw = dict(search=8.0, step=0.5, smooth_vox=1.0, centred_within=1.5)  # the synthetic ridge is thin, unlike a real 2.4 um sheet
        b, a = tx.ct_centering(drifted, ct, **kw), tx.ct_centering(snapped, ct, **kw)
        assert b["centred_fraction"] < 0.2 and a["centred_fraction"] > 0.9
        assert a["contrast"] > b["contrast"]

    def test_points_outside_the_volume_are_nan_not_zero(self):
        v = VolumeRef(np.ones((10, 10, 10), np.float32))
        out = v.sample(np.array([[5.0, 5.0, 5.0], [50.0, 5.0, 5.0]]))
        assert out[0] == pytest.approx(1.0) and np.isnan(out[1])


def test_normals_are_unit_and_perpendicular_to_the_cylinder_axis():
    s = tx.from_coord_map(cylinder_coord_map(), offset_zyx=(100, 0, 0))
    n = vertex_normals(s)
    v = s.valid
    assert np.allclose(np.linalg.norm(n[v], axis=-1), 1.0, atol=1e-6)
    assert np.abs(n[v][:, 2]).max() < 0.05  # normals of a vertical cylinder lie in the xy plane
