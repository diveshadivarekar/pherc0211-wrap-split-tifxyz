"""Tests for src/spiral_fitting/ — the diffeomorphic spiral-fitting module.

Uses a genuine synthetic Archimedean spiral phantom (not the project's main
concentric-ring phantom, which is the wrong test case here — see
features.py's WindingPair docstring for why a true spiral is topologically
one connected component per slice, unlike disjoint concentric rings).

Convergence tests use generous tolerances and short iteration counts:
FitConfig.__doc__ documents that full convergence needs thousands of
iterations, visibly still improving at that point rather than plateaued.
These tests check the model moves correctly and produces sane output, not
that it fully converges in test-suite time.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

torch = pytest.importorskip("torch", reason="spiral_fitting tests require torch")


def make_spiral_volume(
    shape=(12, 80, 80), omega_true=0.5, n_turns=3, thickness=1.5, noise=0.02, seed=0
) -> np.ndarray:
    """A real Archimedean spiral cross-section (parametrically sampled,
    then thickened via distance transform), extruded along z."""
    depth, h, w = shape
    cy, cx = (h - 1) / 2, (w - 1) / 2
    rng = np.random.default_rng(seed)

    theta = np.linspace(1e-3, n_turns * 2 * np.pi, 8000)
    r = theta / omega_true
    px = cx + r * np.cos(theta)
    py = cy - r * np.sin(theta)

    curve_mask = np.zeros((h, w), dtype=bool)
    valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    curve_mask[py[valid].astype(int), px[valid].astype(int)] = True

    dist = distance_transform_edt(~curve_mask)
    cross_section = (dist < thickness / 2).astype(np.float32) * 0.8

    volume = np.repeat(cross_section[np.newaxis, :, :], depth, axis=0).copy()
    if noise > 0:
        volume += rng.normal(0, noise, size=volume.shape).astype(np.float32)
    return volume.astype(np.float32)


class TestCanonicalSpiral:
    def test_spiral_point_radius_matches_theta_over_omega(self):
        from src.spiral_fitting.canonical import spiral_point

        theta = torch.tensor([1.0, 2.0, 3.0])
        z = torch.zeros(3)
        omega = torch.tensor(0.5)
        pts = spiral_point(theta, z, omega)
        radii = torch.linalg.norm(pts[:, :2], dim=-1)
        torch.testing.assert_close(radii, theta / omega, atol=1e-5, rtol=1e-5)

    def test_winding_radius_constant_within_one_winding(self):
        """theta_principal (from arctan2) wraps at odd multiples of pi, not
        at multiples of 2*pi — verified interactively while building this
        module — so a "within one winding" range must avoid crossing pi,
        not just stay under a full 2*pi turn."""
        from src.spiral_fitting.canonical import point_to_winding_radius, spiral_point

        omega = torch.tensor(0.4)
        theta = torch.linspace(0.1, torch.pi - 0.1, 50)
        z = torch.zeros_like(theta)
        pts = spiral_point(theta, z, omega)
        r = point_to_winding_radius(pts[:, :2], omega)
        assert r.std() < 1e-3

    def test_winding_radius_jumps_by_spacing_between_windings(self):
        from src.spiral_fitting.canonical import point_to_winding_radius, spiral_point, winding_spacing

        omega = torch.tensor(0.4)
        theta = torch.tensor([0.5, 0.5 + 2 * torch.pi])
        z = torch.zeros(2)
        pts = spiral_point(theta, z, omega)
        r = point_to_winding_radius(pts[:, :2], omega)
        torch.testing.assert_close(r[1] - r[0], winding_spacing(omega), atol=1e-3, rtol=1e-3)

    def test_nearest_winding_offset_zero_on_winding(self):
        from src.spiral_fitting.canonical import nearest_winding_offset, winding_spacing

        omega = torch.tensor(0.4)
        spacing = winding_spacing(omega)
        r = 2 * spacing  # exactly on a winding
        offset = nearest_winding_offset(r, omega)
        assert offset.abs().item() < 1e-3


class TestFlowFieldDiffeomorphism:
    def test_forward_inverse_round_trip(self):
        from src.spiral_fitting.diffeomorphism import FlowFieldDiffeomorphism

        torch.manual_seed(0)
        diffeo = FlowFieldDiffeomorphism(grid_shape=(4, 6, 6), bounds_min=(-10, -10, 0), bounds_max=(10, 10, 20))
        with torch.no_grad():
            diffeo.velocity.mul_(50)  # meaningfully deform, not near-identity

        pts = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 5.0], [-3.0, 4.0, 10.0]])
        forward = diffeo(pts)
        back = diffeo.inverse(forward)
        torch.testing.assert_close(back, pts, atol=1e-3, rtol=1e-3)

    def test_near_identity_at_init(self):
        from src.spiral_fitting.diffeomorphism import FlowFieldDiffeomorphism

        diffeo = FlowFieldDiffeomorphism(grid_shape=(4, 4, 4), bounds_min=(0, 0, 0), bounds_max=(10, 10, 10))
        pts = torch.tensor([[5.0, 5.0, 5.0]])
        out = diffeo(pts)
        assert (out - pts).abs().max().item() < 0.1


class TestFeatureExtraction:
    def test_true_spiral_gives_one_path_per_slice(self):
        """A genuine spiral is topologically one connected curve per slice
        — this is the key structural difference from the project's main
        concentric-ring phantom, and why WindingPair (ray-casting) exists
        instead of same-slice multi-path ranking."""
        from src.spiral_fitting.features import extract_surface_paths

        vol = make_spiral_volume(shape=(4, 80, 80), n_turns=3)
        paths, pairs = extract_surface_paths(vol, threshold=0.4, z_stride=1)

        by_slice: dict[int, int] = {}
        for p in paths:
            by_slice[p.slice_z] = by_slice.get(p.slice_z, 0) + 1
        assert all(count == 1 for count in by_slice.values())

    def test_winding_pairs_are_found(self):
        from src.spiral_fitting.features import extract_surface_paths

        vol = make_spiral_volume(shape=(4, 80, 80), n_turns=3)
        paths, pairs = extract_surface_paths(vol, threshold=0.4, z_stride=1)
        assert len(pairs) > 0

    def test_estimate_initial_omega_close_to_ground_truth(self):
        """The closed-form estimator (no gradient descent) is the fix that
        actually mattered for convergence — verified interactively to land
        within ~1% of ground truth; this pins a looser 10% bound so the
        test isn't brittle to the phantom's random noise/thickness."""
        from src.spiral_fitting.features import estimate_initial_omega, extract_surface_paths

        omega_true = 0.5
        vol = make_spiral_volume(shape=(8, 128, 128), omega_true=omega_true, n_turns=4)
        paths, pairs = extract_surface_paths(vol, threshold=0.4, z_stride=1)

        estimate = estimate_initial_omega(pairs)
        assert estimate is not None
        assert abs(estimate - omega_true) / omega_true < 0.10

    def test_estimate_initial_omega_none_when_no_pairs(self):
        from src.spiral_fitting.features import estimate_initial_omega

        assert estimate_initial_omega([]) is None

    def test_winding_pair_outer_radius_exceeds_inner(self):
        """Sanity check on the ray-cast direction: the 'outer' point found
        by casting outward should genuinely be farther from centre."""
        from src.spiral_fitting.features import extract_surface_paths

        vol = make_spiral_volume(shape=(4, 80, 80), n_turns=3)
        paths, pairs = extract_surface_paths(vol, threshold=0.4, z_stride=1)
        cy, cx = (vol.shape[1] - 1) / 2, (vol.shape[2] - 1) / 2

        farther_count = 0
        for pair in pairs:
            r_inner = np.hypot(pair.inner_point[1] - cy, pair.inner_point[2] - cx)
            r_outer = np.hypot(pair.outer_point[1] - cy, pair.outer_point[2] - cx)
            farther_count += r_outer > r_inner
        assert farther_count / len(pairs) > 0.9

    def test_normals_point_outward(self):
        from src.spiral_fitting.features import extract_surface_paths

        vol = make_spiral_volume(shape=(4, 80, 80), n_turns=3)
        paths, _ = extract_surface_paths(vol, threshold=0.4, z_stride=1)
        cy, cx = (vol.shape[1] - 1) / 2, (vol.shape[2] - 1) / 2

        path = paths[0]
        radial = path.points[:, 1:] - np.array([cy, cx])
        radial_unit = radial / (np.linalg.norm(radial, axis=-1, keepdims=True) + 1e-8)
        cos_sim = (radial_unit * path.normals[:, 1:]).sum(axis=-1)
        assert cos_sim.mean() > 0.5  # normals broadly agree with radial-outward


class TestFitSpiral:
    @pytest.fixture(scope="class")
    def small_spiral_fit(self):
        from src.spiral_fitting.features import extract_surface_paths
        from src.spiral_fitting.fit import FitConfig, fit_spiral

        vol = make_spiral_volume(shape=(6, 64, 64), omega_true=0.6, n_turns=2)
        paths, pairs = extract_surface_paths(vol, threshold=0.4, z_stride=1)
        config = FitConfig(n_iterations=60, grid_shape=(3, 4, 4), n_euler_steps=3, paths_per_batch=4)
        model, history = fit_spiral(paths, pairs, vol.shape, initial_omega=0.4, config=config, progress=False)
        return model, history, vol

    def test_raises_with_too_few_paths(self):
        from src.spiral_fitting.fit import fit_spiral

        with pytest.raises(ValueError):
            fit_spiral([], [], (10, 20, 20), initial_omega=0.5)

    def test_loss_history_recorded(self, small_spiral_fit):
        _, history, _ = small_spiral_fit
        assert len(history["loss"]) == 60
        assert len(history["omega"]) == 60
        assert all(np.isfinite(history["loss"]))

    def test_omega_moves_toward_true_value(self, small_spiral_fit):
        """Not full convergence (see FitConfig's docstring) — just checks
        omega moved in the right direction from a deliberately-off start."""
        _, history, _ = small_spiral_fit
        true_omega = 0.6
        start_error = abs(0.4 - true_omega)
        end_error = abs(history["omega"][-1] - true_omega)
        assert end_error < start_error

    def test_omega_stays_positive(self, small_spiral_fit):
        model, _, _ = small_spiral_fit
        assert model.omega.item() > 0

    def test_auto_init_with_default_config_converges_well(self):
        """The actual fix for slow/drifting convergence (see FitConfig's
        docstring): auto-estimating the starting omega from winding pairs
        instead of guessing, combined with the now-default omega_lr_scale.
        Verified interactively this reaches within ~1% of ground truth in
        under 100 iterations; this test uses the full default FitConfig
        (n_iterations=400) and a looser 10% bound to stay robust to the
        phantom's random noise while still catching a real regression.
        """
        from src.spiral_fitting.features import extract_surface_paths
        from src.spiral_fitting.fit import FitConfig, fit_spiral
        from src.spiral_fitting.mesh import extract_flattened_page

        omega_true = 0.5
        vol = make_spiral_volume(shape=(16, 128, 128), omega_true=omega_true, n_turns=4, thickness=1.5, noise=0.02)
        paths, pairs = extract_surface_paths(vol, threshold=0.4, z_stride=2)

        config = FitConfig(grid_shape=(3, 5, 5), n_euler_steps=4, paths_per_batch=8)  # defaults otherwise
        model, history = fit_spiral(paths, pairs, vol.shape, initial_omega=None, config=config, progress=False)

        final_omega = history["omega"][-1]
        assert abs(final_omega - omega_true) / omega_true < 0.10

        best_max = 0.0
        for winding in range(4):
            page, _ = extract_flattened_page(model, vol, winding, n_theta=150, z_samples=16)
            best_max = max(best_max, float(page.max()))
        assert best_max > 0.5  # close to true ink density 0.8, not background


class TestExtractFlattenedPage:
    def test_output_shapes(self):
        from src.spiral_fitting.features import extract_surface_paths
        from src.spiral_fitting.fit import FitConfig, fit_spiral
        from src.spiral_fitting.mesh import extract_flattened_page

        vol = make_spiral_volume(shape=(6, 64, 64), omega_true=0.6, n_turns=2)
        paths, pairs = extract_surface_paths(vol, threshold=0.4, z_stride=1)
        config = FitConfig(n_iterations=20, grid_shape=(3, 4, 4), n_euler_steps=3, paths_per_batch=4)
        model, _ = fit_spiral(paths, pairs, vol.shape, initial_omega=0.6, config=config, progress=False)

        page, coord_map = extract_flattened_page(model, vol, n_winding=0, n_theta=50, z_samples=6)
        assert page.shape == (6, 50)
        assert coord_map.shape == (6, 50, 3)

    def test_coord_map_in_bounds_where_not_nan(self):
        from src.spiral_fitting.features import extract_surface_paths
        from src.spiral_fitting.fit import FitConfig, fit_spiral
        from src.spiral_fitting.mesh import extract_flattened_page

        vol = make_spiral_volume(shape=(6, 64, 64), omega_true=0.6, n_turns=2)
        paths, pairs = extract_surface_paths(vol, threshold=0.4, z_stride=1)
        config = FitConfig(n_iterations=20, grid_shape=(3, 4, 4), n_euler_steps=3, paths_per_batch=4)
        model, _ = fit_spiral(paths, pairs, vol.shape, initial_omega=0.6, config=config, progress=False)

        page, coord_map = extract_flattened_page(model, vol, n_winding=0, n_theta=50, z_samples=6)
        valid = ~np.isnan(coord_map[..., 0])
        if valid.any():
            zs, ys, xs = coord_map[valid, 0], coord_map[valid, 1], coord_map[valid, 2]
            assert (zs >= 0).all() and (zs < vol.shape[0]).all()
            assert (ys >= 0).all() and (ys < vol.shape[1]).all()
            assert (xs >= 0).all() and (xs < vol.shape[2]).all()

    def test_exact_omega_with_untrained_model_recovers_real_signal(self):
        """Isolates mesh.py's correctness from fit.py's optimisation
        behaviour: build a model at the *exact* true omega with an
        untrained (near-identity) flow field — no fitting involved — and
        check the extracted page for a real winding matches the phantom's
        true ink density (0.8), not background. This is deliberately not a
        test of the fitting/convergence loop (see FitConfig's docstring for
        that story); it isolates whether the geometry/sampling pipeline
        (canonical spiral -> diffeomorphism -> volume sampling) is correct
        when given the right answer directly.
        """
        from src.spiral_fitting.fit import SpiralFitModel
        from src.spiral_fitting.mesh import extract_flattened_page

        omega_true = 0.5
        vol = make_spiral_volume(shape=(8, 96, 96), omega_true=omega_true, n_turns=3, thickness=1.5, noise=0.0)

        model = SpiralFitModel(
            diffeo_grid_shape=(3, 5, 5), bounds_min=(0.0, 0.0, 0.0), bounds_max=(96.0, 96.0, 8.0), initial_omega=omega_true
        )

        best_max = 0.0
        for winding in range(3):
            page, _ = extract_flattened_page(model, vol, winding, n_theta=100, z_samples=8)
            best_max = max(best_max, float(page.max()))
        assert best_max > 0.5  # meaningfully above background, close to true 0.8


def _shifted(vol, dy, dx):
    return np.roll(np.roll(vol, dy, axis=1), dx, axis=2)


class TestScrollAxisAndHandedness:
    def test_off_centre_spiral_page_is_not_blank(self):
        """The canonical spiral sits on the origin of its own plane; a scroll whose axis is
        elsewhere in the scan used to be sampled in the volume's corner (blank pages)."""
        from src.spiral_fitting.fit import SpiralFitModel
        from src.spiral_fitting.mesh import extract_flattened_page

        vol = _shifted(make_spiral_volume(shape=(8, 128, 128), omega_true=0.5, n_turns=4, noise=0.0), 20, -15)
        model = SpiralFitModel((3, 5, 5), (0, 0, 0), (128.0, 128.0, 8.0), 0.5, canonical_origin=(64.0 - 15, 64.0 + 20, 0.0))
        best = max(extract_flattened_page(model, vol, w, n_theta=120, z_samples=8)[0].max() for w in range(4))
        assert best > 0.5

    def test_mirrored_spiral_is_detected(self):
        from src.spiral_fitting.features import extract_surface_paths
        from src.spiral_fitting.fit import FitConfig, fit_spiral

        vol = make_spiral_volume(shape=(8, 128, 128), omega_true=0.5, n_turns=4, noise=0.0)
        flipped = vol[:, ::-1, :].copy()
        signs = []
        for v in (vol, flipped):
            paths, pairs = extract_surface_paths(v, threshold=0.4, z_stride=2)
            model, _ = fit_spiral(paths, pairs, v.shape, None, FitConfig(n_iterations=2, grid_shape=(3, 4, 4), n_euler_steps=2, paths_per_batch=4), progress=False)
            signs.append(float(model.diffeo.canonical_sign[1]))
        assert signs[0] != signs[1]


class TestFastExtraction:
    def test_longest_path_on_a_line_and_an_l_shape(self):
        from src.spiral_fitting.features import _longest_path_in_skeleton

        line = np.zeros((10, 30), dtype=bool)
        line[5, 2:25] = True
        path = _longest_path_in_skeleton(line)
        assert len(path) == 23 and {p[0] for p in path} == {5}

        bent = np.zeros((20, 20), dtype=bool)
        bent[3, 3:15] = True
        bent[3:15, 14] = True
        assert len(_longest_path_in_skeleton(bent)) >= 22  # the corner may be cut diagonally

    def test_downsampled_extraction_returns_full_resolution_coordinates(self):
        from src.spiral_fitting.features import extract_surface_paths

        vol = make_spiral_volume(shape=(2, 160, 160), omega_true=0.25, n_turns=3, thickness=3.0, noise=0.0)
        paths, _ = extract_surface_paths(vol, threshold=0.3, z_stride=1, downsample=2)
        assert paths
        pts = np.concatenate([p.points[:, 1:] for p in paths])
        assert pts.max() > 100  # in full-resolution voxel units, not the 80-px downsampled grid
        on_sheet = vol[0][pts[:, 0].astype(int).clip(0, 159), pts[:, 1].astype(int).clip(0, 159)]
        assert (on_sheet > 0.3).mean() > 0.6

    def test_many_tiny_fragments_are_capped(self):
        from src.spiral_fitting.features import extract_surface_paths

        vol = np.zeros((1, 100, 100), dtype=np.float32)
        vol[0, ::4, ::4] = 1.0
        vol[0, ::4, 1::4] = 1.0
        paths, _ = extract_surface_paths(vol, threshold=0.5, z_stride=1, min_component_pixels=2, max_components_per_slice=5)
        assert len(paths) <= 5


class TestWindingSpacing:
    def test_recovers_spacing_of_a_regular_spiral(self):
        from src.spiral_fitting.features import estimate_winding_spacing

        vol = make_spiral_volume(shape=(1, 256, 256), omega_true=0.3, n_turns=6, thickness=2.0, noise=0.0)
        spacing = estimate_winding_spacing(vol[0], (127.5, 127.5))
        assert spacing is not None and abs(spacing - 2 * np.pi / 0.3) < 2.0

    def test_returns_none_for_noise(self):
        from src.spiral_fitting.features import estimate_winding_spacing

        noise = np.random.default_rng(0).normal(size=(200, 200)).astype(np.float32)
        assert estimate_winding_spacing(noise, (100.0, 100.0)) is None


class TestRunQualityGate:
    def _write(self, tmp_path, vol):
        import h5py

        path = tmp_path / "spiral_corrected.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("volume", data=vol)
        return path

    def test_good_fit_writes_quality_summary(self, tmp_path):
        import json

        from src.spiral_fitting.fit import FitConfig
        from src.spiral_fitting.run import run

        vol = make_spiral_volume(shape=(8, 128, 128), omega_true=0.5, n_turns=4, noise=0.01)
        out = run(self._write(tmp_path, vol), tmp_path, n_windings=3, n_theta=100, config=FitConfig(n_iterations=60, grid_shape=(3, 4, 4), n_euler_steps=3, paths_per_batch=8))
        summary = json.loads((tmp_path / "spiral_spiral_fit_summary.json").read_text())
        assert out.exists() and len(summary["windings"]) == 3
        assert {"loss_improvement", "fit_ok", "warnings", "handedness"} <= set(summary)
        assert max(w["page_max"] for w in summary["windings"]) > 0.5

    def test_blank_output_is_an_error_not_a_success(self, tmp_path, monkeypatch):
        import src.spiral_fitting.run as spiral_run

        vol = make_spiral_volume(shape=(8, 128, 128), omega_true=0.5, n_turns=4, noise=0.01)
        monkeypatch.setattr(
            spiral_run, "extract_flattened_page", lambda *a, **k: (np.zeros((8, 50), np.float32), np.full((8, 50, 3), np.nan, np.float32))
        )
        with pytest.raises(ValueError, match="blank"):
            spiral_run.run(self._write(tmp_path, vol), tmp_path, n_windings=2, config=spiral_run.FitConfig(n_iterations=5, grid_shape=(3, 4, 4), n_euler_steps=2, paths_per_batch=4))
