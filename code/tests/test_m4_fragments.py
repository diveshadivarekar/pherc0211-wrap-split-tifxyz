"""Tests for the fragment-trained ink model: data loading, tiling, metrics and a tiny end-to-end training run.

Uses a small synthetic 'fragment' laid out exactly like the official download, so no real data is needed.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch", reason="needs torch")

from src.m4_ink_detection import fragments as fr  # noqa: E402
from src.m4_ink_detection.infer import best_fbeta, normalize_stack, predict_stack  # noqa: E402

SIZE = 160
N_SLICES = 4


def make_fragment(root, seed=0, size=SIZE, n_total=8):
    """Papyrus disc with 'ink' stripes that only show up as a subtle high-frequency change in the middle slices."""
    import tifffile

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    mask = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (size * 0.45) ** 2
    ink = mask & (((xx // 8) % 3 == 0) & (yy % 40 < 24))
    (root / "surface_volume").mkdir(parents=True, exist_ok=True)
    for z in range(n_total):
        base = 20000 + 3000 * rng.standard_normal((size, size))
        if 2 <= z < 2 + N_SLICES:
            base = base + ink * 6000  # ink visible only in the slices the model is asked to read
        img = np.clip(base * mask, 0, 65535).astype(np.uint16)
        tifffile.imwrite(root / "surface_volume" / f"{z:02d}.tif", img)
    Image.fromarray((ink * 255).astype(np.uint8)).save(root / "inklabels.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(root / "mask.png")
    return root


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("frags")
    for i, name in enumerate(("frag1", "frag2", "frag3")):
        make_fragment(root / name, seed=i)
    return root


class TestLoading:
    def test_loads_stack_labels_and_mask(self, data):
        frag = fr.load_fragment(data / "frag1", z_start=2, n_slices=N_SLICES)
        assert frag.stack.shape == (N_SLICES, SIZE, SIZE) and frag.stack.dtype == np.uint16
        assert frag.labels.dtype == bool and frag.mask.sum() > frag.labels.sum() > 0
        assert (data / "frag1" / f"stack_z2_n{N_SLICES}.npy").exists()  # cached for next time

    def test_too_few_slices_is_a_clear_error(self, data):
        with pytest.raises(FileNotFoundError):
            fr.build_stack_cache(data / "frag2", z_start=6, n_slices=N_SLICES)

    def test_mismatched_label_size_is_rejected(self, data, tmp_path):
        import shutil

        root = tmp_path / "bad"
        shutil.copytree(data / "frag3", root)
        Image.fromarray(np.zeros((10, 10), np.uint8)).save(root / "mask.png")
        with pytest.raises(ValueError, match="differ in size"):
            fr.load_fragment(root, z_start=2, n_slices=N_SLICES)

    def test_intensity_stats_are_sane(self, data):
        frags = [fr.load_fragment(data / n, 2, N_SLICES) for n in ("frag1", "frag2")]
        mean, std = fr.intensity_stats(frags)
        assert 0.2 < mean < 0.5 and 0.01 < std < 0.3


class TestInference:
    def test_normalize_scales_uint16_only(self):
        x = np.full((1, 2, 2), 65535, dtype=np.uint16)
        assert np.allclose(normalize_stack(x, 0.5, 0.5), 1.0)
        assert np.allclose(normalize_stack(np.full((1, 2, 2), 0.5, np.float32), 0.5, 0.5), 0.0)

    def test_tiling_covers_every_pixel_and_respects_mask(self):
        class Const(torch.nn.Module):
            def forward(self, x):
                return torch.full((x.shape[0], x.shape[2], x.shape[3]), 20.0, device=x.device)  # sigmoid ~ 1

        stack = np.zeros((N_SLICES, 150, 170), dtype=np.uint16)  # not a multiple of the tile size
        mask = np.zeros((150, 170), dtype=bool)
        mask[20:140, 30:160] = True
        prob = predict_stack(Const(), stack, 0.0, 1.0, mask=mask, tile=64, stride=48, device="cpu")
        assert prob.shape == (150, 170)
        assert np.allclose(prob[mask], 1.0, atol=1e-3) and (prob[~mask] == 0).all()

    def test_best_fbeta_on_perfect_and_random_scores(self):
        rng = np.random.default_rng(0)
        labels = rng.random((100, 100)) < 0.2
        mask = np.ones_like(labels)
        perfect = best_fbeta(labels.astype(np.float32) * 0.9 + 0.05, labels, mask)
        noise = best_fbeta(rng.random((100, 100)).astype(np.float32), labels, mask)
        assert perfect["f_beta"] > 0.99 and noise["f_beta"] < 0.35 and abs(perfect["ink_fraction"] - labels.mean()) < 1e-9


class TestModelAndTraining:
    def test_unet_output_shape_and_gradient(self):
        from src.m4_ink_detection.unet import InkUNet

        net = InkUNet(N_SLICES, base=8)
        out = net(torch.randn(2, N_SLICES, 64, 64))
        assert out.shape == (2, 64, 64)
        out.mean().backward()
        assert all(p.grad is not None for p in net.parameters())

    def test_crop_sampler_oversamples_ink_and_stays_in_bounds(self, data):
        from src.m4_ink_detection.train_fragments import CropSampler

        frag = fr.load_fragment(data / "frag1", 2, N_SLICES)
        sampler = CropSampler([frag], crop=64)
        frac = []
        for _ in range(60):
            raw, lab, mask = sampler.sample()
            assert raw.shape == (N_SLICES, 64, 64) and lab.shape == (64, 64) == mask.shape
            frac.append(lab.mean())
        assert np.mean(frac) > 1.2 * frag.labels.mean()  # crops see more ink than the fragment as a whole

    def test_masked_loss_ignores_pixels_outside_the_papyrus(self):
        from src.m4_ink_detection.train_fragments import masked_loss

        target = torch.zeros(1, 8, 8)
        mask = torch.zeros(1, 8, 8)
        mask[:, :4] = 1
        logits = torch.zeros(1, 8, 8)
        logits[:, 4:] = 50.0  # confidently wrong, but only outside the mask
        assert masked_loss(logits, target, mask).item() == pytest.approx(masked_loss(torch.zeros(1, 8, 8), target, mask).item())

    def test_end_to_end_training_beats_the_density_baseline_on_a_held_out_fragment(self, data, tmp_path):
        from src.m4_ink_detection.train_fragments import load_model, train

        out = tmp_path / "models" / "ink.pt"
        report = train(data, ["frag1", "frag2"], "frag3", out, steps=300, batch=8, crop=64, lr=2e-3, base=8, eval_every=150, z_start=2, n_slices=N_SLICES)
        assert out.exists() and out.with_suffix(".json").exists()
        assert report["best"]["f_beta"] > 0.5
        assert report["best"]["f_beta"] > report["all_ink_baseline_f0.5"] + 0.1
        model, meta = load_model(out, device="cpu")
        assert meta["val"] == "frag3" and meta["n_slices"] == N_SLICES and meta["metrics"]["f_beta"] == report["best"]["f_beta"]


class TestScrollPages:
    @staticmethod
    def _tilted_page():
        """A 3D plane z = 0.5*x tilted about the y axis, sampled on a grid (like an M3 coord_map)."""
        h, w = 20, 30
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        coord = np.stack([10 + 0.5 * xx, 5 + yy, 5 + xx], axis=-1)
        return coord

    def test_normals_of_a_plane_are_unit_and_perpendicular(self):
        from src.m4_ink_detection.scroll_pages import surface_normals

        coord = self._tilted_page()
        n = surface_normals(coord)
        assert np.allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-4)
        tangent_x = np.array([0.5, 0, 1.0]) / np.linalg.norm([0.5, 0, 1.0])
        assert np.abs(n[10, 15] @ tangent_x) < 1e-3 and np.abs(n[10, 15, 1]) < 1e-3  # normal is perpendicular to the plane

    def test_normals_extend_over_invalid_pixels(self):
        from src.m4_ink_detection.scroll_pages import surface_normals

        coord = self._tilted_page()
        coord[:, :5] = np.nan
        n = surface_normals(coord)
        assert np.allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-4)
        assert surface_normals(np.full((4, 4, 3), np.nan, np.float32)).shape == (4, 4, 3)

    def test_sampled_stack_follows_the_surface_and_reads_the_depth_profile(self):
        from src.m4_ink_detection.scroll_pages import sample_page_stack

        coord = self._tilted_page()
        zz, yy, xx = np.mgrid[:40, :40, :50].astype(np.float32)
        vol = (zz - 0.5 * xx)  # signed distance-ish from the tilted plane z = 0.5x (+ offset below)
        stack, valid = sample_page_stack(vol, coord, n_slices=5)
        assert stack.shape == (5, 20, 30) and valid.all()
        # a volume that increases along the normal gives a monotonic profile through the stack
        diffs = np.diff(stack[:, 10, 15])
        assert (np.all(diffs > 0) or np.all(diffs < 0))
        flipped, _ = sample_page_stack(vol, coord, n_slices=5, flip_depth=True)
        assert np.allclose(flipped[::-1], stack, atol=1e-4)

    def test_voxel_size_shrinks_the_step_to_match_the_fragments(self):
        from src.m4_ink_detection.scroll_pages import sample_page_stack

        coord = self._tilted_page()
        zz, yy, xx = np.mgrid[:40, :40, :50].astype(np.float32)
        vol = zz - 0.5 * xx
        coarse, _ = sample_page_stack(vol, coord, n_slices=5)
        fine, _ = sample_page_stack(vol, coord, n_slices=5, voxel_size_um=7.91)
        assert np.ptp(fine[:, 10, 15]) < 0.5 * np.ptp(coarse[:, 10, 15])

    def test_standardize_page_is_zero_mean_inside_and_zero_outside(self):
        from src.m4_ink_detection.scroll_pages import standardize_page

        stack = np.random.default_rng(0).normal(5, 3, (4, 10, 10)).astype(np.float32)
        valid = np.zeros((10, 10), bool)
        valid[2:8, 2:8] = True
        x = standardize_page(stack, valid)
        assert abs(x[:, valid].mean()) < 1e-4 and abs(x[:, valid].std() - 1) < 1e-3 and (x[:, ~valid] == 0).all()

    def test_run_with_a_checkpoint_writes_ink_maps_and_honest_attributes(self, tmp_path):
        import h5py

        from src.m4_ink_detection.run import run
        from src.m4_ink_detection.train_fragments import _save
        from src.m4_ink_detection.unet import InkUNet

        ckpt = tmp_path / "ink.pt"
        _save(ckpt, InkUNet(N_SLICES, 8), 0.3, 0.1, 2, N_SLICES, 8, 64, {"f_beta": 0.61}, ["frag1"], "frag3")
        coord = self._tilted_page()
        with h5py.File(tmp_path / "scan_flattened.h5", "w") as f:
            for name, c in (("layer_1", coord), ("layer_2", coord[:10]), ("layer_10", coord[:5])):
                g = f.create_group(name)
                g.create_dataset("page", data=np.zeros(c.shape[:2], np.float32))
                g.create_dataset("coord_map", data=c)
        with h5py.File(tmp_path / "scan_corrected.h5", "w") as f:
            f.create_dataset("volume", data=np.random.default_rng(0).random((40, 40, 50)).astype(np.float32))

        out = run(tmp_path / "scan_flattened.h5", tmp_path / "scan_corrected.h5", tmp_path, model_path=ckpt, voxel_size_um=7.91, max_pages=2)
        with h5py.File(out, "r") as f:
            assert f.attrs["mode"] == "fragment_unet" and not f.attrs["validated"]
            assert "NOT been validated" in f.attrs["notice"] and "0.61" in f.attrs["notice"]
            assert set(f.keys()) == {"layer_1", "layer_2"}  # the two largest pages only
            prob = f["layer_1"]["ink_probability"][...]
            assert prob.shape == (20, 30) and 0 <= prob.min() and prob.max() <= 1


class TestRefineSurface:
    @staticmethod
    def _sheet_volume(centre=21.0):
        """A 4-voxel-thick bright sheet in z (flat, at ``centre``) inside a dark volume."""
        zz = np.arange(40, dtype=np.float32)[:, None, None]
        return np.broadcast_to(np.exp(-((zz - centre) ** 2) / (2 * 1.5**2)), (40, 40, 40)).copy().astype(np.float32) + 0.05

    @staticmethod
    def _page_at(z, h=24, w=24):
        yy, xx = np.mgrid[8:8 + h, 8:8 + w].astype(np.float32)
        return np.stack([np.full_like(yy, z), yy, xx], axis=-1)

    def test_a_surface_a_few_voxels_off_snaps_onto_the_sheet(self):
        from src.m4_ink_detection.scroll_pages import refine_surface

        vol = self._sheet_volume(21.0)
        refined, info = refine_surface(vol, self._page_at(17.0), papyrus_threshold=0.3)
        assert np.nanmean(np.abs(refined[..., 0] - 21.0)) < 0.7 and info["n_dropped"] == 0

    def test_pixels_with_no_papyrus_in_reach_are_dropped(self):
        from src.m4_ink_detection.scroll_pages import refine_surface

        vol = self._sheet_volume(21.0)
        refined, info = refine_surface(vol, self._page_at(2.0), search=4.0, papyrus_threshold=0.3)  # sheet is 19 voxels away
        assert info["n_dropped"] == info["n_pixels"] and np.isnan(refined).all()

    def test_empty_page_is_safe_and_the_shift_is_smooth(self):
        from src.m4_ink_detection.scroll_pages import refine_surface

        out, info = refine_surface(np.zeros((8, 8, 8), np.float32), np.full((4, 4, 3), np.nan, np.float32))
        assert np.isnan(out).all() and info["n_pixels"] == 0
        refined, _ = refine_surface(self._sheet_volume(21.0), self._page_at(19.0), papyrus_threshold=0.3)
        z = refined[..., 0]
        assert np.nanstd(z) < 0.3
