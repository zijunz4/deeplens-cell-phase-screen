"""Tests for deeplens/optics/hybridlens.py — HybridLens."""

import json
import os

import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    """Restore default dtype after each test (HybridLens sets float64 globally)."""
    old_dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(old_dtype)


class TestHybridLensInit:
    """Tests for HybridLens initialization."""

    def test_init_from_json(self, sample_hybridlens):
        """HybridLens loads from JSON with geolens and doe."""
        lens = sample_hybridlens
        assert lens.geolens is not None
        assert lens.doe is not None
        assert len(lens.geolens.surfaces) > 0
        assert hasattr(lens.doe, "d_next")
        assert not hasattr(lens.doe, "d")

    def test_device_transfer(self, sample_hybridlens):
        """to(device) transfers both geolens and doe."""
        lens = sample_hybridlens
        lens.to(torch.device("cpu"))
        assert lens.doe.d_next.device.type == "cpu"


class TestHybridLensPSF:
    """Tests for PSF computation."""

    def test_psf_shape_and_normalization(self, sample_hybridlens):
        """psf() returns [ks, ks] tensor normalized to ~1."""
        lens = sample_hybridlens
        ks = 64
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            psf = lens.psf(points=[0.0, 0.0, -10000.0], ks=ks, spp=1_000_000)
        finally:
            torch.set_default_dtype(old_dtype)
        assert psf.shape == (ks, ks)
        assert psf.sum().item() == pytest.approx(1.0, abs=0.05)
        assert (psf >= 0).all()

    def test_psf_supports_metasurface_array(self, sample_hybridlens, monkeypatch):
        """The hybrid ray-wave path applies an array phase and transmission map."""
        from deeplens.diffractive_surface import Pixel2DMetasurfaceArray

        lens = sample_hybridlens
        doe = Pixel2DMetasurfaceArray(
            d_next=lens.doe.d_next.detach().clone(),
            array_shape=(2, 3),
            aperture_res=(4, 4),
            aperture_ps=0.1,
            aperture_pitch=(0.5, 0.6),
            aperture_shape="square",
            device=lens.device,
        ).astype(torch.float64)
        lens.doe = doe
        lens.geolens.set_sensor_res((64, 64))
        lens.set_sensor(lens.geolens.sensor_size, lens.geolens.sensor_res)

        wavefront = torch.ones(
            doe.res,
            dtype=torch.complex128,
            device=lens.device,
        )
        psfc = [
            torch.tensor(0.0, dtype=torch.float64, device=lens.device),
            torch.tensor(0.0, dtype=torch.float64, device=lens.device),
        ]
        monkeypatch.setattr(
            lens,
            "doe_field",
            lambda *args, **kwargs: (wavefront, psfc),
        )

        psf = lens.psf(
            points=[0.0, 0.0, -10000.0],
            ks=16,
            spp=1,
            upsample_factor=1,
        )

        assert psf.shape == (16, 16)
        assert torch.isfinite(psf).all()
        assert psf.sum().item() == pytest.approx(1.0, abs=1e-6)


class TestHybridLensUtils:
    """Tests for utility methods."""

    def test_calc_scale(self, sample_hybridlens):
        """calc_scale returns a positive float."""
        lens = sample_hybridlens
        scale = lens.calc_scale(depth=-10000.0)
        assert isinstance(scale, float)
        assert scale > 0

    def test_refocus(self, sample_hybridlens):
        """refocus changes geolens d_sensor."""
        lens = sample_hybridlens
        d_before = lens.geolens.d_sensor.clone()
        lens.refocus(foc_dist=-5000.0)
        # d_sensor should change after refocus
        assert lens.geolens.d_sensor is not None


class TestHybridLensIO:
    """Tests for I/O."""

    def test_metasurface_array_roundtrip(self, sample_hybridlens, tmp_path):
        """Hybrid JSON stores and reloads array metadata and phase maps."""
        from deeplens import HybridLens
        from deeplens.diffractive_surface import Pixel2DMetasurfaceArray

        lens = sample_hybridlens
        doe = Pixel2DMetasurfaceArray(
            d_next=lens.doe.d_next.detach().clone(),
            array_shape=(2, 3),
            aperture_res=(4, 4),
            aperture_ps=0.1,
            aperture_pitch=(0.5, 0.6),
            aperture_shape="square",
            device=lens.device,
        ).astype(torch.float64)
        lens.doe = doe
        path = tmp_path / "hybrid_metasurface_array.json"

        lens.write_lens_json(str(path))
        loaded = HybridLens(filename=str(path), device=lens.device)

        assert isinstance(loaded.doe, Pixel2DMetasurfaceArray)
        assert loaded.doe.array_shape == doe.array_shape
        assert torch.allclose(loaded.doe.phase_maps, doe.phase_maps)

    def test_legacy_absolute_d_is_normalized_at_read_boundary(
        self, sample_hybridlens, test_output_dir
    ):
        path = os.path.join(test_output_dir, "legacy_hybridlens.json")
        sample_hybridlens.write_lens_json(path)
        with open(path) as f:
            data = json.load(f)
        doe_gap = data["DOE"].pop("d_next")
        data["DOE"]["d"] = data["d_sensor"] - doe_gap
        with open(path, "w") as f:
            json.dump(data, f)

        from deeplens import HybridLens

        loaded = HybridLens(filename=path)

        assert loaded.doe.d_next.item() == pytest.approx(doe_gap)
        assert not hasattr(loaded.doe, "d")

    def test_write_read_json_roundtrip(self, sample_hybridlens, test_output_dir):
        """write_lens_json then read_lens_json preserves structure."""
        lens = sample_hybridlens
        out_path = os.path.join(test_output_dir, "test_hybridlens_roundtrip.json")
        original_num_surfs = len(lens.geolens.surfaces)

        lens.write_lens_json(out_path)
        assert os.path.exists(out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert "d_next" in data["DOE"]
        assert "d" not in data["DOE"]

        from deeplens import HybridLens

        lens2 = HybridLens(filename=out_path)
        assert lens2.geolens is not None
        assert lens2.doe is not None


class TestHybridLensOptim:
    """Tests for optimization helpers."""

    def test_get_optimizer(self, sample_hybridlens):
        """get_optimizer returns an Adam optimizer."""
        lens = sample_hybridlens
        optimizer = lens.get_optimizer()
        assert isinstance(optimizer, torch.optim.Adam)


class TestHybridLensVis:
    """Smoke test for draw_layout."""

    def test_draw_layout(self, sample_hybridlens, test_output_dir):
        """draw_layout produces a file without crashing."""
        lens = sample_hybridlens
        path = os.path.join(test_output_dir, "test_hybridlens_layout.png")
        lens.draw_layout(save_name=path, dpi=100)
        assert os.path.exists(path)
