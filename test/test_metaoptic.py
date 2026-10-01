"""Tests for multi-aperture meta-optic camera readout."""

import pytest
import torch

from deeplens.diffractive_surface import Pixel2DMetasurfaceArray
from deeplens.imgsim import DetectorModel, MetaOpticChannelArray


def _delta_psf(num_channels=2, channels=3, ks=3):
    """Create unit-sum delta PSFs with shape [N, C, ks, ks]."""
    psfs = torch.zeros(num_channels, channels, ks, ks)
    psfs[..., ks // 2, ks // 2] = 1.0
    return psfs


class TestMetaOpticChannelArray:
    """Tests for tiled multi-aperture detector readout."""

    def test_tiled_forward_matches_channel_crops(self):
        """The full detector contains each aperture's centered feature image."""
        scene = torch.rand(1, 3, 8, 10)
        camera = MetaOpticChannelArray(
            psfs=_delta_psf(),
            channel_rois=((0, 0), (0, 4)),
            feature_size=(3, 3),
            detector_shape=(3, 7),
        )

        detector = camera(scene)

        expected = scene[..., 2:5, 3:6]
        assert detector.shape == (1, 3, 3, 7)
        assert torch.allclose(detector[..., :3], expected)
        assert torch.allclose(detector[..., 4:7], expected)

    def test_positive_negative_channel_pair(self):
        """Paired features subtract non-negative detector measurements."""
        scene = torch.ones(1, 1, 8, 8)
        psfs = _delta_psf(num_channels=2, channels=1)
        psfs[1] *= 0.25
        camera = MetaOpticChannelArray(
            psfs=psfs,
            channel_rois=((0, 0), (0, 4)),
            feature_size=(3, 3),
            detector_shape=(3, 7),
            channel_pairs=((0, 1),),
        )

        features = camera.forward_features(scene, reduce="mean")

        assert features.shape == (1, 1, 1)
        assert features.item() == pytest.approx(0.75)
        assert camera(scene).min().item() >= 0

    def test_gradients_flow_to_psfs(self):
        """Feature extraction remains differentiable with respect to PSFs."""
        scene = torch.rand(1, 1, 7, 7)
        psfs = _delta_psf(num_channels=1, channels=1).requires_grad_(True)
        camera = MetaOpticChannelArray(
            psfs=psfs,
            channel_rois=((0, 0),),
            feature_size=(3, 3),
            detector_shape=(3, 3),
        )

        camera.forward_features(scene).sum().backward()

        assert psfs.grad is not None
        assert psfs.grad.abs().sum().item() > 0

    def test_overlapping_rois_are_rejected(self):
        """Physical detector channels cannot occupy the same pixels."""
        with pytest.raises(ValueError, match="overlap"):
            MetaOpticChannelArray(
                psfs=_delta_psf(),
                channel_rois=((0, 0), (2, 2)),
                feature_size=(3, 3),
                detector_shape=(6, 6),
            )


class TestDetectorModel:
    """Tests for detector gain, noise, and quantization."""

    def test_deterministic_conversion_and_saturation(self):
        detector = DetectorModel(
            bit_depth=12,
            full_well=1000.0,
            gain=1000.0 / 4095.0,
            noise=False,
        )

        signal = detector(torch.tensor([[[[0.5]]], [[[2.0]]]]))

        assert signal[0].item() == pytest.approx(2047.5)
        assert signal[1].item() == pytest.approx(4095.0)

    def test_noise_changes_the_sample(self):
        detector = DetectorModel(
            bit_depth=12,
            full_well=1000.0,
            read_noise=5.0,
            noise=False,
        )
        intensity = torch.full((1, 1, 4, 4), 0.5)
        clean = detector(intensity, add_noise=False)

        torch.manual_seed(0)
        noisy = detector(intensity, add_noise=True)

        assert not torch.allclose(clean, noisy)


class TestPhaseToPSF:
    """Tests for phase-domain to detector-channel conversion."""

    def test_metasurface_array_builds_receiver_psfs(self):
        metasurface = Pixel2DMetasurfaceArray(
            d_next=0.05,
            array_shape=(2, 2),
            aperture_res=(4, 4),
            aperture_ps=0.005,
            aperture_shape="square",
        )

        receiver = MetaOpticChannelArray.from_metasurface_array(
            metasurface_array=metasurface,
            channel_rois=((0, 0), (0, 4), (4, 0), (4, 4)),
            feature_size=(4, 4),
            detector_shape=(8, 8),
        )

        assert receiver.psfs.shape == (4, 1, 4, 4)
        assert torch.allclose(
            receiver.psfs.sum(dim=(-2, -1)),
            torch.ones(4, 1),
            atol=1e-6,
        )

    def test_phase_to_psf_is_differentiable(self):
        metasurface = Pixel2DMetasurfaceArray(
            d_next=0.05,
            array_shape=(1, 1),
            aperture_res=(4, 4),
            aperture_ps=0.005,
            aperture_shape="square",
            phase_maps=torch.randn(1, 1, 4, 4),
        )
        metasurface.phase_maps.requires_grad_(True)

        psfs = metasurface.compute_channel_psfs()
        weighted = torch.linspace(0.0, 1.0, 4).view(1, 1, 4, 1)
        (psfs * weighted).sum().backward()

        assert metasurface.phase_maps.grad is not None
        assert metasurface.phase_maps.grad.abs().sum().item() > 0
