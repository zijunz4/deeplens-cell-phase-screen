"""Tests for the system-level meta-optic flow-cytometry model."""

import pytest
import torch

from deeplens import (
    DetectorModel,
    FlowCytometrySpec,
    FlowTrajectory,
    MetaOpticFlowCytometer,
    extract_pulse_features,
)
from deeplens.imgsim import MetaOpticChannelArray


def _make_receiver(feature_size=(9, 9)):
    psfs = torch.zeros(1, 1, 3, 3)
    psfs[0, 0, 1, 1] = 1.0
    return MetaOpticChannelArray(
        psfs=psfs,
        channel_rois=((0, 0),),
        feature_size=feature_size,
        detector_shape=feature_size,
    )


class TestFlowTrajectory:
    """Tests for particle trajectories."""

    def test_linear_trajectory(self):
        trajectory = FlowTrajectory.linear(
            start=(-0.02, 0.01),
            velocity=(0.02, -0.01),
            num_steps=3,
            dt=0.1,
        )

        assert trajectory.positions.shape == (3, 2)
        assert trajectory.times.tolist() == pytest.approx([0.0, 0.1, 0.2])
        assert trajectory.positions[:, 0].tolist() == pytest.approx(
            [-0.02, -0.018, -0.016]
        )
        assert trajectory.positions[:, 1].tolist() == pytest.approx(
            [0.01, 0.009, 0.008]
        )

    def test_times_must_increase(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            FlowTrajectory(
                positions=((0.0, 0.0), (0.1, 0.0)),
                times=(1.0, 1.0),
            )


class TestFlowCytometrySpec:
    """Tests for practical first-order flow-system specifications."""

    def test_derived_first_order_quantities(self):
        spec = FlowCytometrySpec(
            focal_length=50.0,
            f_number=4.0,
            half_fov_deg=20.0,
            working_distance=100.0,
            detector_pixel_pitch=0.005,
            flow_velocity=1.0,
            integration_time=1e-3,
            channel_count=4,
            back_focal_length=10.0,
            total_track_length=70.0,
            num_elements=5,
        )

        assert spec.aperture_diameter == pytest.approx(12.5)
        assert spec.image_diameter == pytest.approx(
            2 * 50 * torch.tan(torch.deg2rad(torch.tensor(20.0))).item()
        )
        assert spec.detector_nyquist_frequency == pytest.approx(100.0)
        assert spec.flow_pixels_per_frame == pytest.approx(0.2)
        assert spec.validate() == []

    def test_invalid_constraints_are_reported(self):
        spec = FlowCytometrySpec(
            focal_length=50.0,
            f_number=0.0,
            half_fov_deg=20.0,
            working_distance=100.0,
            detector_pixel_pitch=0.005,
            flow_velocity=1.0,
            integration_time=1e-3,
            channel_count=0,
            back_focal_length=70.0,
            total_track_length=60.0,
            num_elements=1,
        )

        violations = spec.validate()

        assert any("f_number" in message for message in violations)
        assert any("channel_count" in message for message in violations)
        assert any("total_track_length" in message for message in violations)
        assert any("num_elements" in message for message in violations)


class TestPulseFeatures:
    """Tests for flow-cytometry event statistics."""

    def test_pulse_features(self):
        signals = torch.tensor([[[1.0], [5.0], [1.0]]])
        times = torch.tensor([0.0, 0.1, 0.2])

        stats = extract_pulse_features(signals, times=times)

        assert stats["peak"].item() == pytest.approx(5.0)
        assert stats["area"].item() == pytest.approx(0.7)
        assert stats["width"].item() == pytest.approx(0.1)
        assert stats["time_to_peak"].item() == pytest.approx(0.1)
        assert stats["baseline"].item() == pytest.approx(1.0)


class TestMetaOpticFlowCytometer:
    """Tests for end-to-end event simulation."""

    def test_default_single_sample(self):
        system = MetaOpticFlowCytometer(receiver=_make_receiver())
        scene = torch.ones(1, 1, 9, 9)

        features = system(scene)

        assert features.shape == (1, 1, 1, 1)
        assert features.item() == pytest.approx(1.0)

    def test_moving_particle_changes_event_features(self):
        system = MetaOpticFlowCytometer(
            receiver=_make_receiver(feature_size=(3, 3)),
            flow_pixel_pitch=0.01,
        )
        scene = torch.zeros(1, 1, 9, 9)
        scene[..., 4, 4] = 1.0
        trajectory = FlowTrajectory.linear(
            start=(-0.03, 0.0),
            velocity=(0.03, 0.0),
            num_steps=3,
        )

        features, detector_frames = system(
            scene,
            trajectory=trajectory,
            return_frames=True,
        )

        assert features.shape == (1, 3, 1, 1)
        assert detector_frames.shape == (1, 3, 1, 3, 3)
        assert not torch.allclose(features[:, 0], features[:, 1])

    def test_frontend_and_backend_are_applied(self):
        backend = torch.nn.Identity()
        system = MetaOpticFlowCytometer(
            receiver=_make_receiver(),
            frontend=lambda scene: scene * 0.5,
            backend=backend,
        )
        scene = torch.ones(1, 1, 9, 9)

        prediction = system.predict(scene)

        assert prediction.item() == pytest.approx(0.5)

    def test_optimizer_contains_receiver_parameters(self):
        system = MetaOpticFlowCytometer(receiver=_make_receiver())

        optimizer = system.get_optimizer(receiver_lr=0.02)

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.02)
        assert system.receiver.psfs.requires_grad

    def test_backend_can_consume_pulse_statistics(self):
        backend = torch.nn.Linear(4, 2)
        system = MetaOpticFlowCytometer(
            receiver=_make_receiver(),
            backend=backend,
            backend_input="pulse_stats",
        )
        scene = torch.ones(1, 1, 9, 9)

        prediction = system.predict(scene)

        assert prediction.shape == (1, 2)

    def test_detector_model_is_applied_before_feature_extraction(self):
        detector = DetectorModel(
            bit_depth=12,
            full_well=1000.0,
            gain=1.0,
            noise=False,
        )
        system = MetaOpticFlowCytometer(
            receiver=_make_receiver(),
            detector_model=detector,
        )
        scene = torch.ones(1, 1, 9, 9)

        features = system(scene)

        assert features.item() == pytest.approx(1000.0)
