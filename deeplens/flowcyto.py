# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/vccimaging/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""System-level simulation for multi-aperture meta-optic flow cytometry."""

import math

import torch
import torch.nn.functional as F

from .base import DeepObj
from .imgsim import DetectorModel, MetaOpticChannelArray


def extract_pulse_features(signals, times=None, threshold=0.5):
    """Extract flow-cytometry pulse features from temporal channel signals.

    The time dimension must be dimension 1. Leading dimension 0 is batch.

    Args:
        signals (torch.Tensor): Signal sequence, shape [B, T, ...].
        times (torch.Tensor or None, optional): Time samples in seconds, shape
            [T]. When None, unit sampling is used. Defaults to None.
        threshold (float, optional): Fraction of the pulse peak above the
            baseline used to define width. Defaults to 0.5.

    Returns:
        features (dict): Dictionary containing `peak`, `area`, `width`,
            `time_to_peak`, and `baseline`. Every value has shape [B, ...].
    """
    if signals.ndim < 2:
        raise ValueError("signals must have shape [B, T, ...].")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")

    if times is None:
        times = torch.arange(signals.shape[1], device=signals.device)
        dt = 1.0
    else:
        times = torch.as_tensor(times, device=signals.device).reshape(-1)
        if times.numel() != signals.shape[1]:
            raise ValueError("times must match the signal time dimension.")
        dt = float(times[1] - times[0]) if times.numel() > 1 else 1.0

    baseline = signals.amin(dim=1)
    peak = signals.amax(dim=1)
    peak_idx = signals.argmax(dim=1)
    peak_level = baseline + threshold * (peak - baseline)
    above = signals >= peak_level.unsqueeze(1)

    time_to_peak = times[peak_idx]

    return {
        "peak": peak,
        "area": signals.sum(dim=1) * dt,
        "width": above.sum(dim=1).to(signals.dtype) * dt,
        "time_to_peak": time_to_peak,
        "baseline": baseline,
    }


class FlowCytometrySpec:
    """First-order specifications and practical constraints for a flow imager.

    Attributes use millimetres for optical distances, millimetres per pixel for
    detector sampling, millimetres per second for flow velocity, and seconds for
    integration time.
    """

    def __init__(
        self,
        focal_length,
        f_number,
        half_fov_deg,
        working_distance,
        detector_pixel_pitch,
        flow_velocity,
        integration_time,
        channel_count,
        back_focal_length=None,
        total_track_length=None,
        num_elements=None,
    ):
        """Initialize a flow-system specification.

        Args:
            focal_length (float): Effective focal length. [mm]
            f_number (float): Working F-number.
            half_fov_deg (float): Half diagonal field of view. [deg]
            working_distance (float): Object distance to the first surface. [mm]
            detector_pixel_pitch (float): Detector pixel pitch. [mm/pixel]
            flow_velocity (float): Flow speed. [mm/s]
            integration_time (float): Detector integration time. [s]
            channel_count (int): Number of optical/detector channels.
            back_focal_length (float or None, optional): Back focal length. [mm]
            total_track_length (float or None, optional): Total track length. [mm]
            num_elements (int or None, optional): Number of refractive elements.
        """
        self.focal_length = float(focal_length)
        self.f_number = float(f_number)
        self.half_fov_deg = float(half_fov_deg)
        self.working_distance = float(working_distance)
        self.detector_pixel_pitch = float(detector_pixel_pitch)
        self.flow_velocity = float(flow_velocity)
        self.integration_time = float(integration_time)
        self.channel_count = int(channel_count)
        self.back_focal_length = (
            None if back_focal_length is None else float(back_focal_length)
        )
        self.total_track_length = (
            None if total_track_length is None else float(total_track_length)
        )
        self.num_elements = None if num_elements is None else int(num_elements)

    @property
    def aperture_diameter(self):
        """Entrance-pupil diameter from focal length and F-number. [mm]"""
        return self.focal_length / self.f_number

    @property
    def image_diameter(self):
        """Paraxial full image diameter from focal length and half-FOV. [mm]"""
        return 2.0 * self.focal_length * math.tan(math.radians(self.half_fov_deg))

    @property
    def detector_nyquist_frequency(self):
        """Detector Nyquist frequency. [line pairs/mm]"""
        return 1.0 / (2.0 * self.detector_pixel_pitch)

    @property
    def flow_pixels_per_frame(self):
        """Lateral image-plane displacement during one integration time. [pixel]"""
        return self.flow_velocity * self.integration_time / self.detector_pixel_pitch

    def validate(self):
        """Return practical specification violations.

        Returns:
            violations (list of str): Empty when the specification is valid.
        """
        violations = []
        positive = {
            "focal_length": self.focal_length,
            "f_number": self.f_number,
            "half_fov_deg": self.half_fov_deg,
            "working_distance": self.working_distance,
            "detector_pixel_pitch": self.detector_pixel_pitch,
            "flow_velocity": self.flow_velocity,
            "integration_time": self.integration_time,
        }
        for name, value in positive.items():
            if value <= 0:
                violations.append(f"{name} must be positive.")
        if self.channel_count < 1:
            violations.append("channel_count must be at least 1.")
        if self.back_focal_length is not None and self.back_focal_length < 0:
            violations.append("back_focal_length must be non-negative.")
        if self.total_track_length is not None:
            if self.total_track_length <= 0:
                violations.append("total_track_length must be positive.")
            if (
                self.back_focal_length is not None
                and self.total_track_length <= self.back_focal_length
            ):
                violations.append("total_track_length must exceed back_focal_length.")
        if self.num_elements is not None and not 2 <= self.num_elements <= 20:
            violations.append("num_elements must be between 2 and 20.")
        return violations

    def require_valid(self):
        """Raise `ValueError` when the specification violates constraints."""
        violations = self.validate()
        if violations:
            raise ValueError(" ".join(violations))

    def summary(self):
        """Return first-order and flow quantities as a dictionary."""
        return {
            "focal_length": self.focal_length,
            "f_number": self.f_number,
            "aperture_diameter": self.aperture_diameter,
            "half_fov_deg": self.half_fov_deg,
            "image_diameter": self.image_diameter,
            "working_distance": self.working_distance,
            "back_focal_length": self.back_focal_length,
            "total_track_length": self.total_track_length,
            "num_elements": self.num_elements,
            "channel_count": self.channel_count,
            "detector_pixel_pitch": self.detector_pixel_pitch,
            "detector_nyquist_frequency": self.detector_nyquist_frequency,
            "flow_velocity": self.flow_velocity,
            "integration_time": self.integration_time,
            "flow_pixels_per_frame": self.flow_pixels_per_frame,
        }


class FlowTrajectory(DeepObj):
    """Particle trajectory and time samples in the flow interrogation region.

    Attributes:
        positions (torch.Tensor): Lateral positions as [x, y] in millimetres,
            shape [T, 2].
        times (torch.Tensor): Time samples in seconds, shape [T].
    """

    def __init__(self, positions, times=None, device="cpu"):
        """Initialize a flow trajectory.

        Args:
            positions (torch.Tensor): Lateral positions as [x, y] in mm,
                shape [T, 2].
            times (torch.Tensor or None, optional): Time samples in seconds,
                shape [T]. When None, unit sampling is used. Defaults to None.
            device (str, optional): Device for trajectory tensors. Defaults to
                "cpu".
        """
        super().__init__()
        positions = torch.as_tensor(positions, device=device)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(
                f"positions must have shape [T, 2], got {positions.shape}."
            )
        if positions.shape[0] < 1:
            raise ValueError("A flow trajectory must contain at least one sample.")

        if times is None:
            times = torch.arange(positions.shape[0], device=device)
        times = torch.as_tensor(times, device=device).reshape(-1)
        if times.numel() != positions.shape[0]:
            raise ValueError("times and positions must contain the same sample count.")
        if bool((times[1:] <= times[:-1]).any()):
            raise ValueError("times must be strictly increasing.")

        self.positions = positions
        self.times = times
        self.to(device)

    @classmethod
    def linear(
        cls,
        start=(0.0, 0.0),
        velocity=(1.0, 0.0),
        num_steps=10,
        dt=1.0,
        device="cpu",
    ):
        """Create a constant-velocity trajectory.

        Args:
            start (tuple, optional): Initial [x, y] position in mm. Defaults to
                `(0.0, 0.0)`.
            velocity (tuple, optional): Velocity [vx, vy] in mm/s. Defaults to
                `(1.0, 0.0)`.
            num_steps (int, optional): Number of time samples. Defaults to 10.
            dt (float, optional): Sampling interval in seconds. Defaults to 1.0.
            device (str, optional): Device for trajectory tensors. Defaults to
                `"cpu"`.

        Returns:
            trajectory (FlowTrajectory): Constant-velocity trajectory.
        """
        if num_steps < 1:
            raise ValueError("num_steps must be positive.")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        start = torch.as_tensor(start, device=device, dtype=torch.get_default_dtype())
        velocity = torch.as_tensor(
            velocity, device=device, dtype=torch.get_default_dtype()
        )
        times = torch.arange(num_steps, device=device, dtype=torch.get_default_dtype())
        times = times * dt
        positions = start.unsqueeze(0) + velocity.unsqueeze(0) * times.unsqueeze(-1)
        return cls(positions=positions, times=times, device=device)


class MetaOpticFlowCytometer(DeepObj):
    """Practical multi-aperture meta-optic flow-cytometry system model.

    The system optionally sends an input scene through a refractive frontend,
    shifts it according to a flow trajectory, and evaluates a tiled
    `MetaOpticChannelArray`. The output contains both physical detector frames
    and signed channel features versus time.

    The refractive `frontend` is responsible for collection, working distance,
    magnification, and baseline aberration correction. The meta-optic receiver
    performs channel encoding and feature extraction.

    Attributes:
        receiver (MetaOpticChannelArray): Multi-aperture detector readout.
        trajectory (FlowTrajectory or None): Default flow trajectory.
        frontend (object or None): Optional refractive or hybrid lens object.
        backend (torch.nn.Module or None): Optional digital classifier.
        flow_pixel_pitch (float): Pixel pitch in the plane where flow motion is
            applied. [mm/pixel]
    """

    def __init__(
        self,
        receiver,
        trajectory=None,
        frontend=None,
        backend=None,
        spec=None,
        detector_model=None,
        flow_pixel_pitch=0.001,
        flow_shift_stage="image",
        backend_input="features",
        object_depth=-10000.0,
        render_method="psf_patch",
        device="cpu",
    ):
        """Initialize a meta-optic flow-cytometry system.

        Args:
            receiver (MetaOpticChannelArray): Multi-aperture receiver and
                detector readout.
            trajectory (FlowTrajectory or None, optional): Default trajectory.
                If None, `forward()` creates a single zero-position sample.
            frontend (object or None, optional): Refractive or hybrid lens with
                a `render()` method, or a callable applied before the receiver.
            backend (torch.nn.Module or None, optional): Optional digital
                classifier applied to the event features.
            spec (FlowCytometrySpec or None, optional): Practical first-order
                specification. When provided, it is validated at initialization.
            detector_model (DetectorModel or None, optional): Detector response
                applied before channel feature extraction.
            flow_pixel_pitch (float, optional): Pixel pitch in the plane where
                lateral flow displacement is applied. [mm/pixel]. Defaults to
                0.001.
            flow_shift_stage (str, optional): Apply flow displacement before the
                frontend (`"object"`) or after it (`"image"`). Defaults to
                `"image"`.
            backend_input (str, optional): Input supplied to the backend,
                either `"features"` or `"pulse_stats"`. Defaults to
                `"features"`.
            object_depth (float, optional): Object depth passed to the
                frontend renderer. Defaults to -10000.0.
            render_method (str, optional): Rendering method passed to the
                frontend. Defaults to `"psf_patch"`.
            device (str, optional): Device for system state. Defaults to
                `"cpu"`.
        """
        super().__init__()
        if not isinstance(receiver, MetaOpticChannelArray):
            raise TypeError("receiver must be a MetaOpticChannelArray.")
        if trajectory is not None and not isinstance(trajectory, FlowTrajectory):
            raise TypeError("trajectory must be a FlowTrajectory or None.")
        if spec is not None:
            if not isinstance(spec, FlowCytometrySpec):
                raise TypeError("spec must be a FlowCytometrySpec or None.")
            spec.require_valid()
        if detector_model is not None and not isinstance(detector_model, DetectorModel):
            raise TypeError("detector_model must be a DetectorModel or None.")
        if flow_pixel_pitch <= 0:
            raise ValueError("flow_pixel_pitch must be positive.")
        if flow_shift_stage not in {"object", "image"}:
            raise ValueError("flow_shift_stage must be 'object' or 'image'.")
        if backend_input not in {"features", "pulse_stats"}:
            raise ValueError("backend_input must be 'features' or 'pulse_stats'.")

        self.receiver = receiver
        self.trajectory = trajectory
        self.frontend = frontend
        self.backend = backend
        self.spec = spec
        self.detector_model = detector_model
        self.flow_pixel_pitch = float(flow_pixel_pitch)
        self.flow_shift_stage = flow_shift_stage
        self.backend_input = backend_input
        self.object_depth = float(object_depth)
        self.render_method = render_method
        self.to(device)

    def __call__(self, scene, **kwargs):
        """Alias for `forward` that preserves keyword arguments."""
        return self.forward(scene, **kwargs)

    def _apply_frontend(self, scene):
        """Apply the refractive or hybrid frontend to an input scene."""
        if self.frontend is None:
            return scene
        if hasattr(self.frontend, "render"):
            return self.frontend.render(
                scene,
                depth=self.object_depth,
                method=self.render_method,
            )
        if callable(self.frontend):
            return self.frontend(scene)
        raise TypeError("frontend must define render() or be callable.")

    def _shift_scene(self, scene, position):
        """Translate a scene by a physical [x, y] displacement in millimetres."""
        batch = scene.shape[0]
        height, width = scene.shape[-2:]
        shift_x = float(position[0]) / self.flow_pixel_pitch
        shift_y = float(position[1]) / self.flow_pixel_pitch

        theta = scene.new_zeros(batch, 2, 3)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = -2.0 * shift_x / width
        theta[:, 1, 2] = -2.0 * shift_y / height
        grid = F.affine_grid(theta, scene.shape, align_corners=False)
        return F.grid_sample(
            scene,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    def forward(
        self,
        scene,
        trajectory=None,
        reduce="mean",
        return_frames=False,
    ):
        """Simulate detector frames and event features over the flow trajectory.

        Args:
            scene (torch.Tensor): Input scene in raw space, shape [B, C, H, W].
            trajectory (FlowTrajectory or None, optional): Trajectory override.
                When None, uses the system trajectory; if that is also None,
                one zero-displacement sample is used.
            reduce (str or None, optional): Feature reduction passed to the
                receiver. `"mean"` returns one scalar per channel. Defaults to
                `"mean"`.
            return_frames (bool, optional): Also return detector frames.
                Defaults to False.

        Returns:
            features (torch.Tensor): Event features. With `reduce="mean"`, shape
                is [B, T, C, F]. Without reduction, shape is
                [B, T, C, F, feature_H, feature_W].
            detector_frames (torch.Tensor): Returned only when
                `return_frames=True`, with shape [B, T, C, detector_H, detector_W].
        """
        trajectory = self.trajectory if trajectory is None else trajectory
        if trajectory is None:
            trajectory = FlowTrajectory.linear(
                start=(0.0, 0.0),
                velocity=(0.0, 0.0),
                num_steps=1,
                dt=1.0,
                device=self.device,
            )

        if self.flow_shift_stage == "image":
            scene = self._apply_frontend(scene)
        detector_frames = []
        feature_sequence = []
        for position in trajectory.positions:
            if self.flow_shift_stage == "object":
                shifted = self._apply_frontend(self._shift_scene(scene, position))
            else:
                shifted = self._shift_scene(scene, position)
            detector = self.receiver.forward(shifted)
            if self.detector_model is not None:
                detector = self.detector_model(detector)
            detector_frames.append(detector)
            feature_sequence.append(
                self.receiver.extract_features(detector, reduce=reduce)
            )

        features = torch.stack(feature_sequence, dim=1)
        if return_frames:
            return features, torch.stack(detector_frames, dim=1)
        return features

    def event_statistics(self, features, trajectory=None, threshold=0.5):
        """Extract pulse statistics from an event feature sequence.

        Args:
            features (torch.Tensor): Event features with shape [B, T, ...].
            trajectory (FlowTrajectory or None, optional): Trajectory providing
                time samples. When None, the system trajectory or unit sampling
                is used.
            threshold (float, optional): Fractional pulse threshold used for
                width. Defaults to 0.5.

        Returns:
            stats (dict): Pulse height, area, width, time-to-peak, and baseline.
        """
        trajectory = self.trajectory if trajectory is None else trajectory
        times = None if trajectory is None else trajectory.times
        return extract_pulse_features(features, times=times, threshold=threshold)

    def _format_backend_input(self, features, trajectory):
        """Convert feature sequences into the configured backend input."""
        if self.backend_input == "features":
            return features

        stats = self.event_statistics(features, trajectory=trajectory)
        keys = ("peak", "area", "width", "time_to_peak")
        return torch.cat([stats[key].flatten(start_dim=1) for key in keys], dim=-1)

    def predict(self, scene, trajectory=None, reduce="mean"):
        """Run the optical system and optional digital classifier.

        Args:
            scene (torch.Tensor): Input scene, shape [B, C, H, W].
            trajectory (FlowTrajectory or None, optional): Trajectory override.
            reduce (str or None, optional): Feature reduction. Defaults to
                `"mean"`.

        Returns:
            prediction (torch.Tensor): Classifier output when a backend exists,
                otherwise the event features.
        """
        features = self.forward(scene, trajectory=trajectory, reduce=reduce)
        if self.backend is None:
            return features
        return self.backend(self._format_backend_input(features, trajectory))

    def get_optimizer_params(
        self,
        frontend_lrs=(1e-4, 1e-4, 1e-2, 1e-5),
        receiver_lr=0.01,
        backend_lr=1e-3,
    ):
        """Collect optimizer parameter groups from the complete system.

        Args:
            frontend_lrs (sequence, optional): Learning rates passed to the
                frontend optimizer.
            receiver_lr (float, optional): Learning rate for receiver PSFs.
            backend_lr (float, optional): Learning rate for the digital backend.

        Returns:
            params (list): Parameter groups for `torch.optim.Adam`.
        """
        params = []
        if self.frontend is not None and hasattr(self.frontend, "get_optimizer_params"):
            params += self.frontend.get_optimizer_params(lrs=frontend_lrs)
        params += self.receiver.get_optimizer_params(lr=receiver_lr)
        if self.backend is not None:
            backend_params = [p for p in self.backend.parameters() if p.requires_grad]
            if backend_params:
                params.append({"params": backend_params, "lr": backend_lr})
        return params

    def get_optimizer(self, **kwargs):
        """Create an Adam optimizer over frontend, receiver, and backend."""
        return torch.optim.Adam(self.get_optimizer_params(**kwargs))
