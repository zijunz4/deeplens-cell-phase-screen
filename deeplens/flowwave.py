# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/vccimaging/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Coherent wave propagation through metalenses and a 2D phase-screen cell."""

import math

import torch
import torch.nn.functional as F

from .base import DeepObj
from .imgsim import DetectorModel
from .light import AngularSpectrumMethod


def _align_field(field, target_shape):
    """Center-crop or zero-pad a field to the common simulation grid."""
    height, width = field.shape[-2:]
    target_h, target_w = target_shape
    if height > target_h or width > target_w:
        top = (height - target_h) // 2
        left = (width - target_w) // 2
        return field[..., top : top + target_h, left : left + target_w]
    if height < target_h or width < target_w:
        pad_top = (target_h - height) // 2
        pad_bottom = target_h - height - pad_top
        pad_left = (target_w - width) // 2
        pad_right = target_w - width - pad_left
        return F.pad(
            field,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )
    return field


class PhaseScreenCell(DeepObj):
    """A thin, lossless two-dimensional cell represented by a phase map.

    Attributes:
        phase_map (torch.Tensor): Cell phase [H, W], in radians.
        pixel_pitch (float): Cell-plane pixel pitch [mm].
    """

    def __init__(self, phase_map, pixel_pitch, device="cpu"):
        """Initialize a 2D phase screen.

        Args:
            phase_map (torch.Tensor): Real phase map [H, W]. [rad]
            pixel_pitch (float): Phase-map sample spacing. [mm]
            device (str, optional): Tensor device. Defaults to ``"cpu"``.

        Raises:
            ValueError: If the phase map or pitch is invalid.
        """
        super().__init__()
        if not torch.is_tensor(phase_map) or phase_map.ndim != 2:
            raise ValueError("phase_map must be a real tensor with shape [H, W].")
        if phase_map.is_complex() or not bool(torch.isfinite(phase_map).all()):
            raise ValueError("phase_map must contain finite real phase values.")
        if not math.isfinite(float(pixel_pitch)) or pixel_pitch <= 0:
            raise ValueError("pixel_pitch must be finite and positive.")
        self.phase_map = phase_map.to(device=device)
        self.pixel_pitch = float(pixel_pitch)
        self.to(device)

    def transmission(self, position=None, target_shape=None, dtype=None):
        """Return the translated unit-amplitude complex transmission map.

        Args:
            position (tensor or None): Lateral shift ``[x, y]``. [mm]
            target_shape (tuple or None): Optional common grid shape ``(H, W)``.
            dtype (torch.dtype or None): Optional complex output dtype.

        Returns:
            transmission (torch.Tensor): ``exp(i*phase)`` on the requested grid.
        """
        phase = self.phase_map
        if position is not None:
            shift = torch.as_tensor(
                position, device=phase.device, dtype=phase.dtype
            ).reshape(-1)
            if shift.numel() != 2:
                raise ValueError("position must contain [x, y].")
            height, width = phase.shape
            theta = phase.new_zeros(1, 2, 3)
            theta[:, 0, 0] = 1.0
            theta[:, 1, 1] = 1.0
            theta[:, 0, 2] = -2 * shift[0] / (self.pixel_pitch * width)
            theta[:, 1, 2] = -2 * shift[1] / (self.pixel_pitch * height)
            grid = F.affine_grid(theta, (1, 1, height, width), align_corners=False)
            phase = F.grid_sample(
                phase[None, None],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[0, 0]
        if target_shape is not None:
            phase = _align_field(phase[None, None], target_shape)[0, 0]
        if dtype is None:
            dtype = (
                torch.complex128 if phase.dtype == torch.float64 else torch.complex64
            )
        return torch.exp(1j * phase).to(dtype=dtype)


class CoherentMetaOpticFlowSystem(DeepObj):
    """Coherent source-array / 2D-cell / receiver-array wave-optics system.

    `source_metasurfaces` contains one or more serial metalens arrays. Each
    surface's ``d_next`` is the free-space distance to the following plane;
    ``source_to_cell`` is the final source-plane-to-cell gap. The cell is a
    single phase screen. The receiver is followed by a square-law detector.
    """

    def __init__(
        self,
        source_metasurfaces,
        receiver_metasurface,
        res,
        ps,
        source_to_cell,
        cell_to_receiver,
        receiver_to_sensor,
        cell=None,
        detector_model=None,
        device="cpu",
    ):
        """Configure the planes and propagation distances.

        Args:
            source_metasurfaces (sequence): One or more source-side metalenses.
            receiver_metasurface (DiffractiveSurface): Receiver metalens array.
            res (tuple): Common simulation grid ``(H, W)``. [pixel]
            ps (float): Grid pixel pitch. [mm]
            source_to_cell (float): Last source metalens to cell distance. [mm]
            cell_to_receiver (float): Cell to receiver metalens distance. [mm]
            receiver_to_sensor (float): Receiver to sensor distance. [mm]
            cell (PhaseScreenCell or None): Thin cell phase screen.
            detector_model (DetectorModel or None): Optional detector response.
            device (str): Complex-field device.
        """
        super().__init__()
        if (
            not isinstance(source_metasurfaces, (list, tuple))
            or not source_metasurfaces
        ):
            raise ValueError("source_metasurfaces must be a nonempty sequence.")
        if len(res) != 2 or any(int(v) <= 0 for v in res):
            raise ValueError("res must contain two positive integers.")
        if ps <= 0:
            raise ValueError("ps must be positive.")
        for name, distance in {
            "source_to_cell": source_to_cell,
            "cell_to_receiver": cell_to_receiver,
            "receiver_to_sensor": receiver_to_sensor,
        }.items():
            if distance <= 0:
                raise ValueError(f"{name} must be positive.")
        if cell is not None and not isinstance(cell, PhaseScreenCell):
            raise TypeError("cell must be a PhaseScreenCell or None.")
        surfaces = [*source_metasurfaces, receiver_metasurface]
        if any(not hasattr(surface, "get_phase_map") for surface in surfaces):
            raise TypeError("All metalenses must provide a diffractive phase map.")
        if any(
            not math.isclose(float(surface.ps), float(ps), rel_tol=0, abs_tol=1e-12)
            for surface in surfaces
        ):
            raise ValueError("All metalenses must use the simulation pixel pitch.")
        if cell is not None and not math.isclose(
            cell.pixel_pitch, float(ps), rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(
                "The cell phase screen must use the simulation pixel pitch."
            )
        if detector_model is not None and not isinstance(detector_model, DetectorModel):
            raise TypeError("detector_model must be a DetectorModel or None.")

        self.source_metasurfaces = list(source_metasurfaces)
        self.receiver_metasurface = receiver_metasurface
        self.cell = cell
        self.detector_model = detector_model
        self.res = (int(res[0]), int(res[1]))
        self.ps = float(ps)
        self.source_to_cell = float(source_to_cell)
        self.cell_to_receiver = float(cell_to_receiver)
        self.receiver_to_sensor = float(receiver_to_sensor)
        self.to(device)

    def __call__(self, trajectory, **kwargs):
        """Alias for ``forward`` preserving keyword arguments."""
        return self.forward(trajectory, **kwargs)

    def _apply_metasurface(self, field, metasurface, wvln):
        """Apply one surface's phase and opaque-aperture transmission."""
        phase = _align_field(
            metasurface.get_phase_map(wvln)[None, None].to(
                device=field.device, dtype=field.real.dtype
            ),
            self.res,
        )[0, 0]
        transmission = _align_field(
            metasurface.get_transmission_map(
                dtype=field.real.dtype, device=field.device
            )[None, None],
            self.res,
        )[0, 0]
        return field * transmission * torch.exp(1j * phase)

    def _propagate(self, field, distance, wvln):
        """Propagate the complex field by the angular-spectrum method."""
        return AngularSpectrumMethod(
            field, z=distance, wvln=wvln, ps=self.ps, padding=True
        )

    def simulate_frame(self, position=None, wvln=0.55, return_records=False):
        """Simulate a plane wave at one lateral cell position.

        Returns:
            intensity (torch.Tensor): Sensor intensity ``[1, 1, H, W]``.
            records (dict, optional): Complex field at every physical plane.
        """
        if position is None:
            position = torch.zeros(2, device=self.device)
        position = torch.as_tensor(position, device=self.device).reshape(-1)
        if position.numel() != 2:
            raise ValueError("position must contain [x, y].")

        field = torch.ones(1, 1, *self.res, device=self.device, dtype=torch.complex128)
        records = {"laser": field.detach().clone()}
        for index, metasurface in enumerate(self.source_metasurfaces):
            field = self._apply_metasurface(field, metasurface, wvln)
            key = f"source_metasurface_{index + 1}"
            records[key] = field.detach().clone()
            distance = (
                self.source_to_cell
                if index == len(self.source_metasurfaces) - 1
                else float(metasurface.d_next.detach())
            )
            field = self._propagate(field, distance, wvln)
            if index < len(self.source_metasurfaces) - 1:
                records[f"source_gap_{index + 1}"] = field.detach().clone()

        records["cell_incident"] = field.detach().clone()
        if self.cell is not None:
            transmission = self.cell.transmission(
                position=position, target_shape=self.res, dtype=field.dtype
            )
            field = field * transmission[None, None]
        records["cell"] = field.detach().clone()
        field = self._propagate(field, self.cell_to_receiver, wvln)
        records["receiver_incident"] = field.detach().clone()
        field = self._apply_metasurface(field, self.receiver_metasurface, wvln)
        records["receiver_metasurface"] = field.detach().clone()
        field = self._propagate(field, self.receiver_to_sensor, wvln)
        intensity = field.abs().square()
        if return_records:
            records["detector_field"] = field.detach().clone()
            return intensity, records
        return intensity

    def forward(self, trajectory, wvln=0.55, return_intensity=False):
        """Simulate sensor intensity across a lateral cell trajectory."""
        intensity_frames = [
            self.simulate_frame(position=position, wvln=wvln)
            for position in trajectory.positions
        ]
        intensity_sequence = torch.stack(intensity_frames, dim=1)
        detector_sequence = (
            intensity_sequence
            if self.detector_model is None
            else self.detector_model(intensity_sequence)
        )
        if return_intensity:
            return detector_sequence, intensity_sequence
        return detector_sequence

    def get_optimizer_params(self, source_lr=0.01, receiver_lr=0.01):
        """Collect optimizer groups for all source and receiver metalenses."""
        params = []
        seen = set()
        for metasurface in [*self.source_metasurfaces, self.receiver_metasurface]:
            if id(metasurface) in seen:
                continue
            seen.add(id(metasurface))
            is_source = any(
                metasurface is source for source in self.source_metasurfaces
            )
            lr = source_lr if is_source else receiver_lr
            params += metasurface.get_optimizer_params(lr=lr)
        return params

    def get_optimizer(self, **kwargs):
        """Create an Adam optimizer over all metalens phase maps."""
        return torch.optim.Adam(self.get_optimizer_params(**kwargs))
