# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/vccimaging/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Multi-aperture metasurface array built from independent Pixel2D phase maps."""

import math

import torch
import torch.nn.functional as F

from ..utils import diff_quantize
from .base_diffractive import DiffractiveSurface


class Pixel2DMetasurfaceArray(DiffractiveSurface):
    """Planar array of independently addressable Pixel2D metasurface apertures.

    Each aperture contains an independent `[H, W]` phase map. The apertures are
    placed on a centered rectangular lattice in one plane, and the class
    composes them into a single full-array phase map for efficient wave-optics
    propagation. Opaque regions between or outside apertures are represented by
    a transmission mask.

    The global array grid uses the same pixel pitch as the aperture phase maps.
    Therefore, `aperture_pitch` must be an integer multiple of `aperture_ps`.

    Attributes:
        phase_maps (torch.Tensor): Independent phase maps at the design
            wavelength, shape [array_rows, array_cols, aperture_H, aperture_W].
            [rad]
        array_shape (tuple): Array size as (rows, cols).
        aperture_res (tuple): Resolution of one aperture as (H, W). [pixel]
        aperture_ps (float): Pixel pitch of the phase maps and global array.
            [mm]
        aperture_pitch (tuple): Center-to-center lattice pitch as
            (pitch_y, pitch_x). [mm]
        aperture_shape (str): Active aperture shape, `"circle"` or `"square"`.
        fill_factor (float): Fraction of the aperture tile covered by the
            active aperture. Must be in `(0, 1]`.
    """

    def __init__(
        self,
        d_next,
        array_shape=(2, 2),
        aperture_res=(128, 128),
        aperture_ps=0.01,
        aperture_pitch=None,
        aperture_shape="circle",
        fill_factor=1.0,
        phase_maps=None,
        phase_maps_path=None,
        mat="fused_silica",
        wvln0=0.55,
        fab_ps=0.001,
        fab_step=16,
        device="cpu",
    ):
        """Initialize a Pixel2D multi-aperture metasurface array.

        Args:
            d_next (float): Axial distance to the next plane. [mm]
            array_shape (int or tuple, optional): Array size as `(rows, cols)`.
                An int creates a square array. Defaults to `(2, 2)`.
            aperture_res (int or tuple, optional): Resolution of each aperture
                as `(H, W)`. An int creates a square aperture. Defaults to
                `(128, 128)`.
            aperture_ps (float, optional): Phase-map pixel pitch. [mm].
                Defaults to 0.01.
            aperture_pitch (float or tuple, optional): Center-to-center pitch
                as `(pitch_y, pitch_x)`. A scalar applies to both axes. When
                None, apertures are contiguous and the pitch equals their
                physical size. [mm]. Defaults to None.
            aperture_shape (str, optional): `"circle"` or `"square"`.
                Defaults to `"circle"`.
            fill_factor (float, optional): Active-area scale within each
                aperture tile. Must be in `(0, 1]`. Defaults to 1.0.
            phase_maps (torch.Tensor or None, optional): Initial aperture
                phase maps with shape `[rows, cols, H, W]`. Defaults to None.
            phase_maps_path (str or None, optional): Path to a saved
                `phase_maps` tensor. If None, phase maps start with small
                random values. Defaults to None.
            mat (str, optional): Metasurface material. Defaults to
                `"fused_silica"`.
            wvln0 (float, optional): Design wavelength. [um]. Defaults to 0.55.
            fab_ps (float, optional): Fabrication pixel size. [mm]. Defaults
                to 0.001.
            fab_step (int, optional): Number of fabrication quantization
                levels. Defaults to 16.
            device (str, optional): Device for phase maps and masks. Defaults
                to `"cpu"`.

        Raises:
            ValueError: If the array geometry is invalid or the lattice pitch
                is not an integer multiple of `aperture_ps`.
        """
        array_shape = (
            (array_shape, array_shape) if isinstance(array_shape, int) else array_shape
        )
        aperture_res = (
            (aperture_res, aperture_res)
            if isinstance(aperture_res, int)
            else aperture_res
        )
        if len(array_shape) != 2 or any(int(v) <= 0 for v in array_shape):
            raise ValueError("array_shape must contain two positive integers.")
        if len(aperture_res) != 2 or any(int(v) <= 0 for v in aperture_res):
            raise ValueError("aperture_res must contain two positive integers.")
        if aperture_ps <= 0:
            raise ValueError("aperture_ps must be positive.")
        if not 0 < fill_factor <= 1:
            raise ValueError("fill_factor must be in (0, 1].")

        self.array_shape = (int(array_shape[0]), int(array_shape[1]))
        self.aperture_res = (int(aperture_res[0]), int(aperture_res[1]))
        self.aperture_ps = float(aperture_ps)
        self.aperture_shape = str(aperture_shape).lower()
        if self.aperture_shape not in {"circle", "square"}:
            raise ValueError("aperture_shape must be 'circle' or 'square'.")
        self.fill_factor = float(fill_factor)

        aperture_h = self.aperture_res[0] * self.aperture_ps
        aperture_w = self.aperture_res[1] * self.aperture_ps
        if aperture_pitch is None:
            pitch_y, pitch_x = aperture_h, aperture_w
        elif isinstance(aperture_pitch, (int, float)):
            pitch_y = pitch_x = float(aperture_pitch)
        elif len(aperture_pitch) == 2:
            pitch_y, pitch_x = (float(v) for v in aperture_pitch)
        else:
            raise ValueError("aperture_pitch must be a scalar or (pitch_y, pitch_x).")
        if pitch_y < aperture_h or pitch_x < aperture_w:
            raise ValueError(
                "aperture_pitch must be at least the aperture size on each axis."
            )

        self.aperture_pitch = (pitch_y, pitch_x)
        pitch_y_px = self._pitch_to_pixels(pitch_y)
        pitch_x_px = self._pitch_to_pixels(pitch_x)

        global_h = (self.array_shape[0] - 1) * pitch_y_px + self.aperture_res[0]
        global_w = (self.array_shape[1] - 1) * pitch_x_px + self.aperture_res[1]
        super().__init__(
            d_next=d_next,
            res=(global_h, global_w),
            mat=mat,
            fab_ps=fab_ps,
            fab_step=fab_step,
            wvln0=wvln0,
            design_ps=self.aperture_ps,
            is_square=global_h == global_w,
            device=device,
        )

        self.aperture_mask = self._make_aperture_mask()
        self.transmission_mask = self._make_transmission_mask(pitch_y_px, pitch_x_px)

        if phase_maps is not None and phase_maps_path is not None:
            raise ValueError("Provide phase_maps or phase_maps_path, not both.")
        if phase_maps is not None:
            if not torch.is_tensor(phase_maps):
                raise ValueError("phase_maps must be a torch.Tensor.")
            self.phase_maps = phase_maps.to(device=device)
        elif phase_maps_path is None:
            self.phase_maps = (
                torch.randn((*self.array_shape, *self.aperture_res), device=self.device)
                * 1e-3
            )
        elif isinstance(phase_maps_path, str):
            self.phase_maps = torch.load(
                phase_maps_path, map_location=device, weights_only=True
            )
        else:
            raise ValueError(f"Invalid phase_maps_path: {phase_maps_path}")

        expected_shape = (*self.array_shape, *self.aperture_res)
        if tuple(self.phase_maps.shape) != expected_shape:
            raise ValueError(
                f"phase_maps must have shape {expected_shape}, got "
                f"{tuple(self.phase_maps.shape)}."
            )
        self.to(device)

    def _pitch_to_pixels(self, pitch):
        """Convert an on-lattice pitch in mm to an integer pixel count."""
        pitch_px = pitch / self.aperture_ps
        pitch_px_round = round(pitch_px)
        if not math.isclose(pitch_px, pitch_px_round, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(
                "aperture_pitch must be an integer multiple of aperture_ps so "
                "that all apertures land exactly on the global pixel grid."
            )
        return pitch_px_round

    def _make_aperture_mask(self):
        """Create the active mask inside one aperture tile."""
        half_w = self.aperture_res[1] * self.aperture_ps / 2
        half_h = self.aperture_res[0] * self.aperture_ps / 2
        y, x = torch.meshgrid(
            torch.linspace(
                half_h - self.aperture_ps / 2,
                -half_h + self.aperture_ps / 2,
                self.aperture_res[0],
                device=self.device,
            ),
            torch.linspace(
                -half_w + self.aperture_ps / 2,
                half_w - self.aperture_ps / 2,
                self.aperture_res[1],
                device=self.device,
            ),
            indexing="xy",
        )
        if self.aperture_shape == "circle":
            radius = self.fill_factor * min(half_w, half_h)
            return x**2 + y**2 <= radius**2

        return (x.abs() <= half_w * self.fill_factor) & (
            y.abs() <= half_h * self.fill_factor
        )

    def _make_transmission_mask(self, pitch_y_px, pitch_x_px):
        """Compose aperture masks into a full-array transmission map."""
        mask = torch.zeros(self.res, dtype=torch.bool, device=self.device)
        for row in range(self.array_shape[0]):
            top = row * pitch_y_px
            for col in range(self.array_shape[1]):
                left = col * pitch_x_px
                mask[
                    top : top + self.aperture_res[0],
                    left : left + self.aperture_res[1],
                ] = self.aperture_mask
        return mask

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize a metasurface array from a serialized dictionary."""
        return cls(
            d_next=doe_dict["d_next"],
            array_shape=doe_dict.get("array_shape", (2, 2)),
            aperture_res=doe_dict.get("aperture_res", (128, 128)),
            aperture_ps=doe_dict.get(
                "aperture_ps",
                doe_dict.get("fab_ps", 0.01),
            ),
            aperture_pitch=doe_dict.get("aperture_pitch"),
            aperture_shape=doe_dict.get("aperture_shape", "circle"),
            fill_factor=doe_dict.get("fill_factor", 1.0),
            phase_maps=doe_dict.get("phase_maps"),
            phase_maps_path=doe_dict.get("phase_maps_path"),
            mat=doe_dict.get("mat", "fused_silica"),
            wvln0=doe_dict.get("wvln0", 0.55),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
        )

    def phase_func(self):
        """Compose the independent aperture maps into one global phase map.

        Returns:
            phase_map (torch.Tensor): Full-array phase map. [H, W]. [rad]
        """
        phase_map = self.phase_maps.new_zeros(self.res)
        pitch_y_px = self._pitch_to_pixels(self.aperture_pitch[0])
        pitch_x_px = self._pitch_to_pixels(self.aperture_pitch[1])
        for row in range(self.array_shape[0]):
            top = row * pitch_y_px
            for col in range(self.array_shape[1]):
                left = col * pitch_x_px
                phase_map[
                    top : top + self.aperture_res[0],
                    left : left + self.aperture_res[1],
                ] = self.phase_maps[row, col] * self.aperture_mask
        return phase_map

    def get_transmission_map(self, dtype=None, device=None):
        """Return the full-array amplitude transmission mask.

        Args:
            dtype (torch.dtype, optional): Output dtype. Defaults to the phase
                map dtype.
            device (torch.device or str, optional): Output device. Defaults to
                the surface device.

        Returns:
            transmission (torch.Tensor): Binary transmission map. [H, W]
        """
        target_dtype = self.phase_maps.dtype if dtype is None else dtype
        target_device = self.device if device is None else device
        return self.transmission_mask.to(dtype=target_dtype, device=target_device)

    def compute_channel_psfs(
        self,
        wvln=None,
        upsample_factor=1,
        normalize=True,
    ):
        """Compute one PSF for every aperture using differentiable ASM.

        Each aperture phase map is applied independently to a unit plane wave,
        propagated by `d_next`, and converted to intensity. This bridges the
        phase-domain metasurface array and the detector-channel readout model.

        Args:
            wvln (float or None, optional): Wavelength in micrometres. When
                None, uses the design wavelength. Defaults to None.
            upsample_factor (int, optional): Integer field upsampling factor.
                Defaults to 1.
            normalize (bool, optional): Normalize each PSF to unit energy.
                Defaults to True.

        Returns:
            psfs (torch.Tensor): Per-aperture intensity PSFs, shape
                [N, 1, H * upsample_factor, W * upsample_factor].
        """
        from ..light import AngularSpectrumMethod

        if not isinstance(upsample_factor, int) or upsample_factor < 1:
            raise ValueError("upsample_factor must be a positive integer.")
        wvln = self.wvln0 if wvln is None else float(wvln)

        n = self.mat.refractive_index(wvln)
        phase_scale = (self.wvln0 / wvln) * (n - 1) / (self.n0 - 1)
        tile_h, tile_w = self.aperture_res
        ps = self.aperture_ps / upsample_factor
        complex_dtype = (
            torch.complex128
            if self.phase_maps.dtype == torch.float64
            else torch.complex64
        )

        aperture_mask = self.aperture_mask.to(dtype=self.phase_maps.dtype)
        if upsample_factor > 1:
            aperture_mask = F.interpolate(
                aperture_mask[None, None],
                scale_factor=upsample_factor,
                mode="nearest",
            )[0, 0]

        psfs = []
        for row in range(self.array_shape[0]):
            for col in range(self.array_shape[1]):
                phase = torch.remainder(self.phase_maps[row, col], 2 * torch.pi)
                phase = diff_quantize(phase, levels=self.fab_step) * phase_scale
                phase = phase * self.aperture_mask
                if upsample_factor > 1:
                    phase = F.interpolate(
                        phase[None, None],
                        scale_factor=upsample_factor,
                        mode="nearest",
                    )[0, 0]

                field = aperture_mask * torch.exp(1j * phase)
                field = field.to(complex_dtype)[None, None]
                propagated = AngularSpectrumMethod(
                    field,
                    z=float(self.d_next.detach()),
                    wvln=wvln,
                    ps=ps,
                    padding=True,
                )
                psf = propagated.abs().square()[0, 0]
                if normalize:
                    psf = psf / (psf.sum() + torch.finfo(psf.dtype).eps)
                psfs.append(psf)

        return torch.stack(psfs, dim=0).unsqueeze(1)

    def get_optimizer_params(self, lr=0.01):
        """Return an Adam parameter group for all aperture phase maps."""
        self.phase_maps.requires_grad = True
        return [{"params": [self.phase_maps], "lr": lr}]

    def surf_dict(self, phase_maps_path=None):
        """Return a serializable dictionary and save the aperture phase maps.

        Args:
            phase_maps_path (str or None, optional): Destination for the
                phase-map tensor. When None, a default filename is used.

        Returns:
            surf_dict (dict): Surface metadata including the saved tensor path.
        """
        if phase_maps_path is None:
            phase_maps_path = "./metasurface_array_phase_maps.pth"
        surf_dict = super().surf_dict()
        surf_dict.update(
            {
                "array_shape": self.array_shape,
                "aperture_res": self.aperture_res,
                "aperture_ps": self.aperture_ps,
                "aperture_pitch": self.aperture_pitch,
                "aperture_shape": self.aperture_shape,
                "fill_factor": self.fill_factor,
                "phase_maps_path": phase_maps_path,
            }
        )
        torch.save(self.phase_maps.detach().cpu(), phase_maps_path)
        return surf_dict
