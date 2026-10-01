# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/vccimaging/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Detector response model for image and event simulation."""

import torch

from ..base import DeepObj


class DetectorModel(DeepObj):
    """Convert normalized optical intensity to noisy, quantized detector data.

    The optical input is interpreted as a fraction of the detector full well.
    Photoelectrons are generated with quantum efficiency, saturated, corrupted
    by shot and read noise, converted to digital numbers by gain, offset by the
    black level, and clipped to the detector bit depth.

    Attributes:
        bit_depth (int): Quantization bit depth.
        full_well (float): Full-well capacity in photoelectrons.
        read_noise (float): RMS read noise in electrons.
        dark_current (float): Dark current in electrons per second.
        exposure_time (float): Integration time in seconds.
        gain (float): Conversion gain in electrons per digital number.
        black_level (float): Digital black-level offset.
        quantum_efficiency (float): Photon-to-electron conversion efficiency.
        noise (bool): Whether stochastic shot and read noise are enabled.
    """

    def __init__(
        self,
        bit_depth=10,
        full_well=10000.0,
        read_noise=5.0,
        dark_current=0.0,
        exposure_time=1.0,
        gain=1.0,
        black_level=0.0,
        quantum_efficiency=1.0,
        noise=True,
        device="cpu",
    ):
        """Initialize a detector response model.

        Args:
            bit_depth (int, optional): Quantization bit depth. Defaults to 10.
            full_well (float, optional): Full-well capacity in electrons.
                Defaults to 10000.0.
            read_noise (float, optional): RMS read noise in electrons. Defaults
                to 5.0.
            dark_current (float, optional): Dark current in electrons per
                second. Defaults to 0.0.
            exposure_time (float, optional): Integration time in seconds.
                Defaults to 1.0.
            gain (float, optional): Conversion gain in electrons per digital
                number. Defaults to 1.0.
            black_level (float, optional): Digital black-level offset.
                Defaults to 0.0.
            quantum_efficiency (float, optional): Photon-to-electron conversion
                efficiency in `(0, 1]`. Defaults to 1.0.
            noise (bool, optional): Enable shot and read noise. Defaults to
                True.
            device (str, optional): Device for the detector model. Defaults to
                "cpu".
        """
        super().__init__()
        if not 1 <= int(bit_depth) <= 32:
            raise ValueError("bit_depth must be between 1 and 32.")
        if full_well <= 0:
            raise ValueError("full_well must be positive.")
        if read_noise < 0:
            raise ValueError("read_noise must be non-negative.")
        if dark_current < 0:
            raise ValueError("dark_current must be non-negative.")
        if exposure_time <= 0:
            raise ValueError("exposure_time must be positive.")
        if gain <= 0:
            raise ValueError("gain must be positive.")
        if black_level < 0:
            raise ValueError("black_level must be non-negative.")
        if not 0 < quantum_efficiency <= 1:
            raise ValueError("quantum_efficiency must be in (0, 1].")

        self.bit_depth = int(bit_depth)
        self.full_well = float(full_well)
        self.read_noise = float(read_noise)
        self.dark_current = float(dark_current)
        self.exposure_time = float(exposure_time)
        self.gain = float(gain)
        self.black_level = float(black_level)
        self.quantum_efficiency = float(quantum_efficiency)
        self.noise = bool(noise)
        self.to(device)

    @property
    def max_digital_value(self):
        """Maximum unsigned digital value for this bit depth."""
        return float(2**self.bit_depth - 1)

    def __call__(self, intensity, add_noise=None):
        """Alias for `forward` that preserves the noise override."""
        return self.forward(intensity, add_noise=add_noise)

    def forward(self, intensity, add_noise=None):
        """Convert normalized intensity to detector digital numbers.

        Args:
            intensity (torch.Tensor): Non-negative optical intensity.
            add_noise (bool or None, optional): Override the detector noise
                setting for this call. Defaults to None.

        Returns:
            digital (torch.Tensor): Clipped detector digital numbers.
        """
        if bool((intensity < 0).any()):
            raise ValueError("intensity must be non-negative.")

        use_noise = self.noise if add_noise is None else bool(add_noise)
        electrons = (
            intensity * self.full_well * self.quantum_efficiency
            + self.dark_current * self.exposure_time
        )
        electrons = electrons.clamp(0.0, self.full_well)

        if use_noise:
            shot_sigma = torch.sqrt(electrons.clamp_min(0.0))
            electrons = electrons + torch.randn_like(electrons) * shot_sigma
            electrons = electrons + torch.randn_like(electrons) * self.read_noise

        digital = electrons / self.gain + self.black_level
        return digital.clamp(self.black_level, self.max_digital_value)
