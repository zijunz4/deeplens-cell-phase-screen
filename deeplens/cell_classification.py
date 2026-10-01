"""Cell-scale synthetic phase objects and a coherent optical classifier.

Lengths in configuration are micrometres. Conversion to DeepLens millimetres
occurs only at surface construction and propagation. These are shape phantoms,
not a model or dataset of biological cell types.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .diffractive_surface import Pixel2DMetasurfaceArray
from .light.wave import BandLimitedASM
from .utils import diff_quantize


@dataclass(frozen=True)
class CellOpticsConfig:
    """Physical experiment settings; all lengths below are in um.

    The homogeneous propagation medium has index 1.335. The metasurface is an
    ideal scalar phase sheet at a single wavelength, not a meta-atom model.
    Each of four aperture tiles has independent Pixel2D parameters.
    """

    tile_pixels: int = 32
    gap_pixels: int = 4
    pixel_um: float = 0.5
    wavelength_um: float = 0.55
    medium_index: float = 1.335
    cell_to_array_um: float = 20.0
    array_to_sensor_um: float = 40.0
    diameter_um: tuple = (10.0, 16.0)
    thickness_um: tuple = (3.0, 6.0)
    delta_n: tuple = (0.02, 0.04)
    shift_um: float = 2.0
    detector_bins: int = 8
    phase_levels: int = 0

    @property
    def pixels(self):
        """Side length of the common physical grid."""
        return 2 * self.tile_pixels + self.gap_pixels

    def __post_init__(self):
        """Reject unsupported geometry and nonphysical settings."""
        for name in ("tile_pixels", "gap_pixels", "detector_bins", "phase_levels"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.tile_pixels < 2 or not 1 <= self.detector_bins <= self.pixels:
            raise ValueError("Invalid tile size or detector bin count")
        if self.phase_levels == 1:
            raise ValueError("phase_levels must be zero (continuous) or >= 2")
        if not 0.1 < self.wavelength_um < 10:
            raise ValueError("wavelength_um must be between 0.1 and 10")
        for name in ("pixel_um", "medium_index"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("cell_to_array_um", "array_to_sensor_um", "shift_um"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("diameter_um", "thickness_um", "delta_n"):
            lo, hi = getattr(self, name)
            if not 0 < lo <= hi or not math.isfinite(hi):
                raise ValueError(f"Invalid {name} interval")
        # Largest rotated equal-area rectangle has aspect ratio 1.5.
        radius = self.diameter_um[1] * math.sqrt(math.pi * (1.5 + 1 / 1.5)) / 4
        if radius + self.shift_um >= self.pixels * self.pixel_um / 2:
            raise ValueError("Cell footprints may clip the simulation window")


def phase_cells(count, config, seed):
    """Generate balanced, paired circle/rectangle phase phantoms.

    Args:
        count (int): Even number of samples, at least four.
        config (CellOpticsConfig): Physical settings.
        seed (int): Local RNG seed; use distinct seeds for each data split.

    Returns:
        tuple: Phase [B, 1, H, W] in radians, labels [B] (circle=0), and
        per-sample metadata [B, 7]: diameter, thickness, delta_n, x, y, angle,
        rectangle aspect ratio. Length columns are in um, angle in radians.

    Both labels share identical nuisance draws within each pair, including
    geometric area pi*d^2/4. Pixelization introduces a small area mismatch.
    Phase is 2*pi*delta_n*thickness/vacuum_wavelength. The rectangle is a slab,
    and the circular mask is also a constant-thickness slab, not a sphere.
    """
    if count < 4 or count % 2:
        raise ValueError("count must be even and at least four")
    rng = torch.Generator().manual_seed(seed)
    r = torch.rand(count // 2, 7, generator=rng)
    for col, bounds in enumerate(
        (config.diameter_um, config.thickness_um, config.delta_n)
    ):
        r[:, col] = bounds[0] + r[:, col] * (bounds[1] - bounds[0])
    r[:, 3:5] = (r[:, 3:5] * 2 - 1) * config.shift_um
    r[:, 5] *= 2 * math.pi
    r[:, 6] = 1.0 + 0.5 * r[:, 6]
    metadata = r.repeat_interleave(2, dim=0)
    labels = torch.arange(count) % 2
    coord = (torch.arange(config.pixels) - (config.pixels - 1) / 2) * config.pixel_um
    y, x = torch.meshgrid(coord, coord, indexing="ij")
    d, t, dn, cx, cy, angle, aspect = metadata.T[:, :, None, None]
    x, y = x - cx, y - cy
    xr = x * angle.cos() + y * angle.sin()
    yr = -x * angle.sin() + y * angle.cos()
    area = math.pi * d.square() / 4
    circle = x.square() + y.square() <= d.square() / 4
    rectangle = (xr.abs() <= (area * aspect).sqrt() / 2) & (
        yr.abs() <= (area / aspect).sqrt() / 2
    )
    support = torch.where(labels[:, None, None] == 0, circle, rectangle)
    phase = support * (2 * math.pi * dn * t / config.wavelength_um)
    order = torch.randperm(count, generator=rng)
    return phase[order, None], labels[order], metadata[order]


class CellOpticalEncoder(nn.Module):
    """Coherent propagation through a shared 2x2 Pixel2D aperture array.

    All aperture fields interfere before square-law detection. No replicated
    input beams, incoherent PSF convolution, or per-image energy normalization
    are used. Pooling averages intensity over sensor regions in software.
    """

    def __init__(self, config):
        """Create registered phase parameters and aperture transmission buffers."""
        super().__init__()
        self.config = config
        self.surface = Pixel2DMetasurfaceArray(
            d_next=config.array_to_sensor_um * 1e-3,
            array_shape=(2, 2),
            aperture_res=config.tile_pixels,
            aperture_ps=config.pixel_um * 1e-3,
            aperture_pitch=(config.tile_pixels + config.gap_pixels)
            * config.pixel_um
            * 1e-3,
            aperture_shape="square",
            wvln0=config.wavelength_um,
        )
        self.phase = nn.Parameter(self.surface.phase_maps.clone())
        self.register_buffer("transmission", self.surface.get_transmission_map())

    def phase_map(self):
        """Assemble the registered trainable tiles on the global array grid."""
        # DeepObj is not nn.Module: bind its view to the registered Parameter
        # on every call so .to(), state_dict(), and optimization stay coherent.
        self.surface.phase_maps = self.phase
        self.surface.aperture_mask = self.surface.aperture_mask.to(self.phase.device)
        phase = self.surface.phase_func()
        if self.config.phase_levels:
            phase = diff_quantize(
                phase.remainder(2 * math.pi), self.config.phase_levels
            )
        return phase

    def propagate(self, field, distance_um):
        """Use padded band-limited ASM; the output window may lose energy."""
        if distance_um == 0:
            return field
        c = self.config
        return BandLimitedASM(
            field,
            distance_um * 1e-3,
            c.wavelength_um,
            c.pixel_um * 1e-3,
            n=c.medium_index,
            padding=True,
        )

    def forward(self, phase, return_records=False):
        """Encode a batch of real cell phases, optionally exposing complex planes."""
        if phase.ndim != 4 or phase.shape[1:] != (
            1,
            self.config.pixels,
            self.config.pixels,
        ):
            raise ValueError("phase must have shape [B, 1, pixels, pixels]")
        cell = torch.exp(1j * phase)
        incident = self.propagate(cell, self.config.cell_to_array_um)
        encoded = incident * self.transmission * torch.exp(1j * self.phase_map())
        sensor = self.propagate(encoded, self.config.array_to_sensor_um)
        intensity = sensor.abs().square()
        features = F.adaptive_avg_pool2d(intensity, self.config.detector_bins).flatten(
            1
        )
        if return_records:
            return features, {
                "cell": cell,
                "array_incident": incident,
                "array_exit": encoded,
                "sensor": sensor,
                "intensity": intensity,
            }
        return features


class CellClassifier(nn.Module):
    """Trainable optical encoder followed by a small digital MLP."""

    def __init__(self, config):
        """Initialize optics and a two-class backend."""
        super().__init__()
        self.encoder = CellOpticalEncoder(config)
        self.backend = nn.Sequential(
            nn.Linear(config.detector_bins**2, 32), nn.GELU(), nn.Linear(32, 2)
        )

    def forward(self, phase):
        """Predict logits from detected optical intensity only."""
        return self.backend(self.encoder(phase))


def supervised_contrastive(features, labels, temperature=0.1):
    """Contrast detected features using other same-label samples as positives.

    This is supervised contrastive pretraining, not self-supervised biological
    representation learning. Require at least two examples of every label.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = F.normalize(features, dim=1)
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = (labels[:, None] == labels[None, :]) & ~diagonal
    if not bool(positive.any(dim=1).all()):
        raise ValueError("Every label needs another positive in the batch")
    logits = (z @ z.T / temperature).masked_fill(diagonal, -torch.inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob.masked_fill(~positive, 0).sum(1) / positive.sum(1)).mean()
