"""Tests for serial source metalenses and a 2D phase-screen cell."""

import pytest
import torch

from deeplens import CoherentMetaOpticFlowSystem, FlowTrajectory, PhaseScreenCell
from deeplens.diffractive_surface import Pixel2DMetasurfaceArray


def _metasurface(phase=0.0):
    """Create a small, fully transmitting, independently addressable plane."""
    surface = Pixel2DMetasurfaceArray(
        d_next=0.1,
        array_shape=(1, 1),
        aperture_res=(8, 8),
        aperture_ps=0.01,
        aperture_shape="square",
    )
    with torch.no_grad():
        surface.phase_maps.fill_(phase)
    return surface


def _system(source_metasurfaces=None, cell=None):
    return CoherentMetaOpticFlowSystem(
        source_metasurfaces=[_metasurface()]
        if source_metasurfaces is None
        else source_metasurfaces,
        receiver_metasurface=_metasurface(),
        res=(8, 8),
        ps=0.01,
        source_to_cell=0.1,
        cell_to_receiver=0.1,
        receiver_to_sensor=0.1,
        cell=cell,
    )


def test_phase_screen_is_unit_amplitude_and_shifted_in_two_dimensions():
    """A 2D cell changes only phase, and its requested displacement is applied."""
    phase = torch.zeros(8, 8)
    phase[3:5, 3:5] = torch.pi / 2
    cell = PhaseScreenCell(phase, pixel_pitch=0.01)
    centered = cell.transmission()
    shifted = cell.transmission(position=(0.01, 0.0))
    assert centered.shape == (8, 8)
    assert torch.allclose(centered.abs(), torch.ones_like(phase))
    assert not torch.allclose(centered, shifted)
    assert torch.count_nonzero(centered.imag) == 4
    with pytest.raises(ValueError, match=r"shape \[H, W\]"):
        PhaseScreenCell(torch.zeros(2, 8, 8), pixel_pitch=0.01)


def test_multiple_source_metalenses_are_applied_in_order():
    """Each serial source array has a separately recorded complex field."""
    source_a = _metasurface(torch.pi / 2)
    source_b = _metasurface(torch.pi / 3)
    system = _system([source_a, source_b])
    _, records = system.simulate_frame(return_records=True)
    assert list(records) == [
        "laser",
        "source_metasurface_1",
        "source_gap_1",
        "source_metasurface_2",
        "cell_incident",
        "cell",
        "receiver_incident",
        "receiver_metasurface",
        "detector_field",
    ]
    assert torch.allclose(records["source_metasurface_1"], 1j * records["laser"])
    assert torch.allclose(
        records["source_metasurface_2"],
        records["source_gap_1"]
        * torch.exp(1j * source_b.get_phase_map(0.55))[None, None],
    )


def test_empty_phase_screen_preserves_a_plane_wave():
    """No cell phase contrast leaves the uniform-field sensor unchanged."""
    cell = PhaseScreenCell(torch.zeros(8, 8), pixel_pitch=0.01)
    trajectory = FlowTrajectory(positions=((0.0, 0.0), (0.01, 0.0)), times=(0.0, 0.1))
    detector, intensity = _system(cell=cell)(trajectory, return_intensity=True)
    assert detector.shape == (1, 2, 1, 8, 8)
    assert torch.isfinite(detector).all()
    assert intensity.mean().item() == pytest.approx(1.0, rel=2e-2)
    assert torch.allclose(intensity[:, 0], intensity[:, 1], atol=1e-6)


def test_moving_2d_phase_screen_changes_sensor_intensity():
    """Translation of one thin phase map changes the coherent sensor field."""
    phase = torch.zeros(8, 8)
    phase[2:6, 2:6] = 1.3
    cell = PhaseScreenCell(phase, pixel_pitch=0.01)
    trajectory = FlowTrajectory(positions=((-0.02, 0.0), (0.02, 0.0)), times=(0.0, 0.1))
    intensity = _system(cell=cell)(trajectory, return_intensity=True)[1]
    assert not torch.allclose(intensity[:, 0], intensity[:, 1])


def test_multiple_surface_phase_gradients_and_records():
    """Both source planes and the receiver remain differentiable."""
    source_a, source_b, receiver = _metasurface(), _metasurface(), _metasurface()
    system = CoherentMetaOpticFlowSystem(
        [source_a, source_b], receiver, (8, 8), 0.01, 0.1, 0.1, 0.1
    )
    for surface in (source_a, source_b, receiver):
        surface.phase_maps.requires_grad_(True)
    intensity, records = system.simulate_frame(return_records=True)
    intensity.sum().backward()
    for surface in (source_a, source_b, receiver):
        assert surface.phase_maps.grad is not None
        assert surface.phase_maps.grad.abs().sum() > 0
    assert all(field.is_complex() for field in records.values())
    assert torch.allclose(records["detector_field"].abs().square(), intensity)
