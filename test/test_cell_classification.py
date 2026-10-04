"""Independent physical and learning checks for cell phase classification."""

from dataclasses import replace

import pytest
import torch

from deeplens.cell_classification import (
    CellClassifier,
    CellOpticalEncoder,
    CellOpticsConfig,
    detector_response_metrics,
    maximize_class_difference,
    optical_energy,
    phase_cells,
    supervised_contrastive,
)


def test_phantom_units_area_and_split_reproducibility():
    """OPD formula, cell footprint, paired nuisance and RNG checks."""
    c = CellOpticsConfig()
    phase, labels, metadata = phase_cells(16, c, 1)
    assert torch.equal(phase, phase_cells(16, c, 1)[0])
    assert not torch.equal(phase, phase_cells(16, c, 2)[0])
    expected = 2 * torch.pi * metadata[:, 1] * metadata[:, 2] / c.wavelength_um
    torch.testing.assert_close(phase.flatten(1).max(1).values, expected)
    torch.testing.assert_close(torch.exp(1j * phase).abs(), torch.ones_like(phase))
    areas = (phase > 0).sum((1, 2, 3)) * c.pixel_um**2
    torch.testing.assert_close(
        areas, torch.pi * metadata[:, 0] ** 2 / 4, rtol=0.06, atol=0
    )
    for row in metadata:
        assert bool((metadata == row).all(1).sum() >= 1)
    assert phase[:, :, 0].count_nonzero() == 0
    assert phase[:, :, -1].count_nonzero() == 0


def test_no_propagation_hides_phase_and_gaps_are_opaque():
    """A pure phase object cannot be classified by contact intensity."""
    c = CellOpticsConfig(cell_to_array_um=0, array_to_sensor_um=0)
    encoder = CellOpticalEncoder(c)
    phase = phase_cells(8, c, 3)[0]
    features, records = encoder(phase, True)
    torch.testing.assert_close(features, features[:1].expand_as(features))
    expected = encoder.transmission.expand_as(records["intensity"])
    torch.testing.assert_close(records["intensity"], expected)
    assert records["array_exit"][..., encoder.transmission == 0].count_nonzero() == 0


def test_passivity_and_square_law():
    """Padded/cropped propagation may lose energy, never create it."""
    c = CellOpticsConfig()
    _, records = CellOpticalEncoder(c)(phase_cells(4, c, 4)[0], True)
    previous = None
    for name in ("cell", "array_incident", "array_exit", "sensor"):
        energy = records[name].abs().square().sum((-2, -1))
        if previous is not None:
            assert bool((energy <= previous * (1 + 1e-5)).all())
        previous = energy
    torch.testing.assert_close(records["intensity"], records["sensor"].abs().square())


def test_optical_gradient_finite_difference_and_state_roundtrip():
    """Check a continuous phase derivative independently of autograd."""
    torch.manual_seed(0)
    c = CellOpticsConfig()
    model = CellClassifier(c).double()
    phase, labels, _ = phase_cells(4, c, 5)
    phase = phase.double()
    loss = torch.nn.functional.cross_entropy(model(phase), labels)
    loss.backward()
    gradient = model.encoder.phase.grad
    assert bool(torch.isfinite(gradient).all())
    index = int(gradient.abs().argmax())
    analytic = gradient.flatten()[index].item()
    assert abs(analytic) > 1e-9
    step = 1e-4
    with torch.no_grad():
        parameter = model.encoder.phase.flatten()
        original = parameter[index].item()
        parameter[index] = original + step
        plus = torch.nn.functional.cross_entropy(model(phase), labels).item()
        parameter[index] = original - step
        minus = torch.nn.functional.cross_entropy(model(phase), labels).item()
        parameter[index] = original
    assert (plus - minus) / (2 * step) == pytest.approx(analytic, rel=2e-3, abs=1e-9)
    restored = CellClassifier(c).double()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(phase), restored(phase))


def test_contrastive_gradients_and_invalid_inputs():
    """Pretraining reaches the optical parameters; bad batches fail explicitly."""
    c = CellOpticsConfig()
    encoder = CellOpticalEncoder(c)
    phase, labels, _ = phase_cells(8, c, 7)
    loss = supervised_contrastive(encoder(phase), labels)
    loss.backward()
    assert bool(torch.isfinite(encoder.phase.grad).all())
    assert encoder.phase.grad.abs().max() > 0
    with pytest.raises(ValueError):
        supervised_contrastive(torch.rand(2, 8), torch.tensor([0, 1]))
    with pytest.raises(ValueError):
        replace(c, diameter_um=(100, 150))
    assert phase_cells(7, c, 1)[1].unique().numel() == 3


def test_plane_wave_analytic_propagation():
    """Independent analytic zero-frequency solution on a periodic grid."""
    from deeplens.light.wave import BandLimitedASM

    c = CellOpticsConfig()
    field = torch.ones(1, 1, 32, 32, dtype=torch.complex128)
    z_mm = 0.01
    result = BandLimitedASM(
        field, z_mm, c.wavelength_um, c.pixel_um * 1e-3, n=c.medium_index, padding=False
    )
    angle = torch.tensor(
        2 * torch.pi * c.medium_index * z_mm / (c.wavelength_um * 1e-3),
        dtype=torch.float64,
    )
    torch.testing.assert_close(result, field * torch.exp(1j * angle))


def test_detector_response_metrics_are_interpretable():
    features = torch.tensor([[4.0, 1.0], [3.0, 1.0], [1.0, 4.0], [1.0, 3.0]])
    labels = torch.tensor([0, 0, 1, 1])
    metrics = detector_response_metrics(features, labels)
    assert torch.equal(metrics["preferred_bin"], torch.tensor([0, 1]))
    assert torch.all(metrics["contrast"] > 0)
    assert torch.allclose(metrics["cross_talk"], torch.tensor([1 / 3.5, 1 / 3.5]))


def test_optical_energy_rejects_real_inputs():
    field = torch.ones(2, 4, 4, dtype=torch.complex64)
    assert torch.equal(optical_energy(field), torch.tensor([16.0, 16.0]))
    with pytest.raises(ValueError, match="complex"):
        optical_energy(field.real)


def test_class_difference_loss_prefers_separated_centroids():
    labels = torch.tensor([0, 0, 1, 1])
    close = torch.tensor([[0.0], [0.1], [0.0], [0.1]])
    far = torch.tensor([[0.0], [0.1], [2.0], [2.1]])
    assert maximize_class_difference(far, labels) < maximize_class_difference(
        close, labels
    )
