"""Audit a saved classifier with physical identities and grid refinement."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from deeplens.cell_classification import CellClassifier, CellOpticsConfig, phase_cells


def main():
    """Export measured checks, including convergence errors without hiding failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="output/cell_classification/joint")
    args = parser.parse_args()
    folder = Path(args.run)
    torch.set_num_threads(4)
    checkpoint = torch.load(folder / "model.pt", weights_only=True)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    c = CellOpticsConfig(**checkpoint["config"])
    model = CellClassifier(c).double()
    model.load_state_dict(checkpoint["state_dict"])
    data = torch.load(folder / "data.pt", weights_only=True)["test"]
    phase = data[0][:8].double()
    model.eval()
    with torch.no_grad():
        features, records = model.encoder(phase, True)
        amplitude_error = (records["cell"].abs() - 1).abs().max().item()
        square_law_error = (
            (records["intensity"] - records["sensor"].abs().square()).abs().max().item()
        )
        gap_amplitude = (
            records["array_exit"][..., model.encoder.transmission == 0]
            .abs()
            .max()
            .item()
            if c.gap_pixels
            else 0.0
        )
        energy = {
            k: v.abs().square().sum((-2, -1))
            for k, v in records.items()
            if k != "intensity"
        }
        energy_increases = [
            (energy[b] - energy[a]).max().item()
            for a, b in zip(
                ("cell", "array_incident", "array_exit"),
                ("array_incident", "array_exit", "sensor"),
            )
        ]
        contact = CellClassifier(
            replace(c, cell_to_array_um=0, array_to_sensor_um=0)
        ).double()
        contact.load_state_dict(checkpoint["state_dict"])
        contact_features = contact.encoder(phase)
        contact_difference = (
            (contact_features - contact_features[:1]).abs().max().item()
        )
        refinements = []
        previous_intensity = records["intensity"]
        previous_features = features
        for factor in (2, 4):
            finer_config = replace(
                c,
                tile_pixels=c.tile_pixels * factor,
                gap_pixels=c.gap_pixels * factor,
                pixel_um=c.pixel_um / factor,
            )
            finer = CellClassifier(finer_config).double()
            finer.backend.load_state_dict(model.backend.state_dict())
            # Hold physical design pixels fixed; do not introduce new optical DOFs.
            finer.encoder.phase.copy_(
                model.encoder.phase.repeat_interleave(factor, -2).repeat_interleave(
                    factor, -1
                )
            )
            finer_phase = phase_cells(
                report["settings"]["test_samples"],
                finer_config,
                report["settings"]["seed"] + 303,
            )[0][:8].double()
            _, f_records = finer.encoder(finer_phase, True)
            intensity = F.avg_pool2d(f_records["intensity"], factor)
            # Keep the identical detector regions after refinement, including
            # the base grid's adaptive-pooling edge conventions.
            f_features = F.adaptive_avg_pool2d(intensity, c.detector_bins).flatten(1)
            relative_error = (
                intensity - previous_intensity
            ).norm() / previous_intensity.norm()
            feature_error = (
                f_features - previous_features
            ).norm() / previous_features.norm()
            refinements.append(
                {
                    "factor": factor,
                    "pixel_um": finer_config.pixel_um,
                    "intensity_relative_l2_vs_previous": relative_error.item(),
                    "feature_relative_l2_vs_previous": feature_error.item(),
                    "predictions_first_8": finer.backend(f_features).argmax(1).tolist(),
                }
            )
            previous_intensity, previous_features = intensity, f_features
    # Smooth, unquantized gradient verification. Quantization uses STE, whose
    # surrogate backward derivative must not be described as a finite difference.
    gradient_audit = {"status": "skipped: quantized phase uses STE"}
    if c.phase_levels == 0:
        loss = F.cross_entropy(model(phase), data[1][:8])
        loss.backward()
        flat = model.encoder.phase.flatten()
        grad = model.encoder.phase.grad.flatten()
        index = int(grad.abs().argmax())
        analytic = grad[index].item()
        with torch.no_grad():
            initial = flat[index].item()
            step = 1e-4
            flat[index] = initial + step
            plus = F.cross_entropy(model(phase), data[1][:8]).item()
            flat[index] = initial - step
            minus = F.cross_entropy(model(phase), data[1][:8]).item()
            flat[index] = initial
        finite = (plus - minus) / (2 * step)
        gradient_audit = {
            "autograd": analytic,
            "finite_difference": finite,
            "absolute_error": abs(analytic - finite),
            "relative_error": abs(analytic - finite) / max(abs(analytic), 1e-12),
        }
    audit = {
        "phase_only_amplitude_max_error": amplitude_error,
        "square_law_max_error": square_law_error,
        "opaque_gap_max_amplitude": gap_amplitude,
        "maximum_energy_increase_each_stage": energy_increases,
        "contact_features_max_difference": contact_difference,
        "gradient_check": gradient_audit,
        "base_predictions_first_8": model(phase).argmax(1).tolist(),
        "grid_refinement": refinements,
        "refinement_note": "Same 8 physical cells, fixed physical phase tiles, fixed FOV and distances. Phantoms are rerasterized analytically. Relative L2 compares factor 2 to base and factor 4 to factor 2, after intensity integration to the base grid. This probes sampling, not padding/FOV convergence or vector physics.",
    }
    (folder / "verification.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    if (
        amplitude_error > 1e-10
        or square_law_error > 1e-10
        or gap_amplitude > 1e-10
        or max(energy_increases) > 1e-7
        or contact_difference > 1e-10
    ):
        raise RuntimeError("A physical identity failed; inspect verification.json")


if __name__ == "__main__":
    main()
