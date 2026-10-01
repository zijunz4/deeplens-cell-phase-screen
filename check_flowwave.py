"""Human-readable checks for the 2D coherent meta-optic flow simulation."""

from pathlib import Path

import torch

from deeplens import CoherentMetaOpticFlowSystem, FlowTrajectory, PhaseScreenCell
from deeplens.diffractive_surface import Pixel2DMetasurfaceArray

REPORT_PATH = Path("output/advisor_update/manual_wave_verification.txt")


def make_metasurface():
    """Create a transparent 8 x 8 metasurface plane."""
    return Pixel2DMetasurfaceArray(
        d_next=0.1,
        array_shape=(1, 1),
        aperture_res=(8, 8),
        aperture_ps=0.01,
        aperture_shape="square",
    )


def check(name, condition, lines):
    """Print, save, and enforce one independently readable result."""
    result = f"[{'PASS' if condition else 'FAIL'}] {name}"
    print(result)
    lines.append(result)
    if not condition:
        raise RuntimeError(name)


def main():
    """Check 2D phase-only cell behavior and serial source metalenses."""
    lines = []
    cell_phase = torch.zeros(8, 8)
    cell_phase[2:6, 2:6] = 1.2
    cell = PhaseScreenCell(cell_phase, pixel_pitch=0.01)
    transmission = cell.transmission()
    check(
        "2D phase screen preserves unit amplitude",
        torch.allclose(transmission.abs(), torch.ones_like(cell_phase)),
        lines,
    )
    source = [make_metasurface(), make_metasurface(), make_metasurface()]
    system = CoherentMetaOpticFlowSystem(
        source_metasurfaces=source,
        receiver_metasurface=make_metasurface(),
        res=(8, 8),
        ps=0.01,
        source_to_cell=0.1,
        cell_to_receiver=0.1,
        receiver_to_sensor=0.1,
        cell=cell,
    )
    trajectory = FlowTrajectory(positions=((-0.02, 0.0), (0.02, 0.0)), times=(0.0, 0.1))
    _, intensity = system(trajectory, return_intensity=True)
    check(
        "Cell translation changes sensor intensity",
        float((intensity[:, 0] - intensity[:, 1]).abs().mean()) > 1e-8,
        lines,
    )
    _, records = system.simulate_frame(return_records=True)
    expected = {
        "laser",
        "source_metasurface_1",
        "source_metasurface_2",
        "source_metasurface_3",
        "cell_incident",
        "cell",
        "receiver_incident",
        "receiver_metasurface",
        "detector_field",
    }
    check(
        "Fields are recorded at every requested plane", set(records) == expected, lines
    )
    check(
        "All recorded fields remain finite and complex",
        all(
            field.is_complex() and bool(torch.isfinite(field).all())
            for field in records.values()
        ),
        lines,
    )
    check(
        "Sensor intensity equals squared detector-field amplitude",
        torch.allclose(
            records["detector_field"].abs().square(), system.simulate_frame()
        ),
        lines,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Verification report written to {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
