"""Generate a visual report for the coherent meta-optic flow-wave system.

Run from the repository root:

```bash
python visualize_flowwave.py
```
"""

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch, Rectangle

from deeplens import CoherentMetaOpticFlowSystem, FlowTrajectory, PhaseScreenCell
from deeplens.diffractive_surface import Pixel2DMetasurfaceArray

OUTPUT_DIR = Path("output/flowwave_report")


def _metasurface():
    return Pixel2DMetasurfaceArray(
        d_next=0.1,
        array_shape=(1, 1),
        aperture_res=(8, 8),
        aperture_ps=0.01,
        aperture_shape="square",
    )


def build_system():
    """Build and run the coherent source/receiver flow wave simulation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = [_metasurface() for _ in range(3)]
    receiver = _metasurface()

    yy, xx = torch.meshgrid(
        torch.linspace(-0.03, 0.03, 8),
        torch.linspace(-0.03, 0.03, 8),
        indexing="ij",
    )
    cell_mask = ((xx / 0.012) ** 2 + (yy / 0.008) ** 2 <= 1.0).to(torch.float32)
    cell = PhaseScreenCell(phase_map=1.3 * cell_mask, pixel_pitch=0.01)
    system = CoherentMetaOpticFlowSystem(
        source_metasurfaces=source,
        receiver_metasurface=receiver,
        res=(8, 8),
        ps=0.01,
        source_to_cell=0.1,
        cell_to_receiver=0.1,
        receiver_to_sensor=0.1,
        cell=cell,
    )
    trajectory = FlowTrajectory.linear(
        start=(-0.02, 0.0),
        velocity=(0.01, 0.0),
        num_steps=5,
        dt=0.1,
    )
    detector, intensity = system(trajectory, return_intensity=True)
    _, records = system.simulate_frame(return_records=True)
    return {
        "source": source,
        "receiver": receiver,
        "cell": cell,
        "trajectory": trajectory,
        "detector": detector,
        "intensity": intensity,
        "records": records,
    }


def save_plane_schematic():
    """Save an optical-axis schematic with coordinate-plane labels."""
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    ax.add_patch(
        FancyArrowPatch(
            (0.7, 2.55),
            (12.2, 2.55),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2,
            color="#667780",
        )
    )
    ax.text(12.25, 2.72, "optical axis z", color="#667780", fontsize=11)

    planes = [
        (0.8, "Coherent\nlaser", "$z=0$"),
        (2.8, "Source\nmetalens 1", "$z=z_1$"),
        (4.4, "Source\nmetalens 2", "$z=z_2$"),
        (6.0, "Source\nmetalens 3", "$z=z_3$"),
        (7.8, "2D cell\nphase screen", "$z=z_c$"),
        (10.0, "Receiver\nmetalens array", "$z=z_r$"),
        (12.0, "Detector", "$z=z_s$"),
    ]
    for x, label, coordinate in planes:
        ax.add_patch(
            Rectangle(
                (x - 0.12, 1.5),
                0.24,
                2.1,
                facecolor="#E1F0EC",
                edgecolor="#207C78",
                linewidth=2,
            )
        )
        ax.text(x, 3.75, label, ha="center", va="center", fontsize=11)
        ax.text(x, 1.15, coordinate, ha="center", va="center", fontsize=11)

    distances = list(
        zip(
            [p[0] for p in planes[:-1]],
            [p[0] for p in planes[1:]],
            ["$d_1$", "$d_2$", "$d_3$", "$z_{sc}$", "$z_{cr}$", "$z_{rs}$"],
        )
    )
    for start, end, label in distances:
        ax.annotate(
            "",
            xy=(end, 2.9),
            xytext=(start, 2.9),
            arrowprops={"arrowstyle": "<->", "color": "#D88B2A", "linewidth": 1.8},
        )
        ax.text(
            (start + end) / 2,
            3.08,
            label,
            ha="center",
            fontsize=10,
            color="#9A5E16",
        )

    ax.annotate(
        "",
        xy=(7.8, 0.55),
        xytext=(7.8, 4.25),
        arrowprops={"arrowstyle": "<->", "color": "#B9554F", "linewidth": 2},
    )
    ax.text(
        8.0,
        4.25,
        "cell flow direction",
        color="#B9554F",
        fontsize=11,
        va="center",
    )
    ax.text(
        6.5,
        0.2,
        "Scalar wave propagation follows +z; cell displacement is lateral.",
        ha="center",
        fontsize=11,
        color="#667780",
    )
    path = OUTPUT_DIR / "optical_planes.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_schematic():
    """Save the target physical system schematic."""
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    labels = [
        "Coherent laser",
        "Multiple serial\nsource metalenses",
        "2D cell phase\nscreen",
        "Receiver-side\nmetasurface",
        "Detector array",
        "Digital backend",
    ]
    centers = [0.8, 2.9, 5.25, 7.7, 9.9, 11.7]
    for index, (center, label) in enumerate(zip(centers, labels)):
        color = "#E1F0EC" if index in (0, 1, 3, 4, 5) else "#F8EBD8"
        edge = "#207C78" if index in (0, 1, 3, 4, 5) else "#D88B2A"
        width = 1.65
        ax.add_patch(
            Rectangle(
                (center - width / 2, 1.75),
                width,
                1.25,
                facecolor=color,
                edgecolor=edge,
                linewidth=2,
            )
        )
        ax.text(center, 2.37, label, ha="center", va="center", fontsize=10.5)
    for left, right in zip(centers[:-1], centers[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (left + 0.85, 2.37),
                (right - 0.85, 2.37),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=2,
                color="#207C78",
            )
        )
    ax.annotate(
        "",
        xy=(5.25, 1.35),
        xytext=(5.25, 3.35),
        arrowprops={"arrowstyle": "<->", "color": "#B9554F", "linewidth": 2},
    )
    ax.text(
        5.45,
        3.35,
        "cell flow direction",
        color="#B9554F",
        fontsize=11,
        va="center",
    )
    ax.text(
        6.5,
        4.25,
        "Coherent multi-metalens 2D phase-screen system",
        ha="center",
        fontsize=19,
        fontweight="bold",
    )
    path = OUTPUT_DIR / "schematic.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_field_records(records):
    """Save intensity snapshots at selected physical planes."""
    selected = [
        ("Laser", "laser"),
        ("Source metalens 1", "source_metasurface_1"),
        ("Source metalens 2", "source_metasurface_2"),
        ("Cell phase screen", "cell"),
        ("Receiver input", "receiver_incident"),
        ("Receiver metalens", "receiver_metasurface"),
        ("Detector field", "detector_field"),
    ]
    fig, axes = plt.subplots(1, len(selected), figsize=(16, 3.2))
    for ax, (title, key) in zip(axes, selected):
        image = records[key].abs().square()[0, 0].detach().cpu().numpy()
        ax.imshow(image, cmap="magma")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("Complex-field intensity at every physical plane", fontsize=16)
    path = OUTPUT_DIR / "plane_intensities.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_cell_figure(cell):
    """Save the cell's single 2D phase mask."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(cell.phase_map, cmap="inferno")
    ax.set_title("2D cell phase screen")
    ax.axis("off")
    fig.colorbar(image, ax=ax, shrink=0.8, label="phase [rad]")
    path = OUTPUT_DIR / "cell_phase_screen.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_sensor_figure(intensity, times):
    """Save the sensor intensity sequence."""
    fig, axes = plt.subplots(1, intensity.shape[1], figsize=(13, 3))
    for index, ax in enumerate(axes):
        image = intensity[0, index, 0].detach().cpu().numpy()
        ax.imshow(image, cmap="magma")
        ax.set_title(f"t = {times[index]:.1f} s")
        ax.axis("off")
    fig.suptitle("Coherent sensor intensity during cell flow", fontsize=16)
    path = OUTPUT_DIR / "sensor_sequence.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_signal_figure(intensity, times):
    """Save integrated detector signal versus time."""
    signal = intensity.mean(dim=(-1, -2))[0, :, 0].detach().cpu().numpy()
    time = times.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(time, signal, marker="o", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean sensor intensity")
    ax.set_title("Event signal from coherent wave simulation")
    ax.grid(alpha=0.25)
    path = OUTPUT_DIR / "event_signal.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(paths):
    """Write a static HTML report."""
    cards = "".join(
        f"<section><h2>{html.escape(title)}</h2>"
        f'<img src="{path.name}" alt="{html.escape(title)}"></section>'
        for title, path in paths
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Coherent Flow Wave Simulation</title>
  <style>
    body {{
      margin: 0;
      background: #f7f6f1;
      color: #12232d;
      font-family: Arial, sans-serif;
    }}
    header {{
      padding: 42px clamp(24px, 6vw, 80px) 26px;
      background: #173d4b;
      color: white;
    }}
    header h1 {{ margin: 0 0 10px; font-size: clamp(30px, 5vw, 52px); }}
    header p {{ margin: 0; color: #cde2df; }}
    main {{ padding: 30px clamp(24px, 6vw, 80px) 60px; }}
    section {{
      margin-bottom: 28px;
      padding: 22px;
      background: white;
      border: 1px solid #d5dddf;
    }}
    h2 {{ margin-top: 0; }}
    img {{ display: block; width: 100%; height: auto; }}
  </style>
</head>
<body>
  <header>
    <h1>Coherent Flow Wave Simulation</h1>
    <p>Serial source metalenses, one 2D cell phase screen, receiver metasurface array, and detector</p>
  </header>
  <main>{cards}</main>
</body>
</html>
"""
    (OUTPUT_DIR / "index.html").write_text(document, encoding="utf-8")


def main():
    """Build and save the coherent flow-wave report."""
    result = build_system()
    paths = [
        ("Optical-axis coordinate definition", save_plane_schematic()),
        ("Physical system schematic", save_schematic()),
        (
            "Field intensity at each physical plane",
            save_field_records(result["records"]),
        ),
        ("2D cell phase screen", save_cell_figure(result["cell"])),
        (
            "Coherent sensor sequence",
            save_sensor_figure(result["intensity"], result["trajectory"].times),
        ),
        (
            "Event signal",
            save_signal_figure(result["intensity"], result["trajectory"].times),
        ),
    ]
    write_report(paths)
    print(f"Report written to {(OUTPUT_DIR / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
