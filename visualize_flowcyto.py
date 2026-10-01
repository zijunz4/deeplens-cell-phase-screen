"""Generate a visual HTML report for the flow-cytometry simulator.

Run from the repository root:

```bash
python visualize_flowcyto.py
```
"""

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, Rectangle

from deeplens import (
    DetectorModel,
    FlowTrajectory,
    MetaOpticFlowCytometer,
    extract_pulse_features,
)
from deeplens.diffractive_surface import Pixel2DMetasurfaceArray
from deeplens.imgsim import MetaOpticChannelArray

OUTPUT_DIR = Path("output/flowcyto_report")


def build_system():
    """Build a small deterministic multi-aperture flow-cytometry system."""
    out = OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    wavelength = 0.55
    focal_length = 0.05
    aperture_ps = 0.005
    aperture_res = 4
    phase_map = Pixel2DMetasurfaceArray(
        d_next=focal_length,
        array_shape=(2, 2),
        aperture_res=aperture_res,
        aperture_ps=aperture_ps,
        aperture_shape="square",
    )

    half_size = aperture_res * aperture_ps / 2
    coords = torch.linspace(
        -half_size + aperture_ps / 2,
        half_size - aperture_ps / 2,
        aperture_res,
    )
    yy, xx = torch.meshgrid(coords, coords, indexing="xy")
    radius = torch.sqrt(xx**2 + yy**2)
    k = 2 * torch.pi / (wavelength * 1e-3)
    focusing_phase = -k * (torch.sqrt(radius**2 + focal_length**2) - focal_length)
    with torch.no_grad():
        phase_map.phase_maps.copy_(
            focusing_phase.expand(2, 2, aperture_res, aperture_res)
        )

    psfs = phase_map.compute_channel_psfs(wvln=wavelength)
    receiver = MetaOpticChannelArray(
        psfs=psfs,
        channel_rois=((0, 0), (0, 4), (4, 0), (4, 4)),
        feature_size=(4, 4),
        detector_shape=(8, 8),
    )
    detector = DetectorModel(
        bit_depth=12,
        full_well=1000.0,
        gain=1000.0 / 4095.0,
        read_noise=2.0,
        dark_current=1.0,
        exposure_time=1e-3,
        noise=False,
    )
    system = MetaOpticFlowCytometer(
        receiver=receiver,
        detector_model=detector,
        flow_pixel_pitch=aperture_ps,
    )

    coordinate = torch.linspace(-0.02, 0.02, 9)
    y_grid, x_grid = torch.meshgrid(coordinate, coordinate, indexing="ij")
    sigma = 0.007
    scene = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))[None, None]
    trajectory = FlowTrajectory.linear(
        start=(-0.01, 0.0),
        velocity=(0.005, 0.0),
        num_steps=5,
        dt=0.1,
    )
    features, frames = system(
        scene,
        trajectory=trajectory,
        return_frames=True,
    )
    stats = extract_pulse_features(features, times=trajectory.times)
    return {
        "phase_map": phase_map,
        "psfs": psfs,
        "receiver": receiver,
        "frames": frames,
        "features": features,
        "stats": stats,
        "trajectory": trajectory,
    }


def save_architecture_figure():
    """Save target-system and implemented-model schematics side by side."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def draw_box(x, y, width, height, label, implemented=True):
        box = Rectangle(
            (x, y),
            width,
            height,
            facecolor="#E1F0EC" if implemented else "#F8EBD8",
            edgecolor="#207C78" if implemented else "#D88B2A",
            linewidth=2,
            linestyle="-" if implemented else "--",
        )
        ax.add_patch(box)
        ax.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            color="#12232D",
        )
        return box

    def draw_row(y, labels, implemented):
        xs = np.linspace(0.45, 11.75, len(labels))
        for x, label, is_impl in zip(xs, labels, implemented):
            draw_box(x, y, 1.58, 1.0, label, implemented=is_impl)
        for x0, x1 in zip(xs[:-1], xs[1:]):
            ax.add_patch(
                FancyArrowPatch(
                    (x0 + 1.58, y + 0.5),
                    (x1, y + 0.5),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    color="#207C78",
                    linewidth=1.8,
                )
            )

    ax.text(
        7,
        7.55,
        "Target physical system vs implemented simulation",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#12232D",
    )
    ax.text(
        0.45,
        6.8,
        "Target physical architecture",
        ha="left",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#12232D",
    )
    target_labels = [
        "Coherent laser",
        "Source-side\nmetasurface array",
        "Microfluidic\nchannel and cell",
        "Receiver-side\nmetasurface array",
        "High-speed\nAPD or detector",
        "Digital\nclassifier",
    ]
    draw_row(5.25, target_labels, [True, False, False, False, True, True])
    ax.text(
        7,
        4.85,
        "Cell motion provides time-folding across the 1D metasurface array",
        ha="center",
        va="center",
        fontsize=11,
        color="#667780",
    )

    ax.text(
        0.45,
        4.0,
        "Implemented simulation model",
        ha="left",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#12232D",
    )
    implemented_labels = [
        "Input scene\nor cell phantom",
        "Optional refractive\nfrontend",
        "Flow trajectory\nand shift",
        "Pixel2D\nphase array",
        "ASM\nper-aperture PSF",
        "Detector ROI\nand noise",
        "Pulse features\nand classifier",
    ]
    xs = np.linspace(0.45, 11.45, len(implemented_labels))
    for x, label in zip(xs, implemented_labels):
        draw_box(x, 2.2, 1.55, 1.05, label, implemented=True)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (x0 + 1.55, 2.72),
                (x1, 2.72),
                arrowstyle="-|>",
                mutation_scale=14,
                color="#207C78",
                linewidth=1.8,
            )
        )

    ax.add_patch(
        Rectangle(
            (11.82, 5.25),
            0.23,
            1.0,
            facecolor="#E1F0EC",
            edgecolor="#207C78",
            linewidth=1.6,
        )
    )
    ax.text(12.15, 5.75, "Implemented", va="center", fontsize=10)
    ax.add_patch(
        Rectangle(
            (11.82, 3.88),
            0.23,
            1.0,
            facecolor="#F8EBD8",
            edgecolor="#D88B2A",
            linewidth=1.6,
            linestyle="--",
        )
    )
    ax.text(12.15, 4.38, "Planned / not yet", va="center", fontsize=10)

    path = OUTPUT_DIR / "architecture.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_phase_and_psf_figure(phase_map, psfs):
    """Save aperture phase maps and their propagated PSFs."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for index, ax in enumerate(axes[0]):
        row, col = divmod(index, 2)
        image = phase_map.phase_maps[row, col].detach().cpu().numpy()
        ax.imshow(image, cmap="twilight")
        ax.set_title(f"Aperture {index + 1} phase")
        ax.axis("off")
    for index, ax in enumerate(axes[1]):
        image = psfs[index, 0].detach().cpu().numpy()
        ax.imshow(image, cmap="magma")
        ax.set_title(f"Aperture {index + 1} PSF")
        ax.axis("off")
    fig.tight_layout()
    path = OUTPUT_DIR / "phase_and_psf.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_event_figure(frames, features, times):
    """Save detector frames and event signals."""
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for index in range(5):
        image = frames[0, index, 0].detach().cpu().numpy()
        axes[0, index].imshow(image, cmap="magma")
        axes[0, index].set_title(f"t = {times[index]:.1f} s")
        axes[0, index].axis("off")

    signal = features[0].detach().cpu().numpy()
    time = times.detach().cpu().numpy()
    for index in range(signal.shape[-1]):
        axes[1, index].plot(time, signal[:, 0, index], marker="o")
        axes[1, index].set_title(f"Channel {index + 1}")
        axes[1, index].set_xlabel("Time (s)")
        axes[1, index].set_ylabel("Detector signal")
        axes[1, index].grid(alpha=0.25)
    fig.tight_layout()
    path = OUTPUT_DIR / "event_sequence.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_html(paths, stats):
    """Write a static HTML report that embeds the generated figures."""
    cards = []
    for title, path in paths:
        cards.append(
            '<section class="report-card">'
            f"<h2>{html.escape(title)}</h2>"
            f'<img src="{path.name}" alt="{html.escape(title)}">'
            "</section>"
        )
    values = {
        "Peak": float(stats["peak"].mean()),
        "Area": float(stats["area"].mean()),
        "Width": float(stats["width"].mean()),
        "Time to peak": float(stats["time_to_peak"].mean()),
    }
    metric_html = "".join(
        f'<div class="metric"><span>{key}</span><strong>{value:.3f}</strong></div>'
        for key, value in values.items()
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Flow Cytometry Simulation Report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #12232d;
      --muted: #667780;
      --line: #d5dddf;
      --paper: #f7f6f1;
      --panel: #ffffff;
      --teal: #207c78;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: Arial, sans-serif;
    }}
    header {{
      padding: 48px clamp(24px, 6vw, 80px) 28px;
      background: #173d4b;
      color: white;
    }}
    header h1 {{ margin: 0 0 10px; font-size: clamp(32px, 5vw, 56px); }}
    header p {{ margin: 0; color: #cde2df; font-size: 18px; }}
    main {{ padding: 32px clamp(24px, 6vw, 80px) 64px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 14px;
      margin-bottom: 28px;
    }}
    .metric {{
      padding: 16px;
      background: var(--panel);
      border: 1px solid var(--line);
    }}
    .metric span {{ display: block; color: var(--muted); font-size: 14px; }}
    .metric strong {{ display: block; margin-top: 8px; color: var(--teal); font-size: 26px; }}
    .report-card {{
      margin-bottom: 28px;
      padding: 24px;
      background: var(--panel);
      border: 1px solid var(--line);
    }}
    .report-card h2 {{ margin: 0 0 18px; font-size: 24px; }}
    .report-card img {{ display: block; width: 100%; height: auto; }}
    .note {{ color: var(--muted); line-height: 1.6; max-width: 900px; }}
  </style>
</head>
<body>
  <header>
    <h1>Flow Cytometry Simulation Report</h1>
    <p>Multi-aperture meta-optic system, detector sequence, and event features</p>
  </header>
  <main>
    <div class="metrics">{metric_html}</div>
    <p class="note">
      This report is generated directly from the Python simulation. It does not
      require Zemax or another optical CAD application.
    </p>
    {"".join(cards)}
  </main>
</body>
</html>
"""
    (OUTPUT_DIR / "index.html").write_text(document, encoding="utf-8")


def main():
    """Generate all report figures and the HTML entry point."""
    result = build_system()
    paths = [
        ("Target vs implemented schematic", save_architecture_figure()),
        (
            "Aperture phase maps and PSFs",
            save_phase_and_psf_figure(result["phase_map"], result["psfs"]),
        ),
        (
            "Moving-cell detector sequence",
            save_event_figure(
                result["frames"],
                result["features"],
                result["trajectory"].times,
            ),
        ),
    ]
    write_html(paths, result["stats"])
    print(f"Report written to {(OUTPUT_DIR / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
