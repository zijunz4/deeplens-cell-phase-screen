"""Standalone smoke check for the multi-aperture flow-cytometry simulator.

Run from the repository root:

```bash
python check_flowcyto.py
```
"""

import torch

from deeplens import (
    DetectorModel,
    FlowTrajectory,
    MetaOpticFlowCytometer,
    extract_pulse_features,
)
from deeplens.diffractive_surface import Pixel2DMetasurfaceArray
from deeplens.imgsim import MetaOpticChannelArray


def check(name, condition):
    """Print one check result and raise when it fails."""
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise RuntimeError(name)


def main():
    """Run a deterministic end-to-end smoke check."""
    torch.manual_seed(0)

    metasurface = Pixel2DMetasurfaceArray(
        d_next=0.05,
        array_shape=(2, 2),
        aperture_res=(4, 4),
        aperture_ps=0.005,
        aperture_shape="square",
    )
    metasurface.phase_maps.requires_grad_(True)

    psfs = metasurface.compute_channel_psfs()
    check("phase maps produce four aperture PSFs", tuple(psfs.shape) == (4, 1, 4, 4))
    check(
        "each PSF is normalized",
        torch.allclose(
            psfs.sum(dim=(-2, -1)),
            torch.ones(4, 1, dtype=psfs.dtype),
            atol=1e-6,
        ),
    )

    (psfs[..., 0, 0].sum()).backward()
    check(
        "phase-to-PSF path is differentiable",
        metasurface.phase_maps.grad is not None
        and metasurface.phase_maps.grad.abs().sum().item() > 0,
    )
    metasurface.phase_maps.grad = None

    receiver = MetaOpticChannelArray.from_metasurface_array(
        metasurface_array=metasurface,
        channel_rois=((0, 0), (0, 4), (4, 0), (4, 4)),
        feature_size=(4, 4),
        detector_shape=(8, 8),
    )
    detector = DetectorModel(
        bit_depth=12,
        full_well=1000.0,
        gain=1000.0 / 4095.0,
        noise=False,
    )
    system = MetaOpticFlowCytometer(
        receiver=receiver,
        detector_model=detector,
        flow_pixel_pitch=0.005,
    )

    scene = torch.zeros(1, 1, 9, 9)
    scene[..., 4, 4] = 1.0
    trajectory = FlowTrajectory.linear(
        start=(-0.01, 0.0),
        velocity=(0.01, 0.0),
        num_steps=3,
        dt=0.1,
    )
    features, frames = system(
        scene,
        trajectory=trajectory,
        return_frames=True,
    )
    check(
        "flow simulation returns feature sequence",
        tuple(features.shape) == (1, 3, 1, 4),
    )
    check(
        "flow simulation returns detector frames",
        tuple(frames.shape) == (1, 3, 1, 8, 8),
    )
    check("detector output is finite", bool(torch.isfinite(frames).all()))

    stats = extract_pulse_features(features, times=trajectory.times)
    check("pulse peak is finite", bool(torch.isfinite(stats["peak"]).all()))
    check("pulse area is finite", bool(torch.isfinite(stats["area"]).all()))
    check("pulse width is finite", bool(torch.isfinite(stats["width"]).all()))

    print("\nAll flow-cytometry checks passed.")


if __name__ == "__main__":
    main()
