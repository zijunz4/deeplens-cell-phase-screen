"""Reproducible optical/digital shape-classification experiment and audit exports."""

import argparse
import copy
import csv
import json
import platform
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml

from deeplens.cell_classification import (
    CellClassifier,
    CellOpticsConfig,
    phase_cells,
    supervised_contrastive,
)


def write_csv(path, rows):
    """Write a nonempty sequence of flat records for spreadsheet inspection."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(model, data):
    """Evaluate a complete held-out split without updates."""
    model.eval()
    logits = model(data[0])
    return {
        "loss": F.cross_entropy(logits, data[1]).item(),
        "accuracy": (logits.argmax(1) == data[1]).float().mean().item(),
    }, logits


def run_experiment(config, settings, output, mode):
    """Train one controlled run; select weights using validation loss only."""
    output.mkdir(parents=True, exist_ok=True)
    seed = settings["seed"]
    torch.manual_seed(seed)
    model = CellClassifier(config)
    initial_phase = model.encoder.phase.detach().clone()
    datasets = [
        phase_cells(settings[f"{split}_samples"], config, seed + offset)
        for split, offset in (("train", 101), ("validation", 202), ("test", 303))
    ]
    train, validation, test = datasets
    # Save exact inputs and split identifiers; no test samples enter training.
    torch.save(dict(zip(("train", "validation", "test"), datasets)), output / "data.pt")
    model.encoder.phase.requires_grad_(mode != "frozen")
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": settings["optical_lr"]},
            {"params": model.backend.parameters(), "lr": settings["digital_lr"]},
        ]
    )
    history = []
    if mode == "contrastive_joint":
        # A full balanced training batch guarantees positives. No projection
        # head: separation must occur in the actual detected optical features.
        for epoch in range(settings["contrastive_epochs"]):
            optimizer.zero_grad()
            loss = supervised_contrastive(model.encoder(train[0]), train[1])
            loss.backward()
            optimizer.step()
            history.append(
                {
                    "stage": "contrastive",
                    "epoch": epoch + 1,
                    "train_loss": loss.item(),
                    "validation_loss": "",
                    "validation_accuracy": "",
                }
            )
        # Start joint optimization with fresh Adam moments for a clean stage.
        optimizer.state.clear()
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    generator = torch.Generator().manual_seed(seed + 404)
    for epoch in range(settings["epochs"]):
        model.train()
        permutation = torch.randperm(len(train[1]), generator=generator)
        total_loss = 0.0
        for indices in permutation.split(settings["batch_size"]):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(train[0][indices]), train[1][indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        metrics, _ = evaluate(model, validation)
        history.append(
            {
                "stage": "joint" if mode != "frozen" else "digital_only",
                "epoch": epoch + 1,
                "train_loss": total_loss / len(train[1]),
                "validation_loss": metrics["loss"],
                "validation_accuracy": metrics["accuracy"],
            }
        )
        if metrics["loss"] < best_loss:
            best_loss, best_epoch = metrics["loss"], epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"{mode} epoch {epoch + 1}: validation accuracy={metrics['accuracy']:.3f}",
            flush=True,
        )
    model.load_state_dict(best_state)
    metrics, logits = evaluate(model, test)
    prediction = logits.argmax(1)
    class_count = int(test[1].max().item()) + 1
    confusion = torch.bincount(
        test[1] * class_count + prediction, minlength=class_count**2
    ).reshape(class_count, class_count)
    with torch.no_grad():
        features, records = model.encoder(test[0][:8], return_records=True)
        energies = {
            name: field.abs().square().sum((-2, -1)).flatten().tolist()
            for name, field in records.items()
            if name != "intensity"
        }
        phase_map = model.encoder.phase_map()
        # Fixed fabricated phase levels evaluated without retraining.
        from dataclasses import replace

        quantized = CellClassifier(replace(config, phase_levels=16))
        quantized.load_state_dict(best_state)
        quantized_metrics, _ = evaluate(quantized, test)
    audit = {
        "config": asdict(config),
        "settings": settings,
        "mode": mode,
        "best_validation_epoch": best_epoch,
        "test": metrics,
        "test_confusion_rows_true_columns_predicted": confusion.tolist(),
        "test_16_level_postquantization": quantized_metrics,
        "phase_change_l2": (model.encoder.phase.detach() - initial_phase).norm().item(),
        "energy_pixel_sums_first_8": energies,
        "energy_area_multiplier_um2": config.pixel_um**2,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "phase_formula": "2*pi*delta_n*thickness_um/wavelength_um",
        "class_names": ["circle", "rectangle", "triangle"],
        "limitations": "Synthetic thin phase slabs; ideal coherent scalar optics; no shot/read noise, polarization or meta-atom response. One seed is not evidence of optical advantage.",
    }
    (output / "report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    torch.save(
        {"config": asdict(config), "state_dict": best_state}, output / "model.pt"
    )
    torch.save(
        {
            "phase": test[0][:8],
            "labels": test[1][:8],
            "features": features,
            "records": records,
            "array_phase": phase_map,
            "transmission": model.encoder.transmission,
        },
        output / "physical_planes.pt",
    )
    write_csv(output / "history.csv", history)
    probabilities = logits.softmax(1)
    write_csv(
        output / "predictions.csv",
        [
            {
                "sample": i,
                "true": int(test[1][i]),
                "predicted": int(prediction[i]),
                "p_circle": float(probabilities[i, 0]),
                "p_rectangle": float(probabilities[i, 1]),
                "p_triangle": float(probabilities[i, 2]),
                **dict(
                    zip(
                        (
                            "diameter_um",
                            "thickness_um",
                            "delta_n",
                            "x_um",
                            "y_um",
                            "angle_rad",
                            "aspect",
                        ),
                        test[2][i].tolist(),
                    )
                ),
            }
            for i in range(len(test[1]))
        ],
    )
    extent = [
        -config.pixels * config.pixel_um / 2,
        config.pixels * config.pixel_um / 2,
    ] * 2
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    for row, label in enumerate((0, 1, 2)):
        index = (
            int((test[1][:8] == label).nonzero()[0])
            if bool((test[1][:8] == label).any())
            else 0
        )
        items = [
            (test[0][index, 0], "Cell phase [rad]"),
            (phase_map * model.encoder.transmission, "Array phase [rad]; gaps=0"),
            (records["intensity"][index, 0], "Detected intensity [relative]"),
        ]
        for ax, (values, title) in zip(axes[row], items):
            im = ax.imshow(values.detach().numpy(), extent=extent, origin="lower")
            ax.set(title=title, xlabel="x [um]", ylabel="y [um]")
            fig.colorbar(im, ax=ax)
    fig.savefig(output / "physical_planes.png", dpi=150)
    plt.close(fig)
    return audit


def main():
    """Run matched frozen/joint baselines and optional contrastive pretraining."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cell_classification.yml")
    parser.add_argument("--output", default="output/cell_classification")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--contrastive-epochs", type=int)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        setup = yaml.safe_load(handle)
    config = CellOpticsConfig(**setup["optics"])
    settings = setup["training"]
    if args.epochs is not None:
        settings["epochs"] = args.epochs
    if args.contrastive_epochs is not None:
        settings["contrastive_epochs"] = args.contrastive_epochs
    if (
        settings["epochs"] < 1
        or settings["batch_size"] < 1
        or settings["contrastive_epochs"] < 0
    ):
        raise ValueError("Invalid epoch count or batch size")
    torch.set_num_threads(settings["threads"])
    torch.use_deterministic_algorithms(True)
    output = Path(args.output)
    modes = ["frozen", "joint"]
    if settings["contrastive_epochs"]:
        modes.append("contrastive_joint")
    reports = [run_experiment(config, settings, output / mode, mode) for mode in modes]
    write_csv(
        output / "comparison.csv",
        [
            {
                "mode": r["mode"],
                "test_accuracy": r["test"]["accuracy"],
                "best_epoch": r["best_validation_epoch"],
                "phase_change_l2": r["phase_change_l2"],
            }
            for r in reports
        ],
    )
    print(f"Audit artifacts: {output.resolve()}")


if __name__ == "__main__":
    main()
