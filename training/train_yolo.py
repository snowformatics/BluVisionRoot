"""
Standalone YOLO segmentation training script.

This script only trains models and creates a training overview table.
It does not preprocess real experiment images and it does not run prediction.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable

import torch
from ultralytics import YOLO

from training_config import (
    AUGMENTATION,
    BATCH,
    DATA_YAML,
    DEVICE,
    EPOCHS_LIST,
    IMG_SIZES,
    LR0_VALUES,
    MODELS,
    OPTIMIZERS,
    PATIENCE,
    TRAINING_OUTPUT_ROOT,
    WEIGHT_DECAY_VALUES,
    WORKERS,
)
from training_overview import create_training_overview


def resolve_device(device_setting: str) -> str:
    """Resolve 'auto' to CUDA when available, otherwise CPU."""
    if device_setting.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_setting


def make_run_name(
    model_name: str,
    imgsz: int,
    epochs: int,
    optimizer: str,
    lr0: float,
    weight_decay: float,
) -> str:
    """Create a readable YOLO run name from training parameters."""
    model_label = model_name.replace(".pt", "")
    return (
        f"{model_label}"
        f"_img{imgsz}"
        f"_ep{epochs}"
        f"_{optimizer}"
        f"_lr{lr0}"
        f"_wd{weight_decay}"
    )


def train_single_run(
    data_yaml: Path,
    output_root: Path,
    model_name: str,
    imgsz: int,
    epochs: int,
    optimizer: str,
    lr0: float,
    weight_decay: float,
    device: str,
) -> Path:
    """Train one YOLO segmentation model and return the run directory."""
    run_name = make_run_name(
        model_name=model_name,
        imgsz=imgsz,
        epochs=epochs,
        optimizer=optimizer,
        lr0=lr0,
        weight_decay=weight_decay,
    )

    print("\n" + "=" * 80)
    print(f"Training run: {run_name}")
    print("=" * 80)

    model = YOLO(model_name)

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=BATCH,
        optimizer=optimizer,
        lr0=lr0,
        weight_decay=weight_decay,
        project=str(output_root),
        name=run_name,
        workers=WORKERS,
        patience=PATIENCE,
        device=device,
        **AUGMENTATION,
    )

    return output_root / run_name


def train_grid(
    data_yaml: Path,
    output_root: Path,
    models: Iterable[str],
    img_sizes: Iterable[int],
    epochs_list: Iterable[int],
    optimizers: Iterable[str],
    lr0_values: Iterable[float],
    weight_decay_values: Iterable[float],
    device: str,
) -> None:
    """Run a small grid search over YOLO segmentation training settings."""
    for model_name, imgsz, epochs, optimizer, lr0, weight_decay in itertools.product(
        models,
        img_sizes,
        epochs_list,
        optimizers,
        lr0_values,
        weight_decay_values,
    ):
        train_single_run(
            data_yaml=data_yaml,
            output_root=output_root,
            model_name=model_name,
            imgsz=imgsz,
            epochs=epochs,
            optimizer=optimizer,
            lr0=lr0,
            weight_decay=weight_decay,
            device=device,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO segmentation models for Fusarium/root segmentation."
    )

    parser.add_argument(
        "-s",
        "--source_path",
        default=str(DATA_YAML),
        help="Path to YOLO data.yaml file.",
    )
    parser.add_argument(
        "-d",
        "--destination_path",
        default=str(TRAINING_OUTPUT_ROOT),
        help="Directory where training runs will be stored.",
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
        help="Training device: auto, cpu, 0, cuda, etc.",
    )
    parser.add_argument(
        "--overview_only",
        action="store_true",
        help="Only create model_training_overview.csv from existing runs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_yaml = Path(args.source_path)
    output_root = Path(args.destination_path)
    output_root.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Data YAML: {data_yaml}")
    print(f"Training output: {output_root}")

    if not args.overview_only:
        train_grid(
            data_yaml=data_yaml,
            output_root=output_root,
            models=MODELS,
            img_sizes=IMG_SIZES,
            epochs_list=EPOCHS_LIST,
            optimizers=OPTIMIZERS,
            lr0_values=LR0_VALUES,
            weight_decay_values=WEIGHT_DECAY_VALUES,
            device=device,
        )

    create_training_overview(output_root)
    print("\nTraining finished.")


if __name__ == "__main__":
    main()
