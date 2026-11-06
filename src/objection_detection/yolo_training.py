"""Future agents: maintain this file's implementation memory in this region.

# 2026-11-06

Canonical implementation moved here from qichen/src/object_detection/yolo_training.py. Legacy qichen modules re-
export these APIs.
This public module depends only on sibling public modules and installed third-party packages.
Run from the public repository: python -m src.objection_detection.yolo_training --help
Direct execution is also supported: python src/objection_detection/yolo_training.py --help
Public defaults are rooted in this repository and use the portable yolo26n.pt model name.
The legacy entry point preserves qichen dataset/output/model defaults through a Config subclass.

Train an Ultralytics YOLO object-detection model on the TAC dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, CliApp


if __package__:
    from ..utility import validate_directory
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utility import validate_directory


# region Constants and configuration

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_DIR / "data" / "handwriting_tac_det-v0816"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output" / "v0816" / "handwriting_tac-yolo26n"
SUPPORTED_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
REQUIRED_SPLITS = ("train", "val")


@dataclass(frozen=True)
class SplitSummary:
    """Counts for one validated dataset split."""

    name: str
    image_count: int
    matched_label_count: int


class TrainingConfig(BaseSettings, cli_parse_args=True):
    """Command-line configuration for YOLO object-detection training."""

    dataset_dir: Path = Field(default=DEFAULT_DATASET_DIR, description="YOLO dataset root directory.")
    class_file: Path | None = Field(
        default=None,
        description="Class file; defaults to <dataset_dir>/class.txt.",
    )
    model: str = Field(
        default="yolo26n.pt",
        description="YOLO detection checkpoint or model configuration.",
    )
    epochs: int = Field(default=100, gt=0, description="Number of training epochs.")
    image_size: int = Field(default=640, gt=0, description="Training image size.")
    batch_size: int = Field(default=8, gt=0, description="Images per training batch.")
    device: str | None = Field(
        default=None,
        description="Training device, such as 0, 0,1, cpu, or mps; null selects automatically.",
    )
    workers: int = Field(default=8, ge=0, description="Data-loader worker processes.")
    output_dir: Path = Field(default=DEFAULT_OUTPUT_DIR, description="Training experiment output directory.")
    patience: int = Field(default=50, ge=0, description="Early-stopping patience in epochs.")
    seed: int = Field(default=0, ge=0, description="Random seed.")
    cache: bool = Field(default=False, description="Cache dataset images in memory.")
    amp: bool = Field(default=True, description="Use automatic mixed precision when supported.")
    exist_ok: bool = Field(default=False, description="Allow reuse of an existing run directory.")


def write_training_config(config: TrainingConfig, output_path: Path) -> None:
    """
    Persist a validated training configuration as JSON.

    Args:
        config (`TrainingConfig`):
            Parsed command-line training configuration.
        output_path (`Path`):
            Destination JSON path.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")


# endregion


# region Dataset validation and runtime configuration


def summarize_split(dataset_dir: Path, split: str) -> SplitSummary:
    """
    Validate a dataset split and count images with matching labels.

    Args:
        dataset_dir (`Path`):
            YOLO dataset root directory.
        split (`str`):
            Dataset split name.

    Returns:
        `SplitSummary`: Image and matching-label counts for the split.
    """

    image_dir: Path = dataset_dir / "images" / split
    label_dir: Path = dataset_dir / "labels" / split
    validate_directory(image_dir, f"{split} image directory")
    validate_directory(label_dir, f"{split} label directory")

    image_paths: list[Path] = sorted(
        (path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )
    if not image_paths:
        raise ValueError(f"No supported images were found in {image_dir}")

    matched_label_count: int = sum((label_dir / f"{image_path.stem}.txt").is_file() for image_path in image_paths)
    if matched_label_count == 0:
        raise ValueError(f"No image has a matching YOLO label in {label_dir}")
    return SplitSummary(
        name=split,
        image_count=len(image_paths),
        matched_label_count=matched_label_count,
    )


def read_class_names(class_file: Path) -> tuple[str, ...]:
    """
    Read contiguous class IDs and names from a class file.

    Args:
        class_file (`Path`):
            Text file containing lines such as `0 handwriting`.

    Returns:
        `tuple[str, ...]`: Class names ordered by class ID.
    """

    if not class_file.is_file():
        raise FileNotFoundError(f"Class file does not exist: {class_file}")

    indexed_names: list[tuple[int, str]] = []
    class_text: str = class_file.read_text(encoding="utf-8-sig")
    for line_number, raw_line in enumerate(class_text.splitlines(), start=1):
        line: str = raw_line.strip()
        if not line:
            continue
        fields: list[str] = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"{class_file}:{line_number}: expected '<class_id> <class_name>'")
        try:
            class_id: int = int(fields[0])
        except ValueError as error:
            raise ValueError(f"{class_file}:{line_number}: class ID must be an integer") from error
        class_name: str = fields[1].strip()
        if class_id < 0 or not class_name:
            raise ValueError(f"{class_file}:{line_number}: invalid class ID or name")
        indexed_names.append((class_id, class_name))

    indexed_names.sort(key=lambda item: item[0])
    actual_ids: list[int] = [class_id for class_id, _ in indexed_names]
    expected_ids: list[int] = list(range(len(indexed_names)))
    if not indexed_names or actual_ids != expected_ids:
        raise ValueError(f"Class IDs in {class_file} must be unique and contiguous from 0; found {actual_ids}")
    return tuple(class_name for _, class_name in indexed_names)


def write_runtime_dataset_yaml(
    dataset_dir: Path,
    class_names: tuple[str, ...],
    output_path: Path,
) -> None:
    """
    Write a dataset YAML whose root path is valid on the current machine.

    Polygon labels remain unchanged because Ultralytics detection loaders convert
    them to their minimum enclosing boxes during label verification.

    Args:
        dataset_dir (`Path`):
            YOLO dataset root directory.
        class_names (`tuple[str, ...]`):
            Class names ordered by class ID.
        output_path (`Path`):
            Destination YAML path.
    """

    yaml_lines: list[str] = [
        f"path: {json.dumps(str(dataset_dir.resolve()), ensure_ascii=False)}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    yaml_lines.extend(
        f"  {class_id}: {json.dumps(class_name, ensure_ascii=False)}"
        for class_id, class_name in enumerate(class_names)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


# endregion


# region YOLO training


def train(config: TrainingConfig, dataset_yaml: Path, ultralytics_output_dir: Path) -> None:
    """
    Load the requested YOLO detection model and start training.

    Args:
        config (`TrainingConfig`):
            Parsed training configuration.
        dataset_yaml (`Path`):
            Runtime dataset YAML path.
        ultralytics_output_dir (`Path`):
            Directory where Ultralytics writes training artifacts.
    """

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed in the active Python environment. "
            "Install it with 'python -m pip install ultralytics'."
        ) from error

    model: YOLO = YOLO(config.model, task="detect")
    model.train(
        data=str(dataset_yaml),
        epochs=config.epochs,
        imgsz=config.image_size,
        batch=config.batch_size,
        device=config.device,
        workers=config.workers,
        save_dir=ultralytics_output_dir,
        patience=config.patience,
        seed=config.seed,
        cache=config.cache,
        amp=config.amp,
        exist_ok=config.exist_ok,
        task="detect",
    )


# endregion


# region Application entry point


def main(config_type: type[TrainingConfig] = TrainingConfig) -> None:
    """
    Validate the TAC dataset, create its YAML, and launch detection training.

    Args:
        config_type (`type[TrainingConfig]`, *optional*, defaults to `TrainingConfig`):
            Settings class; legacy callers supply a subclass retaining original path and model defaults.
    """

    config: TrainingConfig = CliApp.run(config_type)
    validate_directory(config.dataset_dir, "Dataset directory")

    split_summaries: tuple[SplitSummary, ...] = tuple(
        summarize_split(config.dataset_dir, split) for split in REQUIRED_SPLITS
    )
    class_file: Path = config.class_file or config.dataset_dir / "class.txt"
    class_names: tuple[str, ...] = read_class_names(class_file)
    config_dir: Path = config.output_dir / "config"
    dataset_yaml: Path = config_dir / "handwriting_tac_det.yaml"
    write_runtime_dataset_yaml(config.dataset_dir, class_names, dataset_yaml)
    config_path: Path = config_dir / "config.json"
    write_training_config(config, config_path)
    ultralytics_output_dir: Path = config.output_dir / "ultralytics_output"

    for summary in split_summaries:
        print(f"{summary.name}: {summary.image_count} images, {summary.matched_label_count} matching labels")
    print(f"classes: {', '.join(class_names)}")
    print("detection boxes: derived automatically from the dataset polygon labels")
    print(f"dataset config: {dataset_yaml}")
    print(f"training config: {config_path}")
    print(f"model: {config.model}")
    print(f"Ultralytics output: {ultralytics_output_dir}")

    train(config, dataset_yaml, ultralytics_output_dir)


if __name__ == "__main__":
    main()


# endregion
