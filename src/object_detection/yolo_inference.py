"""Future agents: maintain this file's implementation memory in this region.

# 2026-11-06

The public detection package is src.object_detection; import and CLI paths use this directory name.
Canonical implementation moved here from qichen/src/object_detection/yolo_inference.py. Legacy qichen modules re-
export these APIs.
This public module depends only on sibling public modules and installed third-party packages.
Run from the public repository: python -m src.object_detection.yolo_inference --help
Direct execution is also supported: python src/object_detection/yolo_inference.py --help

Run YOLO object-detection inference and export bounding-box predictions.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, CliApp
from ultralytics import YOLO
from ultralytics.engine.results import Results


if __package__:
    from ..utility import archive_existing_output
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utility import archive_existing_output


# region Constants and configuration

RESULT_FILE_NAME = "result.json"
CONFIG_FILE_NAME = "inference_config.json"


class InferenceConfig(BaseSettings, cli_parse_args=True):
    """Command-line configuration for YOLO object-detection inference."""

    model: Path = Field(description="Checkpoint file or training output directory.")
    image_dir: Path = Field(description="Directory containing inference images.")
    output_dir: Path | None = Field(
        default=None,
        description="Output parent for a checkpoint file; must be omitted for a training output directory.",
    )
    experiment_name: str = Field(default="default", min_length=1, description="Inference experiment directory name.")
    group_id: int = Field(default=0, description="Team or submission group identifier.")
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="Prediction confidence threshold.")
    iou_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="NMS IoU threshold for non-end-to-end models.",
    )
    image_size: int = Field(default=640, gt=0, description="Inference image size.")
    batch_size: int = Field(default=8, gt=0, description="Inference batch size.")
    max_det: int = Field(default=300, gt=0, description="Maximum predictions retained per image.")
    device: str | None = Field(default=None, description="Inference device such as 0, cpu, or mps.")
    verbose: bool = Field(default=False, description="Print per-image Ultralytics inference logs.")
    image_count: int | None = Field(
        default=None,
        gt=0,
        description="Number of images processed for debugging; process all images when omitted.",
    )

    @model_validator(mode="after")
    def setup(self) -> Self:
        """
        Validate paths, resolve defaults, archive old output, and persist the config.

        Returns:
            `InferenceConfig`: The validated configuration with resolved paths.
        """

        experiment_path: Path = Path(self.experiment_name)
        if experiment_path.is_absolute() or len(experiment_path.parts) != 1 or self.experiment_name in {".", ".."}:
            raise ValueError("experiment_name must be a single directory name")

        self.image_dir = self.image_dir.expanduser().resolve()
        if not self.image_dir.is_dir():
            raise NotADirectoryError(f"Inference image directory does not exist: {self.image_dir}")

        # region Resolve the checkpoint and output directory
        model_path: Path = self.model.expanduser()
        if model_path.is_dir():
            if self.output_dir is not None:
                raise ValueError("output_dir must be None when model is a training output directory")
            checkpoint_path: Path = resolve_training_checkpoint(model_path)
            resolved_output_dir: Path = model_path / "inference_result" / self.experiment_name
        elif model_path.is_file():
            if self.output_dir is None:
                raise ValueError("output_dir is required when model is a checkpoint file")
            checkpoint_path = model_path
            resolved_output_dir = self.output_dir.expanduser().resolve() / self.experiment_name
        else:
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        self.model = checkpoint_path.resolve()
        self.output_dir = resolved_output_dir.resolve()
        # endregion

        existing_entries: list[Path] = (
            [entry for entry in self.output_dir.iterdir() if entry.name != "history"]
            if self.output_dir.is_dir()
            else []
        )
        if any(entry.suffix != ".log" for entry in existing_entries):
            archive_existing_output(self.output_dir)

        config_dir: Path = self.output_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path: Path = config_dir / CONFIG_FILE_NAME
        config_path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return self


def resolve_training_checkpoint(training_output_dir: Path) -> Path:
    """
    Select the preferred checkpoint from a training output directory.

    Args:
        training_output_dir (`Path`):
            Training experiment directory.

    Returns:
        `Path`: Path to `best.pt`, or `last.pt` when no best checkpoint exists.
    """

    weights_dir: Path = training_output_dir / "ultralytics_output" / "weights"
    checkpoint_candidates: tuple[Path, ...] = (weights_dir / "best.pt", weights_dir / "last.pt")
    for checkpoint_path in checkpoint_candidates:
        if checkpoint_path.is_file():
            return checkpoint_path
    raise FileNotFoundError(
        f"No best.pt or last.pt checkpoint was found in the training output directory: {weights_dir}"
    )


# endregion


# region Detection result schema


class ObjectPrediction(BaseModel):
    """One predicted object represented by an absolute pixel bounding box."""

    prediction_id: int
    class_id: int
    class_name: str
    confidence_score: float
    bbox_xyxy: tuple[float, float, float, float]


class ImagePrediction(BaseModel):
    """All detection predictions for one image."""

    image_path: str
    width: int
    height: int
    predictions: list[ObjectPrediction]


class ResultSummary(BaseModel):
    """Counts for a complete inference run."""

    image_count: int
    prediction_count: int


class InferenceResult(BaseModel):
    """Versioned object-detection inference result file."""

    schema_version: Literal["1.0"] = "1.0"
    task: Literal["object_detection"] = "object_detection"
    group_id: int
    generated_at: datetime
    model: str
    coordinate_system: Literal["absolute_image_pixels"] = "absolute_image_pixels"
    summary: ResultSummary
    images: list[ImagePrediction]


# endregion


# region Result conversion and inference


def get_class_name(names: dict[int, str] | list[str], class_id: int) -> str:
    """
    Resolve a class ID from an Ultralytics class-name collection.

    Args:
        names (`dict[int, str]` or `list[str]`):
            Ultralytics class-name mapping.
        class_id (`int`):
            Predicted numeric class ID.

    Returns:
        `str`: Class name or the numeric ID as a fallback.
    """

    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return names[class_id] if 0 <= class_id < len(names) else str(class_id)


def serialize_workspace_path(path: Path, workspace_dir: Path) -> str:
    """
    Serialize an image path relative to the workspace when possible.

    Args:
        path (`Path`):
            Image path to serialize.
        workspace_dir (`Path`):
            Workspace directory used as the relative-path root.

    Returns:
        `str`: POSIX workspace-relative path or an absolute fallback.
    """

    absolute_path: Path = path.expanduser().resolve()
    try:
        return absolute_path.relative_to(workspace_dir).as_posix()
    except ValueError:
        return absolute_path.as_posix()


def convert_result(result: Results, workspace_dir: Path) -> ImagePrediction:
    """
    Convert one Ultralytics detection result into the JSON schema.

    Args:
        result (`Results`):
            Ultralytics result for one image.
        workspace_dir (`Path`):
            Workspace directory used for stored image paths.

    Returns:
        `ImagePrediction`: Serializable image metadata and bounding-box predictions.
    """

    height: int = int(result.orig_shape[0])
    width: int = int(result.orig_shape[1])
    image_path: str = serialize_workspace_path(Path(result.path), workspace_dir)
    if result.boxes is None or len(result.boxes) == 0:
        return ImagePrediction(image_path=image_path, width=width, height=height, predictions=[])

    bbox_rows: list[list[float]] = result.boxes.xyxy.cpu().tolist()
    confidence_scores: list[float] = result.boxes.conf.cpu().tolist()
    class_ids: list[int] = [int(value) for value in result.boxes.cls.cpu().tolist()]
    predictions: list[ObjectPrediction] = []
    for prediction_id, (bbox_row, confidence_score, class_id) in enumerate(
        zip(bbox_rows, confidence_scores, class_ids, strict=True)
    ):
        bbox_xyxy: tuple[float, float, float, float] = (
            float(bbox_row[0]),
            float(bbox_row[1]),
            float(bbox_row[2]),
            float(bbox_row[3]),
        )
        predictions.append(
            ObjectPrediction(
                prediction_id=prediction_id,
                class_id=class_id,
                class_name=get_class_name(result.names, class_id),
                confidence_score=float(confidence_score),
                bbox_xyxy=bbox_xyxy,
            )
        )
    return ImagePrediction(image_path=image_path, width=width, height=height, predictions=predictions)


def run_inference(config: InferenceConfig) -> Path:
    """
    Run streaming detection inference and write `result.json`.

    Args:
        config (`InferenceConfig`):
            Resolved inference configuration.

    Returns:
        `Path`: Saved inference result path.
    """

    if config.output_dir is None:
        raise RuntimeError("InferenceConfig.setup() did not resolve output_dir")

    model: YOLO = YOLO(str(config.model), task="detect")
    raw_results: Iterable[Results] = model.predict(
        source=str(config.image_dir),
        conf=config.confidence_threshold,
        iou=config.iou_threshold,
        imgsz=config.image_size,
        batch=config.batch_size,
        max_det=config.max_det,
        device=config.device,
        stream=True,
        save=False,
        verbose=config.verbose,
    )
    selected_results: Iterable[Results] = (
        raw_results if config.image_count is None else islice(raw_results, config.image_count)
    )
    workspace_dir: Path = Path.cwd().resolve()
    image_predictions: list[ImagePrediction] = [convert_result(result, workspace_dir) for result in selected_results]
    prediction_count: int = sum(len(image.predictions) for image in image_predictions)
    inference_result = InferenceResult(
        group_id=config.group_id,
        generated_at=datetime.now().astimezone(),
        model=str(config.model),
        summary=ResultSummary(image_count=len(image_predictions), prediction_count=prediction_count),
        images=image_predictions,
    )
    result_path: Path = config.output_dir / RESULT_FILE_NAME
    result_path.write_text(inference_result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result_path


# endregion


# region Application entry point


def main() -> None:
    """Parse configuration, run detection inference, and report the result path."""

    config: InferenceConfig = CliApp.run(InferenceConfig)
    result_path: Path = run_inference(config)
    print(f"Inference results saved to: {result_path}")


if __name__ == "__main__":
    main()


# endregion
