"""Future agents: maintain this file's implementation memory in this region.

# 2026-11-06

Canonical implementation moved here from qichen/src/object_detection/yolo_evaluation.py. Legacy qichen modules re-
export these APIs.
This public module depends only on sibling public modules and installed third-party packages.
Run from the public repository: python -m src.objection_detection.yolo_evaluation --help
Direct execution is also supported: python src/objection_detection/yolo_evaluation.py --help

Evaluate exported YOLO object-detection boxes against YOLO labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self, cast

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, CliApp
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ap_per_class, box_iou


if __package__:
    from ..apply_label import Annotation, read_annotations
    from ..utility import archive_existing_output, validate_directory
    from .yolo_inference import RESULT_FILE_NAME, ImagePrediction, InferenceResult, ResultSummary
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.apply_label import Annotation, read_annotations
    from src.objection_detection.yolo_inference import (
        RESULT_FILE_NAME,
        ImagePrediction,
        InferenceResult,
        ResultSummary,
    )
    from src.utility import archive_existing_output, validate_directory


# region Constants and configuration

EVALUATION_RESULT_FILE_NAME = "evaluation_result.json"
EVALUATION_CONFIG_FILE_NAME = "evaluation_config.json"


class EvaluationConfig(BaseSettings, cli_parse_args=True):
    """Command-line configuration for object-detection evaluation."""

    inference_result_path: Path = Field(description="Path to result.json or its inference output directory.")
    label_dir: Path = Field(description="Directory containing YOLO ground-truth labels.")
    output_dir: Path | None = Field(default=None, description="Explicit evaluation output directory.")
    experiment_name: str | None = Field(
        default=None,
        description="Named directory below inference_output_dir/evaluation_result.",
    )

    @model_validator(mode="after")
    def setup(self) -> Self:
        """
        Resolve paths, archive an existing output, and save the configuration.

        Returns:
            `EvaluationConfig`: The validated configuration with resolved paths.
        """

        # region Resolve the inference result
        input_path: Path = self.inference_result_path.expanduser().resolve()
        if input_path.is_dir():
            inference_output_dir: Path = input_path
            resolved_result_path: Path = input_path / RESULT_FILE_NAME
        elif input_path.is_file():
            inference_output_dir = input_path.parent
            resolved_result_path = input_path
        else:
            raise FileNotFoundError(f"Inference result path does not exist: {input_path}")
        if not resolved_result_path.is_file():
            raise FileNotFoundError(f"Inference result does not exist: {resolved_result_path}")
        # endregion

        # region Resolve the evaluation output
        if self.output_dir is not None:
            if self.experiment_name is not None:
                raise ValueError("experiment_name must be None when output_dir is specified")
            resolved_output_dir: Path = self.output_dir.expanduser().resolve()
        else:
            output_name: str = self.experiment_name or datetime.now().astimezone().strftime("%Y-%m-%d-%H-%M-%S")
            output_name_path: Path = Path(output_name)
            if output_name_path.is_absolute() or len(output_name_path.parts) != 1 or output_name in {".", ".."}:
                raise ValueError("experiment_name must be a single directory name")
            resolved_output_dir = inference_output_dir / "evaluation_result" / output_name
        # endregion

        self.inference_result_path = resolved_result_path.resolve()
        self.output_dir = resolved_output_dir.resolve()
        self.label_dir = self.label_dir.expanduser().resolve()
        validate_directory(self.label_dir, "Label directory")

        archive_existing_output(self.output_dir)
        config_dir: Path = self.output_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path: Path = config_dir / EVALUATION_CONFIG_FILE_NAME
        config_path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return self


# endregion


# region Evaluation result schema


class Metric(BaseModel):
    """Aggregate precision, recall, and average precision values."""

    precision: float = Field(ge=0.0, le=1.0, description="Mean precision at the maximum-F1 threshold.")
    recall: float = Field(ge=0.0, le=1.0, description="Mean recall at the maximum-F1 threshold.")
    map: float = Field(ge=0.0, le=1.0, description="Mean AP over IoU thresholds 0.50 through 0.95.")
    map50: float = Field(ge=0.0, le=1.0, description="Mean AP at IoU threshold 0.50.")


class ClassMetric(Metric):
    """Precision, recall, and average precision values for one class."""

    class_id: int = Field(ge=0)


class EvaluationSummary(BaseModel):
    """Counts describing the evaluated inference result."""

    image_count: int = Field(ge=0)
    prediction_count: int = Field(ge=0)
    target_count: int = Field(ge=0)


class EvaluationResult(BaseModel):
    """Serializable object-detection evaluation result."""

    evaluated_at: datetime
    inference_result_path: str
    group_id: int
    summary: EvaluationSummary
    box_level: Metric
    class_level: list[ClassMetric]


# endregion


# region Internal evaluation data models


@dataclass(frozen=True)
class GroundTruthObject:
    """One ground-truth object in absolute image coordinates."""

    class_id: int
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class ImageMatchResult:
    """Matched prediction arrays for one evaluated image."""

    box_correct: np.ndarray
    confidence_scores: np.ndarray
    predicted_classes: np.ndarray
    target_classes: np.ndarray


@dataclass(frozen=True)
class CalculatedMetrics:
    """Aggregate and per-class metrics returned by Ultralytics."""

    overall: Metric
    per_class: list[ClassMetric]


# endregion


# region Input parsing and ground-truth conversion


def read_inference_result(result_path: Path) -> InferenceResult:
    """
    Read and validate an object-detection inference result.

    Args:
        result_path (`Path`):
            Path to `result.json`.

    Returns:
        `InferenceResult`: Validated inference result.
    """

    result: InferenceResult = InferenceResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    actual_prediction_count: int = sum(len(image.predictions) for image in result.images)
    expected_summary: ResultSummary = result.summary
    if (
        expected_summary.image_count != len(result.images)
        or expected_summary.prediction_count != actual_prediction_count
    ):
        raise ValueError("Inference result summary does not match its image and prediction records")
    return result


def annotation_to_ground_truth(
    annotation: Annotation,
    image_width: int,
    image_height: int,
) -> GroundTruthObject:
    """
    Convert a normalized YOLO box or polygon into its absolute enclosing box.

    Args:
        annotation (`Annotation`):
            Parsed YOLO box or polygon annotation.
        image_width (`int`):
            Original image width in pixels.
        image_height (`int`):
            Original image height in pixels.

    Returns:
        `GroundTruthObject`: Class ID and absolute `xyxy` bounding box.
    """

    absolute_points: tuple[tuple[float, float], ...] = tuple(
        (
            min(max(x * image_width, 0.0), float(image_width)),
            min(max(y * image_height, 0.0), float(image_height)),
        )
        for x, y in annotation.points
    )
    x_coordinates: tuple[float, ...] = tuple(point[0] for point in absolute_points)
    y_coordinates: tuple[float, ...] = tuple(point[1] for point in absolute_points)
    bbox_xyxy: tuple[float, float, float, float] = (
        min(x_coordinates),
        min(y_coordinates),
        max(x_coordinates),
        max(y_coordinates),
    )
    if bbox_xyxy[0] >= bbox_xyxy[2] or bbox_xyxy[1] >= bbox_xyxy[3]:
        raise ValueError(f"Ground-truth annotation has a degenerate bounding box: {annotation}")
    return GroundTruthObject(class_id=annotation.class_id, bbox_xyxy=bbox_xyxy)


def read_ground_truths(image_prediction: ImagePrediction, label_dir: Path) -> list[GroundTruthObject]:
    """
    Read and convert ground-truth objects for one prediction image.

    Exact duplicate annotations are removed to match Ultralytics label loading.

    Args:
        image_prediction (`ImagePrediction`):
            Image metadata and exported predictions.
        label_dir (`Path`):
            Directory containing YOLO label files.

    Returns:
        `list[GroundTruthObject]`: Ground-truth boxes in source annotation order.
    """

    image_name: str = Path(image_prediction.image_path).name
    label_path: Path = label_dir / f"{Path(image_name).stem}.txt"
    if not label_path.is_file():
        return []
    annotations: list[Annotation] = list(dict.fromkeys(read_annotations(label_path)))
    return [
        annotation_to_ground_truth(annotation, image_prediction.width, image_prediction.height)
        for annotation in annotations
    ]


# endregion


# region IoU and matching


def evaluate_image(
    image_prediction: ImagePrediction,
    label_dir: Path,
    validator: DetectionValidator,
) -> ImageMatchResult:
    """
    Calculate class-aware bounding-box matches for one image.

    Args:
        image_prediction (`ImagePrediction`):
            Exported predictions and image metadata.
        label_dir (`Path`):
            Directory containing YOLO labels.
        validator (`DetectionValidator`):
            Ultralytics validator used for official class-aware matching.

    Returns:
        `ImageMatchResult`: Correctness, confidence, prediction-class, and target-class arrays.
    """

    ground_truths: list[GroundTruthObject] = read_ground_truths(image_prediction, label_dir)
    target_boxes: torch.Tensor = torch.tensor(
        [target.bbox_xyxy for target in ground_truths], dtype=torch.float32
    ).reshape(-1, 4)
    prediction_boxes: torch.Tensor = torch.tensor(
        [prediction.bbox_xyxy for prediction in image_prediction.predictions], dtype=torch.float32
    ).reshape(-1, 4)
    target_classes: np.ndarray = np.asarray([target.class_id for target in ground_truths], dtype=np.int64)
    predicted_classes: np.ndarray = np.asarray(
        [prediction.class_id for prediction in image_prediction.predictions], dtype=np.int64
    )
    confidence_scores: np.ndarray = np.asarray(
        [prediction.confidence_score for prediction in image_prediction.predictions], dtype=np.float64
    )
    box_iou_matrix: torch.Tensor = box_iou(target_boxes, prediction_boxes)
    box_correct: np.ndarray = validator.match_predictions(
        torch.from_numpy(predicted_classes),
        torch.from_numpy(target_classes),
        box_iou_matrix,
    ).numpy()
    return ImageMatchResult(
        box_correct=box_correct,
        confidence_scores=confidence_scores,
        predicted_classes=predicted_classes,
        target_classes=target_classes,
    )


# endregion


# region Metric aggregation and output


def calculate_metrics(
    correct: np.ndarray,
    confidence_scores: np.ndarray,
    predicted_classes: np.ndarray,
    target_classes: np.ndarray,
) -> CalculatedMetrics:
    """
    Calculate aggregate and per-class box metrics from prediction matches.

    Args:
        correct (`np.ndarray`):
            Prediction correctness matrix over ten IoU thresholds.
        confidence_scores (`np.ndarray`):
            Confidence score for every prediction.
        predicted_classes (`np.ndarray`):
            Class ID for every prediction.
        target_classes (`np.ndarray`):
            Class ID for every ground-truth object.

    Returns:
        `CalculatedMetrics`: Aggregate and per-class P, R, mAP50-95, and mAP50 values.
    """

    if target_classes.size == 0:
        raise ValueError("Cannot calculate mAP because the selected labels contain no ground-truth objects")
    ap_results: tuple[np.ndarray, ...] = cast(
        tuple[np.ndarray, ...],
        ap_per_class(correct, confidence_scores, predicted_classes, target_classes),
    )
    precision_values: np.ndarray = ap_results[2]
    recall_values: np.ndarray = ap_results[3]
    average_precisions: np.ndarray = ap_results[5]
    unique_class_ids: np.ndarray = ap_results[6]
    per_class: list[ClassMetric] = [
        ClassMetric(
            class_id=int(class_id),
            precision=float(precision_values[class_index]),
            recall=float(recall_values[class_index]),
            map=float(average_precisions[class_index].mean()),
            map50=float(average_precisions[class_index, 0]),
        )
        for class_index, class_id in enumerate(unique_class_ids)
    ]
    overall = Metric(
        precision=float(precision_values.mean()) if precision_values.size else 0.0,
        recall=float(recall_values.mean()) if recall_values.size else 0.0,
        map=float(average_precisions.mean()) if average_precisions.size else 0.0,
        map50=float(average_precisions[:, 0].mean()) if average_precisions.size else 0.0,
    )
    return CalculatedMetrics(overall=overall, per_class=per_class)


def serialize_workspace_path(path: Path) -> str:
    """
    Serialize a workspace path as relative when possible.

    Args:
        path (`Path`):
            Resolved filesystem path.

    Returns:
        `str`: POSIX workspace-relative path or an absolute fallback.
    """

    workspace_dir: Path = Path.cwd().resolve()
    try:
        return path.resolve().relative_to(workspace_dir).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def evaluate(config: EvaluationConfig) -> Path:
    """
    Evaluate an inference result and write `evaluation_result.json`.

    Args:
        config (`EvaluationConfig`):
            Resolved evaluation configuration.

    Returns:
        `Path`: Saved evaluation result path.
    """

    if config.output_dir is None:
        raise RuntimeError("EvaluationConfig.setup() did not resolve output_dir")
    inference_result: InferenceResult = read_inference_result(config.inference_result_path)
    validator: DetectionValidator = DetectionValidator(
        save_dir=config.output_dir,
        args={"task": "detect", "plots": False},
    )
    image_match_results: list[ImageMatchResult] = [
        evaluate_image(image_prediction, config.label_dir, validator) for image_prediction in inference_result.images
    ]
    if not image_match_results:
        raise ValueError("Inference result does not contain any images to evaluate")

    box_correct: np.ndarray = np.concatenate([result.box_correct for result in image_match_results], axis=0)
    confidence_scores: np.ndarray = np.concatenate(
        [result.confidence_scores for result in image_match_results], axis=0
    )
    predicted_classes: np.ndarray = np.concatenate(
        [result.predicted_classes for result in image_match_results], axis=0
    )
    target_classes: np.ndarray = np.concatenate([result.target_classes for result in image_match_results], axis=0)
    calculated_metrics: CalculatedMetrics = calculate_metrics(
        box_correct,
        confidence_scores,
        predicted_classes,
        target_classes,
    )

    evaluation_result = EvaluationResult(
        evaluated_at=datetime.now().astimezone(),
        inference_result_path=serialize_workspace_path(config.inference_result_path),
        group_id=inference_result.group_id,
        summary=EvaluationSummary(
            image_count=len(inference_result.images),
            prediction_count=int(confidence_scores.size),
            target_count=int(target_classes.size),
        ),
        box_level=calculated_metrics.overall,
        class_level=calculated_metrics.per_class,
    )
    result_path: Path = config.output_dir / EVALUATION_RESULT_FILE_NAME
    result_path.write_text(evaluation_result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result_path


# endregion


# region Application entry point


def main() -> None:
    """Parse configuration, evaluate detection boxes, and report the result path."""

    config: EvaluationConfig = CliApp.run(EvaluationConfig)
    result_path: Path = evaluate(config)
    print(f"Evaluation results saved to: {result_path}")


if __name__ == "__main__":
    main()


# endregion
