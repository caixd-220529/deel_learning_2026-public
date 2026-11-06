"""Future agents: maintain this file's implementation memory in this region.

# 2026-11-06

Canonical implementation moved here from qichen/src/apply_label.py. Legacy qichen modules re-export these APIs.
This public module depends only on sibling public modules and installed third-party packages.
Defaults are relative to the public repository; legacy wrappers preserve original qichen defaults.
Run from the public repository: python -m src.apply_label --help
Direct execution is also supported: python src/apply_label.py --help

# 2026-11-05

Render merged YOLO handwriting=0, tick=1 and cross=2 labels directly from the preprocessed tac-cxd-v0818 datasets.
Default to with_actual, which contains manual TAC annotations without synthetic marks.
Use --dataset_dir and --split to select a variant; explicit image_dir/label_dir overrides remain supported.
--render_mode object_detection draws polygon envelopes; instance_segmentation preserves source polygon outlines
and existing box labels. The latter remains the default for compatibility; no segmentation masks are invented
from box-only annotations. Colors are green, blue and red for handwriting, tick and cross, including text tags.
Preserve public Annotation/read_annotations APIs and optional rendering argument defaults used by other scripts.

Run from the repository root:

```bash
conda run -n deep_learning_2026 python -m qichen.src.apply_label \
  --dataset_dir qichen/data/tac-cxd-v0818/with_actual --split val \
  --render_mode object_detection --image_count 10 --output_dir qichen/temp/v1105/detection_labels
```
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as PILImage
from PIL.ImageDraw import ImageDraw as PILImageDraw
from PIL.ImageFont import FreeTypeFont
from PIL.ImageFont import ImageFont as PILImageFont
from pydantic import Field
from pydantic_settings import BaseSettings, CliApp


if __package__:
    from .utility import validate_directory
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utility import validate_directory


# region Constants and data models

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_DIR / "data" / "tac-cxd-v0818" / "with_actual"
DEFAULT_IMAGE_DIR = DEFAULT_DATASET_DIR / "images" / "train"
DEFAULT_LABEL_DIR = DEFAULT_DATASET_DIR / "labels" / "train"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output" / "v0807" / "label_preview"

SUPPORTED_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
BOX_COLOR = (50, 205, 50)
CLASS_COLORS = (BOX_COLOR, (30, 144, 255), (255, 69, 0), (186, 85, 211), (255, 165, 0))
TEXT_COLOR = (255, 255, 255)
DEFAULT_FONT_CANDIDATES = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
)

NormalizedPoint: TypeAlias = tuple[float, float]
PixelPoint: TypeAlias = tuple[int, int]
ShapeType: TypeAlias = Literal["polygon", "box"]
RenderMode: TypeAlias = Literal["object_detection", "instance_segmentation"]
RGBColor: TypeAlias = tuple[int, int, int]


@dataclass(frozen=True)
class Annotation:
    """One parsed YOLO annotation in normalized image coordinates."""

    class_id: int
    shape_type: ShapeType
    points: tuple[NormalizedPoint, ...]


@dataclass(frozen=True)
class ImageLabelPair:
    """An image and its matching YOLO label file."""

    image_path: Path
    label_path: Path


class Config(BaseSettings, cli_parse_args=True):
    """Command-line configuration for label visualization."""

    dataset_dir: Path = Field(default=DEFAULT_DATASET_DIR, description="Preprocessed dataset variant directory.")
    split: Literal["train", "val"] = "train"
    image_dir: Path | None = Field(default=None, description="Override dataset_dir/images/split.")
    label_dir: Path | None = Field(default=None, description="Override dataset_dir/labels/split.")
    render_mode: RenderMode = Field(
        default="instance_segmentation", description="Draw polygon outlines or axis-aligned detection boxes."
    )
    output_dir: Path = Field(default=DEFAULT_OUTPUT_DIR, description="Annotated image directory.")
    image_count: int = Field(
        default=10, gt=0, description="Number of sorted images to process when no paths are given."
    )
    image_start_index: int = Field(
        default=0,
        ge=0,
        description="Offset in the sorted image list when no paths are given.",
    )
    image_path_list: list[Path] = Field(
        default_factory=list,
        description="Optional image paths to process in the specified order.",
    )
    line_width: int = Field(default=3, gt=0, description="Outline width in pixels.")
    class_name: str = Field(default="handwriting", description="Display name used for class 0.")
    class_names_path: Path | None = Field(
        default=None,
        description="Optional class-name file with one class per line.",
    )
    font_path: Path | None = Field(default=None, description="Optional TrueType/OpenType font path.")


# endregion


# region Input discovery and YOLO parsing


def discover_image_label_pairs(image_dir: Path, label_dir: Path) -> list[ImageLabelPair]:
    """Return deterministically sorted images that have a matching label file."""

    image_paths: list[Path] = sorted(
        (path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )
    pairs: list[ImageLabelPair] = []
    for image_path in image_paths:
        label_path: Path = label_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            pairs.append(ImageLabelPair(image_path=image_path, label_path=label_path))
    return pairs


def resolve_image_label_pairs(image_paths: list[Path], label_dir: Path) -> list[ImageLabelPair]:
    """Resolve explicitly requested images and their matching label files."""

    pairs: list[ImageLabelPair] = []
    for image_path in image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix for {image_path}: {image_path.suffix}")

        label_path: Path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Label file does not exist for image {image_path}: {label_path}")
        pairs.append(ImageLabelPair(image_path=image_path, label_path=label_path))
    return pairs


def parse_annotation_line(line: str, label_path: Path, line_number: int) -> Annotation:
    """Parse one YOLO box or segmentation line."""

    tokens: list[str] = line.split()
    location: str = f"{label_path}:{line_number}"
    if len(tokens) < 5:
        raise ValueError(f"{location}: expected a class id and at least four coordinates")

    # region Get class_id, coordinates
    try:
        class_id: int = int(tokens[0])
        coordinates: list[float] = [float(token) for token in tokens[1:]]
    except ValueError as error:
        raise ValueError(f"{location}: class id or coordinate is not numeric") from error

    if class_id < 0:
        raise ValueError(f"{location}: class id must be non-negative")
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError(f"{location}: coordinates must be finite numbers")
    # endregion

    # region Get annotation
    # box coordinates: center_x center_y width height
    # polygon coordinates: x1 y1 x2 y2 x3 y3 ...
    if len(coordinates) == 4:
        center_x, center_y, box_width, box_height = coordinates
        left: float = center_x - box_width / 2.0
        top: float = center_y - box_height / 2.0
        right: float = center_x + box_width / 2.0
        bottom: float = center_y + box_height / 2.0
        points: tuple[NormalizedPoint, ...] = (
            (max(0.0, left), max(0.0, top)),
            (min(1.0, right), min(1.0, bottom)),
        )
        return Annotation(class_id=class_id, shape_type="box", points=points)

    if len(coordinates) < 6 or len(coordinates) % 2 != 0:
        raise ValueError(f"{location}: a polygon needs at least three x/y coordinate pairs")
    polygon_points: tuple[NormalizedPoint, ...] = tuple(
        (coordinates[index], coordinates[index + 1]) for index in range(0, len(coordinates), 2)
    )
    return Annotation(class_id=class_id, shape_type="polygon", points=polygon_points)
    # endregion


def read_annotations(label_path: Path) -> list[Annotation]:
    """Read all non-empty YOLO annotations from a label file."""

    annotations: list[Annotation] = []
    label_text: str = label_path.read_text(encoding="utf-8-sig")
    for line_number, raw_line in enumerate(label_text.splitlines(), start=0):
        line: str = raw_line.strip()
        if line:
            annotations.append(parse_annotation_line(line, label_path, line_number))
    return annotations


def read_class_names(config: Config) -> list[str]:
    """Read class names from plain lines or indexed `<class_id> <name>` lines."""

    if config.class_names_path is None:
        return [config.class_name, "tick", "cross"]
    if not config.class_names_path.is_file():
        raise FileNotFoundError(f"Class-name file does not exist: {config.class_names_path}")
    class_lines: list[str] = [
        line.strip() for line in config.class_names_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()
    ]
    if not class_lines:
        raise ValueError(f"Class-name file is empty: {config.class_names_path}")

    indexed_names: list[tuple[int, str]] = []
    for class_line in class_lines:
        fields: list[str] = class_line.split(maxsplit=1)
        if len(fields) != 2:
            return class_lines
        try:
            class_id: int = int(fields[0])
        except ValueError:
            return class_lines
        indexed_names.append((class_id, fields[1]))

    indexed_names.sort(key=lambda item: item[0])
    actual_ids: list[int] = [class_id for class_id, _ in indexed_names]
    expected_ids: list[int] = list(range(len(indexed_names)))
    if actual_ids != expected_ids:
        raise ValueError(
            f"Indexed class IDs in {config.class_names_path} must be unique and contiguous from 0; found {actual_ids}"
        )
    return [class_name for _, class_name in indexed_names]


# endregion


# region Rendering and output


def resolve_font(font_path: Path | None, font_size: int) -> PILImageFont | FreeTypeFont:
    """Load a requested font or the first available CJK-capable fallback."""

    candidate_paths: tuple[Path, ...]
    if font_path is not None:
        candidate_paths = (font_path,)
    else:
        candidate_paths = DEFAULT_FONT_CANDIDATES

    for candidate_path in candidate_paths:
        if candidate_path.is_file():
            return ImageFont.truetype(str(candidate_path), size=font_size)
    if font_path is not None:
        raise FileNotFoundError(f"Font file does not exist: {font_path}")
    return ImageFont.load_default()


def to_pixel_points(
    points: tuple[NormalizedPoint, ...], image_width: int, image_height: int
) -> tuple[PixelPoint, ...]:
    """Convert normalized coordinates into clamped pixel coordinates."""

    max_x: int = image_width - 1
    max_y: int = image_height - 1
    return tuple(
        (
            min(max(round(x * image_width), 0), max_x),
            min(max(round(y * image_height), 0), max_y),
        )
        for x, y in points
    )


def draw_text_label(
    draw: PILImageDraw,
    anchor: PixelPoint,
    text: str,
    font: PILImageFont | FreeTypeFont,
    color: RGBColor = BOX_COLOR,
) -> None:
    """Draw a class-colored tag immediately above a shape's top-left point."""

    text_box: tuple[int, int, int, int] = draw.textbbox((0, 0), text, font=font)  # pyright: ignore[reportAssignmentType]
    text_width: int = text_box[2] - text_box[0]
    text_height: int = text_box[3] - text_box[1]
    padding_x: int = max(3, text_height // 4)
    padding_y: int = max(2, text_height // 8)
    box_width: int = text_width + 2 * padding_x
    box_height: int = text_height + 2 * padding_y
    left: int = anchor[0]
    top: int = max(0, anchor[1] - box_height)
    draw.rectangle((left, top, left + box_width, top + box_height), fill=color)
    draw.text(
        (left + padding_x, top + padding_y - text_box[1]),
        text,
        fill=TEXT_COLOR,
        font=font,
    )


def draw_annotation(
    draw: PILImageDraw,
    annotation: Annotation,
    image_width: int,
    image_height: int,
    line_width: int,
    label_text: str,
    font: PILImageFont | FreeTypeFont,
    render_mode: RenderMode = "instance_segmentation",
) -> None:
    """Draw a class-colored polygon or its detection envelope without changing source annotations."""

    if render_mode not in {"object_detection", "instance_segmentation"}:
        raise ValueError(f"Unsupported render mode: {render_mode}")
    color: RGBColor = CLASS_COLORS[annotation.class_id % len(CLASS_COLORS)]
    pixel_points: tuple[PixelPoint, ...] = to_pixel_points(annotation.points, image_width, image_height)
    if annotation.shape_type == "box" or render_mode == "object_detection":
        top_left: PixelPoint = (min(point[0] for point in pixel_points), min(point[1] for point in pixel_points))
        bottom_right: PixelPoint = (max(point[0] for point in pixel_points), max(point[1] for point in pixel_points))
        draw.rectangle((top_left, bottom_right), outline=color, width=line_width)
    else:
        closed_points: tuple[PixelPoint, ...] = pixel_points + (pixel_points[0],)
        draw.line(closed_points, fill=color, width=line_width, joint="curve")

    anchor: PixelPoint = (
        min(point[0] for point in pixel_points),
        min(point[1] for point in pixel_points),
    )
    draw_text_label(draw, anchor, label_text, font, color)


def save_image(image: PILImage, output_path: Path, source_format: str | None) -> None:
    """Save an annotated image while avoiding unnecessary JPEG degradation."""

    save_options: dict[str, int] = {}
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        save_options = {"quality": 95, "subsampling": 0}
    image.save(output_path, format=source_format, **save_options)


def annotate_image(
    pair: ImageLabelPair,
    output_dir: Path,
    class_names: list[str],
    line_width: int,
    font_path: Path | None,
    render_mode: RenderMode = "instance_segmentation",
) -> int:
    """Render and save a single labeled image, returning its object count."""

    annotations: list[Annotation] = read_annotations(pair.label_path)
    with Image.open(pair.image_path) as source_image:
        source_format: str | None = source_image.format
        image: PILImage = source_image.convert("RGB")
        image_width, image_height = image.size
        font_size: int = max(14, round(min(image_width, image_height) * 0.018))
        font: PILImageFont | FreeTypeFont = resolve_font(font_path, font_size)
        draw: PILImageDraw = ImageDraw.Draw(image)

        for display_index, annotation in enumerate(annotations, start=0):
            class_name: str = (
                class_names[annotation.class_id]
                if annotation.class_id < len(class_names)
                else f"class_{annotation.class_id}"
            )
            label_text: str = f"{display_index} {class_name}"
            draw_annotation(
                draw,
                annotation,
                image_width,
                image_height,
                line_width,
                label_text,
                font,
                render_mode,
            )

        output_path: Path = output_dir / pair.image_path.name
        save_image(image, output_path, source_format)
    return len(annotations)


# endregion


# region Application entry point


def main(config_type: type[Config] = Config) -> None:
    """
    Visualize labels for explicitly requested images or a sorted image slice.

    Args:
        config_type (`type[Config]`, *optional*, defaults to `Config`):
            Settings class; legacy callers supply a subclass retaining original path defaults.
    """

    config: Config = CliApp.run(config_type)
    image_dir: Path = config.image_dir or config.dataset_dir / "images" / config.split
    label_dir: Path = config.label_dir or config.dataset_dir / "labels" / config.split
    validate_directory(label_dir, "Label directory")
    class_names: list[str] = read_class_names(config)

    all_pairs: list[ImageLabelPair] | None = None
    selected_pairs: list[ImageLabelPair]
    if config.image_path_list:
        selected_pairs = resolve_image_label_pairs(config.image_path_list, label_dir)
    else:
        validate_directory(image_dir, "Image directory")
        all_pairs = discover_image_label_pairs(image_dir, label_dir)
        selected_pairs = all_pairs[config.image_start_index : config.image_start_index + config.image_count]

    if not selected_pairs:
        available_pair_count: int = len(all_pairs) if all_pairs is not None else 0
        raise ValueError(
            "No matched image/label pairs were selected. "
            f"Found {available_pair_count} pairs; image_start_index={config.image_start_index}."
        )
    output_dir: Path = config.output_dir.resolve()
    for pair in selected_pairs:
        if pair.image_path.resolve().parent == output_dir:
            raise ValueError(f"Output directory must differ from the input image directory: {pair.image_path.parent}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    total_annotations: int = 0
    for processed_count, pair in enumerate(selected_pairs):
        annotation_count: int = annotate_image(
            pair,
            config.output_dir,
            class_names,
            config.line_width,
            config.font_path,
            config.render_mode,
        )
        total_annotations += annotation_count
        print(f"[{processed_count}/{len(selected_pairs)}] {pair.image_path.name}: {annotation_count} annotations")

    if not config.image_path_list and len(selected_pairs) < config.image_count:
        print(
            f"Warning: requested {config.image_count} images, but only {len(selected_pairs)} "
            "matched pairs remained in the selected range."
        )
    print(f"Done: saved {len(selected_pairs)} images with {total_annotations} annotations to {config.output_dir}")


if __name__ == "__main__":
    main()


# endregion
