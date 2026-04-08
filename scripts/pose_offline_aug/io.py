from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .structures import BBox, ImageAnnotation, Keypoint, ObjectAnnotation


REVIEW_VISIBLE_COLOR = (64, 224, 208)
REVIEW_OCCLUDED_COLOR = (0, 180, 255)
REVIEW_BOX_COLOR = (48, 48, 48)
REVIEW_TEXT_COLOR = (255, 255, 255)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def format_float(value: float) -> str:
    return f"{value:.10f}".rstrip("0").rstrip(".") or "0"


def read_image_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to decode image: {path}")
    return image


def write_image_bgr(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix.lower()
    ext = ".png" if suffix == ".png" else ".jpg"
    params: list[int] = []
    if ext == ".jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def parse_pose_label_line(
    line: str,
    keypoint_count: int,
    image_width: int,
    image_height: int,
) -> ObjectAnnotation:
    parts = line.split()
    expected = 5 + keypoint_count * 3
    if len(parts) != expected:
        raise ValueError(f"Expected {expected} values, found {len(parts)}.")

    class_id = int(float(parts[0]))
    center_x = float(parts[1]) * image_width
    center_y = float(parts[2]) * image_height
    width = float(parts[3]) * image_width
    height = float(parts[4]) * image_height
    bbox = BBox(
        x1=center_x - width / 2.0,
        y1=center_y - height / 2.0,
        x2=center_x + width / 2.0,
        y2=center_y + height / 2.0,
    )

    keypoints: list[Keypoint] = []
    offset = 5
    for index in range(keypoint_count):
        x_coord = float(parts[offset + index * 3]) * image_width
        y_coord = float(parts[offset + index * 3 + 1]) * image_height
        visibility = int(round(float(parts[offset + index * 3 + 2])))
        keypoints.append(Keypoint(x=x_coord, y=y_coord, v=visibility))
    return ObjectAnnotation(class_id=class_id, bbox=bbox, keypoints=keypoints)


def read_single_pose_annotation(
    image_path: Path,
    label_path: Path,
    split: str,
    sample_id: str,
    keypoint_count: int,
) -> ImageAnnotation:
    image = read_image_bgr(image_path)
    image_height, image_width = image.shape[:2]
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"Expected exactly one instance per label file, found {len(lines)} in {label_path}")
    annotation = parse_pose_label_line(lines[0], keypoint_count=keypoint_count, image_width=image_width, image_height=image_height)
    return ImageAnnotation(
        sample_id=sample_id,
        split=split,
        image_path=image_path.resolve(),
        label_path=label_path.resolve(),
        image_width=image_width,
        image_height=image_height,
        object_annotation=annotation,
    )


def object_annotation_to_yolo_pose_line(
    annotation: ObjectAnnotation,
    image_width: int,
    image_height: int,
) -> str:
    bbox = annotation.bbox
    center_x, center_y = bbox.center
    values = [
        str(annotation.class_id),
        format_float(center_x / image_width),
        format_float(center_y / image_height),
        format_float(bbox.width / image_width),
        format_float(bbox.height / image_height),
    ]
    for keypoint in annotation.keypoints:
        if keypoint.v <= 0:
            x_coord = 0.0
            y_coord = 0.0
        else:
            x_coord = min(max(keypoint.x, 0.0), image_width - 1.0)
            y_coord = min(max(keypoint.y, 0.0), image_height - 1.0)
        values.append(format_float(x_coord / image_width))
        values.append(format_float(y_coord / image_height))
        values.append(str(int(keypoint.v)))
    return " ".join(values) + "\n"


def write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv_file(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def draw_review_overlay(
    image: np.ndarray,
    annotation: ObjectAnnotation,
    title: str,
) -> np.ndarray:
    rendered = image.copy()
    bbox = annotation.bbox
    x1 = int(round(bbox.x1))
    y1 = int(round(bbox.y1))
    x2 = int(round(bbox.x2))
    y2 = int(round(bbox.y2))
    cv2.rectangle(rendered, (x1, y1), (x2, y2), REVIEW_BOX_COLOR, 2, lineType=cv2.LINE_AA)

    for index, keypoint in enumerate(annotation.keypoints):
        if keypoint.v <= 0:
            continue
        point = (int(round(keypoint.x)), int(round(keypoint.y)))
        color = REVIEW_VISIBLE_COLOR if keypoint.v >= 2 else REVIEW_OCCLUDED_COLOR
        cv2.circle(rendered, point, 3, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(
            rendered,
            str(index),
            (point[0] + 4, point[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            REVIEW_TEXT_COLOR,
            1,
            lineType=cv2.LINE_AA,
        )

    cv2.putText(
        rendered,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        REVIEW_TEXT_COLOR,
        2,
        lineType=cv2.LINE_AA,
    )
    return rendered
