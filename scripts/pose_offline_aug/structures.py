from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    v: int


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class ObjectAnnotation:
    class_id: int
    bbox: BBox
    keypoints: list[Keypoint]


@dataclass(frozen=True)
class ImageAnnotation:
    sample_id: str
    split: str
    image_path: Path
    label_path: Path
    image_width: int
    image_height: int
    object_annotation: ObjectAnnotation


@dataclass(frozen=True)
class OcclusionRegion:
    x1: float
    y1: float
    x2: float
    y2: float
    kind: str = "cutout"

    def contains(self, x_coord: float, y_coord: float) -> bool:
        return self.x1 <= x_coord <= self.x2 and self.y1 <= y_coord <= self.y2


@dataclass(frozen=True)
class SampleMetrics:
    visible_keypoints: int
    bbox_width_px: float
    bbox_height_px: float
    bbox_area_ratio: float
    out_of_frame_ratio: float
    bbox_max_side_input_px: float = 0.0


@dataclass
class AugmentRecord:
    source_split: str
    source_sample_id: str
    source_image: str
    source_label: str
    output_image: str
    output_label: str
    template: str
    slot_index: int
    attempt_index: int
    valid: bool
    reject_reason: str
    transforms: dict[str, Any]
    metrics: SampleMetrics | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_split": self.source_split,
            "source_sample_id": self.source_sample_id,
            "source_image": self.source_image,
            "source_label": self.source_label,
            "output_image": self.output_image,
            "output_label": self.output_label,
            "template": self.template,
            "slot_index": self.slot_index,
            "attempt_index": self.attempt_index,
            "valid": self.valid,
            "reject_reason": self.reject_reason,
            "transforms": self.transforms,
        }
        if self.metrics is not None:
            payload["metrics"] = {
                "visible_keypoints": self.metrics.visible_keypoints,
                "bbox_width_px": self.metrics.bbox_width_px,
                "bbox_height_px": self.metrics.bbox_height_px,
                "bbox_area_ratio": self.metrics.bbox_area_ratio,
                "out_of_frame_ratio": self.metrics.out_of_frame_ratio,
                "bbox_max_side_input_px": self.metrics.bbox_max_side_input_px,
            }
        return payload

    def to_csv_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "source_split": self.source_split,
            "source_sample_id": self.source_sample_id,
            "source_image": self.source_image,
            "source_label": self.source_label,
            "output_image": self.output_image,
            "output_label": self.output_label,
            "template": self.template,
            "slot_index": self.slot_index,
            "attempt_index": self.attempt_index,
            "valid": self.valid,
            "reject_reason": self.reject_reason,
            "transforms_json": json.dumps(self.transforms, sort_keys=True, ensure_ascii=False),
        }
        if self.metrics is not None:
            row.update(
                {
                    "visible_keypoints": self.metrics.visible_keypoints,
                    "bbox_width_px": self.metrics.bbox_width_px,
                    "bbox_height_px": self.metrics.bbox_height_px,
                    "bbox_area_ratio": self.metrics.bbox_area_ratio,
                    "out_of_frame_ratio": self.metrics.out_of_frame_ratio,
                    "bbox_max_side_input_px": self.metrics.bbox_max_side_input_px,
                }
            )
        return row
