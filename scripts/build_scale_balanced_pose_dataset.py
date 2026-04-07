#!/usr/bin/env python3
"""Build a scale-balanced YOLO pose dataset with offline geometric augmentation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
REVIEW_POINT_COLOR = (64, 224, 208)
REVIEW_BOX_COLOR = (48, 48, 48)
REVIEW_TEXT_COLOR = (255, 255, 255)


@dataclass(frozen=True)
class BucketSpec:
    name: str
    min_value: float
    max_value: float | None

    def contains(self, value: float) -> bool:
        upper_ok = self.max_value is None or value < self.max_value
        return value >= self.min_value and upper_ok


@dataclass
class PoseSample:
    sample_id: str
    split: str
    image_path: Path
    label_path: Path
    image_width: int
    image_height: int
    class_id: int
    bbox_norm: tuple[float, float, float, float]
    bbox_center_px: tuple[float, float]
    bbox_size_px: tuple[float, float]
    bbox_xyxy_px: tuple[float, float, float, float]
    keypoints_norm: list[tuple[float, float, int]]
    keypoints_px: list[tuple[float, float, int]]
    visible_count: int
    bbox_width_px_input: float
    bbox_height_px_input: float
    metric_px: float
    area_ratio: float
    kp_bbox_width_px: float
    kp_bbox_height_px: float
    mean_nn_px: float
    scale_to_input: float
    bucket: str


@dataclass
class AttemptRecord:
    sample_id: str
    split: str
    strategy: str
    source_image: str
    donor_image: str
    source_bucket: str
    target_bucket: str
    target_max_side_px: float
    actual_max_side_px: float | None
    visible_kpts: int | None
    mean_nn_px: float | None
    status: str
    reject_reason: str
    output_image: str
    output_label: str


@dataclass
class GeneratedSample:
    record: AttemptRecord
    image: np.ndarray
    label_text: str
    split: str
    target_bucket: str
    strategy: str
    source_sample_id: str
    output_image_path: Path
    output_label_path: Path
    image_width: int
    image_height: int
    bbox_center_px: tuple[float, float]
    bbox_size_px: tuple[float, float]
    keypoints_px: list[tuple[float, float, int]]


@dataclass(frozen=True)
class EligibleSource:
    sample: PoseSample
    min_target_px: float
    max_target_px: float

    def can_reach(self, target_metric: float) -> bool:
        return self.min_target_px <= target_metric <= self.max_target_px


def print_info(message: str) -> None:
    print(f"[build_scale_balanced_pose_dataset] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a scale-balanced YOLO pose dataset.")
    parser.add_argument(
        "--config",
        default="configs/dataset_scale_balance_v8_moderate.yaml",
        help="Path to the builder YAML config.",
    )
    parser.add_argument(
        "--src-root",
        help="Optional override for source.root.",
    )
    parser.add_argument(
        "--dst-root",
        help="Optional override for output.root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full planning/build loop without writing any files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional override for runtime.seed.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return payload


def sanitize_for_dump(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_for_dump(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_dump(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_dump(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = resolve_path(args.config, REPO_ROOT)
    config = load_yaml_file(config_path)
    required_top_level = [
        "source",
        "output",
        "buckets",
        "targets",
        "strategies",
        "quality",
        "backgrounds",
        "review",
        "runtime",
    ]
    missing = [key for key in required_top_level if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    source_cfg = config["source"]
    output_cfg = config["output"]
    runtime_cfg = config["runtime"]
    if not isinstance(source_cfg, dict) or not isinstance(output_cfg, dict) or not isinstance(runtime_cfg, dict):
        raise ValueError("'source', 'output', and 'runtime' must be mappings.")

    if args.src_root:
        source_cfg["root"] = args.src_root
    if args.dst_root:
        output_cfg["root"] = args.dst_root
    if args.seed is not None:
        runtime_cfg["seed"] = args.seed

    source_cfg["root"] = str(resolve_path(str(source_cfg["root"]), REPO_ROOT))
    output_cfg["root"] = str(resolve_path(str(output_cfg["root"]), REPO_ROOT))
    source_data_yaml = source_cfg.get("data_yaml") or (Path(source_cfg["root"]) / "data.yaml")
    source_cfg["data_yaml"] = str(resolve_path(str(source_data_yaml), REPO_ROOT))

    if Path(source_cfg["root"]).resolve() == Path(output_cfg["root"]).resolve():
        raise ValueError("Source and output roots must differ.")

    runtime_cfg.setdefault("seed", 52)
    runtime_cfg.setdefault("clean_output", True)
    runtime_cfg.setdefault("max_attempts_per_source", 3)
    runtime_cfg.setdefault("max_donor_attempts", 24)
    return config


def build_bucket_specs(config: dict[str, Any]) -> list[BucketSpec]:
    defs = config["buckets"].get("definitions")
    if not isinstance(defs, dict) or not defs:
        raise ValueError("buckets.definitions must be a non-empty mapping.")
    result: list[BucketSpec] = []
    for name, payload in defs.items():
        if not isinstance(payload, dict):
            raise ValueError(f"Bucket definition must be a mapping for {name}")
        min_value = float(payload["min"])
        max_value = payload.get("max")
        result.append(BucketSpec(name=name, min_value=min_value, max_value=None if max_value is None else float(max_value)))
    return result


def bucket_for_value(value: float, buckets: list[BucketSpec]) -> str:
    for bucket in buckets:
        if bucket.contains(value):
            return bucket.name
    raise ValueError(f"Value {value} did not match any bucket.")


def image_file_map(images_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            mapping[path.stem] = path.resolve()
    return mapping


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


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


def parse_pose_label(line: str, keypoint_count: int) -> tuple[int, tuple[float, float, float, float], list[tuple[float, float, int]]]:
    parts = line.split()
    expected = 5 + keypoint_count * 3
    if len(parts) != expected:
        raise ValueError(f"Expected {expected} values, found {len(parts)}.")
    class_id = int(float(parts[0]))
    bbox_norm = tuple(float(parts[index]) for index in range(1, 5))
    keypoints: list[tuple[float, float, int]] = []
    offset = 5
    for index in range(keypoint_count):
        x_coord = float(parts[offset + index * 3])
        y_coord = float(parts[offset + index * 3 + 1])
        visibility = int(round(float(parts[offset + index * 3 + 2])))
        keypoints.append((x_coord, y_coord, visibility))
    return class_id, bbox_norm, keypoints


def bbox_xyxy_from_norm(
    bbox_norm: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    center_x, center_y, width, height = bbox_norm
    box_width = width * image_width
    box_height = height * image_height
    x1 = center_x * image_width - box_width / 2.0
    y1 = center_y * image_height - box_height / 2.0
    x2 = x1 + box_width
    y2 = y1 + box_height
    return (x1, y1, x2, y2)


def bbox_center_size_from_xyxy(box: tuple[float, float, float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0), (width, height)


def compute_scaled_keypoint_stats(
    keypoints_px: list[tuple[float, float, int]],
    scale_to_input: float,
) -> tuple[int, float, float, float]:
    visible_points = [(x_coord * scale_to_input, y_coord * scale_to_input) for x_coord, y_coord, visibility in keypoints_px if visibility > 0]
    visible_count = len(visible_points)
    if not visible_points:
        return 0, 0.0, 0.0, 0.0

    xs = [point[0] for point in visible_points]
    ys = [point[1] for point in visible_points]
    kp_bbox_width = max(xs) - min(xs)
    kp_bbox_height = max(ys) - min(ys)
    if len(visible_points) == 1:
        return visible_count, kp_bbox_width, kp_bbox_height, 0.0

    nearest_distances = []
    for index, (x1, y1) in enumerate(visible_points):
        best = float("inf")
        for other_index, (x2, y2) in enumerate(visible_points):
            if index == other_index:
                continue
            distance = math.hypot(x1 - x2, y1 - y2)
            if distance < best:
                best = distance
        nearest_distances.append(best)
    mean_nn = sum(nearest_distances) / len(nearest_distances)
    return visible_count, kp_bbox_width, kp_bbox_height, mean_nn


def format_float(value: float) -> str:
    return f"{value:.10f}".rstrip("0").rstrip(".") or "0"


def build_label_text(
    class_id: int,
    image_width: int,
    image_height: int,
    bbox_center_px: tuple[float, float],
    bbox_size_px: tuple[float, float],
    keypoints_px: list[tuple[float, float, int]],
) -> str:
    center_x, center_y = bbox_center_px
    box_width, box_height = bbox_size_px
    values = [
        str(class_id),
        format_float(center_x / image_width),
        format_float(center_y / image_height),
        format_float(box_width / image_width),
        format_float(box_height / image_height),
    ]
    for x_coord, y_coord, visibility in keypoints_px:
        values.append(format_float(x_coord / image_width))
        values.append(format_float(y_coord / image_height))
        values.append(str(int(visibility)))
    return " ".join(values) + "\n"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clip_rect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    clipped_x1 = int(math.floor(clamp(x1, 0, image_width - 1)))
    clipped_y1 = int(math.floor(clamp(y1, 0, image_height - 1)))
    clipped_x2 = int(math.ceil(clamp(x2, 1, image_width)))
    clipped_y2 = int(math.ceil(clamp(y2, 1, image_height)))
    if clipped_x2 <= clipped_x1:
        clipped_x2 = min(image_width, clipped_x1 + 1)
    if clipped_y2 <= clipped_y1:
        clipped_y2 = min(image_height, clipped_y1 + 1)
    return clipped_x1, clipped_y1, clipped_x2, clipped_y2


def inscribed_rect(region_width: int, region_height: int, aspect_ratio: float) -> tuple[int, int]:
    if region_width <= 0 or region_height <= 0:
        return (0, 0)
    width = region_width
    height = int(round(width / aspect_ratio))
    if height > region_height:
        height = region_height
        width = int(round(height * aspect_ratio))
    return max(0, width), max(0, height)


def build_feather_mask(height: int, width: int, feather_px: int) -> np.ndarray:
    if feather_px <= 0:
        return np.ones((height, width, 1), dtype=np.float32)
    effective = max(1, min(feather_px, width // 2 if width > 1 else 1, height // 2 if height > 1 else 1))
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    x_dist = np.minimum(x, (width - 1) - x)
    y_dist = np.minimum(y, (height - 1) - y)

    def alpha_curve(distance: np.ndarray) -> np.ndarray:
        inside = np.clip(distance / float(effective), 0.0, 1.0)
        return 0.5 - 0.5 * np.cos(np.pi * inside)

    alpha_x = alpha_curve(x_dist)
    alpha_y = alpha_curve(y_dist)
    mask = np.outer(alpha_y, alpha_x).astype(np.float32)
    return mask[..., None]


def transform_points(matrix: np.ndarray, points: list[tuple[float, float, int]]) -> list[tuple[float, float, int]]:
    transformed: list[tuple[float, float, int]] = []
    for x_coord, y_coord, visibility in points:
        new_x = float(matrix[0, 0] * x_coord + matrix[0, 1] * y_coord + matrix[0, 2])
        new_y = float(matrix[1, 0] * x_coord + matrix[1, 1] * y_coord + matrix[1, 2])
        transformed.append((new_x, new_y, visibility))
    return transformed


def scale_factor_for_target(sample: PoseSample, target_metric_px: float) -> float:
    return target_metric_px / sample.metric_px


def compute_target_interval(
    sample: PoseSample,
    strategy: str,
    target_range: tuple[float, float],
    min_mean_nn_px: float,
    min_bbox_px: float,
) -> tuple[float, float] | None:
    lower, upper = target_range
    if strategy in {"A", "B"}:
        upper = min(upper, sample.metric_px)
    if strategy == "C":
        lower = max(lower, sample.metric_px + 1e-6)

    lower = max(lower, min_bbox_px)
    if sample.mean_nn_px <= 0.0:
        return None
    lower = max(lower, min_mean_nn_px * sample.metric_px / sample.mean_nn_px)
    if lower > upper:
        return None
    return (lower, upper)


def candidate_target_metric(
    sample: PoseSample,
    strategy: str,
    target_range: tuple[float, float],
    quality_cfg: dict[str, Any],
    rng: random.Random,
) -> float | None:
    interval = compute_target_interval(
        sample=sample,
        strategy=strategy,
        target_range=target_range,
        min_mean_nn_px=float(quality_cfg["min_mean_nn_px"]),
        min_bbox_px=float(quality_cfg["min_bbox_max_side_px"]),
    )
    if interval is None:
        return None
    lower, upper = interval
    if math.isclose(lower, upper):
        return lower
    return rng.uniform(lower, upper)


def evenly_spaced_targets(
    target_range: tuple[float, float],
    count: int,
    rng: random.Random,
) -> list[float]:
    if count <= 0:
        return []
    lower, upper = target_range
    if count == 1 or math.isclose(lower, upper):
        return [lower if math.isclose(lower, upper) else rng.uniform(lower, upper)]

    targets: list[float] = []
    width = upper - lower
    for index in range(count):
        slot_low = lower + width * index / count
        slot_high = lower + width * (index + 1) / count
        targets.append(rng.uniform(slot_low, slot_high))
    targets.sort()
    return targets


def validate_geometry(
    *,
    image_width: int,
    image_height: int,
    bbox_center_px: tuple[float, float],
    bbox_size_px: tuple[float, float],
    keypoints_px: list[tuple[float, float, int]],
    source_visible_count: int,
    target_bucket: str,
    buckets: list[BucketSpec],
    imgsz: int,
    quality_cfg: dict[str, Any],
) -> tuple[str, float, int, float]:
    center_x, center_y = bbox_center_px
    box_width, box_height = bbox_size_px
    if box_width <= 0 or box_height <= 0:
        raise ValueError("bbox_non_positive")

    x1 = center_x - box_width / 2.0
    y1 = center_y - box_height / 2.0
    x2 = x1 + box_width
    y2 = y1 + box_height
    if bool(quality_cfg.get("bbox_center_in_frame", True)):
        if not (0.0 <= center_x < image_width and 0.0 <= center_y < image_height):
            raise ValueError("bbox_center_out_of_frame")
    if x1 < 0.0 or y1 < 0.0 or x2 > image_width or y2 > image_height:
        raise ValueError("bbox_out_of_frame")

    visible_points = []
    visible_count = 0
    for x_coord, y_coord, visibility in keypoints_px:
        if visibility <= 0:
            continue
        if not (0.0 <= x_coord < image_width and 0.0 <= y_coord < image_height):
            raise ValueError("visible_keypoint_out_of_frame")
        visible_count += 1
        visible_points.append((x_coord, y_coord, visibility))

    if bool(quality_cfg.get("preserve_visible_count", True)) and visible_count != source_visible_count:
        raise ValueError("visible_count_changed")
    if visible_count < int(quality_cfg["min_visible_keypoints"]):
        raise ValueError("visible_keypoints_below_threshold")

    scale_to_input = min(float(imgsz) / image_width, float(imgsz) / image_height)
    actual_metric = max(box_width * scale_to_input, box_height * scale_to_input)
    if actual_metric < float(quality_cfg["min_bbox_max_side_px"]):
        raise ValueError("bbox_too_small")
    actual_bucket = bucket_for_value(actual_metric, buckets)
    if bool(quality_cfg.get("require_target_bucket_match", True)) and actual_bucket != target_bucket:
        raise ValueError(f"bucket_mismatch:{actual_bucket}")

    mean_nn_px = compute_scaled_keypoint_stats(visible_points, scale_to_input)[3]
    if mean_nn_px < float(quality_cfg["min_mean_nn_px"]):
        raise ValueError("mean_nn_below_threshold")
    return actual_bucket, actual_metric, visible_count, mean_nn_px


def review_overlay(
    image: np.ndarray,
    bbox_center_px: tuple[float, float],
    bbox_size_px: tuple[float, float],
    keypoints_px: list[tuple[float, float, int]],
    title: str,
) -> np.ndarray:
    rendered = image.copy()
    center_x, center_y = bbox_center_px
    box_width, box_height = bbox_size_px
    x1 = int(round(center_x - box_width / 2.0))
    y1 = int(round(center_y - box_height / 2.0))
    x2 = int(round(center_x + box_width / 2.0))
    y2 = int(round(center_y + box_height / 2.0))
    cv2.rectangle(rendered, (x1, y1), (x2, y2), REVIEW_BOX_COLOR, 2, lineType=cv2.LINE_AA)
    for index, (x_coord, y_coord, visibility) in enumerate(keypoints_px):
        if visibility <= 0:
            continue
        point = (int(round(x_coord)), int(round(y_coord)))
        cv2.circle(rendered, point, 3, REVIEW_POINT_COLOR, thickness=-1, lineType=cv2.LINE_AA)
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


class DatasetBuilder:
    def __init__(self, config: dict[str, Any], dry_run: bool) -> None:
        self.config = config
        self.dry_run = dry_run
        self.source_root = Path(config["source"]["root"]).resolve()
        self.output_root = Path(config["output"]["root"]).resolve()
        self.source_data_yaml = Path(config["source"]["data_yaml"]).resolve()
        self.imgsz = int(config["source"]["imgsz"])
        self.buckets = build_bucket_specs(config)
        self.target_ranges = {
            name: (float(payload["min"]), float(payload["max"]))
            for name, payload in config["buckets"]["target_ranges"].items()
        }
        self.rng = random.Random(int(config["runtime"]["seed"]))
        self.base_data_config = load_yaml_file(self.source_data_yaml)
        self.keypoint_count = int(self.base_data_config["kpt_shape"][0])
        self.samples_by_split: dict[str, list[PoseSample]] = {}
        self.raw_lists: dict[str, list[Path]] = {}
        self.generated_attempts: list[AttemptRecord] = []
        self.generated_samples: list[GeneratedSample] = []
        self.review_rows: list[dict[str, Any]] = []
        self.usage_total: Counter[str] = Counter()
        self.usage_bucket: Counter[tuple[str, str]] = Counter()
        self.group_attempt_counter: Counter[str] = Counter()
        self.localized_raw_lists: dict[str, list[Path]] = {}

    def prepare_output_root(self) -> None:
        if self.dry_run:
            return
        if self.output_root.exists() and bool(self.config["runtime"].get("clean_output", True)):
            shutil.rmtree(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def split_dirs(self, split: str) -> tuple[Path, Path]:
        images_dir = self.output_root / split / "images"
        labels_dir = self.output_root / split / "labels"
        return images_dir, labels_dir

    def localize_raw_split(self, source_split: str, target_split: str) -> list[Path]:
        images_dir, labels_dir = self.split_dirs(target_split)
        localized_images: list[Path] = []
        for sample in self.samples_by_split[source_split]:
            target_image = images_dir / sample.image_path.name
            target_label = labels_dir / sample.label_path.name
            if not self.dry_run:
                images_dir.mkdir(parents=True, exist_ok=True)
                labels_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sample.image_path, target_image)
                shutil.copy2(sample.label_path, target_label)
            localized_images.append(target_image.resolve())
        self.localized_raw_lists[target_split] = localized_images
        return localized_images

    def localize_raw_splits(self) -> None:
        self.localize_raw_split("train", "train")
        self.localize_raw_split("valid", "valid")
        self.localize_raw_split("valid", "valid_raw")
        self.localize_raw_split("test", "test")

    def scan_source_dataset(self) -> None:
        print_info(f"Scanning source dataset: {self.source_root}")
        for split in ("train", "valid", "test"):
            images_dir = self.source_root / split / "images"
            labels_dir = self.source_root / split / "labels"
            if not images_dir.exists() or not labels_dir.exists():
                raise FileNotFoundError(f"Missing split directories for '{split}': {images_dir} / {labels_dir}")

            images = image_file_map(images_dir)
            labels = {path.stem: path.resolve() for path in sorted(labels_dir.glob("*.txt"))}
            if set(images) != set(labels):
                missing_images = sorted(set(labels) - set(images))
                missing_labels = sorted(set(images) - set(labels))
                raise ValueError(
                    f"Image/label mismatch for split={split}. Missing images={missing_images[:5]}, missing labels={missing_labels[:5]}"
                )

            split_samples: list[PoseSample] = []
            raw_paths: list[Path] = []
            for stem in sorted(labels):
                image_path = images[stem]
                label_path = labels[stem]
                lines = [line for line in read_text(label_path).splitlines() if line.strip()]
                if len(lines) != 1:
                    raise ValueError(f"Expected exactly one instance per label file, found {len(lines)} in {label_path}")
                image = read_image_bgr(image_path)
                image_height, image_width = image.shape[:2]
                class_id, bbox_norm, keypoints_norm = parse_pose_label(lines[0], self.keypoint_count)
                bbox_xyxy_px = bbox_xyxy_from_norm(bbox_norm, image_width=image_width, image_height=image_height)
                bbox_center_px, bbox_size_px = bbox_center_size_from_xyxy(bbox_xyxy_px)
                keypoints_px = [
                    (x_coord * image_width, y_coord * image_height, visibility)
                    for x_coord, y_coord, visibility in keypoints_norm
                ]
                scale_to_input = min(float(self.imgsz) / image_width, float(self.imgsz) / image_height)
                visible_count, kp_bbox_width_px, kp_bbox_height_px, mean_nn_px = compute_scaled_keypoint_stats(
                    keypoints_px,
                    scale_to_input,
                )
                bbox_width_input = bbox_size_px[0] * scale_to_input
                bbox_height_input = bbox_size_px[1] * scale_to_input
                metric_px = max(bbox_width_input, bbox_height_input)
                bucket = bucket_for_value(metric_px, self.buckets)
                split_samples.append(
                    PoseSample(
                        sample_id=stem,
                        split=split,
                        image_path=image_path,
                        label_path=label_path,
                        image_width=image_width,
                        image_height=image_height,
                        class_id=class_id,
                        bbox_norm=bbox_norm,
                        bbox_center_px=bbox_center_px,
                        bbox_size_px=bbox_size_px,
                        bbox_xyxy_px=bbox_xyxy_px,
                        keypoints_norm=keypoints_norm,
                        keypoints_px=keypoints_px,
                        visible_count=visible_count,
                        bbox_width_px_input=bbox_width_input,
                        bbox_height_px_input=bbox_height_input,
                        metric_px=metric_px,
                        area_ratio=float(bbox_norm[2] * bbox_norm[3]),
                        kp_bbox_width_px=kp_bbox_width_px,
                        kp_bbox_height_px=kp_bbox_height_px,
                        mean_nn_px=mean_nn_px,
                        scale_to_input=scale_to_input,
                        bucket=bucket,
                    )
                )
                raw_paths.append(image_path)
            self.samples_by_split[split] = split_samples
            self.raw_lists[split] = raw_paths

    def source_stats_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for split, samples in self.samples_by_split.items():
            for sample in samples:
                rows.append(
                    {
                        "split": split,
                        "sample_id": sample.sample_id,
                        "image_path": str(sample.image_path),
                        "label_path": str(sample.label_path),
                        "image_width": sample.image_width,
                        "image_height": sample.image_height,
                        "class_id": sample.class_id,
                        "bbox_width_px_raw": sample.bbox_size_px[0],
                        "bbox_height_px_raw": sample.bbox_size_px[1],
                        "bbox_width_px_input": sample.bbox_width_px_input,
                        "bbox_height_px_input": sample.bbox_height_px_input,
                        "bbox_metric_px_input": sample.metric_px,
                        "bbox_area_ratio": sample.area_ratio,
                        "visible_keypoints": sample.visible_count,
                        "kp_bbox_width_px_input": sample.kp_bbox_width_px,
                        "kp_bbox_height_px_input": sample.kp_bbox_height_px,
                        "mean_nn_px_input": sample.mean_nn_px,
                        "bucket": sample.bucket,
                    }
                )
        return rows

    def raw_bucket_counts(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for split, samples in self.samples_by_split.items():
            counter = Counter(sample.bucket for sample in samples)
            counts[split] = {bucket.name: int(counter.get(bucket.name, 0)) for bucket in self.buckets}
        return counts

    def target_total_counts(self) -> dict[str, dict[str, int]]:
        raw_counts = self.raw_bucket_counts()
        totals: dict[str, dict[str, int]] = {}
        for split, bucket_targets in self.config["targets"].items():
            current = raw_counts.get(split, {})
            totals[split] = {}
            for bucket_name in [bucket.name for bucket in self.buckets]:
                target_value = bucket_targets.get(bucket_name, "keep")
                if str(target_value).lower() == "keep":
                    totals[split][bucket_name] = int(current.get(bucket_name, 0))
                else:
                    totals[split][bucket_name] = int(target_value)
        return totals

    def deficits(self) -> dict[str, dict[str, int]]:
        raw_counts = self.raw_bucket_counts()
        target_counts = self.target_total_counts()
        deficits: dict[str, dict[str, int]] = {}
        for split, bucket_targets in target_counts.items():
            deficits[split] = {}
            for bucket_name, total in bucket_targets.items():
                raw_count = raw_counts.get(split, {}).get(bucket_name, 0)
                deficit = int(total) - int(raw_count)
                if deficit < 0:
                    raise ValueError(f"Target for split={split} bucket={bucket_name} is below raw count.")
                deficits[split][bucket_name] = deficit
        return deficits

    def strategy_for(self, split: str, bucket_name: str) -> str:
        mapping = self.config["strategies"]["split_bucket_map"]
        return str(mapping.get(split, {}).get(bucket_name, "none")).upper()

    def reuse_limits(self, split: str) -> dict[str, int]:
        limits = self.config["backgrounds"]["source_reuse_limits"][split]
        return {
            "max_total_derivatives": int(limits["max_total_derivatives"]),
            "per_bucket_max": int(limits["per_bucket_max"]),
        }

    def feasible_interval_for(
        self,
        sample: PoseSample,
        target_bucket: str,
        strategy: str,
    ) -> tuple[float, float] | None:
        return compute_target_interval(
            sample=sample,
            strategy=strategy,
            target_range=self.target_ranges[target_bucket],
            min_mean_nn_px=float(self.config["quality"]["min_mean_nn_px"]),
            min_bbox_px=float(self.config["quality"]["min_bbox_max_side_px"]),
        )

    def eligible_sources(self, split: str, target_bucket: str, strategy: str) -> list[EligibleSource]:
        target_range = self.target_ranges[target_bucket]
        samples = list(self.samples_by_split[split])
        shuffled = list(samples)
        self.rng.shuffle(shuffled)

        def rank(sample: PoseSample) -> tuple[int, int, float]:
            same_bucket = 1 if sample.bucket == target_bucket else 0
            below_target_hi = 1 if sample.metric_px < target_range[1] else 0
            return (same_bucket, below_target_hi, -sample.metric_px)

        ordered = sorted(shuffled, key=rank)
        eligible: list[EligibleSource] = []
        for sample in ordered:
            interval = self.feasible_interval_for(sample, target_bucket=target_bucket, strategy=strategy)
            if interval is None:
                continue
            eligible.append(EligibleSource(sample=sample, min_target_px=interval[0], max_target_px=interval[1]))
        return eligible

    def output_paths(self, split: str, sample_id: str) -> tuple[Path, Path]:
        images_dir, labels_dir = self.split_dirs(split)
        image_path = images_dir / f"{sample_id}.jpg"
        label_path = labels_dir / f"{sample_id}.txt"
        return image_path, label_path

    def build_group_sample_id(self, split: str, target_bucket: str, strategy: str) -> str:
        group = f"{split}_{target_bucket}_{strategy}"
        self.group_attempt_counter[group] += 1
        return f"{group}_{self.group_attempt_counter[group]:05d}"

    def can_use_source(self, sample: PoseSample, target_bucket: str) -> bool:
        limits = self.reuse_limits(sample.split)
        if self.usage_total[sample.sample_id] >= limits["max_total_derivatives"]:
            return False
        if self.usage_bucket[(sample.sample_id, target_bucket)] >= limits["per_bucket_max"]:
            return False
        return True

    def select_background_patch(self, donor_sample: PoseSample, output_width: int, output_height: int) -> np.ndarray | None:
        donor_image = read_image_bgr(donor_sample.image_path)
        donor_height, donor_width = donor_image.shape[:2]
        aspect_ratio = output_width / float(output_height)
        margin_ratio = float(self.config["backgrounds"]["donor_exclusion_margin_ratio"])
        margin = max(donor_sample.bbox_size_px) * margin_ratio
        x1, y1, x2, y2 = donor_sample.bbox_xyxy_px
        fx1 = int(math.floor(clamp(x1 - margin, 0.0, donor_width)))
        fy1 = int(math.floor(clamp(y1 - margin, 0.0, donor_height)))
        fx2 = int(math.ceil(clamp(x2 + margin, 0.0, donor_width)))
        fy2 = int(math.ceil(clamp(y2 + margin, 0.0, donor_height)))

        regions = [
            (0, 0, donor_width, fy1),
            (0, fy2, donor_width, donor_height),
            (0, 0, fx1, donor_height),
            (fx2, 0, donor_width, donor_height),
        ]
        candidates: list[tuple[int, tuple[int, int, int, int]]] = []
        for rx1, ry1, rx2, ry2 in regions:
            region_width = rx2 - rx1
            region_height = ry2 - ry1
            crop_width, crop_height = inscribed_rect(region_width, region_height, aspect_ratio)
            if crop_width < 32 or crop_height < 32:
                continue
            max_x1 = rx2 - crop_width
            max_y1 = ry2 - crop_height
            crop_x1 = rx1 if max_x1 <= rx1 else self.rng.randint(rx1, max_x1)
            crop_y1 = ry1 if max_y1 <= ry1 else self.rng.randint(ry1, max_y1)
            candidates.append((crop_width * crop_height, (crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height)))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        top = candidates[: min(4, len(candidates))]
        _, rect = self.rng.choice(top)
        crop_x1, crop_y1, crop_x2, crop_y2 = rect
        crop = donor_image[crop_y1:crop_y2, crop_x1:crop_x2]
        return cv2.resize(crop, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

    def generate_with_strategy_a(
        self,
        sample: PoseSample,
        split: str,
        target_bucket: str,
        sample_id: str,
        target_metric: float | None = None,
    ) -> GeneratedSample:
        target_metric = target_metric if target_metric is not None else candidate_target_metric(
            sample,
            "A",
            self.target_ranges[target_bucket],
            self.config["quality"],
            self.rng,
        )
        if target_metric is None:
            raise ValueError("infeasible_target_interval")
        scale = scale_factor_for_target(sample, target_metric)

        image = read_image_bgr(sample.image_path)
        jitter_ratio = float(self.config["strategies"]["A"][f"{split}_jitter_ratio"])
        max_dx = jitter_ratio * sample.image_width
        max_dy = jitter_ratio * sample.image_height
        offset_x = self.rng.uniform(-max_dx, max_dx) if max_dx > 0 else 0.0
        offset_y = self.rng.uniform(-max_dy, max_dy) if max_dy > 0 else 0.0

        source_center = sample.bbox_center_px
        destination_center = (source_center[0] + offset_x, source_center[1] + offset_y)
        matrix = np.array(
            [
                [scale, 0.0, destination_center[0] - scale * source_center[0]],
                [0.0, scale, destination_center[1] - scale * source_center[1]],
            ],
            dtype=np.float32,
        )
        transformed_image = cv2.warpAffine(
            image,
            matrix,
            (sample.image_width, sample.image_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        transformed_keypoints = transform_points(matrix, sample.keypoints_px)
        transformed_center = (
            float(matrix[0, 0] * source_center[0] + matrix[0, 2]),
            float(matrix[1, 1] * source_center[1] + matrix[1, 2]),
        )
        transformed_size = (sample.bbox_size_px[0] * scale, sample.bbox_size_px[1] * scale)
        actual_bucket, actual_metric, visible_count, mean_nn_px = validate_geometry(
            image_width=sample.image_width,
            image_height=sample.image_height,
            bbox_center_px=transformed_center,
            bbox_size_px=transformed_size,
            keypoints_px=transformed_keypoints,
            source_visible_count=sample.visible_count,
            target_bucket=target_bucket,
            buckets=self.buckets,
            imgsz=self.imgsz,
            quality_cfg=self.config["quality"],
        )
        label_text = build_label_text(
            class_id=sample.class_id,
            image_width=sample.image_width,
            image_height=sample.image_height,
            bbox_center_px=transformed_center,
            bbox_size_px=transformed_size,
            keypoints_px=transformed_keypoints,
        )
        image_path, label_path = self.output_paths(split, sample_id)
        record = AttemptRecord(
            sample_id=sample_id,
            split=split,
            strategy="A",
            source_image=str(sample.image_path),
            donor_image="",
            source_bucket=sample.bucket,
            target_bucket=target_bucket,
            target_max_side_px=float(target_metric),
            actual_max_side_px=float(actual_metric),
            visible_kpts=visible_count,
            mean_nn_px=float(mean_nn_px),
            status="accepted",
            reject_reason="",
            output_image=str(image_path),
            output_label=str(label_path),
        )
        return GeneratedSample(
            record=record,
            image=transformed_image,
            label_text=label_text,
            split=split,
            target_bucket=actual_bucket,
            strategy="A",
            source_sample_id=sample.sample_id,
            output_image_path=image_path,
            output_label_path=label_path,
            image_width=sample.image_width,
            image_height=sample.image_height,
            bbox_center_px=transformed_center,
            bbox_size_px=transformed_size,
            keypoints_px=transformed_keypoints,
        )

    def generate_with_strategy_c(
        self,
        sample: PoseSample,
        split: str,
        target_bucket: str,
        sample_id: str,
        target_metric: float | None = None,
    ) -> GeneratedSample:
        target_metric = target_metric if target_metric is not None else candidate_target_metric(
            sample,
            "C",
            self.target_ranges[target_bucket],
            self.config["quality"],
            self.rng,
        )
        if target_metric is None:
            raise ValueError("infeasible_target_interval")
        if target_metric <= sample.metric_px:
            raise ValueError("strategy_c_requires_upscale")
        scale = scale_factor_for_target(sample, target_metric)
        image = read_image_bgr(sample.image_path)
        source_center = sample.bbox_center_px
        matrix = np.array(
            [
                [scale, 0.0, source_center[0] - scale * source_center[0]],
                [0.0, scale, source_center[1] - scale * source_center[1]],
            ],
            dtype=np.float32,
        )
        transformed_image = cv2.warpAffine(
            image,
            matrix,
            (sample.image_width, sample.image_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        transformed_keypoints = transform_points(matrix, sample.keypoints_px)
        transformed_center = sample.bbox_center_px
        transformed_size = (sample.bbox_size_px[0] * scale, sample.bbox_size_px[1] * scale)
        actual_bucket, actual_metric, visible_count, mean_nn_px = validate_geometry(
            image_width=sample.image_width,
            image_height=sample.image_height,
            bbox_center_px=transformed_center,
            bbox_size_px=transformed_size,
            keypoints_px=transformed_keypoints,
            source_visible_count=sample.visible_count,
            target_bucket=target_bucket,
            buckets=self.buckets,
            imgsz=self.imgsz,
            quality_cfg=self.config["quality"],
        )
        label_text = build_label_text(
            class_id=sample.class_id,
            image_width=sample.image_width,
            image_height=sample.image_height,
            bbox_center_px=transformed_center,
            bbox_size_px=transformed_size,
            keypoints_px=transformed_keypoints,
        )
        image_path, label_path = self.output_paths(split, sample_id)
        record = AttemptRecord(
            sample_id=sample_id,
            split=split,
            strategy="C",
            source_image=str(sample.image_path),
            donor_image="",
            source_bucket=sample.bucket,
            target_bucket=target_bucket,
            target_max_side_px=float(target_metric),
            actual_max_side_px=float(actual_metric),
            visible_kpts=visible_count,
            mean_nn_px=float(mean_nn_px),
            status="accepted",
            reject_reason="",
            output_image=str(image_path),
            output_label=str(label_path),
        )
        return GeneratedSample(
            record=record,
            image=transformed_image,
            label_text=label_text,
            split=split,
            target_bucket=actual_bucket,
            strategy="C",
            source_sample_id=sample.sample_id,
            output_image_path=image_path,
            output_label_path=label_path,
            image_width=sample.image_width,
            image_height=sample.image_height,
            bbox_center_px=transformed_center,
            bbox_size_px=transformed_size,
            keypoints_px=transformed_keypoints,
        )

    def generate_with_strategy_b(
        self,
        sample: PoseSample,
        split: str,
        target_bucket: str,
        sample_id: str,
        target_metric: float | None = None,
    ) -> GeneratedSample:
        target_metric = target_metric if target_metric is not None else candidate_target_metric(
            sample,
            "B",
            self.target_ranges[target_bucket],
            self.config["quality"],
            self.rng,
        )
        if target_metric is None:
            raise ValueError("infeasible_target_interval")
        scale = scale_factor_for_target(sample, target_metric)
        source_image = read_image_bgr(sample.image_path)
        output_height, output_width = source_image.shape[:2]

        context_lo, context_hi = self.config["strategies"]["B"]["context_scale_range"]
        context_scale = self.rng.uniform(float(context_lo), float(context_hi))
        crop_width = sample.bbox_size_px[0] * context_scale
        crop_height = sample.bbox_size_px[1] * context_scale
        crop_x1, crop_y1, crop_x2, crop_y2 = clip_rect(
            sample.bbox_center_px[0] - crop_width / 2.0,
            sample.bbox_center_px[1] - crop_height / 2.0,
            sample.bbox_center_px[0] + crop_width / 2.0,
            sample.bbox_center_px[1] + crop_height / 2.0,
            output_width,
            output_height,
        )
        roi = source_image[crop_y1:crop_y2, crop_x1:crop_x2]
        roi_height, roi_width = roi.shape[:2]
        resized_width = max(2, int(round(roi_width * scale)))
        resized_height = max(2, int(round(roi_height * scale)))
        interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LINEAR
        resized_roi = cv2.resize(roi, (resized_width, resized_height), interpolation=interpolation)

        donor_pool = list(self.samples_by_split[str(self.config["strategies"]["B"]["donor_split"])])
        self.rng.shuffle(donor_pool)
        background: np.ndarray | None = None
        donor_path = ""
        max_donor_attempts = int(self.config["runtime"]["max_donor_attempts"])
        for donor_sample in donor_pool[:max_donor_attempts]:
            background = self.select_background_patch(donor_sample, output_width=output_width, output_height=output_height)
            if background is not None:
                donor_path = str(donor_sample.image_path)
                break
        if background is None:
            raise ValueError("no_valid_background_patch")

        if resized_width >= output_width or resized_height >= output_height:
            raise ValueError("resized_roi_does_not_fit_canvas")
        paste_x = self.rng.randint(0, output_width - resized_width)
        paste_y = self.rng.randint(0, output_height - resized_height)

        feather_px = int(self.config["strategies"]["B"]["feather_px"])
        alpha = build_feather_mask(resized_height, resized_width, feather_px)
        canvas = background.astype(np.float32)
        patch = resized_roi.astype(np.float32)
        view = canvas[paste_y : paste_y + resized_height, paste_x : paste_x + resized_width]
        view[:] = patch * alpha + view * (1.0 - alpha)
        transformed_image = np.clip(canvas, 0, 255).astype(np.uint8)

        local_bbox_center = (
            sample.bbox_center_px[0] - crop_x1,
            sample.bbox_center_px[1] - crop_y1,
        )
        transformed_center = (
            local_bbox_center[0] * scale + paste_x,
            local_bbox_center[1] * scale + paste_y,
        )
        transformed_size = (sample.bbox_size_px[0] * scale, sample.bbox_size_px[1] * scale)
        transformed_keypoints = [
            ((x_coord - crop_x1) * scale + paste_x, (y_coord - crop_y1) * scale + paste_y, visibility)
            for x_coord, y_coord, visibility in sample.keypoints_px
        ]

        actual_bucket, actual_metric, visible_count, mean_nn_px = validate_geometry(
            image_width=output_width,
            image_height=output_height,
            bbox_center_px=transformed_center,
            bbox_size_px=transformed_size,
            keypoints_px=transformed_keypoints,
            source_visible_count=sample.visible_count,
            target_bucket=target_bucket,
            buckets=self.buckets,
            imgsz=self.imgsz,
            quality_cfg=self.config["quality"],
        )
        label_text = build_label_text(
            class_id=sample.class_id,
            image_width=output_width,
            image_height=output_height,
            bbox_center_px=transformed_center,
            bbox_size_px=transformed_size,
            keypoints_px=transformed_keypoints,
        )
        image_path, label_path = self.output_paths(split, sample_id)
        record = AttemptRecord(
            sample_id=sample_id,
            split=split,
            strategy="B",
            source_image=str(sample.image_path),
            donor_image=donor_path,
            source_bucket=sample.bucket,
            target_bucket=target_bucket,
            target_max_side_px=float(target_metric),
            actual_max_side_px=float(actual_metric),
            visible_kpts=visible_count,
            mean_nn_px=float(mean_nn_px),
            status="accepted",
            reject_reason="",
            output_image=str(image_path),
            output_label=str(label_path),
        )
        return GeneratedSample(
            record=record,
            image=transformed_image,
            label_text=label_text,
            split=split,
            target_bucket=actual_bucket,
            strategy="B",
            source_sample_id=sample.sample_id,
            output_image_path=image_path,
            output_label_path=label_path,
            image_width=output_width,
            image_height=output_height,
            bbox_center_px=transformed_center,
            bbox_size_px=transformed_size,
            keypoints_px=transformed_keypoints,
        )

    def try_generate(
        self,
        sample: PoseSample,
        split: str,
        target_bucket: str,
        strategy: str,
        sample_id: str,
        target_metric: float,
    ) -> GeneratedSample:
        if strategy == "A":
            return self.generate_with_strategy_a(sample, split, target_bucket, sample_id, target_metric=target_metric)
        if strategy == "B":
            return self.generate_with_strategy_b(sample, split, target_bucket, sample_id, target_metric=target_metric)
        if strategy == "C":
            return self.generate_with_strategy_c(sample, split, target_bucket, sample_id, target_metric=target_metric)
        raise ValueError(f"Unsupported strategy: {strategy}")

    def accept_generated_sample(self, generated: GeneratedSample) -> None:
        if not self.dry_run:
            write_image_bgr(generated.output_image_path, generated.image)
            generated.output_label_path.parent.mkdir(parents=True, exist_ok=True)
            generated.output_label_path.write_text(generated.label_text, encoding="utf-8")
        generated.image = np.zeros((1, 1, 3), dtype=np.uint8)
        generated.label_text = ""
        self.generated_samples.append(generated)
        self.generated_attempts.append(generated.record)
        self.usage_total[generated.source_sample_id] += 1
        self.usage_bucket[(generated.source_sample_id, generated.target_bucket)] += 1

    def reject_record(
        self,
        sample_id: str,
        sample: PoseSample,
        split: str,
        strategy: str,
        target_bucket: str,
        target_metric: float,
        donor_image: str,
        reason: str,
    ) -> None:
        image_path, label_path = self.output_paths(split, sample_id)
        self.generated_attempts.append(
            AttemptRecord(
                sample_id=sample_id,
                split=split,
                strategy=strategy,
                source_image=str(sample.image_path),
                donor_image=donor_image,
                source_bucket=sample.bucket,
                target_bucket=target_bucket,
                target_max_side_px=float(target_metric),
                actual_max_side_px=None,
                visible_kpts=None,
                mean_nn_px=None,
                status="rejected",
                reject_reason=reason,
                output_image=str(image_path),
                output_label=str(label_path),
            )
        )

    def build_generated_split(self, split: str, deficits: dict[str, int]) -> None:
        max_attempts_per_source = int(self.config["runtime"]["max_attempts_per_source"])
        for target_bucket in [bucket.name for bucket in self.buckets]:
            deficit = int(deficits.get(target_bucket, 0))
            if deficit <= 0:
                continue
            strategy = self.strategy_for(split, target_bucket)
            if strategy == "NONE":
                if deficit != 0:
                    raise ValueError(f"Target deficit remains for split={split}, bucket={target_bucket}, but strategy is none.")
                continue
            eligible = self.eligible_sources(split, target_bucket, strategy)
            planned_targets = evenly_spaced_targets(self.target_ranges[target_bucket], deficit, self.rng)
            print_info(
                f"split={split} bucket={target_bucket} strategy={strategy} deficit={deficit} eligible_sources={len(eligible)}"
            )
            accepted = 0
            for target_metric in planned_targets:
                candidates = [
                    entry
                    for entry in eligible
                    if entry.can_reach(target_metric) and self.can_use_source(entry.sample, target_bucket)
                ]
                candidates.sort(
                    key=lambda entry: (
                        entry.max_target_px,
                        entry.min_target_px,
                        self.usage_total[entry.sample.sample_id],
                        entry.sample.metric_px,
                    )
                )
                success = False
                for entry in candidates:
                    for _ in range(max_attempts_per_source):
                        sample_id = self.build_group_sample_id(split, target_bucket, strategy)
                        sample = entry.sample
                        output_image_path, output_label_path = self.output_paths(split, sample_id)
                        donor_image = ""
                        if strategy == "B":
                            donor_image = "dynamic_train_donor"
                        try:
                            generated = self.try_generate(
                                sample,
                                split,
                                target_bucket,
                                strategy,
                                sample_id,
                                target_metric=target_metric,
                            )
                        except Exception as exc:
                            self.generated_attempts.append(
                                AttemptRecord(
                                    sample_id=sample_id,
                                    split=split,
                                    strategy=strategy,
                                    source_image=str(sample.image_path),
                                    donor_image=donor_image,
                                    source_bucket=sample.bucket,
                                    target_bucket=target_bucket,
                                    target_max_side_px=float(target_metric),
                                    actual_max_side_px=None,
                                    visible_kpts=None,
                                    mean_nn_px=None,
                                    status="rejected",
                                    reject_reason=str(exc),
                                    output_image=str(output_image_path),
                                    output_label=str(output_label_path),
                                )
                            )
                            continue
                        self.accept_generated_sample(generated)
                        accepted += 1
                        success = True
                        break
                    if success:
                        break
                if not success:
                    raise RuntimeError(
                        f"Failed to generate enough samples for split={split}, bucket={target_bucket}, target_metric={target_metric:.4f}."
                    )
            if accepted != deficit:
                raise RuntimeError(
                    f"Failed to generate enough samples for split={split}, bucket={target_bucket}. Needed {deficit}, accepted {accepted}."
                )

    def balanced_lists(self) -> dict[str, list[Path]]:
        generated_by_split: dict[str, list[Path]] = defaultdict(list)
        for sample in self.generated_samples:
            generated_by_split[sample.split].append(sample.output_image_path.resolve())

        combined = {
            "train_balanced": list(self.localized_raw_lists.get("train", [])) + sorted(generated_by_split["train"]),
            "valid_balanced": list(self.localized_raw_lists.get("valid", [])) + sorted(generated_by_split["valid"]),
            "valid_raw": list(self.localized_raw_lists.get("valid_raw", [])),
            "test_raw": list(self.localized_raw_lists.get("test", [])),
        }
        return combined

    def localized_split_file_counts(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for split in ("train", "valid", "valid_raw", "test"):
            images_dir, labels_dir = self.split_dirs(split)
            image_count = sum(1 for path in images_dir.iterdir() if path.is_file()) if images_dir.exists() else 0
            label_count = sum(1 for path in labels_dir.iterdir() if path.is_file()) if labels_dir.exists() else 0
            counts[split] = {
                "images": image_count,
                "labels": label_count,
            }
        return counts

    def combined_counts(self) -> dict[str, dict[str, int]]:
        raw_counts = self.raw_bucket_counts()
        generated_counter: dict[str, Counter[str]] = defaultdict(Counter)
        for sample in self.generated_samples:
            generated_counter[sample.split][sample.target_bucket] += 1
        result: dict[str, dict[str, int]] = {}
        for split in ("train", "valid", "test"):
            result[split] = {}
            for bucket in [bucket.name for bucket in self.buckets]:
                result[split][bucket] = int(raw_counts.get(split, {}).get(bucket, 0) + generated_counter.get(split, Counter()).get(bucket, 0))
        return result

    def verify_final_counts(self) -> None:
        targets = self.target_total_counts()
        combined = self.combined_counts()
        for split, buckets in targets.items():
            for bucket_name, target_count in buckets.items():
                actual_count = combined.get(split, {}).get(bucket_name, 0)
                if abs(actual_count - target_count) > 1:
                    raise RuntimeError(
                        f"Combined counts mismatch for split={split}, bucket={bucket_name}: actual={actual_count}, target={target_count}"
                    )

    def review_groups(self) -> dict[str, list[GeneratedSample]]:
        groups: dict[str, list[GeneratedSample]] = defaultdict(list)
        for sample in self.generated_samples:
            key = f"{sample.split}_{sample.target_bucket}_{sample.strategy}"
            groups[key].append(sample)
        return groups

    def export_review_images(self) -> None:
        if self.dry_run:
            return
        per_group = int(self.config["review"]["per_group"])
        for group, samples in sorted(self.review_groups().items()):
            chosen = sorted(samples, key=lambda item: item.record.sample_id)[:per_group]
            for sample in chosen:
                review_dir = self.output_root / "analysis" / "review" / group
                review_path = review_dir / f"{sample.record.sample_id}.jpg"
                title = f"{group} {sample.record.sample_id}"
                image = read_image_bgr(sample.output_image_path)
                rendered = review_overlay(
                    image=image,
                    bbox_center_px=sample.bbox_center_px,
                    bbox_size_px=sample.bbox_size_px,
                    keypoints_px=sample.keypoints_px,
                    title=title,
                )
                write_image_bgr(review_path, rendered)
                self.review_rows.append(
                    {
                        "group": group,
                        "sample_id": sample.record.sample_id,
                        "image_path": str(sample.output_image_path),
                        "label_path": str(sample.output_label_path),
                        "rendered_path": str(review_path),
                    }
                )

    def write_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        if self.dry_run:
            return
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

    def write_json(self, path: Path, payload: Any) -> None:
        if self.dry_run:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sanitize_for_dump(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def write_text_file(self, path: Path, text: str) -> None:
        if self.dry_run:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def write_list_file(self, path: Path, items: list[Path]) -> None:
        lines = []
        for item in items:
            resolved = item.resolve()
            try:
                relative = resolved.relative_to(self.output_root)
            except ValueError as exc:
                raise ValueError(f"Audit manifest item escaped derived root: {resolved}") from exc
            lines.append(str(relative))
        self.write_text_file(path, "\n".join(lines) + "\n")

    def write_dataset_yaml(self, path: Path, train_value: str, val_value: str, test_value: str) -> None:
        payload = {
            "train": train_value,
            "val": val_value,
            "test": test_value,
            "kpt_shape": list(self.base_data_config["kpt_shape"]),
            "flip_idx": list(self.base_data_config["flip_idx"]),
            "nc": int(self.base_data_config["nc"]),
            "names": list(self.base_data_config["names"]),
        }
        if "roboflow" in self.base_data_config:
            payload["roboflow"] = self.base_data_config["roboflow"]
        if not self.dry_run:
            path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")

    def write_outputs(self) -> None:
        source_rows = self.source_stats_rows()
        target_counts = self.target_total_counts()
        raw_counts = self.raw_bucket_counts()
        combined_counts = self.combined_counts()
        generated_rows = [asdict(record) for record in self.generated_attempts]
        rejection_rows = [row for row in generated_rows if row["status"] != "accepted"]

        source_summary = {
            "source_root": str(self.source_root),
            "imgsz": self.imgsz,
            "raw_counts": raw_counts,
            "targets": target_counts,
            "records": source_rows,
        }
        generated_summary = {
            "output_root": str(self.output_root),
            "dry_run": self.dry_run,
            "generated_count": len(self.generated_samples),
            "attempt_count": len(self.generated_attempts),
            "combined_counts": combined_counts,
            "localized_split_file_counts": self.localized_split_file_counts(),
            "records": generated_rows,
        }

        self.write_csv(self.output_root / "analysis" / "source_stats.csv", source_rows)
        self.write_json(self.output_root / "analysis" / "source_stats.json", source_summary)
        self.write_csv(self.output_root / "analysis" / "generated_stats.csv", generated_rows)
        self.write_json(self.output_root / "analysis" / "generated_stats.json", generated_summary)
        self.write_csv(self.output_root / "analysis" / "rejections.csv", rejection_rows)
        self.write_csv(self.output_root / "analysis" / "review_manifest.csv", self.review_rows)

        lists = self.balanced_lists()
        self.write_list_file(self.output_root / "train_balanced.txt", lists["train_balanced"])
        self.write_list_file(self.output_root / "valid_balanced.txt", lists["valid_balanced"])
        self.write_list_file(self.output_root / "valid_raw.txt", lists["valid_raw"])
        self.write_list_file(self.output_root / "test_raw.txt", lists["test_raw"])

        self.write_dataset_yaml(
            self.output_root / "data.yaml",
            train_value="train/images",
            val_value="valid/images",
            test_value="test/images",
        )
        self.write_dataset_yaml(
            self.output_root / "data.raw_eval.yaml",
            train_value="train/images",
            val_value="valid_raw/images",
            test_value="test/images",
        )

    def build(self) -> dict[str, Any]:
        self.scan_source_dataset()
        self.prepare_output_root()
        self.localize_raw_splits()
        raw_counts = self.raw_bucket_counts()
        deficits = self.deficits()
        for split in ("train", "valid"):
            self.build_generated_split(split, deficits.get(split, {}))
        self.verify_final_counts()
        self.export_review_images()
        self.write_outputs()

        summary = {
            "source_root": str(self.source_root),
            "output_root": str(self.output_root),
            "dry_run": self.dry_run,
            "imgsz": self.imgsz,
            "raw_counts": raw_counts,
            "targets": self.target_total_counts(),
            "deficits": deficits,
            "combined_counts": self.combined_counts(),
            "localized_split_file_counts": self.localized_split_file_counts(),
            "generated_count": len(self.generated_samples),
            "attempt_count": len(self.generated_attempts),
            "accepted_by_group": {
                group: len(items) for group, items in self.review_groups().items()
            },
        }
        return summary


def main() -> int:
    try:
        args = parse_args()
        config = load_runtime_config(args)
        builder = DatasetBuilder(config=config, dry_run=bool(args.dry_run))
        summary = builder.build()
        print_info("Build summary:")
        print(yaml.safe_dump(sanitize_for_dump(summary), sort_keys=False, allow_unicode=False))
        return 0
    except Exception as exc:
        print_info(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
