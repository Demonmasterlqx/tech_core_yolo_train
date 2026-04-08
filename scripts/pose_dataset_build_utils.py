#!/usr/bin/env python3
"""Shared helpers for building self-contained YOLO pose datasets."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
STANDARD_SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class PoseDatasetSample:
    dataset_name: str
    split: str
    stem: str
    image_path: Path
    label_path: Path


def print_info(tool_name: str, message: str) -> None:
    print(f"[{tool_name}] {message}")


def resolve_path(value: str | Path, base_dir: Path = REPO_ROOT) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return payload


def write_yaml_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def dataset_yaml_path(dataset_root: Path) -> Path:
    path = dataset_root / "data.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {path}")
    return path


def load_dataset_metadata(dataset_root: Path) -> dict[str, Any]:
    data_yaml = dataset_yaml_path(dataset_root)
    payload = load_yaml_file(data_yaml)
    required = ("kpt_shape", "nc", "names")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Dataset metadata missing keys {missing}: {data_yaml}")
    return payload


def image_file_map(images_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            mapping[path.stem] = path.resolve()
    return mapping


def split_dirs(dataset_root: Path, split: str) -> tuple[Path, Path]:
    images_dir = dataset_root / split / "images"
    labels_dir = dataset_root / split / "labels"
    return images_dir, labels_dir


def collect_split_samples(dataset_root: Path, split: str, dataset_name: str | None = None) -> list[PoseDatasetSample]:
    images_dir, labels_dir = split_dirs(dataset_root, split)
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Missing split directories for split={split}: {images_dir} / {labels_dir}")

    images = image_file_map(images_dir)
    labels = {path.stem: path.resolve() for path in sorted(labels_dir.glob("*.txt"))}
    if set(images) != set(labels):
        missing_images = sorted(set(labels) - set(images))
        missing_labels = sorted(set(images) - set(labels))
        raise ValueError(
            f"Image/label mismatch for dataset={dataset_root}, split={split}. "
            f"Missing images={missing_images[:5]}, missing labels={missing_labels[:5]}"
        )

    samples: list[PoseDatasetSample] = []
    resolved_name = dataset_name or dataset_root.name
    for stem in sorted(images):
        samples.append(
            PoseDatasetSample(
                dataset_name=resolved_name,
                split=split,
                stem=stem,
                image_path=images[stem],
                label_path=labels[stem],
            )
        )
    return samples


def count_split_files(dataset_root: Path, split: str) -> dict[str, int]:
    images_dir, labels_dir = split_dirs(dataset_root, split)
    image_count = sum(1 for path in images_dir.iterdir() if path.is_file()) if images_dir.exists() else 0
    label_count = sum(1 for path in labels_dir.glob("*.txt")) if labels_dir.exists() else 0
    return {"images": image_count, "labels": label_count}


def copy_sample(
    sample: PoseDatasetSample,
    output_root: Path,
    split: str,
    target_image_name: str,
    target_label_name: str,
) -> tuple[Path, Path]:
    images_dir, labels_dir = split_dirs(output_root, split)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    target_image_path = images_dir / target_image_name
    target_label_path = labels_dir / target_label_name
    shutil.copy2(sample.image_path, target_image_path)
    shutil.copy2(sample.label_path, target_label_path)
    return target_image_path.resolve(), target_label_path.resolve()


def build_dataset_yaml_payload(
    metadata: dict[str, Any],
    *,
    train_value: str,
    val_value: str,
    test_value: str,
) -> dict[str, Any]:
    payload = {
        "train": train_value,
        "val": val_value,
        "test": test_value,
        "kpt_shape": list(metadata["kpt_shape"]),
        "nc": int(metadata["nc"]),
        "names": list(metadata["names"]),
    }
    if "flip_idx" in metadata:
        payload["flip_idx"] = list(metadata["flip_idx"])
    if "roboflow" in metadata:
        payload["roboflow"] = metadata["roboflow"]
    return payload


def require_matching_schema(metadatas: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    if not metadatas:
        raise ValueError("At least one dataset metadata entry is required.")

    first_name, first_metadata = metadatas[0]
    reference = {
        "kpt_shape": list(first_metadata["kpt_shape"]),
        "nc": int(first_metadata["nc"]),
        "names": list(first_metadata["names"]),
    }
    for name, metadata in metadatas[1:]:
        current = {
            "kpt_shape": list(metadata["kpt_shape"]),
            "nc": int(metadata["nc"]),
            "names": list(metadata["names"]),
        }
        if current != reference:
            raise ValueError(
                f"Dataset schema mismatch between '{first_name}' and '{name}'. "
                f"Expected {reference}, got {current}."
            )
    return first_metadata


def default_realonly_output_root(source_root: Path) -> Path:
    if source_root.name.endswith(".yolov8"):
        base_name = source_root.name[: -len(".yolov8")]
        return source_root.with_name(f"{base_name}.realonly_v1.yolov8")
    return source_root.with_name(f"{source_root.name}.realonly_v1.yolov8")
