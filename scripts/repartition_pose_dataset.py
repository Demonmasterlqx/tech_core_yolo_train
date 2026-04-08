#!/usr/bin/env python3
"""Repartition a YOLO pose dataset into new train/valid/test splits."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

from pose_dataset_build_utils import (
    STANDARD_SPLITS,
    PoseDatasetSample,
    build_dataset_yaml_payload,
    collect_split_samples,
    copy_sample,
    count_split_files,
    image_file_map,
    load_yaml_file,
    print_info,
    resolve_path,
    write_csv_file,
    write_json_file,
    write_yaml_file,
)


TOOL_NAME = "repartition_pose_dataset"
LIST_FILE_SPLIT_KEYS = {"train": "train", "valid": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repartition a YOLO pose dataset into new train/valid/test splits.")
    parser.add_argument(
        "--dataset-root",
        default="data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.yolov8",
        help="Source dataset root containing train/valid/test directories.",
    )
    parser.add_argument(
        "--output-root",
        help="Output dataset root. Defaults to <dataset-root>.resplit_v1.yolov8.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Ratio assigned to the new train split.")
    parser.add_argument("--valid-ratio", type=float, default=0.2, help="Ratio assigned to the new valid split.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Ratio assigned to the new test split.")
    parser.add_argument(
        "--seed",
        type=int,
        default=52,
        help="Random seed used for the repartition shuffle.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the repartition plan and summaries without writing dataset files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional debugging limit on the total number of combined samples considered after shuffle.",
    )
    return parser.parse_args()


def default_output_root(dataset_root: Path) -> Path:
    if dataset_root.name.endswith(".yolov8"):
        base_name = dataset_root.name[: -len(".yolov8")]
        return dataset_root.with_name(f"{base_name}.resplit_v1.yolov8")
    return dataset_root.with_name(f"{dataset_root.name}.resplit_v1.yolov8")


def validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    ratios = (float(train_ratio), float(valid_ratio), float(test_ratio))
    if any(ratio < 0.0 or ratio > 1.0 for ratio in ratios):
        raise ValueError(f"All ratios must be within [0, 1], got {ratios}")
    total = sum(ratios)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.10f}")
    return ratios


def largest_remainder_counts(total_count: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    split_names = ("train", "valid", "test")
    raw_counts = [total_count * ratio for ratio in ratios]
    floored = [math.floor(value) for value in raw_counts]
    remainder = total_count - sum(floored)
    ranked = sorted(
        range(len(split_names)),
        key=lambda index: (raw_counts[index] - floored[index], ratios[index], -index),
        reverse=True,
    )
    counts = list(floored)
    for index in ranked[:remainder]:
        counts[index] += 1
    return {name: counts[idx] for idx, name in enumerate(split_names)}


def renamed_output_names(sample: PoseDatasetSample) -> tuple[str, str]:
    return f"{sample.split}__{sample.image_path.name}", f"{sample.split}__{sample.label_path.name}"


def ensure_unique_target_names(allocations: dict[str, list[PoseDatasetSample]]) -> None:
    seen: set[tuple[str, str, str]] = set()
    for target_split, samples in allocations.items():
        for sample in samples:
            image_name, label_name = renamed_output_names(sample)
            key = (target_split, image_name, label_name)
            if key in seen:
                raise ValueError(
                    f"Output name collision detected for split={target_split}: {image_name} / {label_name}. "
                    "Rename the source files or adjust the repartition strategy."
                )
            seen.add(key)


def dataset_yaml_candidates(dataset_root: Path) -> list[Path]:
    return [dataset_root / "data.yaml", dataset_root / "dataset.yaml"]


def resolve_dataset_yaml_path(dataset_root: Path) -> Path:
    for candidate in dataset_yaml_candidates(dataset_root):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find data.yaml or dataset.yaml under {dataset_root}")


def load_dataset_metadata_compat(dataset_root: Path) -> tuple[dict[str, Any], Path]:
    dataset_yaml_path = resolve_dataset_yaml_path(dataset_root)
    metadata = load_yaml_file(dataset_yaml_path)
    required = ("kpt_shape", "nc", "names")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Dataset metadata missing keys {missing}: {dataset_yaml_path}")
    return metadata, dataset_yaml_path


def is_directory_dataset(dataset_root: Path) -> bool:
    return all((dataset_root / split / "images").exists() and (dataset_root / split / "labels").exists() for split in STANDARD_SPLITS)


def normalize_source_split(split: str) -> str:
    normalized = split.strip().lower()
    if normalized == "val":
        return "valid"
    if normalized not in STANDARD_SPLITS:
        raise ValueError(f"Unsupported split name: {split}")
    return normalized


def resolve_listfile_entries(dataset_yaml_path: Path, split_value: str) -> list[Path]:
    resolved_value = resolve_path(split_value, dataset_yaml_path.parent)
    if resolved_value.is_file():
        return [resolved_value]
    raise ValueError(
        f"Only text file split definitions are supported for list-file datasets, got: {split_value} "
        f"(resolved to {resolved_value})"
    )


def label_path_for_listfile_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        image_index = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"Image path does not contain an 'images' directory: {image_path}") from exc
    parts[image_index] = "labels"
    label_path = Path(*parts).with_suffix(".txt")
    return label_path.resolve()


def collect_listfile_samples(dataset_root: Path, dataset_yaml_path: Path) -> dict[str, list[PoseDatasetSample]]:
    data_cfg = load_yaml_file(dataset_yaml_path)
    sample_map: dict[str, PoseDatasetSample] = {}
    split_samples: dict[str, list[PoseDatasetSample]] = {split: [] for split in STANDARD_SPLITS}

    for target_split, yaml_key in LIST_FILE_SPLIT_KEYS.items():
        split_value = data_cfg.get(yaml_key)
        if split_value in (None, ""):
            continue
        for split_file in resolve_listfile_entries(dataset_yaml_path, str(split_value)):
            lines = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            for raw_line in lines:
                image_path = resolve_path(raw_line, split_file.parent)
                if not image_path.exists():
                    raise FileNotFoundError(f"Split entry image does not exist: {image_path}")
                label_path = label_path_for_listfile_image(image_path)
                if not label_path.exists():
                    raise FileNotFoundError(f"Derived label path does not exist: {label_path}")
                sample_key = str(image_path.resolve())
                sample = sample_map.get(sample_key)
                if sample is None:
                    sample = PoseDatasetSample(
                        dataset_name=dataset_root.name,
                        split=target_split,
                        stem=image_path.stem,
                        image_path=image_path.resolve(),
                        label_path=label_path.resolve(),
                    )
                    sample_map[sample_key] = sample
                split_samples[target_split].append(sample)
    return split_samples


def collect_samples_by_split(dataset_root: Path) -> tuple[dict[str, list[PoseDatasetSample]], dict[str, Any], Path, str]:
    metadata, dataset_yaml_path = load_dataset_metadata_compat(dataset_root)
    if is_directory_dataset(dataset_root):
        return (
            {split: collect_split_samples(dataset_root, split) for split in STANDARD_SPLITS},
            metadata,
            dataset_yaml_path,
            "directory",
        )
    return collect_listfile_samples(dataset_root, dataset_yaml_path), metadata, dataset_yaml_path, "listfile"


def main() -> int:
    try:
        args = parse_args()
        dataset_root = resolve_path(args.dataset_root)
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

        output_root = resolve_path(args.output_root) if args.output_root else default_output_root(dataset_root)
        if output_root.exists() and not args.dry_run:
            raise FileExistsError(f"Output dataset root already exists: {output_root}")

        ratios = validate_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)
        samples_by_split, metadata, dataset_yaml_path, dataset_mode = collect_samples_by_split(dataset_root)

        combined: list[PoseDatasetSample] = []
        source_summary: dict[str, dict[str, int]] = {}
        for split in STANDARD_SPLITS:
            samples = list(samples_by_split[split])
            combined.extend(samples)
            source_summary[split] = {
                "source_images": len(samples),
                "source_labels": len(samples),
            }

        rng = random.Random(int(args.seed))
        rng.shuffle(combined)
        if args.limit is not None:
            combined = combined[: int(args.limit)]

        total_count = len(combined)
        target_counts = largest_remainder_counts(total_count, ratios)

        allocations: dict[str, list[PoseDatasetSample]] = {split: [] for split in STANDARD_SPLITS}
        train_end = target_counts["train"]
        valid_end = train_end + target_counts["valid"]
        allocations["train"] = combined[:train_end]
        allocations["valid"] = combined[train_end:valid_end]
        allocations["test"] = combined[valid_end:]
        ensure_unique_target_names(allocations)

        manifest_rows: list[dict[str, Any]] = []
        for target_split in STANDARD_SPLITS:
            for sample in allocations[target_split]:
                target_image_name, target_label_name = renamed_output_names(sample)
                if not args.dry_run:
                    target_image_path, target_label_path = copy_sample(
                        sample=sample,
                        output_root=output_root,
                        split=target_split,
                        target_image_name=target_image_name,
                        target_label_name=target_label_name,
                    )
                else:
                    target_image_path = (output_root / target_split / "images" / target_image_name).resolve()
                    target_label_path = (output_root / target_split / "labels" / target_label_name).resolve()

                manifest_rows.append(
                    {
                        "source_split": sample.split,
                        "target_split": target_split,
                        "source_image": str(sample.image_path),
                        "source_label": str(sample.label_path),
                        "output_image": str(target_image_path),
                        "output_label": str(target_label_path),
                    }
                )

        summary = {
            "source_root": str(dataset_root),
            "dataset_yaml": str(dataset_yaml_path),
            "dataset_mode": dataset_mode,
            "output_root": str(output_root),
            "dry_run": bool(args.dry_run),
            "seed": int(args.seed),
            "limit": None if args.limit is None else int(args.limit),
            "ratios": {
                "train": ratios[0],
                "valid": ratios[1],
                "test": ratios[2],
            },
            "source_summary": source_summary,
            "combined_sample_count": total_count,
            "target_split_counts": target_counts,
            "written_split_counts": {
                split: count_split_files(output_root, split) if not args.dry_run else {"images": len(allocations[split]), "labels": len(allocations[split])}
                for split in STANDARD_SPLITS
            },
        }

        if not args.dry_run:
            write_yaml_file(
                output_root / "data.yaml",
                build_dataset_yaml_payload(
                    metadata,
                    train_value="train/images",
                    val_value="valid/images",
                    test_value="test/images",
                ),
            )
            write_csv_file(output_root / "analysis" / "repartition_manifest.csv", manifest_rows)
            write_json_file(output_root / "analysis" / "repartition_summary.json", summary)

        print_info(TOOL_NAME, "Completed successfully.")
        print_info(
            TOOL_NAME,
            f"combined={total_count} train={target_counts['train']} valid={target_counts['valid']} test={target_counts['test']}",
        )
        return 0
    except Exception as exc:
        print_info(TOOL_NAME, f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
