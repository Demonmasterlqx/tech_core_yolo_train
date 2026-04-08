#!/usr/bin/env python3
"""Repartition a self-contained YOLO pose dataset into new train/valid/test splits."""

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
    load_dataset_metadata,
    print_info,
    resolve_path,
    write_csv_file,
    write_json_file,
    write_yaml_file,
)


TOOL_NAME = "repartition_pose_dataset"


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
        metadata = load_dataset_metadata(dataset_root)

        combined: list[PoseDatasetSample] = []
        source_summary: dict[str, dict[str, int]] = {}
        for split in STANDARD_SPLITS:
            samples = collect_split_samples(dataset_root, split)
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
