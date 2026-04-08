#!/usr/bin/env python3
"""Extract a self-contained real-only YOLO pose dataset from an existing dataset root."""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path
from typing import Any

from pose_dataset_build_utils import (
    STANDARD_SPLITS,
    build_dataset_yaml_payload,
    collect_split_samples,
    copy_sample,
    count_split_files,
    default_realonly_output_root,
    load_dataset_metadata,
    print_info,
    resolve_path,
    write_csv_file,
    write_json_file,
    write_yaml_file,
)


TOOL_NAME = "build_real_only_pose_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract real-domain pose samples into a standalone dataset.")
    parser.add_argument(
        "--dataset-root",
        default="data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.yolov8",
        help="Source dataset root containing train/valid/test.",
    )
    parser.add_argument(
        "--output-root",
        help="Output dataset root. Defaults to <dataset-root>.realonly_v1.yolov8.",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["frame_*.jpg", "1_*.jpg", "3_*.jpg"],
        help="Filename glob patterns that define the real-domain subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=52,
        help="Recorded in the summary for reproducibility. Selection is deterministic by pattern matching.",
    )
    return parser.parse_args()


def first_matching_pattern(filename: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        if fnmatch.fnmatch(filename, pattern):
            return pattern
    return None


def main() -> int:
    try:
        args = parse_args()
        dataset_root = resolve_path(args.dataset_root)
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

        output_root = resolve_path(args.output_root) if args.output_root else default_realonly_output_root(dataset_root)
        if output_root.exists():
            raise FileExistsError(f"Output dataset root already exists: {output_root}")

        metadata = load_dataset_metadata(dataset_root)
        manifest_rows: list[dict[str, Any]] = []
        split_summary: dict[str, dict[str, int]] = {}

        print_info(TOOL_NAME, f"Extracting real-only dataset from: {dataset_root}")
        print_info(TOOL_NAME, f"Output root: {output_root}")
        print_info(TOOL_NAME, f"Patterns: {args.patterns}")

        for split in STANDARD_SPLITS:
            selected = []
            for sample in collect_split_samples(dataset_root, split):
                pattern = first_matching_pattern(sample.image_path.name, args.patterns)
                if pattern is None:
                    continue
                target_image_path, target_label_path = copy_sample(
                    sample=sample,
                    output_root=output_root,
                    split=split,
                    target_image_name=sample.image_path.name,
                    target_label_name=sample.label_path.name,
                )
                selected.append(sample)
                manifest_rows.append(
                    {
                        "split": split,
                        "matched_pattern": pattern,
                        "source_image": str(sample.image_path),
                        "source_label": str(sample.label_path),
                        "output_image": str(target_image_path.relative_to(output_root)),
                        "output_label": str(target_label_path.relative_to(output_root)),
                    }
                )

            if not selected:
                raise FileNotFoundError(
                    f"No samples matched patterns {args.patterns} in split '{split}' under {dataset_root / split / 'images'}"
                )

            split_summary[split] = {
                "selected_images": len(selected),
                "selected_labels": len(selected),
            }

        data_yaml = build_dataset_yaml_payload(
            metadata,
            train_value="train/images",
            val_value="valid/images",
            test_value="test/images",
        )
        write_yaml_file(output_root / "data.yaml", data_yaml)
        write_csv_file(output_root / "analysis" / "selection_manifest.csv", manifest_rows)
        write_json_file(
            output_root / "analysis" / "selection_summary.json",
            {
                "source_root": str(dataset_root),
                "output_root": str(output_root),
                "patterns": list(args.patterns),
                "seed": int(args.seed),
                "splits": {split: count_split_files(output_root, split) for split in STANDARD_SPLITS},
            },
        )

        print_info(TOOL_NAME, "Completed successfully.")
        for split in STANDARD_SPLITS:
            counts = split_summary[split]
            print_info(
                TOOL_NAME,
                f"split={split} selected_images={counts['selected_images']} selected_labels={counts['selected_labels']}",
            )
        return 0
    except Exception as exc:
        print_info(TOOL_NAME, f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
