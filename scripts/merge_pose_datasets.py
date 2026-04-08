#!/usr/bin/env python3
"""Merge self-contained YOLO pose datasets by per-split sampling ratios."""

from __future__ import annotations

import argparse
import copy
import math
import random
from pathlib import Path
from typing import Any

from build_scale_balanced_pose_dataset import DatasetBuilder
from pose_dataset_build_utils import (
    STANDARD_SPLITS,
    PoseDatasetSample,
    build_dataset_yaml_payload,
    collect_split_samples,
    copy_sample,
    count_split_files,
    load_dataset_metadata,
    load_yaml_file,
    print_info,
    require_matching_schema,
    resolve_path,
    write_csv_file,
    write_json_file,
    write_yaml_file,
)


TOOL_NAME = "merge_pose_datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge YOLO pose datasets by per-split sampling ratios.")
    parser.add_argument(
        "--config",
        default="configs/dataset_merge_example.yaml",
        help="Path to the merge YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute selections and summaries without writing any dataset files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional override for sampling.seed.",
    )
    return parser.parse_args()


def sample_count(total_count: int, ratio: float) -> int:
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"Sampling ratio must be within [0, 1], got {ratio}")
    return math.floor(total_count * ratio)


def validate_config(config: dict[str, Any]) -> None:
    required_top = ["output", "inputs", "sampling", "merge", "postprocess"]
    missing = [key for key in required_top if key not in config]
    if missing:
        raise ValueError(f"Merge config missing keys: {missing}")
    if not isinstance(config["inputs"], list) or not config["inputs"]:
        raise ValueError("'inputs' must be a non-empty list.")
    if config["merge"].get("rename_mode") != "prefix_input_name":
        raise ValueError("Only merge.rename_mode='prefix_input_name' is supported.")


def intermediate_merge_root(final_output_root: Path) -> Path:
    return final_output_root.parent / f"{final_output_root.name}.merge_source"


def relative_to_root(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def run_scale_balance_postprocess(
    *,
    merge_root: Path,
    final_output_root: Path,
    builder_config_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    builder_config = load_yaml_file(builder_config_path)
    builder_config = copy.deepcopy(builder_config)
    builder_config.setdefault("source", {})
    builder_config.setdefault("output", {})
    builder_config["source"]["root"] = str(merge_root)
    builder_config["source"]["data_yaml"] = str(merge_root / "data.yaml")
    builder_config["output"]["root"] = str(final_output_root)

    builder = DatasetBuilder(config=builder_config, dry_run=dry_run)
    return builder.build()


def main() -> int:
    try:
        args = parse_args()
        config_path = resolve_path(args.config)
        config = load_yaml_file(config_path)
        validate_config(config)
        if args.seed is not None:
            config.setdefault("sampling", {})
            config["sampling"]["seed"] = args.seed

        output_root = resolve_path(config["output"]["root"])
        if output_root.exists():
            raise FileExistsError(f"Output dataset root already exists: {output_root}")

        scale_balance_cfg = config["postprocess"].get("scale_balance", {})
        scale_balance_enabled = bool(scale_balance_cfg.get("enabled", False))
        merge_root = intermediate_merge_root(output_root) if scale_balance_enabled else output_root
        if merge_root.exists():
            raise FileExistsError(f"Intermediate merge root already exists: {merge_root}")

        rng = random.Random(int(config["sampling"].get("seed", 52)))
        input_metadatas: list[tuple[str, dict[str, Any]]] = []
        input_samples: list[tuple[dict[str, Any], dict[str, list[PoseDatasetSample]]]] = []

        for entry in config["inputs"]:
            if not isinstance(entry, dict):
                raise ValueError("Each input dataset entry must be a mapping.")
            name = str(entry["name"]).strip()
            if not name or "__" in name or "/" in name or "\\" in name:
                raise ValueError(f"Invalid input dataset name: {name!r}")
            dataset_root = resolve_path(entry["root"])
            metadata = load_dataset_metadata(dataset_root)
            input_metadatas.append((name, metadata))
            samples_by_split = {
                split: collect_split_samples(dataset_root, split, dataset_name=name)
                for split in STANDARD_SPLITS
            }
            input_samples.append((entry, samples_by_split))

        merged_metadata = require_matching_schema(input_metadatas)
        manifest_rows: list[dict[str, Any]] = []
        per_input_summary: dict[str, Any] = {}

        for entry, samples_by_split in input_samples:
            name = str(entry["name"]).strip()
            per_input_summary[name] = {"root": str(resolve_path(entry["root"])), "splits": {}}
            ratios = entry["ratios"]
            for split in STANDARD_SPLITS:
                samples = list(samples_by_split[split])
                count = sample_count(len(samples), float(ratios[split]))
                selected = sorted(rng.sample(samples, count), key=lambda sample: sample.image_path.name) if count else []
                per_input_summary[name]["splits"][split] = {
                    "source_count": len(samples),
                    "ratio": float(ratios[split]),
                    "selected_count": count,
                }
                for sample in selected:
                    target_image_name = f"{name}__{sample.image_path.name}"
                    target_label_name = f"{name}__{sample.label_path.name}"
                    if not args.dry_run:
                        target_image_path, target_label_path = copy_sample(
                            sample=sample,
                            output_root=merge_root,
                            split=split,
                            target_image_name=target_image_name,
                            target_label_name=target_label_name,
                        )
                    else:
                        target_image_path = (merge_root / split / "images" / target_image_name).resolve()
                        target_label_path = (merge_root / split / "labels" / target_label_name).resolve()

                    manifest_rows.append(
                        {
                            "input_name": name,
                            "input_root": str(resolve_path(entry["root"])),
                            "split": split,
                            "ratio": float(ratios[split]),
                            "source_image": str(sample.image_path),
                            "source_label": str(sample.label_path),
                            "output_image": relative_to_root(target_image_path, merge_root),
                            "output_label": relative_to_root(target_label_path, merge_root),
                        }
                    )

        if not args.dry_run:
            write_yaml_file(
                merge_root / "data.yaml",
                build_dataset_yaml_payload(
                    merged_metadata,
                    train_value="train/images",
                    val_value="valid/images",
                    test_value="test/images",
                ),
            )
            write_csv_file(merge_root / "analysis" / "merge_manifest.csv", manifest_rows)

        merge_summary: dict[str, Any] = {
            "config_path": str(config_path),
            "dry_run": bool(args.dry_run),
            "rename_mode": str(config["merge"]["rename_mode"]),
            "sampling_seed": int(config["sampling"].get("seed", 52)),
            "merge_output_root": str(merge_root),
            "final_output_root": str(output_root),
            "inputs": per_input_summary,
            "merged_split_counts": {
                split: count_split_files(merge_root, split) if not args.dry_run else None for split in STANDARD_SPLITS
            },
            "postprocess": {
                "scale_balance": {
                    "enabled": scale_balance_enabled,
                }
            },
        }

        if scale_balance_enabled:
            builder_config_value = scale_balance_cfg.get("builder_config")
            if not builder_config_value:
                raise ValueError("postprocess.scale_balance.builder_config is required when scale_balance.enabled=true")
            builder_config_path = resolve_path(builder_config_value)
            merge_summary["postprocess"]["scale_balance"]["builder_config"] = str(builder_config_path)
            merge_summary["postprocess"]["scale_balance"]["result"] = run_scale_balance_postprocess(
                merge_root=merge_root,
                final_output_root=output_root,
                builder_config_path=builder_config_path,
                dry_run=bool(args.dry_run),
            )
        else:
            merge_summary["final_output_root"] = str(merge_root)

        if not args.dry_run:
            write_json_file(merge_root / "analysis" / "merge_summary.json", merge_summary)
            if scale_balance_enabled:
                write_csv_file(output_root / "analysis" / "merge_manifest.csv", manifest_rows)
                write_json_file(output_root / "analysis" / "merge_summary.json", merge_summary)

        print_info(TOOL_NAME, "Completed successfully.")
        for name, payload in per_input_summary.items():
            for split in STANDARD_SPLITS:
                split_payload = payload["splits"][split]
                print_info(
                    TOOL_NAME,
                    f"input={name} split={split} source_count={split_payload['source_count']} "
                    f"selected_count={split_payload['selected_count']} ratio={split_payload['ratio']}",
                )
        return 0
    except Exception as exc:
        print_info(TOOL_NAME, f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
