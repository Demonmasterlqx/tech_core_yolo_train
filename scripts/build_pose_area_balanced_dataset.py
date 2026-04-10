#!/usr/bin/env python3
"""Build a train-ready YOLO pose dataset with explicit bbox area-ratio balancing."""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from pose_area_balance import (
    annotation_area_ratio_after_letterbox,
    bin_for_ratio,
    build_area_bins,
    build_bin_histogram,
    build_generation_tasks,
    generated_target_count_per_bin,
    synthesize_area_balanced_sample,
)
from pose_dataset_build_utils import (
    build_dataset_yaml_payload,
    collect_split_samples,
    copy_sample,
    load_yaml_file,
    resolve_path,
    write_yaml_file,
)
from pose_offline_aug.io import (
    draw_review_overlay,
    object_annotation_to_yolo_pose_line,
    read_single_pose_annotation,
    read_image_bgr,
    write_csv_file,
    write_image_bgr,
    write_json_file,
    write_text_file,
)


TOOL_NAME = "build_pose_area_balanced_dataset"


def print_info(message: str) -> None:
    print(f"[{TOOL_NAME}] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an area-balanced YOLO pose dataset.")
    parser.add_argument("--config", required=True, help="Path to the builder YAML config.")
    parser.add_argument("--src-root", help="Optional override for source.root.")
    parser.add_argument("--dst-root", help="Optional override for output.root.")
    parser.add_argument("--dry-run", action="store_true", help="Preview build decisions without writing the full dataset.")
    parser.add_argument("--seed", type=int, help="Optional override for runtime.seed.")
    parser.add_argument("--limit", type=int, help="Optional limit for train samples.")
    return parser.parse_args()


def require_mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping for '{name}'.")
    return value


def resolve_dataset_yaml_path(source_root: Path, explicit_path: str | None) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(resolve_path(explicit_path))
    else:
        candidates.extend((source_root / "data.yaml", source_root / "dataset.yaml"))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Dataset YAML does not exist. Searched: {searched}")


def load_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml_file(resolve_path(args.config))
    required = ["source", "output", "runtime", "metric", "geometry", "appearance", "occlusion", "filter", "review"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    source_cfg = require_mapping("source", config["source"])
    output_cfg = require_mapping("output", config["output"])
    runtime_cfg = require_mapping("runtime", config["runtime"])
    metric_cfg = require_mapping("metric", config["metric"])
    require_mapping("geometry", config["geometry"])
    require_mapping("appearance", config["appearance"])
    require_mapping("occlusion", config["occlusion"])
    require_mapping("filter", config["filter"])
    require_mapping("review", config["review"])

    if args.src_root:
        source_cfg["root"] = args.src_root
    if args.dst_root:
        output_cfg["root"] = args.dst_root
    if args.seed is not None:
        runtime_cfg["seed"] = int(args.seed)

    source_root = resolve_path(source_cfg["root"])
    data_yaml = resolve_dataset_yaml_path(source_root, source_cfg.get("data_yaml"))
    output_root = resolve_path(output_cfg["root"])
    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset root does not exist: {source_root}")

    source_cfg["root"] = str(source_root)
    source_cfg["data_yaml"] = str(data_yaml)
    output_cfg["root"] = str(output_root)
    output_cfg.setdefault("clean_output", True)
    runtime_cfg.setdefault("seed", 52)
    runtime_cfg.setdefault("generated_multiplier", int(metric_cfg["bin_count"]))
    runtime_cfg.setdefault("keep_original_train", True)
    runtime_cfg.setdefault("max_attempts_per_bin", 10)
    runtime_cfg.setdefault("image_suffix", ".jpg")
    runtime_cfg.setdefault("label_suffix", ".txt")
    runtime_cfg.setdefault("strategy_preference", "auto")
    metric_cfg.setdefault("target_margin_ratio", 0.15)
    return config


def load_dataset_metadata(data_yaml_path: Path) -> dict[str, Any]:
    metadata = load_yaml_file(data_yaml_path)
    required = ("kpt_shape", "nc", "names")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Dataset metadata missing keys {missing}: {data_yaml_path}")
    return metadata


class DatasetBuilder:
    def __init__(self, config: dict[str, Any], *, dry_run: bool, limit: int | None) -> None:
        self.config = config
        self.dry_run = dry_run
        self.limit = limit

        self.source_root = Path(config["source"]["root"]).resolve()
        self.data_yaml_path = Path(config["source"]["data_yaml"]).resolve()
        self.requested_output_root = Path(config["output"]["root"]).resolve()
        self.output_root = self.requested_output_root.with_name(f"{self.requested_output_root.name}.dryrun_preview") if dry_run else self.requested_output_root
        self.clean_output = bool(config["output"].get("clean_output", True))
        self.metadata = load_dataset_metadata(self.data_yaml_path)
        self.keypoint_count = int(self.metadata["kpt_shape"][0])
        self.imgsz = float(config["metric"]["imgsz"])
        self.bins = build_area_bins(
            min_area_ratio=float(config["metric"]["min_area_ratio"]),
            max_area_ratio=float(config["metric"]["max_area_ratio"]),
            bin_count=int(config["metric"]["bin_count"]),
        )

        import random

        self.rng = random.Random(int(config["runtime"]["seed"]))
        self.keep_original_train = bool(config["runtime"]["keep_original_train"])
        self.generated_multiplier = int(config["runtime"]["generated_multiplier"])
        self.max_attempts_per_bin = int(config["runtime"]["max_attempts_per_bin"])
        self.image_suffix = str(config["runtime"]["image_suffix"])
        self.label_suffix = str(config["runtime"]["label_suffix"])

        self.accepted_rows: list[dict[str, Any]] = []
        self.rejected_rows: list[dict[str, Any]] = []
        self.source_rows: list[dict[str, Any]] = []
        self.review_rows: list[dict[str, Any]] = []
        self.train_manifest: list[Path] = []
        self.valid_manifest: list[Path] = []
        self.valid_raw_manifest: list[Path] = []
        self.test_manifest: list[Path] = []
        self.generated_ratios: list[float] = []
        self.raw_train_ratios: list[float] = []

    def split_dirs(self, split: str) -> tuple[Path, Path]:
        return self.output_root / split / "images", self.output_root / split / "labels"

    def planned_paths(self, split: str, image_name: str, label_name: str) -> tuple[Path, Path]:
        image_dir, label_dir = self.split_dirs(split)
        return image_dir / image_name, label_dir / label_name

    def prepare_output_root(self) -> None:
        if self.output_root.exists() and self.clean_output:
            shutil.rmtree(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def source_split_enabled(self, split: str) -> bool:
        return (self.source_root / split / "images").exists() and (self.source_root / split / "labels").exists()

    def localize_split(self, split: str, target_split: str, manifest: list[Path]) -> None:
        if not self.source_split_enabled(split):
            return
        for sample in collect_split_samples(self.source_root, split):
            target_image_path, target_label_path = self.planned_paths(target_split, sample.image_path.name, sample.label_path.name)
            if not self.dry_run:
                copy_sample(sample, self.output_root, target_split, sample.image_path.name, sample.label_path.name)
            manifest.append(target_image_path.resolve())
            if self.dry_run:
                target_label_path.parent.mkdir(parents=True, exist_ok=True)

    def localize_eval_splits(self) -> None:
        self.localize_split("valid", "valid", self.valid_manifest)
        self.localize_split("valid", "valid_raw", self.valid_raw_manifest)
        self.localize_split("test", "test", self.test_manifest)

    def collect_source_stats(self, samples: list[Any], split: str) -> None:
        for sample in samples:
            annotation = read_single_pose_annotation(
                sample.image_path,
                sample.label_path,
                split=sample.split,
                sample_id=sample.stem,
                keypoint_count=self.keypoint_count,
            )
            ratio = annotation_area_ratio_after_letterbox(
                annotation.object_annotation,
                image_width=annotation.image_width,
                image_height=annotation.image_height,
                imgsz=self.imgsz,
            )
            self.source_rows.append(
                {
                    "split": split,
                    "sample_id": sample.stem,
                    "image_path": str(sample.image_path),
                    "label_path": str(sample.label_path),
                    "image_width": annotation.image_width,
                    "image_height": annotation.image_height,
                    "raw_bbox_area_ratio": annotation.object_annotation.bbox.area / float(annotation.image_width * annotation.image_height),
                    "letterbox_bbox_area_ratio": ratio,
                    "in_target_range": ratio >= self.bins[0].lower and ratio < self.bins[-1].upper,
                }
            )
            if split == "train":
                self.raw_train_ratios.append(ratio)

    def maybe_export_review(self, image: Any, annotation: Any, target_bin_label: str, output_stem: str) -> None:
        if not bool(self.config["review"].get("enabled", True)) or self.dry_run:
            return
        per_bin = int(self.config["review"].get("per_bin", 8))
        current_count = sum(1 for row in self.review_rows if row["target_bin"] == target_bin_label)
        if current_count >= per_bin:
            return
        review_dir = self.output_root / str(self.config["review"].get("export_dir", "analysis/review")) / target_bin_label.replace(",", "_")
        review_path = review_dir / f"{output_stem}.jpg"
        rendered = draw_review_overlay(image, annotation, f"{target_bin_label} {output_stem}")
        write_image_bgr(review_path, rendered)
        self.review_rows.append({"target_bin": target_bin_label, "review_image": str(review_path), "output_stem": output_stem})

    def process_train(self) -> None:
        train_samples = collect_split_samples(self.source_root, "train")
        if self.limit is not None:
            train_samples = train_samples[: self.limit]
        self.collect_source_stats(train_samples, "train")
        for split in ("valid", "test"):
            if self.source_split_enabled(split):
                self.collect_source_stats(collect_split_samples(self.source_root, split), split)

        generated_target_count_per_bin(len(train_samples), self.generated_multiplier, len(self.bins))
        tasks = build_generation_tasks(train_samples, self.bins, self.generated_multiplier)
        tasks_by_source: dict[str, list[Any]] = {}
        for task in tasks:
            tasks_by_source.setdefault(task.source_sample.stem, []).append(task)

        for sample in train_samples:
            if self.keep_original_train:
                target_image_path, _ = self.planned_paths("train", sample.image_path.name, sample.label_path.name)
                if not self.dry_run:
                    copy_sample(sample, self.output_root, "train", sample.image_path.name, sample.label_path.name)
                self.train_manifest.append(target_image_path.resolve())

            source_image = read_image_bgr(sample.image_path)
            annotation = read_single_pose_annotation(
                sample.image_path,
                sample.label_path,
                split=sample.split,
                sample_id=sample.stem,
                keypoint_count=self.keypoint_count,
            ).object_annotation

            for task in tasks_by_source[sample.stem]:
                image_name = f"{task.output_stem}{self.image_suffix}"
                label_name = f"{task.output_stem}{self.label_suffix}"
                output_image_path, output_label_path = self.planned_paths("train", image_name, label_name)
                accepted = False
                for attempt_index in range(self.max_attempts_per_bin):
                    try:
                        generated = synthesize_area_balanced_sample(
                            source_image,
                            annotation,
                            self.config,
                            task.target_bin,
                            self.rng,
                        )
                        label_text = object_annotation_to_yolo_pose_line(
                            generated.annotation,
                            image_width=generated.image.shape[1],
                            image_height=generated.image.shape[0],
                        )
                        if not self.dry_run:
                            write_image_bgr(output_image_path, generated.image)
                            write_text_file(output_label_path, label_text)
                        self.train_manifest.append(output_image_path.resolve())
                        self.generated_ratios.append(generated.actual_area_ratio)
                        self.accepted_rows.append(
                            {
                                "source_sample_id": sample.stem,
                                "target_bin": task.target_bin.label,
                                "slot_index": task.slot_index,
                                "attempt_index": attempt_index,
                                "strategy": generated.strategy,
                                "target_area_ratio": generated.target_area_ratio,
                                "actual_area_ratio": generated.actual_area_ratio,
                                "output_image": str(output_image_path),
                                "output_label": str(output_label_path),
                                "transforms": generated.transforms,
                            }
                        )
                        self.maybe_export_review(generated.image, generated.annotation, task.target_bin.label, task.output_stem)
                        accepted = True
                        break
                    except Exception as exc:
                        self.rejected_rows.append(
                            {
                                "source_sample_id": sample.stem,
                                "target_bin": task.target_bin.label,
                                "slot_index": task.slot_index,
                                "attempt_index": attempt_index,
                                "reject_reason": str(exc),
                                "output_image": str(output_image_path),
                                "output_label": str(output_label_path),
                            }
                        )
                if not accepted:
                    raise RuntimeError(
                        f"Failed to generate source={sample.stem} for bin={task.target_bin.label} after "
                        f"{self.max_attempts_per_bin} attempts."
                    )

    def write_list_file(self, path: Path, items: list[Path]) -> None:
        lines = [str(path_item.resolve().relative_to(self.output_root)) for path_item in items]
        write_text_file(path, "\n".join(lines) + ("\n" if lines else ""))

    def write_dataset_yamls(self) -> None:
        data_yaml = build_dataset_yaml_payload(
            self.metadata,
            train_value="train/images",
            val_value="valid/images" if self.valid_manifest else "none",
            test_value="test/images" if self.test_manifest else "none",
        )
        raw_eval_yaml = build_dataset_yaml_payload(
            self.metadata,
            train_value="train/images",
            val_value="valid_raw/images" if self.valid_raw_manifest else "none",
            test_value="test/images" if self.test_manifest else "none",
        )
        if not self.dry_run:
            write_yaml_file(self.output_root / "data.yaml", data_yaml)
            write_yaml_file(self.output_root / "data.raw_eval.yaml", raw_eval_yaml)

    def write_analysis(self) -> None:
        generated_only_hist = build_bin_histogram(self.generated_ratios, self.bins)
        raw_in_range = [value for value in self.raw_train_ratios if bin_for_ratio(value, self.bins) is not None]
        raw_out_of_range = [value for value in self.raw_train_ratios if bin_for_ratio(value, self.bins) is None]
        full_train_hist = build_bin_histogram(self.generated_ratios + raw_in_range, self.bins)

        summary = {
            "source_root": str(self.source_root),
            "data_yaml": str(self.data_yaml_path),
            "effective_output_root": str(self.output_root),
            "dry_run": self.dry_run,
            "raw_train_count": len(self.raw_train_ratios),
            "generated_count": len(self.generated_ratios),
            "final_train_count": len(self.train_manifest),
            "generated_only_target_count_per_bin": generated_target_count_per_bin(
                len(self.raw_train_ratios),
                self.generated_multiplier,
                len(self.bins),
            ),
            "generated_only_hist": generated_only_hist,
            "raw_train_in_range_hist": build_bin_histogram(raw_in_range, self.bins),
            "raw_train_out_of_range_count": len(raw_out_of_range),
            "full_train_hist": full_train_hist,
            "full_train_out_of_range_count": len(raw_out_of_range),
        }

        write_csv_file(self.output_root / "analysis" / "source_stats.csv", self.source_rows)
        write_json_file(self.output_root / "analysis" / "source_stats.json", self.source_rows)
        write_csv_file(self.output_root / "analysis" / "generated_stats.csv", self.accepted_rows)
        write_json_file(self.output_root / "analysis" / "generated_stats.json", self.accepted_rows)
        write_csv_file(self.output_root / "analysis" / "rejections.csv", self.rejected_rows)
        write_json_file(self.output_root / "analysis" / "bin_summary.json", summary)
        write_csv_file(self.output_root / "analysis" / "review_manifest.csv", self.review_rows)

        self.write_list_file(self.output_root / "train_augmented.txt", self.train_manifest)
        self.write_list_file(self.output_root / "valid_raw.txt", self.valid_raw_manifest)
        self.write_list_file(self.output_root / "test_raw.txt", self.test_manifest)

    def build(self) -> dict[str, Any]:
        self.prepare_output_root()
        self.localize_eval_splits()
        self.process_train()
        self.write_dataset_yamls()
        self.write_analysis()
        return {
            "effective_output_root": str(self.output_root),
            "raw_train_count": len(self.raw_train_ratios),
            "generated_count": len(self.generated_ratios),
            "final_train_count": len(self.train_manifest),
        }


def main() -> int:
    try:
        args = parse_args()
        config = load_runtime_config(args)
        builder = DatasetBuilder(config, dry_run=bool(args.dry_run), limit=args.limit)
        summary = builder.build()
        print_info("Build summary:")
        for key, value in summary.items():
            print_info(f"{key}={value}")
        return 0
    except Exception as exc:
        print_info(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
