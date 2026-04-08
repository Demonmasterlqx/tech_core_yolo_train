#!/usr/bin/env python3
"""Build an offline-augmented YOLO pose dataset from an existing dataset root."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pose_dataset_build_utils import (
    STANDARD_SPLITS,
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
    write_csv_file,
    write_image_bgr,
    write_json_file,
    write_text_file,
)
from pose_offline_aug.pipeline import GeneratedSample, generate_augmented_sample


TOOL_NAME = "build_pose_augmented_dataset"


def print_info(message: str) -> None:
    print(f"[{TOOL_NAME}] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an offline-augmented pose dataset.")
    parser.add_argument(
        "--config",
        default="configs/offline_pose_aug_medium.yaml",
        help="Path to the augmentation YAML config.",
    )
    parser.add_argument("--src-root", help="Optional override for source.root.")
    parser.add_argument("--dst-root", help="Optional override for output.root.")
    parser.add_argument("--dry-run", action="store_true", help="Write only review and analysis artifacts.")
    parser.add_argument("--seed", type=int, help="Optional override for runtime.seed.")
    parser.add_argument("--limit", type=int, help="Only process the first N train samples.")
    parser.add_argument("--visualize", action="store_true", help="Force-export review images.")
    return parser.parse_args()


def require_mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping for '{name}'.")
    return value


def load_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = resolve_path(args.config)
    config = load_yaml_file(config_path)
    required_top_level = [
        "source",
        "output",
        "runtime",
        "geometry",
        "appearance",
        "occlusion",
        "filter",
        "review",
        "templates",
    ]
    missing = [key for key in required_top_level if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    source_cfg = require_mapping("source", config["source"])
    output_cfg = require_mapping("output", config["output"])
    runtime_cfg = require_mapping("runtime", config["runtime"])
    require_mapping("geometry", config["geometry"])
    require_mapping("appearance", config["appearance"])
    require_mapping("occlusion", config["occlusion"])
    require_mapping("filter", config["filter"])
    require_mapping("review", config["review"])
    require_mapping("templates", config["templates"])

    if args.src_root:
        source_cfg["root"] = args.src_root
    if args.dst_root:
        output_cfg["root"] = args.dst_root
    if args.seed is not None:
        runtime_cfg["seed"] = int(args.seed)

    source_root = resolve_path(source_cfg["root"])
    data_yaml = resolve_path(source_cfg.get("data_yaml", source_root / "data.yaml"))
    output_root = resolve_path(output_cfg["root"])

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset root does not exist: {source_root}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data_yaml}")

    source_cfg["root"] = str(source_root)
    source_cfg["data_yaml"] = str(data_yaml)
    output_cfg["root"] = str(output_root)
    output_cfg.setdefault("clean_output", True)

    runtime_cfg.setdefault("seed", 52)
    runtime_cfg.setdefault("num_aug_per_image", 3)
    runtime_cfg.setdefault("keep_original_train", True)
    runtime_cfg.setdefault("image_suffix", ".jpg")
    runtime_cfg.setdefault("label_suffix", ".txt")
    runtime_cfg.setdefault("max_attempts_per_augment", 12)

    geometry_cfg = require_mapping("geometry", config["geometry"])
    geometry_cfg.setdefault("border_mode", "reflect101")

    review_cfg = require_mapping("review", config["review"])
    review_cfg.setdefault("enabled", True)
    review_cfg.setdefault("per_template", 24)
    review_cfg.setdefault("export_dir", "analysis/review")
    return config


def load_dataset_metadata(data_yaml_path: Path) -> dict[str, Any]:
    metadata = load_yaml_file(data_yaml_path)
    required = ("kpt_shape", "nc", "names")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Dataset metadata missing keys {missing}: {data_yaml_path}")
    return metadata


def dry_run_output_root(output_root: Path) -> Path:
    return output_root.with_name(f"{output_root.name}.dryrun_preview")


class DatasetBuilder:
    def __init__(self, config: dict[str, Any], *, dry_run: bool, limit: int | None, visualize: bool) -> None:
        self.config = config
        self.dry_run = dry_run
        self.limit = limit
        self.visualize = visualize

        self.source_root = Path(config["source"]["root"]).resolve()
        self.data_yaml_path = Path(config["source"]["data_yaml"]).resolve()
        self.requested_output_root = Path(config["output"]["root"]).resolve()
        self.output_root = dry_run_output_root(self.requested_output_root) if dry_run else self.requested_output_root
        self.clean_output = bool(config["output"].get("clean_output", True))
        self.metadata = load_dataset_metadata(self.data_yaml_path)
        self.keypoint_count = int(self.metadata["kpt_shape"][0])

        self.keep_original_train = bool(config["runtime"]["keep_original_train"])
        self.num_aug_per_image = int(config["runtime"]["num_aug_per_image"])
        self.image_suffix = str(config["runtime"]["image_suffix"])
        self.label_suffix = str(config["runtime"]["label_suffix"])
        self.max_attempts_per_augment = int(config["runtime"]["max_attempts_per_augment"])
        self.rng_seed = int(config["runtime"]["seed"])

        import random

        self.rng = random.Random(self.rng_seed)
        self.review_enabled = bool(config["review"]["enabled"]) or bool(visualize)
        self.review_limit = int(config["review"]["per_template"])
        self.review_dir = Path(str(config["review"]["export_dir"]))

        self.accepted_records = []
        self.rejected_records = []
        self.review_rows: list[dict[str, Any]] = []
        self.review_counter: Counter[str] = Counter()
        self.accepted_by_template: Counter[str] = Counter()
        self.train_manifest: list[Path] = []
        self.valid_manifest: list[Path] = []
        self.valid_raw_manifest: list[Path] = []
        self.test_manifest: list[Path] = []
        self.total_requested_augmented = 0
        self.processed_train_count = 0
        self.raw_train_total = 0

    def split_dirs(self, split: str) -> tuple[Path, Path]:
        return self.output_root / split / "images", self.output_root / split / "labels"

    def planned_paths(self, split: str, image_name: str, label_name: str) -> tuple[Path, Path]:
        images_dir, labels_dir = self.split_dirs(split)
        return images_dir / image_name, labels_dir / label_name

    def relative_manifest_line(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.output_root))

    def prepare_output_root(self) -> None:
        if self.output_root.exists() and self.clean_output:
            shutil.rmtree(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def maybe_export_review(self, generated: GeneratedSample, output_stem: str) -> None:
        if not self.review_enabled:
            return
        template_name = generated.record.template
        if self.review_counter[template_name] >= self.review_limit:
            return

        review_dir = self.output_root / self.review_dir / template_name
        review_path = review_dir / f"{output_stem}.jpg"
        title = f"{template_name} {output_stem}"
        rendered = draw_review_overlay(generated.image, generated.annotation, title)
        write_image_bgr(review_path, rendered)
        self.review_counter[template_name] += 1
        self.review_rows.append(
            {
                "template": template_name,
                "source_sample_id": generated.record.source_sample_id,
                "output_image": generated.record.output_image,
                "review_image": str(review_path),
            }
        )

    def localize_split(self, split: str, target_split: str, manifest: list[Path]) -> None:
        for sample in collect_split_samples(self.source_root, split):
            target_image_path, target_label_path = self.planned_paths(target_split, sample.image_path.name, sample.label_path.name)
            if not self.dry_run:
                copy_sample(
                    sample=sample,
                    output_root=self.output_root,
                    split=target_split,
                    target_image_name=sample.image_path.name,
                    target_label_name=sample.label_path.name,
                )
            manifest.append(target_image_path.resolve())
            if self.dry_run:
                target_label_path.parent.mkdir(parents=True, exist_ok=True)

    def localize_eval_splits(self) -> None:
        self.localize_split("valid", "valid", self.valid_manifest)
        self.localize_split("valid", "valid_raw", self.valid_raw_manifest)
        self.localize_split("test", "test", self.test_manifest)

    def write_generated_sample(self, generated: GeneratedSample) -> None:
        output_image = Path(generated.record.output_image)
        output_label = Path(generated.record.output_label)
        if not self.dry_run:
            write_image_bgr(output_image, generated.image)
            label_text = object_annotation_to_yolo_pose_line(
                generated.annotation,
                image_width=generated.image.shape[1],
                image_height=generated.image.shape[0],
            )
            write_text_file(output_label, label_text)
        self.train_manifest.append(output_image.resolve())

    def process_train(self) -> None:
        train_samples = collect_split_samples(self.source_root, "train")
        self.raw_train_total = len(train_samples)
        if self.limit is not None:
            train_samples = train_samples[: self.limit]
        self.processed_train_count = len(train_samples)
        self.total_requested_augmented = self.processed_train_count * self.num_aug_per_image

        for sample in train_samples:
            if self.keep_original_train:
                target_image_path, _ = self.planned_paths("train", sample.image_path.name, sample.label_path.name)
                if not self.dry_run:
                    copy_sample(
                        sample=sample,
                        output_root=self.output_root,
                        split="train",
                        target_image_name=sample.image_path.name,
                        target_label_name=sample.label_path.name,
                    )
                self.train_manifest.append(target_image_path.resolve())

            annotation = read_single_pose_annotation(
                sample.image_path,
                sample.label_path,
                split=sample.split,
                sample_id=sample.stem,
                keypoint_count=self.keypoint_count,
            )

            for slot_index in range(self.num_aug_per_image):
                output_stem = f"{sample.stem}_aug_{slot_index:03d}"
                image_name = f"{output_stem}{self.image_suffix}"
                label_name = f"{output_stem}{self.label_suffix}"
                output_image_path, output_label_path = self.planned_paths("train", image_name, label_name)
                accepted = False
                for attempt_index in range(self.max_attempts_per_augment):
                    generated = generate_augmented_sample(
                        annotation,
                        self.config,
                        output_image=str(output_image_path),
                        output_label=str(output_label_path),
                        slot_index=slot_index,
                        attempt_index=attempt_index,
                        rng=self.rng,
                    )
                    if generated.record.valid:
                        self.accepted_records.append(generated.record)
                        self.accepted_by_template[generated.record.template] += 1
                        self.write_generated_sample(generated)
                        self.maybe_export_review(generated, output_stem)
                        accepted = True
                        break
                    self.rejected_records.append(generated.record)
                if not accepted:
                    raise RuntimeError(
                        f"Failed to generate source={sample.stem} slot={slot_index} after "
                        f"{self.max_attempts_per_augment} attempts. Refine the config or increase max_attempts_per_augment."
                    )

    def write_list_file(self, path: Path, items: list[Path]) -> None:
        lines = [self.relative_manifest_line(item) for item in items]
        write_text_file(path, "\n".join(lines) + ("\n" if lines else ""))

    def localized_counts(self) -> dict[str, dict[str, int]]:
        return {
            "train": {"images": len(self.train_manifest), "labels": len(self.train_manifest)},
            "valid": {"images": len(self.valid_manifest), "labels": len(self.valid_manifest)},
            "valid_raw": {"images": len(self.valid_raw_manifest), "labels": len(self.valid_raw_manifest)},
            "test": {"images": len(self.test_manifest), "labels": len(self.test_manifest)},
        }

    def write_dataset_yamls(self) -> None:
        if self.dry_run:
            return
        data_yaml = build_dataset_yaml_payload(
            self.metadata,
            train_value="train/images",
            val_value="valid/images",
            test_value="test/images",
        )
        raw_eval_yaml = build_dataset_yaml_payload(
            self.metadata,
            train_value="train/images",
            val_value="valid_raw/images",
            test_value="test/images",
        )
        write_yaml_file(self.output_root / "data.yaml", data_yaml)
        write_yaml_file(self.output_root / "data.raw_eval.yaml", raw_eval_yaml)

    def write_analysis(self) -> None:
        accepted_rows = [record.to_csv_row() for record in self.accepted_records]
        rejected_rows = [record.to_csv_row() for record in self.rejected_records]
        accepted_json = [record.to_json() for record in self.accepted_records]
        rejected_json = [record.to_json() for record in self.rejected_records]
        summary = {
            "source_root": str(self.source_root),
            "data_yaml": str(self.data_yaml_path),
            "requested_output_root": str(self.requested_output_root),
            "effective_output_root": str(self.output_root),
            "dry_run": self.dry_run,
            "seed": self.rng_seed,
            "limit": self.limit,
            "raw_train_total": self.raw_train_total,
            "processed_train_count": self.processed_train_count,
            "requested_augmented_count": self.total_requested_augmented,
            "accepted_augmented_count": len(self.accepted_records),
            "rejected_attempt_count": len(self.rejected_records),
            "accepted_by_template": dict(self.accepted_by_template),
            "localized_split_file_counts": self.localized_counts(),
        }
        write_csv_file(self.output_root / "analysis" / "augment_log.csv", accepted_rows)
        write_json_file(self.output_root / "analysis" / "augment_log.json", accepted_json)
        write_csv_file(self.output_root / "analysis" / "rejections.csv", rejected_rows)
        write_json_file(self.output_root / "analysis" / "rejections.json", rejected_json)
        write_csv_file(self.output_root / "analysis" / "review_manifest.csv", self.review_rows)
        write_json_file(self.output_root / "analysis" / "summary.json", summary)

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
            "requested_output_root": str(self.requested_output_root),
            "effective_output_root": str(self.output_root),
            "dry_run": self.dry_run,
            "processed_train_count": self.processed_train_count,
            "requested_augmented_count": self.total_requested_augmented,
            "accepted_augmented_count": len(self.accepted_records),
            "rejected_attempt_count": len(self.rejected_records),
        }


def main() -> int:
    try:
        args = parse_args()
        config = load_runtime_config(args)
        builder = DatasetBuilder(config, dry_run=bool(args.dry_run), limit=args.limit, visualize=bool(args.visualize))
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
