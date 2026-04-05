#!/usr/bin/env python3
"""Build a weighted train image list that oversamples hard real-frame samples."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a train image list that repeats matching files to emphasize hard cases."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.yolov8",
        help="Dataset root containing train/valid/test directories.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to export. Usually 'train'.",
    )
    parser.add_argument(
        "--boost-patterns",
        nargs="+",
        default=None,
        help="One or more glob patterns inside <split>/images used to identify hard samples to oversample.",
    )
    parser.add_argument(
        "--boost-glob",
        action="append",
        default=[],
        help="Compatibility alias for adding a single boost glob. Can be repeated.",
    )
    parser.add_argument(
        "--extra-repeats",
        type=int,
        default=7,
        help="How many extra copies to append for each matched image.",
    )
    parser.add_argument(
        "--output",
        default="data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.yolov8/train_weighted_real_boost_r7.txt",
        help="Output .txt file. Each line is an absolute image path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    images_dir = dataset_root / args.split / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")

    base_images = sorted(path.resolve() for path in images_dir.iterdir() if path.is_file())
    if not base_images:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    boost_patterns = args.boost_patterns or args.boost_glob or ["frame_*.jpg", "1_*.jpg", "3_*.jpg"]
    lines = [str(path) for path in base_images]
    boosted_set = {path.resolve() for pattern in boost_patterns for path in images_dir.glob(pattern) if path.is_file()}
    boosted_images = sorted(boosted_set)
    if not boosted_images:
        raise FileNotFoundError(
            f"No images matched boost patterns {boost_patterns} in: {images_dir}"
        )
    for _ in range(args.extra_repeats):
        lines.extend(str(path) for path in boosted_images)

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"dataset_root={dataset_root}")
    print(f"images_dir={images_dir}")
    print(f"base_images={len(base_images)}")
    print(f"boost_patterns={boost_patterns}")
    print(f"boosted_images={len(boosted_images)}")
    print(f"extra_repeats={args.extra_repeats}")
    print(f"total_list_entries={len(lines)}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
