#!/usr/bin/env python3
"""Build a train image list containing only selected real-domain samples."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a train image list that only contains matching real-domain files."
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
        "--patterns",
        nargs="+",
        default=["frame_*.jpg", "1_*.jpg", "3_*.jpg"],
        help="One or more glob patterns inside <split>/images used to select real-only samples.",
    )
    parser.add_argument(
        "--output",
        default="data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.yolov8/train_real_only.txt",
        help="Output .txt file. Each line is an absolute image path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    images_dir = dataset_root / args.split / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")

    matched_set = {path.resolve() for pattern in args.patterns for path in images_dir.glob(pattern) if path.is_file()}
    matched_images = sorted(matched_set)
    if not matched_images:
        raise FileNotFoundError(f"No images matched patterns {args.patterns} in: {images_dir}")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(str(path) for path in matched_images) + "\n", encoding="utf-8")

    print(f"dataset_root={dataset_root}")
    print(f"images_dir={images_dir}")
    print(f"patterns={args.patterns}")
    print(f"matched_images={len(matched_images)}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
