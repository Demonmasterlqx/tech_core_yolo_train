#!/usr/bin/env python3
"""Evaluate YOLO pose predictions on selected real-image subsets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grayscale_preprocess import grayscale_prediction_sources
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pose checkpoints on selected real-image subsets only.")
    parser.add_argument("--weights", required=True, help="Path to a YOLO pose .pt weights file.")
    parser.add_argument(
        "--data",
        default="data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.yolov8/data.yaml",
        help="Dataset YAML path.",
    )
    parser.add_argument("--split", default="valid", help="Dataset split to evaluate, e.g. train/valid/test.")
    parser.add_argument("--device", default="auto", help="Inference device. Use auto, cpu, 0, etc.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold.")
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["frame_*.jpg", "1_*.jpg", "3_*.jpg"],
        help="One or more glob patterns used to select the real-image subset.",
    )
    parser.add_argument("--grayscale", action="store_true", help="Convert prediction sources to grayscale before inference.")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args()


def resolve_existing_path(raw_value: str, base_dir: Path) -> Path:
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return content


def resolve_device(device_value: Any) -> str:
    device = str(device_value)
    if device != "auto":
        return device

    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def resolve_split_source(data_config: dict[str, Any], data_path: Path, split: str) -> Path:
    if split not in data_config:
        raise ValueError(f"Dataset YAML does not define split '{split}': {data_path}")

    source_value = str(data_config[split])
    source_path = Path(source_value)
    if source_path.is_absolute():
        return source_path.resolve()
    candidates = [(data_path.parent / source_path).resolve()]
    if source_path.parts and source_path.parts[0] == "..":
        candidates.append((data_path.parent / Path(*source_path.parts[1:])).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def normalize_split(split: str) -> str:
    normalized = split.strip().lower()
    if normalized == "valid":
        return "val"
    return normalized


def bbox_iou(first: list[float], second: list[float]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    ax1, ay1, ax2, ay2 = ax - aw / 2.0, ay - ah / 2.0, ax + aw / 2.0, ay + ah / 2.0
    bx1, by1, bx2, by2 = bx - bw / 2.0, by - bh / 2.0, bx + bw / 2.0, by + bh / 2.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def parse_gt_label(path: Path) -> tuple[list[float], list[tuple[float, float, float]]] | None:
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return None
    values = [float(item) for item in content.splitlines()[0].split()]
    box = values[1:5]
    keypoints = []
    for index in range(5, len(values), 3):
        if index + 2 >= len(values):
            break
        keypoints.append((values[index], values[index + 1], values[index + 2]))
    return box, keypoints


def normalized_keypoint_error(
    gt_keypoints: list[tuple[float, float, float]],
    pred_keypoints: list[tuple[float, float, float]],
) -> float | None:
    errors: list[float] = []
    for (gx, gy, gv), (px, py, _) in zip(gt_keypoints, pred_keypoints):
        if gv <= 0:
            continue
        errors.append(math.hypot(px - gx, py - gy))
    if not errors:
        return None
    return sum(errors) / len(errors)


def summarize_frame_subset(args: argparse.Namespace) -> dict[str, Any]:
    data_path = resolve_existing_path(args.data, REPO_ROOT)
    weights_path = resolve_existing_path(args.weights, REPO_ROOT)
    data_config = load_yaml_file(data_path)
    yaml_split = normalize_split(args.split)
    image_dir = resolve_split_source(data_config, data_path, yaml_split)
    label_dir = image_dir.parent / "labels"
    frame_paths = sorted({path for pattern in args.patterns for path in image_dir.glob(pattern)})
    if not frame_paths:
        raise FileNotFoundError(f"No files matched patterns {args.patterns} in {image_dir}")

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    with grayscale_prediction_sources(frame_paths, args.grayscale) as predict_sources:
        results = model.predict(
            source=predict_sources,
            device=resolve_device(args.device),
            imgsz=args.imgsz,
            conf=args.conf,
            save=False,
            save_txt=False,
            verbose=False,
        )

    misses = 0
    total_pred_objects = 0
    max_pred_objects = 0
    kp_errors: list[float] = []
    best_ious: list[float] = []
    worst_frames: list[dict[str, Any]] = []

    for frame_path, result in zip(frame_paths, results):
        gt = parse_gt_label(label_dir / f"{frame_path.stem}.txt")
        if gt is None:
            continue
        gt_box, gt_keypoints = gt
        pred_count = 0 if result.boxes is None else len(result.boxes)
        total_pred_objects += pred_count
        max_pred_objects = max(max_pred_objects, pred_count)

        if pred_count == 0 or result.keypoints is None:
            misses += 1
            worst_frames.append({"frame": frame_path.name, "reason": "missing_prediction"})
            continue

        pred_boxes = result.boxes.xywhn.detach().cpu().tolist()
        pred_keypoints = result.keypoints.xyn.detach().cpu().tolist()
        pred_keypoint_conf = result.keypoints.conf.detach().cpu().tolist() if result.keypoints.conf is not None else None
        scored = [(bbox_iou(gt_box, box), box, kpts, pred_keypoint_conf[index] if pred_keypoint_conf else None) for index, (box, kpts) in enumerate(zip(pred_boxes, pred_keypoints))]
        best_iou, _, best_kpts_xy, best_conf = max(scored, key=lambda item: item[0])
        best_ious.append(best_iou)

        combined_pred_keypoints: list[tuple[float, float, float]] = []
        for index, (px, py) in enumerate(best_kpts_xy):
            confidence = float(best_conf[index]) if best_conf is not None else 1.0
            combined_pred_keypoints.append((float(px), float(py), confidence))

        kp_err = normalized_keypoint_error(gt_keypoints, combined_pred_keypoints)
        if kp_err is not None:
            kp_errors.append(kp_err)
            worst_frames.append(
                {
                    "frame": frame_path.name,
                    "kp_err": kp_err,
                    "pred_objects": pred_count,
                    "best_iou": best_iou,
                }
            )

    frame_count = len(frame_paths)
    matched = len(kp_errors)
    summary = {
        "weights": str(weights_path),
        "data": str(data_path),
        "split": args.split,
        "yaml_split": yaml_split,
        "patterns": args.patterns,
        "grayscale": args.grayscale,
        "conf": args.conf,
        "frame_count": frame_count,
        "matched": matched,
        "misses": misses,
        "mean_pred_objects": total_pred_objects / frame_count,
        "max_pred_objects": max_pred_objects,
        "mean_norm_kp_err": (sum(kp_errors) / matched) if matched else None,
        "mean_best_iou": (sum(best_ious) / len(best_ious)) if best_ious else None,
        "worst_frames": sorted(
            worst_frames,
            key=lambda item: (
                item.get("reason") != "missing_prediction",
                -(item.get("kp_err") or 0.0),
                -(item.get("pred_objects") or 0),
            ),
        )[:5],
    }
    return summary


def main() -> int:
    args = parse_args()
    summary = summarize_frame_subset(args)
    text = json.dumps(summary, indent=2, ensure_ascii=True)
    print(text)
    if args.output:
        output_path = resolve_existing_path(args.output, REPO_ROOT)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
