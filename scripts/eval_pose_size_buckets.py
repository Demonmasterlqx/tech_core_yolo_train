#!/usr/bin/env python3
"""Evaluate YOLO pose checkpoints on per-size GT buckets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from grayscale_preprocess import grayscale_prediction_sources, patch_ultralytics_dataset_grayscale
from pose_area_balance.metrics import AreaBin, annotation_area_ratio_after_letterbox, build_area_bins, bin_for_ratio
from pose_offline_aug.io import read_single_pose_annotation


def print_info(message: str) -> None:
    print(f"[eval_pose_size_buckets] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a pose checkpoint on GT size buckets.")
    parser.add_argument("--weights", required=True, help="Path to a YOLO pose .pt weights file.")
    parser.add_argument("--data", required=True, help="Dataset YAML path.")
    parser.add_argument("--split", default="valid", help="Dataset split to evaluate, e.g. valid or test.")
    parser.add_argument("--device", default="auto", help="Inference device. Use auto, cpu, 0, etc.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=8, help="Validation batch size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold.")
    parser.add_argument("--grayscale", action="store_true", help="Apply grayscale preprocessing during evaluation.")
    parser.add_argument(
        "--bins",
        default="0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050",
        help="Comma-separated bin edges for bbox area ratio after letterbox.",
    )
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args()


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return payload


def resolve_existing_path(raw_value: str, base_dir: Path) -> Path:
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def normalize_split_name(split: str) -> str:
    normalized = split.strip().lower()
    return "val" if normalized == "valid" else normalized


def resolve_split_source(data_config: dict[str, Any], data_path: Path, split: str) -> Path:
    if split not in data_config:
        raise ValueError(f"Dataset YAML does not define split '{split}': {data_path}")
    source_value = str(data_config[split])
    source_path = Path(source_value)
    if source_path.is_absolute():
        resolved = source_path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Split source does not exist for '{split}': {resolved}")
        return resolved

    candidates = [(data_path.parent / source_path).resolve()]
    if source_path.parts and source_path.parts[0] == "..":
        candidates.append((data_path.parent / Path(*source_path.parts[1:])).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Split source does not exist for '{split}'. Checked: {checked}")


def resolve_device(device_value: Any) -> str:
    device = str(device_value)
    if device != "auto":
        return device
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def parse_bins(raw_value: str) -> list[AreaBin]:
    parts = [float(item.strip()) for item in raw_value.split(",") if item.strip()]
    if len(parts) < 2:
        raise ValueError("At least two bin edges are required.")
    bins: list[AreaBin] = []
    for index, (lower, upper) in enumerate(zip(parts[:-1], parts[1:])):
        bins.append(AreaBin(index=index, lower=lower, upper=upper))
    return bins


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


def sanitize_for_dump(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_for_dump(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_dump(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def summarize_predictions(
    model,
    image_paths: list[Path],
    *,
    args: argparse.Namespace,
    keypoint_count: int,
) -> dict[str, Any]:
    if not image_paths:
        return {
            "count": 0,
            "misses": 0,
            "mean_pred_objects": None,
            "max_pred_objects": 0,
            "mean_best_iou": None,
            "mean_norm_kp_err": None,
        }

    with grayscale_prediction_sources(image_paths, args.grayscale) as predict_sources:
        predictions = model.predict(
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
    best_ious: list[float] = []
    kp_errors: list[float] = []

    for image_path, result in zip(image_paths, predictions):
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        annotation = read_single_pose_annotation(
            image_path,
            label_path,
            split="eval",
            sample_id=image_path.stem,
            keypoint_count=keypoint_count,
        ).object_annotation
        pred_count = 0 if result.boxes is None else len(result.boxes)
        total_pred_objects += pred_count
        max_pred_objects = max(max_pred_objects, pred_count)

        gt_box_xywhn = [
            annotation.bbox.center[0] / result.orig_shape[1],
            annotation.bbox.center[1] / result.orig_shape[0],
            annotation.bbox.width / result.orig_shape[1],
            annotation.bbox.height / result.orig_shape[0],
        ]
        gt_keypoints = [
            (
                keypoint.x / result.orig_shape[1] if keypoint.v > 0 else 0.0,
                keypoint.y / result.orig_shape[0] if keypoint.v > 0 else 0.0,
                float(keypoint.v),
            )
            for keypoint in annotation.keypoints
        ]

        if pred_count == 0 or result.keypoints is None:
            misses += 1
            continue

        pred_boxes = result.boxes.xywhn.detach().cpu().tolist()
        pred_keypoints = result.keypoints.xyn.detach().cpu().tolist()
        pred_keypoint_conf = result.keypoints.conf.detach().cpu().tolist() if result.keypoints.conf is not None else None
        scored = [
            (
                bbox_iou(gt_box_xywhn, box),
                box,
                kpts,
                pred_keypoint_conf[index] if pred_keypoint_conf else None,
            )
            for index, (box, kpts) in enumerate(zip(pred_boxes, pred_keypoints))
        ]
        best_iou, _, best_kpts_xy, best_conf = max(scored, key=lambda item: item[0])
        best_ious.append(best_iou)
        combined_pred_keypoints: list[tuple[float, float, float]] = []
        for index, (px, py) in enumerate(best_kpts_xy):
            confidence = float(best_conf[index]) if best_conf is not None else 1.0
            combined_pred_keypoints.append((float(px), float(py), confidence))
        kp_err = normalized_keypoint_error(gt_keypoints, combined_pred_keypoints)
        if kp_err is not None:
            kp_errors.append(kp_err)

    count = len(image_paths)
    return {
        "count": count,
        "misses": misses,
        "mean_pred_objects": total_pred_objects / count if count else None,
        "max_pred_objects": max_pred_objects,
        "mean_best_iou": (sum(best_ious) / len(best_ious)) if best_ious else None,
        "mean_norm_kp_err": (sum(kp_errors) / len(kp_errors)) if kp_errors else None,
    }


def run_bucket_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    weights_path = resolve_existing_path(args.weights, REPO_ROOT)
    data_path = resolve_existing_path(args.data, REPO_ROOT)
    data_cfg = load_yaml_file(data_path)
    normalized_split = normalize_split_name(args.split)
    source_path = resolve_split_source(data_cfg, data_path, normalized_split)
    label_dir = source_path.parent / "labels"
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

    image_paths = sorted(
        path.resolve()
        for path in source_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in split source: {source_path}")

    bins = parse_bins(args.bins)
    bucket_images: dict[str, list[Path]] = {area_bin.label: [] for area_bin in bins}
    ignored = 0
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        annotation = read_single_pose_annotation(
            image_path,
            label_path,
            split=normalized_split,
            sample_id=image_path.stem,
            keypoint_count=int(data_cfg["kpt_shape"][0]),
        )
        ratio = annotation_area_ratio_after_letterbox(
            annotation.object_annotation,
            image_width=annotation.image_width,
            image_height=annotation.image_height,
            imgsz=float(args.imgsz),
        )
        matched = bin_for_ratio(ratio, bins)
        if matched is None:
            ignored += 1
            continue
        bucket_images[matched.label].append(image_path)

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    if getattr(model, "task", None) != "pose":
        raise ValueError(f"Weights must resolve to a pose model, but got task={model.task}")

    bucket_results: list[dict[str, Any]] = []
    for area_bin in bins:
        paths = bucket_images[area_bin.label]
        if not paths:
            bucket_results.append({"bin": area_bin.label, "count": 0, "metrics": {}, "prediction_summary": {}})
            continue

        with TemporaryDirectory(prefix="tech_core_pose_bucket_") as temp_dir:
            temp_root = Path(temp_dir)
            list_path = temp_root / "bucket_images.txt"
            yaml_path = temp_root / "bucket_data.yaml"
            list_path.write_text("\n".join(str(path) for path in paths) + "\n", encoding="utf-8")
            bucket_yaml = {
                "train": "none",
                "val": str(list_path),
                "test": "none",
                "kpt_shape": list(data_cfg["kpt_shape"]),
                "nc": int(data_cfg["nc"]),
                "names": list(data_cfg["names"]),
            }
            if "flip_idx" in data_cfg:
                bucket_yaml["flip_idx"] = list(data_cfg["flip_idx"])
            yaml_path.write_text(yaml.safe_dump(bucket_yaml, sort_keys=False, allow_unicode=False), encoding="utf-8")
            with patch_ultralytics_dataset_grayscale(bool(args.grayscale)):
                val_results = model.val(
                    data=str(yaml_path),
                    split="val",
                    device=resolve_device(args.device),
                    imgsz=int(args.imgsz),
                    batch=int(args.batch),
                    plots=False,
                    save_json=False,
                    verbose=False,
                    project=str(temp_root / "val_runs"),
                    name=f"bucket_{area_bin.index:02d}",
                )

        prediction_summary = summarize_predictions(
            model,
            paths,
            args=args,
            keypoint_count=int(data_cfg["kpt_shape"][0]),
        )
        bucket_results.append(
            {
                "bin": area_bin.label,
                "count": len(paths),
                "metrics": sanitize_for_dump(getattr(val_results, "results_dict", {})),
                "prediction_summary": prediction_summary,
            }
        )

    return {
        "weights": str(weights_path),
        "data": str(data_path),
        "requested_split": args.split,
        "split": normalized_split,
        "imgsz": int(args.imgsz),
        "grayscale": bool(args.grayscale),
        "ignored_out_of_range_images": ignored,
        "bins": bucket_results,
    }


def main() -> int:
    args = parse_args()
    summary = run_bucket_evaluation(args)
    text = json.dumps(summary, indent=2, ensure_ascii=True)
    print(text)
    if args.output:
        output_path = resolve_existing_path(args.output, REPO_ROOT)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
