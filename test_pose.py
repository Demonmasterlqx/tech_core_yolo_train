#!/usr/bin/env python3
"""Evaluate YOLO pose weights on a dataset split and save prediction artifacts."""

from __future__ import annotations

import argparse
import cv2
import json
import numpy as np
import os
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent
LOCAL_YOLO_CONFIG_ROOT = (REPO_ROOT / ".ultralytics").resolve()
(LOCAL_YOLO_CONFIG_ROOT / "Ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_YOLO_CONFIG_ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KEYPOINT_NAME_CONF_THRES = 0.25
LABEL_BOX_COLOR = (32, 32, 32)
LABEL_BORDER_COLOR = (96, 96, 96)
LABEL_TEXT_COLOR = (255, 255, 255)
LABEL_CONNECTOR_COLOR = (160, 160, 160)


def print_info(message: str) -> None:
    print(f"[test_pose] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO pose weights and export test-set predictions.")
    parser.add_argument("--weights", required=True, help="Path to a trained YOLO pose .pt weights file.")
    parser.add_argument(
        "--data",
        default="data/Energy_Core_Position_Estimate.v6i.yolov8/data.yaml",
        help="Dataset YAML path.",
    )
    parser.add_argument("--split", default="test", help="Dataset split to evaluate and predict on.")
    parser.add_argument("--device", default="auto", help="Inference device. Use auto, cpu, 0, 0,1, etc.")
    parser.add_argument("--project", default="runs/pose_test", help="Base directory for evaluation artifacts.")
    parser.add_argument("--name", help="Base run name. Defaults to the parent training run name.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=8, help="Validation batch size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold.")
    parser.add_argument("--line-width", dest="line_width", type=int, default=2, help="Prediction line width.")
    return parser.parse_args()


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return content


def looks_like_local_path(value: str) -> bool:
    path = Path(value)
    return path.is_absolute() or value.startswith(".") or "/" in value or "\\" in value


def resolve_existing_path(raw_value: str, base_dir: Path) -> Path:
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def resolve_device(device_value: Any) -> str:
    device = str(device_value)
    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        print_info("CUDA detected, using device=0.")
        return "0"

    print_info("CUDA not available, falling back to CPU.")
    return "cpu"


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


def derive_run_name(weights_path: Path, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name

    if weights_path.parent.name == "weights" and len(weights_path.parents) >= 2:
        return weights_path.parents[1].name
    return weights_path.stem


def count_saved_images(save_dir: Path) -> int:
    if not save_dir.exists():
        return 0
    return sum(1 for path in save_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def count_saved_labels(save_dir: Path) -> int:
    labels_dir = save_dir / "labels"
    if not labels_dir.exists():
        return 0
    return sum(1 for path in labels_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")


def write_summary(base_path: Path, payload: dict[str, Any]) -> None:
    json_path = base_path.with_suffix(".json")
    yaml_path = base_path.with_suffix(".yaml")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def sanitize_for_dump(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_for_dump(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_dump(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def build_keypoint_names(data_config: dict[str, Any]) -> list[str]:
    raw_shape = data_config.get("kpt_shape")
    if not isinstance(raw_shape, (list, tuple)) or not raw_shape:
        raise ValueError("Dataset YAML must define kpt_shape as a non-empty list/tuple.")
    keypoint_count = int(raw_shape[0])
    return [str(index) for index in range(keypoint_count)]


def make_rect(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def clamp_rect(
    rect: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    clamped_x1 = min(max(0, x1), max(0, image_width - width))
    clamped_y1 = min(max(0, y1), max(0, image_height - height))
    return (clamped_x1, clamped_y1, clamped_x1 + width, clamped_y1 + height)


def rect_in_bounds(rect: tuple[int, int, int, int], image_width: int, image_height: int) -> bool:
    x1, y1, x2, y2 = rect
    return x1 >= 0 and y1 >= 0 and x2 <= image_width and y2 <= image_height


def rect_overlap_area(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> int:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def total_overlap_area(
    rect: tuple[int, int, int, int],
    blockers: list[tuple[int, int, int, int]],
) -> int:
    return sum(rect_overlap_area(rect, blocker) for blocker in blockers)


def expand_rect(rect: tuple[int, int, int, int], padding: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    return (x1 - padding, y1 - padding, x2 + padding, y2 + padding)


def build_label_candidates(
    x: int,
    y: int,
    box_width: int,
    box_height: int,
    gap: int,
) -> list[tuple[int, int, int, int]]:
    return [
        make_rect(x + gap, y - gap - box_height, x + gap + box_width, y - gap),
        make_rect(x - gap - box_width, y - gap - box_height, x - gap, y - gap),
        make_rect(x + gap, y + gap, x + gap + box_width, y + gap + box_height),
        make_rect(x - gap - box_width, y + gap, x - gap, y + gap + box_height),
        make_rect(x - box_width // 2, y - gap - box_height, x - box_width // 2 + box_width, y - gap),
        make_rect(x - box_width // 2, y + gap, x - box_width // 2 + box_width, y + gap + box_height),
    ]


def choose_label_rect(
    x: int,
    y: int,
    box_width: int,
    box_height: int,
    image_width: int,
    image_height: int,
    blockers: list[tuple[int, int, int, int]],
    gap: int,
) -> tuple[int, int, int, int]:
    candidates = build_label_candidates(x, y, box_width, box_height, gap)

    for candidate in candidates:
        if rect_in_bounds(candidate, image_width, image_height) and total_overlap_area(candidate, blockers) == 0:
            return candidate

    best_rect: tuple[int, int, int, int] | None = None
    best_score: tuple[int, int] | None = None
    for candidate in candidates:
        clamped = clamp_rect(candidate, image_width, image_height)
        overlap = total_overlap_area(clamped, blockers)
        clamp_penalty = abs(clamped[0] - candidate[0]) + abs(clamped[1] - candidate[1])
        score = (overlap, clamp_penalty)
        if best_score is None or score < best_score:
            best_rect = clamped
            best_score = score

    if best_rect is None:
        raise RuntimeError("Failed to choose a keypoint label position.")
    return best_rect


def nearest_point_on_rect(
    x: int,
    y: int,
    rect: tuple[int, int, int, int],
) -> tuple[int, int]:
    x1, y1, x2, y2 = rect
    return (min(max(x, x1), x2), min(max(y, y1), y2))


def render_prediction_image(
    result: Any,
    keypoint_names: list[str],
    line_width: int,
) -> np.ndarray:
    rendered = result.plot(line_width=line_width, kpt_line=True)
    image = np.ascontiguousarray(rendered.copy())

    if result.keypoints is None:
        return image

    keypoints_data = result.keypoints.data.detach().cpu().numpy()
    if keypoints_data.size == 0:
        return image

    image_height, image_width = image.shape[:2]
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(image_width, image_height) / 1400.0)
    font_thickness = max(1, int(round(font_scale * 2)))
    text_padding = max(2, font_thickness + 1)
    label_gap = max(10, line_width * 5)
    exclusion_radius = max(8, line_width * 4)

    visible_points: list[tuple[int, int, str]] = []
    point_blockers: list[tuple[int, int, int, int]] = []
    for keypoint_set in keypoints_data:
        for keypoint_index, keypoint in enumerate(keypoint_set):
            x_coord = int(round(float(keypoint[0])))
            y_coord = int(round(float(keypoint[1])))
            confidence = float(keypoint[2]) if len(keypoint) > 2 else 1.0
            if confidence < KEYPOINT_NAME_CONF_THRES:
                continue
            if x_coord <= 0 or y_coord <= 0 or x_coord >= image_width or y_coord >= image_height:
                continue
            name = keypoint_names[keypoint_index] if keypoint_index < len(keypoint_names) else str(keypoint_index)
            visible_points.append((x_coord, y_coord, name))
            point_blockers.append(
                make_rect(
                    x_coord - exclusion_radius,
                    y_coord - exclusion_radius,
                    x_coord + exclusion_radius,
                    y_coord + exclusion_radius,
                )
            )

    occupied_boxes: list[tuple[int, int, int, int]] = []
    for x_coord, y_coord, name in sorted(visible_points, key=lambda item: (item[1], item[0], item[2])):
        (text_width, text_height), baseline = cv2.getTextSize(name, font_face, font_scale, font_thickness)
        box_width = text_width + text_padding * 2
        box_height = text_height + baseline + text_padding * 2
        blockers = occupied_boxes + point_blockers
        rect = choose_label_rect(
            x=x_coord,
            y=y_coord,
            box_width=box_width,
            box_height=box_height,
            image_width=image_width,
            image_height=image_height,
            blockers=blockers,
            gap=label_gap,
        )

        anchor_x, anchor_y = nearest_point_on_rect(x_coord, y_coord, rect)
        if abs(anchor_x - x_coord) > 2 or abs(anchor_y - y_coord) > 2:
            cv2.line(
                image,
                (x_coord, y_coord),
                (anchor_x, anchor_y),
                LABEL_CONNECTOR_COLOR,
                thickness=max(1, line_width),
                lineType=cv2.LINE_AA,
            )

        cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), LABEL_BOX_COLOR, thickness=-1, lineType=cv2.LINE_AA)
        cv2.rectangle(
            image,
            (rect[0], rect[1]),
            (rect[2], rect[3]),
            LABEL_BORDER_COLOR,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        text_origin = (rect[0] + text_padding, rect[1] + text_padding + text_height)
        cv2.putText(
            image,
            name,
            text_origin,
            font_face,
            font_scale,
            LABEL_TEXT_COLOR,
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )
        occupied_boxes.append(expand_rect(rect, text_padding))

    return image


def save_prediction_artifacts(
    predictions: list[Any],
    predict_save_dir: Path,
    keypoint_names: list[str],
    line_width: int,
) -> None:
    labels_dir = predict_save_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    for result in predictions:
        image_name = Path(result.path).name
        image_path = predict_save_dir / image_name
        label_path = labels_dir / f"{Path(image_name).stem}.txt"
        if label_path.exists():
            label_path.unlink()
        result.save_txt(label_path)

        rendered = render_prediction_image(result, keypoint_names=keypoint_names, line_width=line_width)
        if not cv2.imwrite(str(image_path), rendered):
            raise IOError(f"Failed to save rendered prediction image: {image_path}")


def run_evaluation(args: argparse.Namespace) -> tuple[Path, Path]:
    weights_path = resolve_existing_path(args.weights, REPO_ROOT)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file does not exist: {weights_path}")

    data_path = resolve_existing_path(args.data, REPO_ROOT)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data_path}")

    data_config = load_yaml_file(data_path)
    if "kpt_shape" not in data_config:
        raise ValueError(f"Dataset YAML is missing 'kpt_shape': {data_path}")
    keypoint_names = build_keypoint_names(data_config)

    source_path = resolve_split_source(data_config, data_path, args.split)
    project_path = resolve_existing_path(args.project, REPO_ROOT)
    project_path.mkdir(parents=True, exist_ok=True)
    run_name = derive_run_name(weights_path, args.name)
    eval_name = f"{run_name}_{args.split}_eval"
    predict_name = f"{run_name}_{args.split}_predict"
    resolved_device = resolve_device(args.device)

    from ultralytics import YOLO
    from ultralytics.utils.files import increment_path

    model = YOLO(str(weights_path))
    if getattr(model, "task", None) != "pose":
        raise ValueError(f"Weights must resolve to a pose model, but got task={model.task}")

    print_info(f"Running evaluation on split='{args.split}' with weights={weights_path}")
    val_results = model.val(
        data=str(data_path),
        split=args.split,
        device=resolved_device,
        project=str(project_path),
        name=eval_name,
        imgsz=args.imgsz,
        batch=args.batch,
        plots=True,
        save_json=False,
        verbose=True,
    )
    eval_save_dir = Path(getattr(val_results, "save_dir", project_path / eval_name)).resolve()
    metrics_summary = {
        "weights": str(weights_path),
        "data": str(data_path),
        "split": args.split,
        "save_dir": str(eval_save_dir),
        "metrics": sanitize_for_dump(getattr(val_results, "results_dict", {})),
        "speed_ms_per_image": sanitize_for_dump(getattr(val_results, "speed", {})),
    }
    write_summary(eval_save_dir / "metrics_summary", metrics_summary)
    print_info(f"Saved evaluation metrics to: {eval_save_dir}")

    print_info(f"Saving rendered predictions for split='{args.split}' from source={source_path}")
    predict_save_dir = increment_path(project_path / predict_name, mkdir=True).resolve()
    predictions = model.predict(
        source=str(source_path),
        device=resolved_device,
        project=str(project_path),
        name=predict_save_dir.name,
        imgsz=args.imgsz,
        conf=args.conf,
        line_width=args.line_width,
        save=False,
        save_txt=False,
        verbose=True,
    )
    save_prediction_artifacts(
        predictions=predictions,
        predict_save_dir=predict_save_dir,
        keypoint_names=keypoint_names,
        line_width=args.line_width,
    )

    predict_summary = {
        "weights": str(weights_path),
        "source": str(source_path),
        "save_dir": str(predict_save_dir),
        "saved_image_files": count_saved_images(predict_save_dir),
        "saved_label_files": count_saved_labels(predict_save_dir),
    }
    write_summary(predict_save_dir / "predict_summary", predict_summary)
    print_info(f"Saved rendered predictions to: {predict_save_dir}")

    return eval_save_dir, predict_save_dir


def main() -> int:
    try:
        args = parse_args()
        eval_dir, predict_dir = run_evaluation(args)
        print_info(f"Finished successfully. Eval dir: {eval_dir}")
        print_info(f"Finished successfully. Predict dir: {predict_dir}")
        return 0
    except Exception as exc:
        print_info(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
