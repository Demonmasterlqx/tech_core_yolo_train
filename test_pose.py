#!/usr/bin/env python3
"""Evaluate YOLO pose weights on a dataset split and save prediction artifacts."""

from __future__ import annotations

import argparse
import json
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

    source_path = resolve_split_source(data_config, data_path, args.split)
    project_path = resolve_existing_path(args.project, REPO_ROOT)
    project_path.mkdir(parents=True, exist_ok=True)
    run_name = derive_run_name(weights_path, args.name)
    eval_name = f"{run_name}_{args.split}_eval"
    predict_name = f"{run_name}_{args.split}_predict"
    resolved_device = resolve_device(args.device)

    from ultralytics import YOLO

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
    predictions = model.predict(
        source=str(source_path),
        device=resolved_device,
        project=str(project_path),
        name=predict_name,
        imgsz=args.imgsz,
        conf=args.conf,
        line_width=args.line_width,
        save=True,
        save_txt=True,
        verbose=True,
    )
    if predictions:
        predict_save_dir = Path(getattr(predictions[0], "save_dir", project_path / predict_name)).resolve()
    else:
        predict_save_dir = (project_path / predict_name).resolve()

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
