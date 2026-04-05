#!/usr/bin/env python3
"""Run the fixed V8 color real-only YOLO11s tuning plan."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
POSE_RUNS_ROOT = RUNS_ROOT / "pose"
POSE_TEST_ROOT = RUNS_ROOT / "pose_test"
PLAN_ROOT = RUNS_ROOT / "v8_color_realonly_plan"
PLAN_CONFIG_ROOT = PLAN_ROOT / "generated_configs"
PLAN_RESULT_ROOT = PLAN_ROOT / "candidate_results"
PLAN_SCAN_ROOT = PLAN_ROOT / "conf_scans"
PLAN_SUMMARY_ROOT = PLAN_ROOT / "summaries"
PLAN_LOG_ROOT = PLAN_ROOT / "monitor_logs"
DATASET_ROOT = REPO_ROOT / "data" / "Energy_Core_Position_Estimate.v8-add-blue-real-marker.yolov8"
DATA_REALONLY_YAML = DATASET_ROOT / "data.realonly.yaml"
TRAIN_REALONLY_LIST = DATASET_ROOT / "train_real_only.txt"
BASELINE_RUN_NAME = "v8_yolo11s_color_s9_1boost_e500"
BASELINE_CHECKPOINT = POSE_RUNS_ROOT / BASELINE_RUN_NAME / "weights" / "best.pt"
BASELINE_CONFIG_SOURCE = RUNS_ROOT / "v8_gray_realboost_plan" / "generated_configs" / f"{BASELINE_RUN_NAME}.train.yaml"
BASELINE_VALID_EVAL = POSE_TEST_ROOT / f"{BASELINE_RUN_NAME}_post_val_eval" / "metrics_summary.yaml"
BASELINE_TEST_EVAL = POSE_TEST_ROOT / f"{BASELINE_RUN_NAME}_post_test_eval" / "metrics_summary.yaml"
BASELINE_PREV_MODEL = REPO_ROOT / "runs" / "pose" / "v8_yolo11s_color_s8_1boost_e12" / "weights" / "best.pt"
BASELINE_DATA_YAML = DATASET_ROOT / "data.1boost_r40.yaml"
REAL_PATTERNS = ["frame_*.jpg", "1_*.jpg", "3_*.jpg"]
REAL_COUNTS = {"train": 45, "valid": 18, "test": 6}
CONF_GRID = [0.25, 0.40, 0.55, 0.70]
IMPORTANT_IMAGE_NAMES = [
    "frame_000002_jpg.rf.3f72539f8f9a752bd4c55082e6c25afb.jpg",
    "frame_000024_jpg.rf.272f46943a1b332728fdd74658dfaf4c.jpg",
    "1_002100_jpg.rf.44d7159796eb22408f100e040a82af1c.jpg",
    "1_001860_jpg.rf.45a18a99628a5c295b3d19ee4d5b52da.jpg",
    "frame_000055_jpg.rf.1d67580f016f922d919d63052ea72a04.jpg",
]
BEST_CONFIG_PATH = REPO_ROOT / "configs" / "energy_core_pose_v8_yolo11s_color_realonly_best.yaml"
SUMMARY_DOC_PATH = REPO_ROOT / "docs" / "v8_color_realonly_tuning_summary.md"
PATH_COMPARE_KEYS = {"data"}
FLOAT_TOLERANCE = 1e-9
COMMAND_ENV_OVERRIDES: dict[str, str] = {}
WANDB_RUNTIME: dict[str, Any] = {
    "enabled": True,
    "project": "tech-core-yolo-pose",
    "entity": None,
    "group": "v8-color-realonly-s",
}

COMMON_WANDB = {
    "enabled": True,
    "project": "tech-core-yolo-pose",
    "entity": None,
    "group": "v8-color-realonly-s",
}

R0_TRAIN = {
    "epochs": 500,
    "batch": 4,
    "imgsz": 960,
    "workers": 8,
    "patience": 50,
    "save": True,
    "save_period": -1,
    "cache": "ram",
    "plots": True,
    "exist_ok": False,
    "verbose": True,
    "optimizer": "AdamW",
    "lr0": 0.00001,
    "lrf": 0.2,
    "warmup_epochs": 1,
    "cos_lr": True,
    "pose": 36.0,
    "amp": False,
    "auto_augment": "none",
    "erasing": 0.0,
}

R0_AUGMENT = {
    "hsv_h": 0.01,
    "hsv_s": 0.04,
    "hsv_v": 0.02,
    "degrees": 0.0,
    "translate": 0.002,
    "scale": 0.02,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}

R1_TRAIN = {
    **copy.deepcopy(R0_TRAIN),
    "auto_augment": "randaugment",
    "erasing": 0.10,
}

R1_AUGMENT = {
    **copy.deepcopy(R0_AUGMENT),
    "hsv_s": 0.15,
    "hsv_v": 0.08,
    "degrees": 1.0,
    "translate": 0.03,
    "scale": 0.12,
    "shear": 1.0,
    "perspective": 0.0002,
    "mosaic": 0.15,
    "mixup": 0.05,
    "copy_paste": 0.05,
}


@dataclass(frozen=True)
class CandidatePlan:
    candidate_id: str
    run_name: str
    seed: int
    train: dict[str, Any]
    augment: dict[str, Any]


def print_info(message: str) -> None:
    print(f"[run_v8_color_realonly_plan] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed V8 color real-only tuning plan.")
    parser.add_argument("--device", default="auto", help="Device passed through to training and evaluation.")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["R0", "R1"],
        choices=["R0", "R1"],
        help="Which fixed candidates to execute.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Do not launch missing training jobs. Existing matching runs are still reused.",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Re-run evaluations even if matching summaries already exist.",
    )
    parser.add_argument("--wandb", dest="wandb_enabled", action="store_true", help="Enable W&B for newly launched training jobs.")
    parser.add_argument("--no-wandb", dest="wandb_enabled", action="store_false", help="Disable W&B for newly launched training jobs.")
    parser.set_defaults(wandb_enabled=True)
    parser.add_argument("--wandb-project", default="tech-core-yolo-pose", help="W&B project for orchestrated training runs.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team.")
    parser.add_argument("--wandb-group", default="v8-color-realonly-s", help="W&B group used for newly launched training jobs.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return content


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def run_command(cmd: list[str]) -> None:
    print_info(f"Running command: {' '.join(str(part) for part in cmd)}")
    command_env = os.environ.copy()
    command_env.update(COMMAND_ENV_OVERRIDES)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=command_env)


def resolve_compare_value(key: str, value: Any) -> Any:
    if key in PATH_COMPARE_KEYS and isinstance(value, str):
        return str((REPO_ROOT / value).resolve()) if not Path(value).is_absolute() else str(Path(value).resolve())
    return value


def values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(expected) == bool(actual)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=FLOAT_TOLERANCE)
    return expected == actual


def resolve_execution_device(requested_device: str) -> tuple[str, dict[str, str]]:
    normalized = requested_device.strip()
    if normalized and normalized != "auto":
        return normalized, {}

    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "cpu", {}

    candidates: list[tuple[int, int, int]] = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        raw_index, raw_mem, raw_util = [part.strip() for part in line.split(",")]
        candidates.append((int(raw_mem), int(raw_util), int(raw_index)))
    if not candidates:
        return "cpu", {}
    candidates.sort()
    return str(candidates[0][2]), {}


def candidate_plans() -> dict[str, CandidatePlan]:
    return {
        "R0": CandidatePlan(
            candidate_id="R0",
            run_name="v8_yolo11s_color_r0_realonly_base_e500",
            seed=42,
            train=copy.deepcopy(R0_TRAIN),
            augment=copy.deepcopy(R0_AUGMENT),
        ),
        "R1": CandidatePlan(
            candidate_id="R1",
            run_name="v8_yolo11s_color_r1_realonly_augopen_e500",
            seed=42,
            train=copy.deepcopy(R1_TRAIN),
            augment=copy.deepcopy(R1_AUGMENT),
        ),
    }


def subset_rank_tuple(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(summary["misses"]),
        float(summary["mean_pred_objects"]),
        int(summary["max_pred_objects"]),
        float(summary["mean_norm_kp_err"]) if summary["mean_norm_kp_err"] is not None else float("inf"),
        -float(summary["mean_best_iou"]) if summary["mean_best_iou"] is not None else 0.0,
    )


def real_aggregate_rank(result: dict[str, Any]) -> tuple[Any, ...]:
    valid = result["subset_valid"]
    test = result["subset_test"]
    valid_err = float(valid["mean_norm_kp_err"]) if valid["mean_norm_kp_err"] is not None else float("inf")
    test_err = float(test["mean_norm_kp_err"]) if test["mean_norm_kp_err"] is not None else float("inf")
    valid_iou = float(valid["mean_best_iou"]) if valid["mean_best_iou"] is not None else 0.0
    test_iou = float(test["mean_best_iou"]) if test["mean_best_iou"] is not None else 0.0
    valid_map = float(result["valid_eval"]["metrics"]["metrics/mAP50-95(P)"])
    test_map = float(result["test_eval"]["metrics"]["metrics/mAP50-95(P)"])
    return (
        int(valid["misses"]) + int(test["misses"]),
        (float(valid["mean_pred_objects"]) + float(test["mean_pred_objects"])) / 2.0,
        max(int(valid["max_pred_objects"]), int(test["max_pred_objects"])),
        (valid_err + test_err) / 2.0,
        -((valid_iou + test_iou) / 2.0),
        -((valid_map + test_map) / 2.0),
    )


def select_best_conf(conf_summaries: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    ordered = sorted(conf_summaries, key=lambda item: (subset_rank_tuple(item), float(item["conf"])))
    return float(ordered[0]["conf"]), ordered[0]


def build_real_only_dataset() -> dict[str, Any]:
    run_command(
        [
            sys.executable,
            "scripts/build_real_only_train_list.py",
            "--dataset-root",
            str(DATASET_ROOT),
            "--split",
            "train",
            "--patterns",
            *REAL_PATTERNS,
            "--output",
            str(TRAIN_REALONLY_LIST),
        ]
    )

    lines = [line.strip() for line in TRAIN_REALONLY_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != REAL_COUNTS["train"]:
        raise ValueError(f"Expected {REAL_COUNTS['train']} lines in {TRAIN_REALONLY_LIST}, got {len(lines)}")

    train_names = [Path(line).name for line in lines]
    if len(set(train_names)) != REAL_COUNTS["train"]:
        raise ValueError(f"Expected {REAL_COUNTS['train']} unique train real images, got {len(set(train_names))}")
    if not all(
        any(Path(line).name.startswith(prefix) for prefix in ("frame_", "1_", "3_")) for line in lines
    ):
        raise ValueError(f"{TRAIN_REALONLY_LIST} contains a non-real image entry.")

    split_counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        image_dir = DATASET_ROOT / split / "images"
        split_counts[split] = len({path.name for pattern in REAL_PATTERNS for path in image_dir.glob(pattern)})
        if split_counts[split] != REAL_COUNTS[split]:
            raise ValueError(f"Expected {REAL_COUNTS[split]} real images in {image_dir}, got {split_counts[split]}")

    data_yaml = load_yaml(DATA_REALONLY_YAML)
    if data_yaml.get("train") != TRAIN_REALONLY_LIST.name:
        raise ValueError(f"{DATA_REALONLY_YAML} must point train to {TRAIN_REALONLY_LIST.name}.")
    if data_yaml.get("val") != "../valid/images" or data_yaml.get("test") != "../test/images":
        raise ValueError(f"{DATA_REALONLY_YAML} must keep val/test on the full v8 splits.")

    summary = {
        "real_patterns": REAL_PATTERNS,
        "real_counts": split_counts,
        "train_real_only_count": len(lines),
        "train_real_only_list": str(TRAIN_REALONLY_LIST),
        "data_realonly_yaml": str(DATA_REALONLY_YAML),
    }
    write_yaml(PLAN_SUMMARY_ROOT / "dataset_validation.yaml", summary)
    write_json(PLAN_SUMMARY_ROOT / "dataset_validation.json", summary)
    return summary


def build_train_config(plan: CandidatePlan, device: str) -> dict[str, Any]:
    return {
        "model": str(BASELINE_CHECKPOINT),
        "data": str(DATA_REALONLY_YAML.relative_to(REPO_ROOT)),
        "device": device,
        "project": "runs/pose",
        "name": plan.run_name,
        "seed": plan.seed,
        "init_mode": "pretrained",
        "preprocess": {
            "grayscale": False,
        },
        "train": copy.deepcopy(plan.train),
        "augment": copy.deepcopy(plan.augment),
        "wandb": {
            **COMMON_WANDB,
            "enabled": bool(WANDB_RUNTIME["enabled"]),
            "project": WANDB_RUNTIME["project"],
            "entity": WANDB_RUNTIME["entity"],
            "group": WANDB_RUNTIME["group"],
            "batch_log_interval": 20,
            "tags": ["pose", "energy-core", "v8", "s", "color", "real-only", plan.candidate_id.lower()],
            "notes": f"Fixed V8 color real-only candidate {plan.candidate_id}.",
        },
        "post_eval": {
            "enabled": True,
            "project": "runs/pose_test",
            "splits": ["valid", "test"],
            "imgsz": 960,
            "batch": 4,
            "conf": 0.25,
            "line_width": 2,
        },
    }


def build_baseline_train_config() -> dict[str, Any]:
    return {
        "model": str(BASELINE_PREV_MODEL),
        "data": str(BASELINE_DATA_YAML.relative_to(REPO_ROOT)),
        "device": "3",
        "project": "runs/pose",
        "name": BASELINE_RUN_NAME,
        "seed": 42,
        "init_mode": "pretrained",
        "preprocess": {
            "grayscale": False,
        },
        "train": {
            "epochs": 500,
            "batch": 4,
            "imgsz": 960,
            "workers": 8,
            "patience": 80,
            "save": True,
            "save_period": -1,
            "cache": "ram",
            "plots": True,
            "exist_ok": False,
            "verbose": True,
            "optimizer": "AdamW",
            "lr0": 0.00001,
            "lrf": 0.2,
            "warmup_epochs": 1,
            "cos_lr": True,
            "pose": 36.0,
            "amp": False,
            "auto_augment": "none",
            "erasing": 0.0,
        },
        "augment": {
            "hsv_h": 0.01,
            "hsv_s": 0.04,
            "hsv_v": 0.02,
            "degrees": 0.0,
            "translate": 0.002,
            "scale": 0.02,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
        },
        "wandb": {
            **COMMON_WANDB,
            "enabled": True,
            "project": "tech-core-yolo-pose",
            "entity": None,
            "group": "v8-color-hardcase-s-long",
            "batch_log_interval": 20,
            "tags": ["pose", "energy-core", "v8", "s", "color", "1boost", "longrun"],
            "notes": "Long-run color continuation from S8 with epochs=500 and early stopping.",
        },
    }


def build_test_config(weights_path: Path, split: str, conf: float, run_name: str, device: str) -> dict[str, Any]:
    return {
        "weights": str(weights_path),
        "data": str(DATA_REALONLY_YAML),
        "split": split,
        "device": device,
        "project": "runs/pose_test",
        "name": f"{run_name}_realonly_plan",
        "imgsz": 960,
        "batch": 4,
        "conf": conf,
        "line_width": 2,
        "preprocess": {
            "grayscale": False,
        },
    }


def build_baseline_test_config(split: str, device: str) -> dict[str, Any]:
    return {
        "weights": str(BASELINE_CHECKPOINT),
        "data": str(BASELINE_DATA_YAML),
        "split": split,
        "device": device,
        "project": "runs/pose_test",
        "name": f"{BASELINE_RUN_NAME}_post",
        "imgsz": 960,
        "batch": 4,
        "conf": 0.25,
        "line_width": 2,
        "preprocess": {
            "grayscale": False,
        },
    }


def candidate_expected_args(config: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "model": config["model"],
        "data": str((REPO_ROOT / config["data"]).resolve()) if not Path(config["data"]).is_absolute() else str(Path(config["data"]).resolve()),
        "epochs": config["train"]["epochs"],
        "patience": config["train"]["patience"],
        "batch": config["train"]["batch"],
        "imgsz": config["train"]["imgsz"],
        "workers": config["train"]["workers"],
        "cache": config["train"]["cache"],
        "optimizer": config["train"]["optimizer"],
        "lr0": config["train"]["lr0"],
        "lrf": config["train"]["lrf"],
        "warmup_epochs": config["train"]["warmup_epochs"],
        "cos_lr": config["train"]["cos_lr"],
        "pose": config["train"]["pose"],
        "amp": config["train"]["amp"],
        "auto_augment": config["train"]["auto_augment"],
        "erasing": config["train"]["erasing"],
        "seed": config["seed"],
        "pretrained": True,
        "name": config["name"],
    }
    for key, value in config["augment"].items():
        expected[key] = value
    return expected


def run_matches_expected(run_dir: Path, expected_args: dict[str, Any]) -> tuple[bool, list[str]]:
    args_path = run_dir / "args.yaml"
    weights_path = run_dir / "weights" / "best.pt"
    if not args_path.exists():
        return False, [f"missing {args_path}"]
    if not weights_path.exists():
        return False, [f"missing {weights_path}"]
    actual_args = load_yaml(args_path)
    mismatches: list[str] = []
    for key, expected in expected_args.items():
        if key not in actual_args:
            mismatches.append(f"{key}:missing")
            continue
        actual = resolve_compare_value(key, actual_args[key])
        if not values_match(expected, actual):
            mismatches.append(f"{key}: expected={expected!r} actual={actual!r}")
    return not mismatches, mismatches


def candidate_run_dir(plan: CandidatePlan) -> Path:
    return POSE_RUNS_ROOT / plan.run_name


def candidate_weights_path(plan: CandidatePlan) -> Path:
    return candidate_run_dir(plan) / "weights" / "best.pt"


def ensure_candidate_run(plan: CandidatePlan, device: str, skip_train: bool) -> dict[str, Any]:
    config = build_train_config(plan, device=device)
    config_path = PLAN_CONFIG_ROOT / f"{plan.run_name}.train.yaml"
    write_yaml(config_path, config)

    run_dir = candidate_run_dir(plan)
    expected_args = candidate_expected_args(config)
    matches, mismatches = run_matches_expected(run_dir, expected_args) if run_dir.exists() else (False, ["run_dir_missing"])
    reused = matches
    if not matches and run_dir.exists() and (run_dir / "weights" / "best.pt").exists():
        raise ValueError(
            f"Existing run directory does not match fixed candidate {plan.candidate_id}: {run_dir}\n" + "\n".join(mismatches)
        )
    if not matches:
        if skip_train:
            raise RuntimeError(f"Candidate {plan.candidate_id} is missing and --skip-train was requested.")
        run_command([sys.executable, "train_pose.py", "--config", str(config_path)])
        matches, mismatches = run_matches_expected(run_dir, expected_args)
        if not matches:
            raise ValueError(
                f"Candidate {plan.candidate_id} still does not match expected args after training: {mismatches}"
            )

    best_checkpoint = candidate_weights_path(plan)
    if not best_checkpoint.exists():
        raise FileNotFoundError(f"Best checkpoint missing for {plan.candidate_id}: {best_checkpoint}")

    return {
        "train_config": config,
        "train_config_path": config_path,
        "run_dir": run_dir,
        "best_checkpoint": best_checkpoint,
        "reused_existing_run": reused,
    }


def subset_output_path(candidate_id: str, split: str, conf: float) -> Path:
    return PLAN_SCAN_ROOT / candidate_id / f"{split}_conf_{conf:.2f}.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_subset_summary(
    candidate_id: str,
    weights_path: Path,
    split: str,
    conf: float,
    device: str,
    force_eval: bool,
) -> dict[str, Any]:
    output_path = subset_output_path(candidate_id, split, conf)
    if output_path.exists() and not force_eval:
        summary = load_json(output_path)
        if (
            summary.get("weights") == str(weights_path)
            and summary.get("data") == str(DATA_REALONLY_YAML.resolve())
            and summary.get("split") == split
            and float(summary.get("conf")) == conf
            and bool(summary.get("grayscale")) is False
            and summary.get("patterns") == REAL_PATTERNS
        ):
            return summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            "scripts/eval_frame_subset.py",
            "--weights",
            str(weights_path),
            "--data",
            str(DATA_REALONLY_YAML),
            "--split",
            split,
            "--device",
            device,
            "--imgsz",
            "960",
            "--conf",
            f"{conf:.2f}",
            "--patterns",
            *REAL_PATTERNS,
            "--output",
            str(output_path),
        ]
    )
    return load_json(output_path)


def full_eval_paths(run_name: str, split: str) -> tuple[Path, Path]:
    normalized_split = "val" if split == "valid" else split
    base_name = f"{run_name}_realonly_plan"
    return (
        POSE_TEST_ROOT / f"{base_name}_{normalized_split}_eval" / "metrics_summary.yaml",
        POSE_TEST_ROOT / f"{base_name}_{normalized_split}_predict" / "predict_summary.yaml",
    )


def cleanup_partial_eval_outputs(metrics_path: Path, predict_path: Path) -> None:
    for directory in (metrics_path.parent, predict_path.parent):
        if directory.exists():
            print_info(f"Removing partial plan output: {directory}")
            shutil.rmtree(directory)


def load_eval_summaries(metrics_path: Path, predict_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return load_yaml(metrics_path), load_yaml(predict_path)


def ensure_full_eval(
    plan: CandidatePlan,
    weights_path: Path,
    split: str,
    conf: float,
    device: str,
    force_eval: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics_path, predict_path = full_eval_paths(plan.run_name, split)
    if metrics_path.exists() and predict_path.exists() and not force_eval:
        metrics_summary, predict_summary = load_eval_summaries(metrics_path, predict_path)
        if (
            metrics_summary.get("weights") == str(weights_path)
            and metrics_summary.get("data") == str(DATA_REALONLY_YAML.resolve())
            and metrics_summary.get("requested_split") == split
            and bool(metrics_summary.get("grayscale")) is False
            and predict_summary.get("weights") == str(weights_path)
        ):
            return metrics_summary, predict_summary

    if force_eval or metrics_path.parent.exists() or predict_path.parent.exists():
        cleanup_partial_eval_outputs(metrics_path=metrics_path, predict_path=predict_path)

    config = build_test_config(weights_path=weights_path, split=split, conf=conf, run_name=plan.run_name, device=device)
    config_path = PLAN_CONFIG_ROOT / f"{plan.run_name}.{split}.eval.yaml"
    write_yaml(config_path, config)
    run_command([sys.executable, "test_pose.py", "--config", str(config_path)])
    metrics_summary, predict_summary = load_eval_summaries(metrics_path, predict_path)
    return metrics_summary, predict_summary


def important_image_locations() -> dict[str, dict[str, Any]]:
    locations: dict[str, dict[str, Any]] = {}
    for image_name in IMPORTANT_IMAGE_NAMES:
        matches = sorted(DATASET_ROOT.glob(f"*/images/{image_name}"))
        if not matches:
            locations[image_name] = {"found": False}
            continue
        path = matches[0]
        locations[image_name] = {
            "found": True,
            "path": str(path),
            "split": path.parts[-3],
        }
    return locations


def candidate_manual_review(result: dict[str, Any], image_locations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    for image_name, location in image_locations.items():
        if not location.get("found"):
            outputs.append({"image": image_name, "found": False})
            continue
        split = location["split"]
        predict_dir = Path(result[f"{split}_predict"]["save_dir"])
        outputs.append(
            {
                "image": image_name,
                "found": True,
                "split": split,
                "dataset_path": location["path"],
                "prediction_path": str((predict_dir / image_name).resolve()),
            }
        )
    return {
        "important_images": outputs,
        "valid_worst_frames": result["subset_valid"]["worst_frames"],
        "test_worst_frames": result["subset_test"]["worst_frames"],
    }


def evaluate_candidate(
    plan: CandidatePlan,
    runtime: dict[str, Any],
    device: str,
    force_eval: bool,
    image_locations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary_path = PLAN_RESULT_ROOT / f"{plan.candidate_id}.yaml"
    if summary_path.exists() and not force_eval:
        existing = load_yaml(summary_path)
        if existing.get("best_checkpoint") == str(runtime["best_checkpoint"]):
            return existing

    valid_conf_summaries = [
        ensure_subset_summary(
            candidate_id=plan.candidate_id,
            weights_path=runtime["best_checkpoint"],
            split="valid",
            conf=conf,
            device=device,
            force_eval=force_eval,
        )
        for conf in CONF_GRID
    ]
    selected_conf, selected_valid_summary = select_best_conf(valid_conf_summaries)

    valid_eval, valid_predict = ensure_full_eval(
        plan=plan,
        weights_path=runtime["best_checkpoint"],
        split="valid",
        conf=selected_conf,
        device=device,
        force_eval=force_eval,
    )
    test_eval, test_predict = ensure_full_eval(
        plan=plan,
        weights_path=runtime["best_checkpoint"],
        split="test",
        conf=selected_conf,
        device=device,
        force_eval=force_eval,
    )
    subset_test = ensure_subset_summary(
        candidate_id=plan.candidate_id,
        weights_path=runtime["best_checkpoint"],
        split="test",
        conf=selected_conf,
        device=device,
        force_eval=force_eval,
    )

    summary = {
        "candidate_id": plan.candidate_id,
        "run_name": plan.run_name,
        "reused_existing_run": runtime["reused_existing_run"],
        "train_config_path": str(runtime["train_config_path"]),
        "train_config": sanitize(runtime["train_config"]),
        "run_dir": str(runtime["run_dir"]),
        "best_checkpoint": str(runtime["best_checkpoint"]),
        "selected_conf": selected_conf,
        "selected_conf_valid_summary": selected_valid_summary,
        "conf_scan_valid": valid_conf_summaries,
        "valid_eval": valid_eval,
        "test_eval": test_eval,
        "valid_predict": valid_predict,
        "test_predict": test_predict,
        "subset_valid": selected_valid_summary,
        "subset_test": subset_test,
    }
    summary["manual_review"] = candidate_manual_review(summary, image_locations=image_locations)
    write_yaml(summary_path, sanitize(summary))
    write_json(summary_path.with_suffix(".json"), sanitize(summary))
    return summary


def current_pose_map(result: dict[str, Any], split: str) -> float:
    key = "valid_eval" if split == "valid" else "test_eval"
    return float(result[key]["metrics"]["metrics/mAP50-95(P)"])


def evaluate_baseline_subset(split: str, device: str, force_eval: bool) -> dict[str, Any]:
    return ensure_subset_summary(
        candidate_id="BASELINE",
        weights_path=BASELINE_CHECKPOINT,
        split=split,
        conf=0.25,
        device=device,
        force_eval=force_eval,
    )


def baseline_full_eval_paths(split: str) -> tuple[Path, Path]:
    normalized_split = "val" if split == "valid" else split
    return (
        POSE_TEST_ROOT / f"{BASELINE_RUN_NAME}_post_{normalized_split}_eval" / "metrics_summary.yaml",
        POSE_TEST_ROOT / f"{BASELINE_RUN_NAME}_post_{normalized_split}_predict" / "predict_summary.yaml",
    )


def ensure_baseline_full_eval(split: str, device: str, force_eval: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics_path, predict_path = baseline_full_eval_paths(split)
    if metrics_path.exists() and predict_path.exists() and not force_eval:
        return load_eval_summaries(metrics_path, predict_path)

    if force_eval or metrics_path.parent.exists() or predict_path.parent.exists():
        cleanup_partial_eval_outputs(metrics_path=metrics_path, predict_path=predict_path)

    config = build_baseline_test_config(split=split, device=device)
    config_path = PLAN_CONFIG_ROOT / f"{BASELINE_RUN_NAME}.{split}.baseline.eval.yaml"
    write_yaml(config_path, config)
    run_command([sys.executable, "test_pose.py", "--config", str(config_path)])
    return load_eval_summaries(metrics_path, predict_path)


def baseline_reference(device: str, force_eval: bool) -> dict[str, Any]:
    train_config = load_yaml(BASELINE_CONFIG_SOURCE) if BASELINE_CONFIG_SOURCE.exists() else build_baseline_train_config()
    valid_eval, _ = ensure_baseline_full_eval(split="valid", device=device, force_eval=force_eval)
    test_eval, _ = ensure_baseline_full_eval(split="test", device=device, force_eval=force_eval)
    subset_valid = evaluate_baseline_subset(split="valid", device=device, force_eval=force_eval)
    subset_test = evaluate_baseline_subset(split="test", device=device, force_eval=force_eval)
    return {
        "candidate_id": "BASELINE",
        "run_name": BASELINE_RUN_NAME,
        "best_checkpoint": str(BASELINE_CHECKPOINT),
        "selected_conf": 0.25,
        "train_config": sanitize(train_config),
        "train_config_path": str(BASELINE_CONFIG_SOURCE),
        "valid_eval": valid_eval,
        "test_eval": test_eval,
        "subset_valid": subset_valid,
        "subset_test": subset_test,
    }


def candidate_acceptance(result: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    valid_map = current_pose_map(result, "valid")
    test_map = current_pose_map(result, "test")
    baseline_valid_map = current_pose_map(baseline, "valid")
    baseline_test_map = current_pose_map(baseline, "test")
    if valid_map < baseline_valid_map - 0.01:
        reasons.append(f"valid full-map guard failed: {valid_map:.6f} < {baseline_valid_map - 0.01:.6f}")
    if test_map < baseline_test_map - 0.01:
        reasons.append(f"test full-map guard failed: {test_map:.6f} < {baseline_test_map - 0.01:.6f}")

    valid = result["subset_valid"]
    test = result["subset_test"]
    baseline_valid = baseline["subset_valid"]
    baseline_test = baseline["subset_test"]
    if int(valid["misses"]) != 0:
        reasons.append(f"real valid misses must be 0, got {valid['misses']}")
    if float(valid["mean_pred_objects"]) > 1.0:
        reasons.append(f"real valid mean_pred_objects must be <= 1.0, got {valid['mean_pred_objects']}")
    if int(valid["max_pred_objects"]) > 1:
        reasons.append(f"real valid max_pred_objects must be <= 1, got {valid['max_pred_objects']}")
    if int(test["misses"]) != 0:
        reasons.append(f"real test misses must be 0, got {test['misses']}")
    if int(test["max_pred_objects"]) > 2:
        reasons.append(f"real test max_pred_objects must be <= 2, got {test['max_pred_objects']}")

    improved = False
    valid_err = float(valid["mean_norm_kp_err"]) if valid["mean_norm_kp_err"] is not None else float("inf")
    test_err = float(test["mean_norm_kp_err"]) if test["mean_norm_kp_err"] is not None else float("inf")
    baseline_valid_err = (
        float(baseline_valid["mean_norm_kp_err"]) if baseline_valid["mean_norm_kp_err"] is not None else float("inf")
    )
    baseline_test_err = (
        float(baseline_test["mean_norm_kp_err"]) if baseline_test["mean_norm_kp_err"] is not None else float("inf")
    )
    if valid_err < baseline_valid_err:
        improved = True
    if test_err < baseline_test_err:
        improved = True
    if float(test["mean_pred_objects"]) < float(baseline_test["mean_pred_objects"]):
        improved = True
    if not improved:
        reasons.append("candidate must strictly improve at least one baseline real metric")

    return (not reasons), reasons if reasons else ["candidate accepted"]


def best_config_payload(best_result: dict[str, Any]) -> dict[str, Any]:
    train_config = copy.deepcopy(best_result["train_config"])
    train_config["recommended_inference"] = {
        "conf": float(best_result["selected_conf"]),
    }
    train_config["best_checkpoint"] = best_result["best_checkpoint"]
    train_config["selection_summary"] = {
        "accepted": bool(best_result.get("accepted", False)),
        "valid_mAP50_95_pose": float(best_result["valid_eval"]["metrics"]["metrics/mAP50-95(P)"]),
        "test_mAP50_95_pose": float(best_result["test_eval"]["metrics"]["metrics/mAP50-95(P)"]),
        "valid_real_frame_count": int(best_result["subset_valid"]["frame_count"]),
        "valid_real_misses": int(best_result["subset_valid"]["misses"]),
        "valid_real_mean_pred_objects": float(best_result["subset_valid"]["mean_pred_objects"]),
        "valid_real_max_pred_objects": int(best_result["subset_valid"]["max_pred_objects"]),
        "valid_real_mean_norm_kp_err": best_result["subset_valid"]["mean_norm_kp_err"],
        "valid_real_mean_best_iou": best_result["subset_valid"]["mean_best_iou"],
        "test_real_frame_count": int(best_result["subset_test"]["frame_count"]),
        "test_real_misses": int(best_result["subset_test"]["misses"]),
        "test_real_mean_pred_objects": float(best_result["subset_test"]["mean_pred_objects"]),
        "test_real_max_pred_objects": int(best_result["subset_test"]["max_pred_objects"]),
        "test_real_mean_norm_kp_err": best_result["subset_test"]["mean_norm_kp_err"],
        "test_real_mean_best_iou": best_result["subset_test"]["mean_best_iou"],
    }
    return train_config


def build_summary_doc(
    dataset_summary: dict[str, Any],
    baseline: dict[str, Any],
    candidate_results: list[dict[str, Any]],
    best_result: dict[str, Any],
) -> str:
    lines = [
        "# V8 彩色 Real-Only 调参总结",
        "",
        "本文记录了基于 `v8_yolo11s_color_s9_1boost_e500` 的彩色 real-only 迁移训练路线、候选结果和最终胜者。",
        "",
        "## 数据与基线",
        "",
        f"- real-only 训练集: `{TRAIN_REALONLY_LIST}`",
        f"- 真实子集模式: `{', '.join(REAL_PATTERNS)}`",
        f"- 真实子集计数: `train={dataset_summary['real_counts']['train']}` / `valid={dataset_summary['real_counts']['valid']}` / `test={dataset_summary['real_counts']['test']}`",
        f"- 基线 checkpoint: `{BASELINE_CHECKPOINT}`",
        f"- 基线全量指标: `valid mAP50-95(P)={current_pose_map(baseline, 'valid'):.6f}` / `test mAP50-95(P)={current_pose_map(baseline, 'test'):.6f}`",
        f"- 基线 real `valid`: `misses={baseline['subset_valid']['misses']}`, `mean_pred_objects={baseline['subset_valid']['mean_pred_objects']:.6f}`, `mean_norm_kp_err={baseline['subset_valid']['mean_norm_kp_err']:.6f}`",
        f"- 基线 real `test`: `misses={baseline['subset_test']['misses']}`, `mean_pred_objects={baseline['subset_test']['mean_pred_objects']:.6f}`, `mean_norm_kp_err={baseline['subset_test']['mean_norm_kp_err']:.6f}`",
        "",
        "## 候选结果",
        "",
    ]
    for result in candidate_results:
        lines.append(
            f"- `{result['candidate_id']}` / `{result['run_name']}`: "
            f"`accepted={result['accepted']}`, "
            f"`conf={result['selected_conf']:.2f}`, "
            f"`valid_real_misses={result['subset_valid']['misses']}`, "
            f"`valid_real_mean_pred_objects={result['subset_valid']['mean_pred_objects']:.6f}`, "
            f"`valid_real_mean_norm_kp_err={result['subset_valid']['mean_norm_kp_err']:.6f}`, "
            f"`test_real_misses={result['subset_test']['misses']}`, "
            f"`test_real_mean_pred_objects={result['subset_test']['mean_pred_objects']:.6f}`, "
            f"`test_real_mean_norm_kp_err={result['subset_test']['mean_norm_kp_err']:.6f}`, "
            f"`valid_mAP50-95(P)={current_pose_map(result, 'valid'):.6f}`, "
            f"`test_mAP50-95(P)={current_pose_map(result, 'test'):.6f}`"
        )
    lines.extend(
        [
            "",
            "## 最终选择",
            "",
            f"- 胜者: `{best_result['candidate_id']}` (`{best_result['run_name']}`)",
            f"- 最佳 checkpoint: `{best_result['best_checkpoint']}`",
            f"- 推荐 `conf`: `{best_result['selected_conf']:.2f}`",
            f"- `accepted={best_result.get('accepted', False)}`",
            f"- `valid mAP50-95(P)={current_pose_map(best_result, 'valid'):.6f}`, `test mAP50-95(P)={current_pose_map(best_result, 'test'):.6f}`",
            f"- real `valid`: `misses={best_result['subset_valid']['misses']}`, `mean_pred_objects={best_result['subset_valid']['mean_pred_objects']:.6f}`, `max_pred_objects={best_result['subset_valid']['max_pred_objects']}`, `mean_norm_kp_err={best_result['subset_valid']['mean_norm_kp_err']:.6f}`, `mean_best_iou={best_result['subset_valid']['mean_best_iou']:.6f}`",
            f"- real `test`: `misses={best_result['subset_test']['misses']}`, `mean_pred_objects={best_result['subset_test']['mean_pred_objects']:.6f}`, `max_pred_objects={best_result['subset_test']['max_pred_objects']}`, `mean_norm_kp_err={best_result['subset_test']['mean_norm_kp_err']:.6f}`, `mean_best_iou={best_result['subset_test']['mean_best_iou']:.6f}`",
            "",
            "## 必查图片",
            "",
        ]
    )
    manual_review = best_result.get("manual_review", {})
    for item in manual_review.get("important_images", []):
        if not item.get("found"):
            lines.append(f"- `{item['image']}`: 未找到")
            continue
        lines.append(f"- `{item['image']}`: `{item['split']}` -> `{item['prediction_path']}`")
    lines.extend(
        [
            "",
            "## 最差样本",
            "",
            f"- real `valid` 最差 5 张: `{json.dumps(best_result['subset_valid']['worst_frames'], ensure_ascii=True)}`",
            f"- real `test` 最差 5 张: `{json.dumps(best_result['subset_test']['worst_frames'], ensure_ascii=True)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    WANDB_RUNTIME.update(
        {
            "enabled": bool(args.wandb_enabled),
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "group": args.wandb_group,
        }
    )
    effective_device, env_overrides = resolve_execution_device(args.device)
    COMMAND_ENV_OVERRIDES.update(env_overrides)
    print_info(
        f"W&B training logging {'enabled' if WANDB_RUNTIME['enabled'] else 'disabled'} "
        f"(project={WANDB_RUNTIME['project']}, group={WANDB_RUNTIME['group']})."
    )
    print_info(f"Using device '{effective_device}' for training and evaluation.")

    PLAN_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    PLAN_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    PLAN_SCAN_ROOT.mkdir(parents=True, exist_ok=True)
    PLAN_SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    PLAN_LOG_ROOT.mkdir(parents=True, exist_ok=True)

    dataset_summary = build_real_only_dataset()
    image_locations = important_image_locations()
    baseline = baseline_reference(device=effective_device, force_eval=args.force_eval)
    write_yaml(PLAN_SUMMARY_ROOT / "baseline.yaml", sanitize(baseline))
    write_json(PLAN_SUMMARY_ROOT / "baseline.json", sanitize(baseline))

    plans = candidate_plans()
    candidate_results: list[dict[str, Any]] = []
    for candidate_id in args.candidates:
        plan = plans[candidate_id]
        runtime = ensure_candidate_run(plan=plan, device=effective_device, skip_train=args.skip_train)
        result = evaluate_candidate(
            plan=plan,
            runtime=runtime,
            device=effective_device,
            force_eval=args.force_eval,
            image_locations=image_locations,
        )
        accepted, reasons = candidate_acceptance(result, baseline=baseline)
        result["accepted"] = accepted
        result["acceptance_reasons"] = reasons
        write_yaml(PLAN_RESULT_ROOT / f"{candidate_id}.yaml", sanitize(result))
        write_json(PLAN_RESULT_ROOT / f"{candidate_id}.json", sanitize(result))
        candidate_results.append(result)

    accepted_results = [result for result in candidate_results if result["accepted"]]
    if accepted_results:
        best_result = sorted(accepted_results, key=real_aggregate_rank)[0]
    else:
        baseline["accepted"] = False
        baseline["acceptance_reasons"] = ["No candidate passed the real-only acceptance and full-map guard checks."]
        best_result = baseline

    best_payload = best_config_payload(best_result)
    write_yaml(BEST_CONFIG_PATH, sanitize(best_payload))

    summary_text = build_summary_doc(
        dataset_summary=dataset_summary,
        baseline=baseline,
        candidate_results=candidate_results,
        best_result=best_result,
    )
    SUMMARY_DOC_PATH.write_text(summary_text, encoding="utf-8")

    overall_summary = {
        "dataset": dataset_summary,
        "baseline": baseline,
        "candidates": candidate_results,
        "best_result": best_result,
        "best_config": str(BEST_CONFIG_PATH),
        "summary_doc": str(SUMMARY_DOC_PATH),
    }
    write_yaml(PLAN_SUMMARY_ROOT / "overall.yaml", sanitize(overall_summary))
    write_json(PLAN_SUMMARY_ROOT / "overall.json", sanitize(overall_summary))
    print_info("Plan execution completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
