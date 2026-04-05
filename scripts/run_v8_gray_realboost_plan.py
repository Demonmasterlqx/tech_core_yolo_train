#!/usr/bin/env python3
"""Run the fixed V8 grayscale + realboost YOLO11n/YOLO11s tuning plan."""

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

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
POSE_RUNS_ROOT = RUNS_ROOT / "pose"
POSE_TEST_ROOT = RUNS_ROOT / "pose_test"
PLAN_ROOT = RUNS_ROOT / "v8_gray_realboost_plan"
PLAN_CONFIG_ROOT = PLAN_ROOT / "generated_configs"
PLAN_RESULT_ROOT = PLAN_ROOT / "candidate_results"
PLAN_SCAN_ROOT = PLAN_ROOT / "conf_scans"
PLAN_SUMMARY_ROOT = PLAN_ROOT / "summaries"
DATASET_ROOT = REPO_ROOT / "data" / "Energy_Core_Position_Estimate.v8-add-blue-real-marker.yolov8"
DATA_YAML = DATASET_ROOT / "data.yaml"
DATA_R3_YAML = DATASET_ROOT / "data.realboost_r3.yaml"
DATA_R7_YAML = DATASET_ROOT / "data.realboost_r7.yaml"
TRAIN_LIST_R3 = DATASET_ROOT / "train_weighted_real_boost_r3.txt"
TRAIN_LIST_R7 = DATASET_ROOT / "train_weighted_real_boost_r7.txt"
REAL_PATTERNS = ["frame_*.jpg", "1_*.jpg", "3_*.jpg"]
REAL_COUNTS = {"train": 45, "valid": 18, "test": 6}
LIST_COUNTS = {"r3": 2811, "r7": 2991}
CONF_GRID = [0.25, 0.40, 0.55, 0.70]
IMPORTANT_IMAGE_NAMES = [
    "frame_000002_jpg.rf.3f72539f8f9a752bd4c55082e6c25afb.jpg",
    "frame_000024_jpg.rf.272f46943a1b332728fdd74658dfaf4c.jpg",
    "1_002100_jpg.rf.44d7159796eb22408f100e040a82af1c.jpg",
]
BEST_CONFIG_PATHS = {
    "n": REPO_ROOT / "configs" / "energy_core_pose_v8_yolo11n_gray_realboost_best.yaml",
    "s": REPO_ROOT / "configs" / "energy_core_pose_v8_yolo11s_gray_realboost_best.yaml",
}
SUMMARY_DOC_PATH = REPO_ROOT / "docs" / "v8_gray_realboost_tuning_summary.md"
PATH_COMPARE_KEYS = {"data"}
FLOAT_TOLERANCE = 1e-9
COMMAND_ENV_OVERRIDES: dict[str, str] = {}
WANDB_RUNTIME: dict[str, Any] = {
    "enabled": False,
    "project": "tech-core-yolo-pose",
    "entity": None,
    "group": None,
}


COMMON_WANDB = {
    "enabled": False,
    "project": "tech-core-yolo-pose",
    "entity": None,
    "group": None,
}

COMMON_GRAY_AUGMENT = {
    "degrees": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}

N0_TRAIN = {
    "epochs": 40,
    "batch": -1,
    "imgsz": 960,
    "workers": 8,
    "patience": 12,
    "save": True,
    "save_period": -1,
    "cache": "ram",
    "plots": True,
    "exist_ok": False,
    "verbose": True,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "warmup_epochs": 5,
    "cos_lr": True,
    "pose": 20.0,
    "amp": False,
    "auto_augment": "none",
    "erasing": 0.0,
}

N0_AUGMENT = {
    **COMMON_GRAY_AUGMENT,
    "hsv_h": 0.01,
    "hsv_s": 0.15,
    "hsv_v": 0.08,
    "translate": 0.05,
    "scale": 0.20,
}

N2_TRAIN_OVERRIDES = {
    "epochs": 18,
    "patience": 8,
    "lr0": 0.00005,
    "lrf": 0.2,
    "warmup_epochs": 1,
    "pose": 26.0,
}

N2_AUGMENT_OVERRIDES = {
    "translate": 0.01,
    "scale": 0.08,
    "hsv_s": 0.12,
    "hsv_v": 0.06,
}

S1_TRAIN = {
    "epochs": 20,
    "batch": -1,
    "imgsz": 960,
    "workers": 8,
    "patience": 10,
    "save": True,
    "save_period": -1,
    "cache": "ram",
    "plots": True,
    "exist_ok": False,
    "verbose": True,
    "optimizer": "AdamW",
    "lr0": 0.00015,
    "lrf": 0.2,
    "warmup_epochs": 1,
    "cos_lr": True,
    "pose": 24.0,
    "amp": False,
    "auto_augment": "none",
    "erasing": 0.0,
}

S1_AUGMENT = {
    **COMMON_GRAY_AUGMENT,
    "hsv_h": 0.01,
    "hsv_s": 0.18,
    "hsv_v": 0.10,
    "translate": 0.02,
    "scale": 0.10,
}

S3_TRAIN_OVERRIDES = {
    "epochs": 15,
    "patience": 8,
    "lr0": 0.000075,
    "lrf": 0.2,
    "pose": 24.0,
}

S4_TRAIN_OVERRIDES = {
    "epochs": 15,
    "patience": 8,
    "lr0": 0.00010,
    "pose": 26.0,
}

S4_AUGMENT_OVERRIDES = {
    "translate": 0.01,
    "scale": 0.08,
    "hsv_s": 0.12,
    "hsv_v": 0.06,
}


@dataclass(frozen=True)
class CandidatePlan:
    candidate_id: str
    lane: str
    run_name: str
    model_ref: str | None
    data_path: Path
    seed: int
    train: dict[str, Any]
    augment: dict[str, Any]
    parent_candidates: tuple[str, ...] = ()
    parent_mode: str | None = None


def print_info(message: str) -> None:
    print(f"[run_v8_gray_realboost_plan] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed V8 grayscale + realboost tuning plan.")
    parser.add_argument("--device", default="auto", help="Device override passed through to training and evaluation.")
    parser.add_argument(
        "--lanes",
        nargs="+",
        default=["n", "s"],
        choices=["n", "s"],
        help="Which fixed lanes to execute.",
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
    parser.add_argument(
        "--wandb-group",
        default="v8-gray-realboost",
        help="Base W&B group name used for newly launched training jobs.",
    )
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


def candidate_run_dir(plan: CandidatePlan) -> Path:
    return POSE_RUNS_ROOT / plan.run_name


def candidate_weights_path(plan: CandidatePlan) -> Path:
    return candidate_run_dir(plan) / "weights" / "best.pt"


def channel_gap_evidence(run_dir: Path) -> dict[str, Any]:
    image_path = run_dir / "train_batch0.jpg"
    if not image_path.exists():
        return {"image": str(image_path), "available": False}
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {"image": str(image_path), "available": False}
    blue, green, red = cv2.split(image)
    return {
        "image": str(image_path),
        "available": True,
        "mean_abs_b_g": float(np.mean(np.abs(blue.astype(np.int16) - green.astype(np.int16)))),
        "mean_abs_g_r": float(np.mean(np.abs(green.astype(np.int16) - red.astype(np.int16)))),
    }


def candidate_plans() -> dict[str, CandidatePlan]:
    return {
        "N0": CandidatePlan(
            candidate_id="N0",
            lane="n",
            run_name="v8_yolo11n_gray_n0_base_e40",
            model_ref="yolo11n-pose.pt",
            data_path=DATA_YAML,
            seed=3407,
            train=copy.deepcopy(N0_TRAIN),
            augment=copy.deepcopy(N0_AUGMENT),
        ),
        "N1": CandidatePlan(
            candidate_id="N1",
            lane="n",
            run_name="v8_yolo11n_gray_n1_r3_e40",
            model_ref="yolo11n-pose.pt",
            data_path=DATA_R3_YAML,
            seed=3407,
            train=copy.deepcopy(N0_TRAIN),
            augment=copy.deepcopy(N0_AUGMENT),
        ),
        "N2": CandidatePlan(
            candidate_id="N2",
            lane="n",
            run_name="v8_yolo11n_gray_n2_r3_refine_e18",
            model_ref=None,
            data_path=DATA_R3_YAML,
            seed=3407,
            train={**copy.deepcopy(N0_TRAIN), **N2_TRAIN_OVERRIDES},
            augment={**copy.deepcopy(N0_AUGMENT), **N2_AUGMENT_OVERRIDES},
            parent_candidates=("N1",),
            parent_mode="candidate",
        ),
        "N3": CandidatePlan(
            candidate_id="N3",
            lane="n",
            run_name="v8_yolo11n_gray_n3_r7_e40",
            model_ref="yolo11n-pose.pt",
            data_path=DATA_R7_YAML,
            seed=3407,
            train=copy.deepcopy(N0_TRAIN),
            augment=copy.deepcopy(N0_AUGMENT),
        ),
        "S0": CandidatePlan(
            candidate_id="S0",
            lane="s",
            run_name="v8_yolo11s_gray_s0_base_e40",
            model_ref="yolo11s-pose.pt",
            data_path=DATA_YAML,
            seed=42,
            train=copy.deepcopy(N0_TRAIN),
            augment=copy.deepcopy(N0_AUGMENT),
        ),
        "S1": CandidatePlan(
            candidate_id="S1",
            lane="s",
            run_name="v8_yolo11s_gray_s1_r3_e20",
            model_ref="yolo11s-pose.pt",
            data_path=DATA_R3_YAML,
            seed=42,
            train=copy.deepcopy(S1_TRAIN),
            augment=copy.deepcopy(S1_AUGMENT),
        ),
        "S2": CandidatePlan(
            candidate_id="S2",
            lane="s",
            run_name="v8_yolo11s_gray_s2_r7_e20",
            model_ref="yolo11s-pose.pt",
            data_path=DATA_R7_YAML,
            seed=42,
            train=copy.deepcopy(S1_TRAIN),
            augment=copy.deepcopy(S1_AUGMENT),
        ),
        "S3": CandidatePlan(
            candidate_id="S3",
            lane="s",
            run_name="v8_yolo11s_gray_s3_refine_e15",
            model_ref=None,
            data_path=DATA_R3_YAML,
            seed=42,
            train={**copy.deepcopy(S1_TRAIN), **S3_TRAIN_OVERRIDES},
            augment=copy.deepcopy(S1_AUGMENT),
            parent_candidates=("S1", "S2"),
            parent_mode="best_of",
        ),
        "S4": CandidatePlan(
            candidate_id="S4",
            lane="s",
            run_name="v8_yolo11s_gray_s4_suppress_e15",
            model_ref=None,
            data_path=DATA_R3_YAML,
            seed=42,
            train={**copy.deepcopy(S1_TRAIN), **S4_TRAIN_OVERRIDES},
            augment={**copy.deepcopy(S1_AUGMENT), **S4_AUGMENT_OVERRIDES},
            parent_candidates=("S1", "S2"),
            parent_mode="best_of",
        ),
    }


def lane_sequences() -> dict[str, list[str]]:
    return {"n": ["N0", "N1", "N2", "N3"], "s": ["S0", "S1", "S2", "S3", "S4"]}


def build_train_config(plan: CandidatePlan, model_ref: str, data_path: Path, device: str) -> dict[str, Any]:
    wandb_group = WANDB_RUNTIME["group"]
    if wandb_group:
        wandb_group = f"{wandb_group}-{plan.lane}"
    return {
        "model": model_ref,
        "data": str(data_path.relative_to(REPO_ROOT)),
        "device": device,
        "project": "runs/pose",
        "name": plan.run_name,
        "seed": plan.seed,
        "init_mode": "pretrained",
        "preprocess": {
            "grayscale": True,
        },
        "train": copy.deepcopy(plan.train),
        "augment": copy.deepcopy(plan.augment),
        "wandb": {
            **COMMON_WANDB,
            "enabled": bool(WANDB_RUNTIME["enabled"]),
            "project": WANDB_RUNTIME["project"],
            "entity": WANDB_RUNTIME["entity"],
            "group": wandb_group,
            "tags": ["pose", "energy-core", "v8", plan.lane, "gray", "realboost"],
            "notes": f"Fixed V8 grayscale realboost candidate {plan.candidate_id}.",
        },
    }


def build_test_config(weights_path: Path, data_path: Path, split: str, conf: float, run_name: str, device: str) -> dict[str, Any]:
    return {
        "weights": str(weights_path),
        "data": str(data_path),
        "split": split,
        "device": device,
        "project": "runs/pose_test",
        "name": f"{run_name}_plan",
        "imgsz": 960,
        "batch": 8,
        "conf": conf,
        "line_width": 2,
        "preprocess": {
            "grayscale": True,
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


def build_boost_lists() -> dict[str, Any]:
    commands = [
        [
            sys.executable,
            "scripts/build_darkboost_train_list.py",
            "--dataset-root",
            str(DATASET_ROOT),
            "--split",
            "train",
            "--boost-patterns",
            *REAL_PATTERNS,
            "--extra-repeats",
            "3",
            "--output",
            str(TRAIN_LIST_R3),
        ],
        [
            sys.executable,
            "scripts/build_darkboost_train_list.py",
            "--dataset-root",
            str(DATASET_ROOT),
            "--split",
            "train",
            "--boost-patterns",
            *REAL_PATTERNS,
            "--extra-repeats",
            "7",
            "--output",
            str(TRAIN_LIST_R7),
        ],
    ]
    for cmd in commands:
        run_command(cmd)

    counts = {}
    for key, output_path in (("r3", TRAIN_LIST_R3), ("r7", TRAIN_LIST_R7)):
        counts[key] = sum(1 for _ in output_path.open("r", encoding="utf-8"))
        expected = LIST_COUNTS[key]
        if counts[key] != expected:
            raise ValueError(f"Expected {expected} lines in {output_path}, got {counts[key]}")

    split_counts = {}
    for split in ("train", "valid", "test"):
        image_dir = DATASET_ROOT / split / "images"
        split_counts[split] = len({path.name for pattern in REAL_PATTERNS for path in image_dir.glob(pattern)})
        expected = REAL_COUNTS[split]
        if split_counts[split] != expected:
            raise ValueError(f"Expected {expected} real images in {image_dir}, got {split_counts[split]}")

    r3_yaml = load_yaml(DATA_R3_YAML)
    r7_yaml = load_yaml(DATA_R7_YAML)
    for label, data_yaml in (("r3", r3_yaml), ("r7", r7_yaml)):
        if data_yaml.get("val") != "../valid/images" or data_yaml.get("test") != "../test/images":
            raise ValueError(f"{label} dataset YAML must keep val/test untouched.")

    summary = {
        "real_patterns": REAL_PATTERNS,
        "real_counts": split_counts,
        "weighted_list_counts": counts,
        "data_yaml": str(DATA_YAML),
        "data_r3_yaml": str(DATA_R3_YAML),
        "data_r7_yaml": str(DATA_R7_YAML),
    }
    write_yaml(PLAN_SUMMARY_ROOT / "dataset_validation.yaml", summary)
    write_json(PLAN_SUMMARY_ROOT / "dataset_validation.json", summary)
    return summary


def resolve_execution_device(requested_device: str) -> tuple[str, dict[str, str]]:
    normalized = requested_device.strip()
    return normalized, {}


def subset_rank_tuple(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(summary["misses"]),
        float(summary["mean_pred_objects"]),
        int(summary["max_pred_objects"]),
        float(summary["mean_norm_kp_err"]) if summary["mean_norm_kp_err"] is not None else float("inf"),
        -float(summary["mean_best_iou"]) if summary["mean_best_iou"] is not None else 0.0,
    )


def candidate_rank_tuple(result: dict[str, Any]) -> tuple[Any, ...]:
    subset = result["subset_valid"]
    valid_map = float(result["valid_eval"]["metrics"]["metrics/mAP50-95(P)"])
    return (*subset_rank_tuple(subset), -valid_map)


def better_result(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    return first if candidate_rank_tuple(first) <= candidate_rank_tuple(second) else second


def resolve_candidate_data_path(plan: CandidatePlan, completed: dict[str, dict[str, Any]]) -> Path:
    if plan.parent_mode == "best_of" and plan.parent_candidates:
        parent_results = [completed[candidate_id] for candidate_id in plan.parent_candidates if candidate_id in completed]
        if len(parent_results) != len(plan.parent_candidates):
            raise ValueError(f"Missing parent results for {plan.candidate_id}: {plan.parent_candidates}")
        parent_result = better_result(parent_results[0], parent_results[1])
        raw_data = str(parent_result["train_config"]["data"])
        return (REPO_ROOT / raw_data).resolve() if not Path(raw_data).is_absolute() else Path(raw_data).resolve()
    return plan.data_path.resolve()


def resolve_parent_model(plan: CandidatePlan, completed: dict[str, dict[str, Any]]) -> str:
    if plan.model_ref is not None:
        return plan.model_ref
    if plan.parent_mode == "candidate":
        parent_result = completed.get(plan.parent_candidates[0])
        if parent_result is None:
            raise ValueError(f"Missing parent result for {plan.candidate_id}: {plan.parent_candidates[0]}")
        return str(parent_result["best_checkpoint"])
    if plan.parent_mode == "best_of":
        parent_results = [completed[candidate_id] for candidate_id in plan.parent_candidates if candidate_id in completed]
        if len(parent_results) != len(plan.parent_candidates):
            raise ValueError(f"Missing parent results for {plan.candidate_id}: {plan.parent_candidates}")
        return str(better_result(parent_results[0], parent_results[1])["best_checkpoint"])
    raise ValueError(f"Unable to resolve parent model for {plan.candidate_id}")


def ensure_candidate_run(plan: CandidatePlan, device: str, skip_train: bool, completed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    model_ref = resolve_parent_model(plan, completed)
    data_path = resolve_candidate_data_path(plan, completed)
    config = build_train_config(plan, model_ref=model_ref, data_path=data_path, device=device)
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
        "data_path": data_path,
        "reused_existing_run": reused,
        "grayscale_provenance": channel_gap_evidence(run_dir),
    }


def subset_output_path(plan: CandidatePlan, split: str, conf: float) -> Path:
    return PLAN_SCAN_ROOT / plan.candidate_id / f"{split}_conf_{conf:.2f}.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_subset_summary(
    plan: CandidatePlan,
    weights_path: Path,
    data_path: Path,
    split: str,
    conf: float,
    force_eval: bool,
) -> dict[str, Any]:
    output_path = subset_output_path(plan, split, conf)
    if output_path.exists() and not force_eval:
        summary = load_json(output_path)
        if (
            summary.get("weights") == str(weights_path)
            and summary.get("data") == str(data_path)
            and summary.get("split") == split
            and float(summary.get("conf")) == conf
            and bool(summary.get("grayscale")) is True
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
            str(data_path),
            "--split",
            split,
            "--imgsz",
            "960",
            "--conf",
            f"{conf:.2f}",
            "--patterns",
            *REAL_PATTERNS,
            "--grayscale",
            "--output",
            str(output_path),
        ]
    )
    return load_json(output_path)


def full_eval_paths(run_name: str, split: str) -> tuple[Path, Path]:
    normalized_split = "val" if split == "valid" else split
    base_name = f"{run_name}_plan"
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
    data_path: Path,
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
            and metrics_summary.get("data") == str(data_path)
            and metrics_summary.get("requested_split") == split
            and bool(metrics_summary.get("grayscale")) is True
            and predict_summary.get("weights") == str(weights_path)
        ):
            return metrics_summary, predict_summary

    if force_eval or metrics_path.parent.exists() or predict_path.parent.exists():
        cleanup_partial_eval_outputs(metrics_path=metrics_path, predict_path=predict_path)

    config = build_test_config(weights_path=weights_path, data_path=data_path, split=split, conf=conf, run_name=plan.run_name, device=device)
    config_path = PLAN_CONFIG_ROOT / f"{plan.run_name}.{split}.eval.yaml"
    write_yaml(config_path, config)
    run_command([sys.executable, "test_pose.py", "--config", str(config_path)])
    metrics_summary, predict_summary = load_eval_summaries(metrics_path, predict_path)
    return metrics_summary, predict_summary


def select_best_conf(conf_summaries: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    ordered = sorted(conf_summaries, key=lambda item: (subset_rank_tuple(item), CONF_GRID.index(float(item["conf"]))))
    return float(ordered[0]["conf"]), ordered[0]


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
            plan=plan,
            weights_path=runtime["best_checkpoint"],
            data_path=runtime["data_path"],
            split="valid",
            conf=conf,
            force_eval=force_eval,
        )
        for conf in CONF_GRID
    ]
    selected_conf, best_valid_conf_summary = select_best_conf(valid_conf_summaries)

    valid_eval, valid_predict = ensure_full_eval(
        plan=plan,
        weights_path=runtime["best_checkpoint"],
        data_path=runtime["data_path"],
        split="valid",
        conf=selected_conf,
        device=device,
        force_eval=force_eval,
    )
    test_eval, test_predict = ensure_full_eval(
        plan=plan,
        weights_path=runtime["best_checkpoint"],
        data_path=runtime["data_path"],
        split="test",
        conf=selected_conf,
        device=device,
        force_eval=force_eval,
    )
    subset_test = ensure_subset_summary(
        plan=plan,
        weights_path=runtime["best_checkpoint"],
        data_path=runtime["data_path"],
        split="test",
        conf=selected_conf,
        force_eval=force_eval,
    )

    summary = {
        "candidate_id": plan.candidate_id,
        "lane": plan.lane,
        "run_name": plan.run_name,
        "reused_existing_run": runtime["reused_existing_run"],
        "train_config_path": str(runtime["train_config_path"]),
        "train_config": sanitize(runtime["train_config"]),
        "run_dir": str(runtime["run_dir"]),
        "best_checkpoint": str(runtime["best_checkpoint"]),
        "grayscale_provenance": runtime["grayscale_provenance"],
        "selected_conf": selected_conf,
        "selected_conf_valid_summary": best_valid_conf_summary,
        "conf_scan_valid": valid_conf_summaries,
        "valid_eval": valid_eval,
        "test_eval": test_eval,
        "valid_predict": valid_predict,
        "test_predict": test_predict,
        "subset_valid": best_valid_conf_summary,
        "subset_test": subset_test,
    }
    summary["manual_review"] = candidate_manual_review(summary, image_locations=image_locations)
    write_yaml(summary_path, sanitize(summary))
    write_json(summary_path.with_suffix(".json"), sanitize(summary))
    return summary


def current_valid_pose_map(result: dict[str, Any]) -> float:
    return float(result["valid_eval"]["metrics"]["metrics/mAP50-95(P)"])


def challenger_beats_current_best(
    lane: str,
    challenger: dict[str, Any],
    current_best: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    challenger_subset = challenger["subset_valid"]
    current_subset = current_best["subset_valid"]
    challenger_map = current_valid_pose_map(challenger)
    current_map = current_valid_pose_map(current_best)

    if challenger_map < current_map - 0.005:
        reasons.append(
            f"valid mAP50-95(P) guard failed: challenger={challenger_map:.6f} current={current_map:.6f}"
        )
        return False, reasons

    if int(challenger_subset["misses"]) != 0:
        reasons.append(f"misses must be 0, got {challenger_subset['misses']}")
        return False, reasons

    if lane == "n":
        if float(challenger_subset["mean_pred_objects"]) > 1.05:
            reasons.append(f"mean_pred_objects must be <= 1.05, got {challenger_subset['mean_pred_objects']}")
            return False, reasons
        if int(challenger_subset["max_pred_objects"]) > 2:
            reasons.append(f"max_pred_objects must be <= 2, got {challenger_subset['max_pred_objects']}")
            return False, reasons
        challenger_kp = float(challenger_subset["mean_norm_kp_err"]) if challenger_subset["mean_norm_kp_err"] is not None else float("inf")
        current_kp = float(current_subset["mean_norm_kp_err"]) if current_subset["mean_norm_kp_err"] is not None else float("inf")
        if not challenger_kp < current_kp:
            reasons.append(f"mean_norm_kp_err must improve current best: challenger={challenger_kp:.6f} current={current_kp:.6f}")
            return False, reasons
    else:
        if float(challenger_subset["mean_pred_objects"]) > 1.2:
            reasons.append(f"mean_pred_objects must be <= 1.2, got {challenger_subset['mean_pred_objects']}")
            return False, reasons
        if int(challenger_subset["max_pred_objects"]) > 2:
            reasons.append(f"max_pred_objects must be <= 2, got {challenger_subset['max_pred_objects']}")
            return False, reasons
        challenger_kp = float(challenger_subset["mean_norm_kp_err"]) if challenger_subset["mean_norm_kp_err"] is not None else float("inf")
        if challenger_kp > 0.020:
            reasons.append(f"mean_norm_kp_err must be <= 0.020, got {challenger_kp:.6f}")
            return False, reasons

    if candidate_rank_tuple(challenger) >= candidate_rank_tuple(current_best):
        reasons.append("ranking tuple did not beat current best")
        return False, reasons

    reasons.append("candidate beat current best")
    return True, reasons


def execute_lane(
    lane: str,
    plans: dict[str, CandidatePlan],
    device: str,
    skip_train: bool,
    force_eval: bool,
    image_locations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ordered_ids = lane_sequences()[lane]
    completed: dict[str, dict[str, Any]] = {}
    current_best_id: str | None = None
    consecutive_failures = 0
    executed: list[str] = []
    skipped: list[dict[str, Any]] = []

    for index, candidate_id in enumerate(ordered_ids):
        plan = plans[candidate_id]
        runtime = ensure_candidate_run(plan=plan, device=device, skip_train=skip_train, completed=completed)
        result = evaluate_candidate(
            plan=plan,
            runtime=runtime,
            device=device,
            force_eval=force_eval,
            image_locations=image_locations,
        )
        executed.append(candidate_id)

        if current_best_id is None:
            result["comparison"] = {
                "beat_current_best": True,
                "reasons": ["baseline candidate"],
            }
            completed[candidate_id] = result
            current_best_id = candidate_id
            consecutive_failures = 0
        else:
            beat_current_best, reasons = challenger_beats_current_best(lane=lane, challenger=result, current_best=completed[current_best_id])
            result["comparison"] = {
                "beat_current_best": beat_current_best,
                "reasons": reasons,
            }
            completed[candidate_id] = result
            if beat_current_best:
                current_best_id = candidate_id
                consecutive_failures = 0
            else:
                consecutive_failures += 1

        write_yaml(PLAN_RESULT_ROOT / f"{candidate_id}.yaml", sanitize(completed[candidate_id]))
        write_json(PLAN_RESULT_ROOT / f"{candidate_id}.json", sanitize(completed[candidate_id]))

        if consecutive_failures >= 2 and index + 1 < len(ordered_ids):
            for remaining_id in ordered_ids[index + 1 :]:
                skipped.append(
                    {
                        "candidate_id": remaining_id,
                        "reason": f"stop rule triggered after {consecutive_failures} consecutive non-improving candidates",
                    }
                )
            break

    if current_best_id is None:
        raise RuntimeError(f"Lane {lane} produced no candidates.")

    executed_results = {candidate_id: completed[candidate_id] for candidate_id in executed}
    summary = {
        "lane": lane,
        "ordered_candidates": ordered_ids,
        "executed_candidates": executed,
        "skipped_candidates": skipped,
        "current_best_candidate_id": current_best_id,
        "best_checkpoint": executed_results[current_best_id]["best_checkpoint"],
        "candidate_results": executed_results,
    }
    write_yaml(PLAN_SUMMARY_ROOT / f"lane_{lane}.yaml", sanitize(summary))
    write_json(PLAN_SUMMARY_ROOT / f"lane_{lane}.json", sanitize(summary))
    return summary


def best_config_payload(best_result: dict[str, Any]) -> dict[str, Any]:
    train_config = copy.deepcopy(best_result["train_config"])
    train_config["recommended_inference"] = {
        "conf": float(best_result["selected_conf"]),
    }
    train_config["best_checkpoint"] = best_result["best_checkpoint"]
    train_config["selection_summary"] = {
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


def build_summary_doc(lane_summaries: dict[str, dict[str, Any]], dataset_summary: dict[str, Any]) -> str:
    n_best = lane_summaries["n"]["candidate_results"][lane_summaries["n"]["current_best_candidate_id"]]
    s_best = lane_summaries["s"]["candidate_results"][lane_summaries["s"]["current_best_candidate_id"]]

    def lane_section(title: str, lane_summary: dict[str, Any], best_result: dict[str, Any]) -> str:
        lines = [f"## {title}", ""]
        lines.append(f"- 最终胜者: `{best_result['candidate_id']}` (`{best_result['run_name']}`)")
        lines.append(f"- 最佳 checkpoint: `{best_result['best_checkpoint']}`")
        lines.append(f"- 推荐推理 `conf`: `{best_result['selected_conf']:.2f}`")
        lines.append(
            f"- `valid mAP50-95(P) = {current_valid_pose_map(best_result):.6f}`, "
            f"`test mAP50-95(P) = {float(best_result['test_eval']['metrics']['metrics/mAP50-95(P)']):.6f}`"
        )
        lines.append(
            f"- `valid` 真实子集: `misses={best_result['subset_valid']['misses']}`, "
            f"`mean_pred_objects={best_result['subset_valid']['mean_pred_objects']:.6f}`, "
            f"`max_pred_objects={best_result['subset_valid']['max_pred_objects']}`, "
            f"`mean_norm_kp_err={best_result['subset_valid']['mean_norm_kp_err']:.6f}`, "
            f"`mean_best_iou={best_result['subset_valid']['mean_best_iou']:.6f}`"
        )
        lines.append(
            f"- `test` 真实子集: `misses={best_result['subset_test']['misses']}`, "
            f"`mean_pred_objects={best_result['subset_test']['mean_pred_objects']:.6f}`, "
            f"`max_pred_objects={best_result['subset_test']['max_pred_objects']}`, "
            f"`mean_norm_kp_err={best_result['subset_test']['mean_norm_kp_err']:.6f}`, "
            f"`mean_best_iou={best_result['subset_test']['mean_best_iou']:.6f}`"
        )
        lines.append("")
        lines.append("### 候选结果")
        lines.append("")
        for candidate_id in lane_summary["executed_candidates"]:
            candidate = lane_summary["candidate_results"][candidate_id]
            comparison = candidate.get("comparison", {})
            lines.append(
                f"- `{candidate_id}` / `{candidate['run_name']}`: "
                f"`conf={candidate['selected_conf']:.2f}`, "
                f"`valid_real_misses={candidate['subset_valid']['misses']}`, "
                f"`valid_real_mean_pred_objects={candidate['subset_valid']['mean_pred_objects']:.6f}`, "
                f"`valid_real_max_pred_objects={candidate['subset_valid']['max_pred_objects']}`, "
                f"`valid_real_mean_norm_kp_err={candidate['subset_valid']['mean_norm_kp_err']:.6f}`, "
                f"`valid_mAP50-95(P)={current_valid_pose_map(candidate):.6f}`, "
                f"`beat_current_best={comparison.get('beat_current_best')}`"
            )
        if lane_summary["skipped_candidates"]:
            lines.append("")
            lines.append("### Stop Rule")
            lines.append("")
            for skipped in lane_summary["skipped_candidates"]:
                lines.append(f"- `{skipped['candidate_id']}`: {skipped['reason']}")
        return "\n".join(lines)

    important_lines = []
    all_review_items = n_best["manual_review"]["important_images"]
    for item in all_review_items:
        if not item["found"]:
            important_lines.append(f"- `{item['image']}`: 未在 `v8` 数据集里找到")
            continue
        important_lines.append(
            f"- `{item['image']}`: `{item['split']}` -> `{item['dataset_path']}`"
        )

    lines = [
        "# V8 灰度 + Realboost 调参总结",
        "",
        "本文记录了 `YOLO11n` 与 `YOLO11s` 在 `v8` 数据集上的固定候选调参流程、最终胜者和交付物。",
        "",
        "## 数据与约束",
        "",
        f"- 基础数据入口: `{DATA_YAML}`",
        f"- 真实子集模式: `{', '.join(REAL_PATTERNS)}`",
        f"- 真实子集计数: `train={dataset_summary['real_counts']['train']}` / `valid={dataset_summary['real_counts']['valid']}` / `test={dataset_summary['real_counts']['test']}`",
        f"- Weighted list 计数: `r3={dataset_summary['weighted_list_counts']['r3']}` / `r7={dataset_summary['weighted_list_counts']['r7']}`",
        "",
        "## 必查图片定位",
        "",
        *important_lines,
        "",
        lane_section("YOLO11n", lane_summaries["n"], n_best),
        "",
        lane_section("YOLO11s", lane_summaries["s"], s_best),
        "",
        "## 部署建议",
        "",
        f"- YOLO11n 最佳参数文件: `{BEST_CONFIG_PATHS['n']}`",
        f"- YOLO11s 最佳参数文件: `{BEST_CONFIG_PATHS['s']}`",
        f"- 两条 lane 均使用 `preprocess.grayscale: true`，并在总结中保留了现有 gray run 的 provenance 证据。",
        "",
        "## 灰度 Provenance 说明",
        "",
        "- 复用的历史 gray run 以固定 run 名、`args.yaml` 参数匹配和 `train_batch0.jpg` 的近灰度通道差异共同佐证。",
        f"- `YOLO11n` 胜者 train batch 证据: `{json.dumps(n_best['grayscale_provenance'], ensure_ascii=True)}`",
        f"- `YOLO11s` 胜者 train batch 证据: `{json.dumps(s_best['grayscale_provenance'], ensure_ascii=True)}`",
    ]
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
    if env_overrides:
        print_info(
            f"Binding subprocesses to physical CUDA device(s) {env_overrides['CUDA_VISIBLE_DEVICES']} "
            f"and using local device '{effective_device}'."
        )
    print_info(
        f"W&B training logging {'enabled' if WANDB_RUNTIME['enabled'] else 'disabled'} "
        f"(project={WANDB_RUNTIME['project']}, group={WANDB_RUNTIME['group']})."
    )
    PLAN_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    PLAN_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    PLAN_SCAN_ROOT.mkdir(parents=True, exist_ok=True)
    PLAN_SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

    dataset_summary = build_boost_lists()
    image_locations = important_image_locations()

    plans = candidate_plans()
    lane_summaries: dict[str, dict[str, Any]] = {}
    for lane in args.lanes:
        print_info(f"Executing lane {lane}")
        lane_summaries[lane] = execute_lane(
            lane=lane,
            plans=plans,
            device=effective_device,
            skip_train=args.skip_train,
            force_eval=args.force_eval,
            image_locations=image_locations,
        )

    for lane, config_path in BEST_CONFIG_PATHS.items():
        if lane not in lane_summaries:
            continue
        best_id = lane_summaries[lane]["current_best_candidate_id"]
        best_payload = best_config_payload(lane_summaries[lane]["candidate_results"][best_id])
        write_yaml(config_path, sanitize(best_payload))

    if {"n", "s"}.issubset(lane_summaries):
        summary_text = build_summary_doc(lane_summaries=lane_summaries, dataset_summary=dataset_summary)
        SUMMARY_DOC_PATH.write_text(summary_text, encoding="utf-8")

    overall_summary = {
        "dataset": dataset_summary,
        "lanes": lane_summaries,
        "best_configs": {lane: str(path) for lane, path in BEST_CONFIG_PATHS.items() if lane in lane_summaries},
        "summary_doc": str(SUMMARY_DOC_PATH) if {"n", "s"}.issubset(lane_summaries) else None,
    }
    write_yaml(PLAN_SUMMARY_ROOT / "overall.yaml", sanitize(overall_summary))
    write_json(PLAN_SUMMARY_ROOT / "overall.json", sanitize(overall_summary))
    print_info("Plan execution completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
