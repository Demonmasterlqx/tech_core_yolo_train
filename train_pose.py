#!/usr/bin/env python3
"""Train Ultralytics YOLO pose models with optional W&B logging."""

from __future__ import annotations

import argparse
import copy
import faulthandler
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from grayscale_preprocess import patch_ultralytics_dataset_grayscale

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_YOLO_CONFIG_ROOT = (REPO_ROOT / ".ultralytics").resolve()
(LOCAL_YOLO_CONFIG_ROOT / "Ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_YOLO_CONFIG_ROOT))
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "model": "yolo11n-pose.yaml",
    "data": "data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.yolov8/data.yaml",
    "device": "auto",
    "project": "runs/pose",
    "name": "v8_yolo11n_pose",
    "seed": 42,
    "init_mode": "scratch",
    "preprocess": {
        "grayscale": False,
    },
    "train": {
        "epochs": 100,
        "batch": 8,
        "imgsz": 640,
        "workers": 4,
        "patience": 50,
        "save": True,
        "save_period": -1,
        "cache": False,
        "plots": True,
        "exist_ok": False,
        "verbose": True,
    },
    "augment": {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    },
    "wandb": {
        "enabled": True,
        "project": "tech-core-yolo-pose",
        "entity": None,
        "group": None,
        "batch_log_interval": 50,
        "upload_checkpoints": True,
        "checkpoint_policy": "live",
        "tags": ["pose", "energy-core"],
        "notes": "YOLO pose training for Energy Core Position Estimate on the v8 dataset.",
    },
    "post_eval": {
        "enabled": True,
        "project": "runs/pose_test",
        "splits": ["valid", "test"],
        "imgsz": 960,
        "batch": 8,
        "conf": 0.25,
        "line_width": 2,
    },
}


def print_info(message: str) -> None:
    print(f"[train_pose] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO pose models with config and CLI overrides.")
    parser.add_argument("--config", default="train_pose.yaml", help="Path to the default YAML config file.")
    parser.add_argument("--model", help="Ultralytics model name or a local .pt/.yaml path.")
    parser.add_argument("--data", help="Dataset YAML path.")
    parser.add_argument("--device", help="Training device. Use auto, cpu, 0, 0,1, etc.")
    parser.add_argument("--project", help="Local output project directory for Ultralytics runs.")
    parser.add_argument("--name", help="Run name.")
    parser.add_argument("--wandb", dest="wandb_enabled", action="store_true", help="Force-enable W&B logging.")
    parser.add_argument("--no-wandb", dest="wandb_enabled", action="store_false", help="Disable W&B logging.")
    parser.set_defaults(wandb_enabled=None)
    parser.add_argument("--scratch", action="store_true", help="Train from scratch. Rejects .pt checkpoints.")
    parser.add_argument("--pretrained", action="store_true", help="Train with pretrained initialization.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config field, e.g. --set train.epochs=1",
    )
    return parser.parse_args()


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return content


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise ValueError(f"Override must look like key=value: {item}")
    key, raw_value = item.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Override key cannot be empty: {item}")
    return key, yaml.safe_load(raw_value)


def set_nested_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def looks_like_local_path(value: str) -> bool:
    path = Path(value)
    return path.is_absolute() or value.startswith(".") or "/" in value or "\\" in value


def resolve_existing_path(raw_value: str, base_dir: Path) -> Path:
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def load_dotenv(path: Path) -> dict[str, str]:
    env_map: dict[str, str] = {}
    if not path.exists():
        return env_map
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        env_map[key] = value
    return env_map


def inject_env_defaults() -> None:
    env_values = load_dotenv(REPO_ROOT / ".env")
    wandb_key = env_values.get("WANDB_API_KEY") or env_values.get("wandb_api_key")
    if wandb_key and not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = wandb_key


def normalize_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = resolve_existing_path(args.config, REPO_ROOT)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    config = deep_merge(DEFAULT_CONFIG, load_yaml_file(config_path))

    for dotted_key, value in (parse_override(item) for item in args.overrides):
        set_nested_value(config, dotted_key, value)

    if args.model:
        config["model"] = args.model
    if args.data:
        config["data"] = args.data
    if args.device:
        config["device"] = args.device
    if args.project:
        config["project"] = args.project
    if args.name:
        config["name"] = args.name
    if args.scratch and args.pretrained:
        raise ValueError("Choose either --scratch or --pretrained, not both.")
    if args.scratch:
        config["init_mode"] = "scratch"
    if args.pretrained:
        config["init_mode"] = "pretrained"
    if args.wandb_enabled is not None:
        config.setdefault("wandb", {})
        config["wandb"]["enabled"] = args.wandb_enabled

    required_top_level = [
        "model",
        "data",
        "device",
        "project",
        "name",
        "seed",
        "init_mode",
        "preprocess",
        "train",
        "augment",
        "wandb",
        "post_eval",
    ]
    missing = [key for key in required_top_level if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    if (
        not isinstance(config["preprocess"], dict)
        or not isinstance(config["train"], dict)
        or not isinstance(config["augment"], dict)
        or not isinstance(config["wandb"], dict)
        or not isinstance(config["post_eval"], dict)
    ):
        raise ValueError("'preprocess', 'train', 'augment', 'wandb', and 'post_eval' must all be mappings.")

    init_mode = str(config["init_mode"]).lower()
    if init_mode not in {"scratch", "pretrained"}:
        raise ValueError("init_mode must be either 'scratch' or 'pretrained'.")
    config["init_mode"] = init_mode
    validate_project_augment_rules(config)
    return config


def validate_project_augment_rules(config: dict[str, Any]) -> None:
    augment = config.get("augment", {})
    for key in ("flipud", "fliplr"):
        value = float(augment.get(key, 0.0) or 0.0)
        if value != 0.0:
            raise ValueError(
                "Energy Core pose training forbids horizontal/vertical flip augmentation because the object "
                "and keypoint layout are not flip-invariant. Set augment.flipud=0.0 and augment.fliplr=0.0."
            )


def validate_and_resolve_paths(config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    resolved = copy.deepcopy(config)
    data_path = resolve_existing_path(str(config["data"]), REPO_ROOT)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data_path}")

    data_config = load_yaml_file(data_path)
    if "kpt_shape" not in data_config:
        raise ValueError(f"Dataset YAML is missing 'kpt_shape': {data_path}")
    resolved["data"] = str(data_path)

    model_value = str(config["model"])
    if config["init_mode"] == "scratch" and model_value.endswith(".pt"):
        raise ValueError("--scratch requires a pose YAML model, not a .pt checkpoint.")

    if looks_like_local_path(model_value):
        model_path = resolve_existing_path(model_value, REPO_ROOT)
        if not model_path.exists():
            raise FileNotFoundError(f"Local model path does not exist: {model_path}")
        resolved["model"] = str(model_path)

    project_path = resolve_existing_path(str(config["project"]), REPO_ROOT)
    project_path.mkdir(parents=True, exist_ok=True)
    resolved["project"] = str(project_path)

    post_eval_project = resolve_existing_path(str(config["post_eval"].get("project", "runs/pose_test")), REPO_ROOT)
    post_eval_project.mkdir(parents=True, exist_ok=True)
    resolved["post_eval"]["project"] = str(post_eval_project)

    if "tags" in resolved["wandb"] and isinstance(resolved["wandb"]["tags"], str):
        resolved["wandb"]["tags"] = [resolved["wandb"]["tags"]]

    post_eval_splits = resolved["post_eval"].get("splits", ["valid", "test"])
    if isinstance(post_eval_splits, str):
        post_eval_splits = [post_eval_splits]
    if not isinstance(post_eval_splits, list) or not all(isinstance(item, str) for item in post_eval_splits):
        raise ValueError("post_eval.splits must be a string or a list of strings.")
    resolved["post_eval"]["splits"] = post_eval_splits

    return resolved, data_path


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


def sanitize_for_wandb(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_for_wandb(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_wandb(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def configure_wandb(config: dict[str, Any]):
    from ultralytics.utils import SETTINGS

    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        SETTINGS.update({"wandb": False})
        print_info("W&B disabled by configuration.")
        return None

    try:
        import wandb
    except ImportError:
        SETTINGS.update({"wandb": False})
        print_info("W&B requested but package is not installed. Continuing with local logging only.")
        return None

    login_ok = False
    api_key = os.environ.get("WANDB_API_KEY")
    try:
        if api_key:
            login_ok = bool(wandb.login(key=api_key, relogin=True))
        else:
            login_ok = bool(wandb.login(anonymous="never", relogin=False))
    except Exception as exc:  # pragma: no cover - depends on local W&B state
        print_info(f"W&B login unavailable ({exc}). Continuing with local logging only.")

    if not login_ok:
        SETTINGS.update({"wandb": False})
        return None

    SETTINGS.update({"wandb": True})
    init_kwargs = {
        "project": wandb_cfg.get("project") or "tech-core-yolo-pose",
        "name": config.get("name"),
        "entity": wandb_cfg.get("entity"),
        "group": wandb_cfg.get("group"),
        "tags": wandb_cfg.get("tags"),
        "notes": wandb_cfg.get("notes"),
        "config": sanitize_for_wandb(config),
    }
    init_kwargs = {key: value for key, value in init_kwargs.items() if value not in (None, [], "")}
    try:
        run = wandb.init(**init_kwargs) if not wandb.run else wandb.run
    except Exception as exc:
        SETTINGS.update({"wandb": False})
        print_info(f"W&B init failed ({exc}). Continuing with local logging only.")
        return None
    print_info(f"W&B enabled. Run: {run.name}")
    return run


def attach_wandb_live_callbacks(model, run: Any, config: dict[str, Any]) -> None:
    if run is None:
        return

    wandb_cfg = config.get("wandb", {})
    interval = int(wandb_cfg.get("batch_log_interval", 0) or 0)

    def on_train_epoch_start(trainer) -> None:
        trainer._tech_core_live_batch_index = 0

    def on_train_batch_end(trainer) -> None:
        if interval <= 0:
            return
        batch_index = int(getattr(trainer, "_tech_core_live_batch_index", 0)) + 1
        trainer._tech_core_live_batch_index = batch_index
        if batch_index % interval != 0:
            return
        try:
            live_metrics = trainer.label_loss_items(trainer.tloss, prefix="train/live")
            live_metrics.update({f"lr/live_pg{index}": group["lr"] for index, group in enumerate(trainer.optimizer.param_groups)})
            live_metrics["train/live_epoch"] = trainer.epoch + 1
            live_metrics["train/live_batch"] = batch_index
            live_metrics["train/live_global_batch"] = trainer.epoch * max(len(trainer.train_loader), 1) + batch_index
            live_metrics["train/live_gpu_mem_gb"] = trainer._get_memory()
            run.log(live_metrics, step=trainer.epoch + 1, commit=False)
        except Exception as exc:  # pragma: no cover - depends on runtime trainer state
            print_info(f"W&B live batch logging skipped ({exc}).")

    def upload_checkpoint(trainer, checkpoint_name: str, force: bool = False) -> None:
        if not bool(wandb_cfg.get("upload_checkpoints", True)):
            return
        save_dir = getattr(trainer, "save_dir", None)
        if save_dir is None:
            return
        save_dir_path = Path(save_dir).resolve()
        checkpoint_path = save_dir_path / "weights" / checkpoint_name
        if not checkpoint_path.exists():
            return

        policy = str(wandb_cfg.get("checkpoint_policy", "live") or "live")
        stat = checkpoint_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        uploaded = getattr(trainer, "_tech_core_uploaded_checkpoints", {})
        if not force and uploaded.get(checkpoint_name) == signature:
            return

        try:
            run.save(str(checkpoint_path), base_path=str(save_dir_path), policy=policy)
            uploaded[checkpoint_name] = signature
            trainer._tech_core_uploaded_checkpoints = uploaded
            run.summary[f"checkpoints/{checkpoint_name}"] = f"weights/{checkpoint_name}"
            run.summary[f"checkpoints/{checkpoint_name}_bytes"] = stat.st_size
            run.summary[f"checkpoints/{checkpoint_name}_epoch"] = trainer.epoch + 1
        except Exception as exc:  # pragma: no cover - depends on W&B runtime state
            print_info(f"W&B checkpoint upload skipped for {checkpoint_name} ({exc}).")

    def on_model_save(trainer) -> None:
        upload_checkpoint(trainer, "best.pt")

    def on_train_end(trainer) -> None:
        upload_checkpoint(trainer, "best.pt")
        upload_checkpoint(trainer, "last.pt")

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_model_save", on_model_save)
    if "on_train_end" in model.callbacks:
        model.callbacks["on_train_end"].insert(0, on_train_end)
    else:
        model.add_callback("on_train_end", on_train_end)


def load_pose_model(model_ref: str):
    from ultralytics import YOLO

    model = YOLO(model_ref)
    if getattr(model, "task", None) != "pose":
        raise ValueError(f"Model must be a pose model, but resolved task is: {model.task}")
    return model


def build_train_args(config: dict[str, Any], resolved_device: str) -> dict[str, Any]:
    train_args = {
        "data": config["data"],
        "device": resolved_device,
        "project": config["project"],
        "name": config["name"],
        "seed": config["seed"],
        "pretrained": config["init_mode"] == "pretrained",
    }
    train_args.update(config["train"])
    train_args.update(config["augment"])
    return train_args


def run_post_training_evaluations(config: dict[str, Any], best_checkpoint: Path) -> None:
    post_eval = config.get("post_eval", {})
    if not post_eval.get("enabled", False):
        print_info("Post-training valid/test evaluation disabled by configuration.")
        return

    if not best_checkpoint.exists():
        raise FileNotFoundError(f"Best checkpoint missing for post-training evaluation: {best_checkpoint}")

    for split in post_eval.get("splits", []):
        eval_config = {
            "weights": str(best_checkpoint),
            "data": config["data"],
            "split": split,
            "device": config["device"],
            "project": post_eval["project"],
            "name": f"{config['name']}_post",
            "imgsz": int(post_eval.get("imgsz", 960)),
            "batch": int(post_eval.get("batch", 8)),
            "conf": float(post_eval.get("conf", 0.25)),
            "line_width": int(post_eval.get("line_width", 2)),
            "preprocess": {
                "grayscale": bool(config.get("preprocess", {}).get("grayscale", False)),
            },
        }
        eval_config_path = Path(config["project"]) / config["name"] / f"post_eval_{split}.yaml"
        eval_config_path.parent.mkdir(parents=True, exist_ok=True)
        eval_config_path.write_text(
            yaml.safe_dump(eval_config, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        print_info(f"Running post-training evaluation for split='{split}' with config={eval_config_path}")
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "test_pose.py"), "--config", str(eval_config_path)],
            check=True,
            cwd=REPO_ROOT,
        )


def run_training(config: dict[str, Any]) -> Path | None:
    model = load_pose_model(str(config["model"]))
    resolved_device = resolve_device(config["device"])
    config["device"] = resolved_device

    wandb_run = configure_wandb(config)
    attach_wandb_live_callbacks(model, wandb_run, config)

    print_info("Resolved training config:")
    print(yaml.safe_dump(sanitize_for_wandb(config), sort_keys=False))

    train_args = build_train_args(config, resolved_device)
    use_grayscale = bool(config.get("preprocess", {}).get("grayscale", False))
    if use_grayscale:
        print_info("Applying grayscale preprocessing during training and validation.")
    with patch_ultralytics_dataset_grayscale(use_grayscale):
        model.train(**train_args)

    trainer = getattr(model, "trainer", None)
    save_dir = Path(trainer.save_dir).resolve() if trainer and getattr(trainer, "save_dir", None) else None
    if save_dir:
        print_info(f"Training outputs saved to: {save_dir}")
        best_checkpoint = save_dir / "weights" / "best.pt"
        if best_checkpoint.exists():
            run_post_training_evaluations(config=config, best_checkpoint=best_checkpoint)
        else:
            print_info("Skipping post-training evaluation because best.pt is missing (train.save may be false).")
    return save_dir


def main() -> int:
    try:
        inject_env_defaults()
        args = parse_args()
        config = normalize_config(args)
        resolved_config, _ = validate_and_resolve_paths(config)
        save_dir = run_training(resolved_config)
        if save_dir is not None:
            print_info(f"Finished successfully. Run directory: {save_dir}")
        else:
            print_info("Finished successfully.")
        return 0
    except Exception as exc:
        print_info(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
