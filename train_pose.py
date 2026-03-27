#!/usr/bin/env python3
"""Train Ultralytics YOLO pose models with optional W&B logging."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str((REPO_ROOT / ".ultralytics").resolve()))

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "model": "yolo11n-pose.yaml",
    "data": "data/Energy_Core_Position_Estimate.v6i.yolov8/data.yaml",
    "device": "auto",
    "project": "runs/pose",
    "name": "v6_yolo11n_pose",
    "seed": 42,
    "init_mode": "scratch",
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
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    },
    "wandb": {
        "enabled": True,
        "project": "tech-core-yolo-pose",
        "entity": None,
        "group": None,
        "tags": ["pose", "energy-core"],
        "notes": "YOLO pose training for Energy Core Position Estimate.",
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

    required_top_level = ["model", "data", "device", "project", "name", "seed", "init_mode", "train", "augment", "wandb"]
    missing = [key for key in required_top_level if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    if not isinstance(config["train"], dict) or not isinstance(config["augment"], dict) or not isinstance(config["wandb"], dict):
        raise ValueError("'train', 'augment', and 'wandb' must all be mappings.")

    init_mode = str(config["init_mode"]).lower()
    if init_mode not in {"scratch", "pretrained"}:
        raise ValueError("init_mode must be either 'scratch' or 'pretrained'.")
    config["init_mode"] = init_mode
    return config


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

    if "tags" in resolved["wandb"] and isinstance(resolved["wandb"]["tags"], str):
        resolved["wandb"]["tags"] = [resolved["wandb"]["tags"]]

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
    run = wandb.init(**init_kwargs) if not wandb.run else wandb.run
    print_info(f"W&B enabled. Run: {run.name}")
    return run


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


def run_training(config: dict[str, Any]) -> Path | None:
    model = load_pose_model(str(config["model"]))
    resolved_device = resolve_device(config["device"])
    config["device"] = resolved_device

    configure_wandb(config)

    print_info("Resolved training config:")
    print(yaml.safe_dump(sanitize_for_wandb(config), sort_keys=False))

    train_args = build_train_args(config, resolved_device)
    model.train(**train_args)

    trainer = getattr(model, "trainer", None)
    save_dir = Path(trainer.save_dir).resolve() if trainer and getattr(trainer, "save_dir", None) else None
    if save_dir:
        print_info(f"Training outputs saved to: {save_dir}")
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
