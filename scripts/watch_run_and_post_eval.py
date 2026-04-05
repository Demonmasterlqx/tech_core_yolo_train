#!/usr/bin/env python3
"""Wait for a training process pattern to disappear, then run full valid/test evaluation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def print_info(message: str) -> None:
    print(f"[watch_run_and_post_eval] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a training run and execute full valid/test post-eval.")
    parser.add_argument("--pattern", required=True, help="Unique pgrep -f pattern for the training command.")
    parser.add_argument("--weights", required=True, help="Best checkpoint path to evaluate after training ends.")
    parser.add_argument("--data", required=True, help="Dataset YAML path for evaluation.")
    parser.add_argument("--device", default="auto", help="Evaluation device.")
    parser.add_argument("--project", default="runs/pose_test", help="Base output project for test_pose.py.")
    parser.add_argument("--name", required=True, help="Base run name for evaluation artifacts.")
    parser.add_argument("--imgsz", type=int, default=960, help="Evaluation image size.")
    parser.add_argument("--batch", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold.")
    parser.add_argument("--line-width", dest="line_width", type=int, default=2, help="Rendered line width.")
    parser.add_argument("--grayscale", action="store_true", help="Enable grayscale preprocessing during evaluation.")
    parser.add_argument("--poll-sec", dest="poll_sec", type=float, default=20.0, help="Polling interval in seconds.")
    return parser.parse_args()


def pattern_pids(pattern: str) -> list[int]:
    completed = subprocess.run(
        ["pgrep", "-f", pattern],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return []
    current_pid = os.getpid()
    pids = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid = int(line)
        if pid != current_pid:
            pids.append(pid)
    return pids


def write_eval_config(args: argparse.Namespace, split: str) -> Path:
    config = {
        "weights": str(Path(args.weights).resolve()),
        "data": str(Path(args.data).resolve()),
        "split": split,
        "device": args.device,
        "project": args.project,
        "name": f"{args.name}_auto",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "line_width": args.line_width,
        "preprocess": {
            "grayscale": bool(args.grayscale),
        },
    }
    config_path = REPO_ROOT / "runs" / "v8_gray_realboost_plan" / "generated_configs" / f"{args.name}.{split}.auto_eval.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return config_path


def run_eval(config_path: Path) -> None:
    print_info(f"Running evaluation with config={config_path}")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "test_pose.py"), "--config", str(config_path)],
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> int:
    args = parse_args()
    print_info(f"Watching pattern: {args.pattern}")
    while True:
        pids = pattern_pids(args.pattern)
        if not pids:
            break
        print_info(f"Training still active for pattern; matching pids={pids}")
        time.sleep(args.poll_sec)

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Best checkpoint missing after watch: {weights_path}")

    for split in ("valid", "test"):
        run_eval(write_eval_config(args=args, split=split))

    print_info("Automatic post-evaluation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
