# YOLO Pose Training for Energy Core Position Estimate

This project packages the local Energy Core pose datasets into a reproducible YOLO training workspace with:

- Git-based project management
- A dedicated Conda environment
- Configurable Ultralytics pose training
- Optional Weights & Biases cloud logging
- Smoke-test friendly defaults for CPU-only environments

## Repository policy

- Source code, configs, and lightweight dataset metadata are versioned.
- Raw dataset images and labels under `data/**/{train,valid,test}` are ignored.
- Training artifacts under `runs/`, `wandb/`, `.ultralytics/`, and generated weights are ignored.
- Local secrets in `.env` are ignored.

## Environment setup

Create the environment directly:

```bash
conda create -n tech_core_yolo python=3.12 pip -y
conda run -n tech_core_yolo pip install ultralytics==8.3.202 wandb pyyaml
```

Or recreate it from the checked-in file:

```bash
conda env create -f environment.yml
```

## W&B login

The training script automatically checks `.env` for either `WANDB_API_KEY` or `wandb_api_key` and maps it to the environment variable expected by W&B.

If you want to login manually:

```bash
set -a
source .env
set +a
export WANDB_API_KEY="${WANDB_API_KEY:-$wandb_api_key}"
conda run -n tech_core_yolo wandb login "$WANDB_API_KEY"
```

## Training usage

Run a scratch training job with the default config:

```bash
conda run -n tech_core_yolo python train_pose.py --scratch
```

Run the planned smoke test for the `v6` dataset:

```bash
conda run -n tech_core_yolo python train_pose.py \
  --scratch \
  --model yolo11n-pose.yaml \
  --data data/Energy_Core_Position_Estimate.v6i.yolov8/data.yaml \
  --name v6_yolo11n_scratch_smoke \
  --set train.epochs=1 \
  --set train.batch=2 \
  --set train.workers=0
```

Override any config value with repeated `--set key=value` flags:

```bash
conda run -n tech_core_yolo python train_pose.py \
  --set train.epochs=10 \
  --set augment.mosaic=0.0 \
  --set wandb.group=compare-yolo11-vs-yolo26
```

## Future YOLO26 support

The script already accepts a generic `--model` input. Once a `yolo26-pose.yaml` or `yolo26-pose.pt` becomes available, you can pass it directly:

```bash
conda run -n tech_core_yolo python train_pose.py \
  --model /path/to/yolo26-pose.yaml \
  --scratch
```

## Common notes

- `device=auto` prefers CUDA when available and otherwise falls back to CPU with a warning.
- `--scratch` rejects `.pt` checkpoints on purpose to avoid accidentally fine-tuning when you intended to train from scratch.
- W&B is optional. If login is unavailable, training continues locally.
