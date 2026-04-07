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
conda run -n tech_core_yolo pip install torch==2.8.0 torchvision==0.23.0 ultralytics==8.4.30 wandb pyyaml
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

Run a scratch training job with the default `v8` config:

```bash
conda run -n tech_core_yolo python train_pose.py --scratch
```

Run a minimal smoke test for the `v8` dataset:

```bash
conda run -n tech_core_yolo python train_pose.py \
  --scratch \
  --model yolo11n-pose.yaml \
  --data data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.yolov8/data.yaml \
  --name v8_yolo11n_scratch_smoke \
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

## YOLO26 usage

The script accepts a generic `--model` input, so you can train supported YOLO11 and YOLO26 pose checkpoints directly:

```bash
conda run -n tech_core_yolo python train_pose.py \
  --model yolo26s-pose.pt \
  --pretrained
```

## Test usage

Run a full test-set evaluation and export rendered predictions:

```bash
conda run -n tech_core_yolo python test_pose.py \
  --config test_pose.yaml \
  --weights runs/pose/<train_run>/weights/best.pt \
  --split valid \
  --grayscale
```

This produces two artifact directories under `runs/pose_test/`:

- `<train_run>_test_eval` for quantitative metrics and evaluation plots
- `<train_run>_test_predict` for rendered test images and YOLO-format prediction labels

## Common notes

- `device=auto` prefers CUDA when available and otherwise falls back to CPU with a warning.
- `--scratch` rejects `.pt` checkpoints on purpose to avoid accidentally fine-tuning when you intended to train from scratch.
- W&B is optional. If login is unavailable, training continues locally.
- This project forbids `augment.fliplr` and `augment.flipud`. The Energy Core object and its custom keypoint ordering are not flip-invariant, so the training script will raise an error if either flip augmentation is set above `0.0`.
- `test_pose.py` now supports both YAML config loading and CLI overrides via `--set key=value`, and accepts `valid` as an alias for the Ultralytics `val` split.

## Experiment Naming

For new experiments created from now on, use the forward-only naming template:

`v{x}_{model_name}_{image_input_mode}_{model_size}{experiment_id}_{feature_or_target_chain}`

Segment meanings:

- `v{x}`: dataset or experiment family version, for example `v8`
- `model_name`: architecture family, for example `yolo11_pose`
- `image_input_mode`: image preprocessing mode, for example `gray` or `color`
- `model_size{experiment_id}`: model size plus experiment serial, for example `s16`, `s17`, `n12`
- `feature_or_target_chain`: concise suffix describing the training target, data variant, or key feature stack

Examples:

- `v8_yolo11_pose_gray_s16_scalebalance_v2_from_s14_ms025_augmax_e500`
- `v8_yolo11_pose_gray_s17_scalebalance_v2_from_s12_ms025_augmax_e500`

Historical experiments keep their existing names and are not renamed retroactively.

## Fixed v8 plan

Run the fixed grayscale + realboost plan for `YOLO11n` and `YOLO11s`:

```bash
conda run -n tech_core_yolo python scripts/run_v8_gray_realboost_plan.py
```

This script rebuilds the `r3/r7` weighted train lists, reuses matching `v8_*gray*` runs when available, trains any missing fixed candidates, scans `conf` on the real subset, and writes the final best-config YAMLs plus the Chinese summary document.

## Scale-balanced pose dataset builder

Build the first moderate scale-balanced derivative of the `v8-add-blue-real-marker` pose dataset:

```bash
python scripts/build_scale_balanced_pose_dataset.py \
  --config configs/dataset_scale_balance_v8_moderate.yaml
```

Preview the planned distribution and feasibility without writing files:

```bash
python scripts/build_scale_balanced_pose_dataset.py \
  --config configs/dataset_scale_balance_v8_moderate.yaml \
  --dry-run
```

The builder writes a new derived dataset root at:

`data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.scalebalance_moderate_v2.yolov8`

Key outputs inside that directory:

- `train/images`, `valid/images`, `valid_raw/images`, and `test/images`: self-contained localized split directories
- `data.yaml`: train on `train/images`, validate on `valid/images`, test on `test/images`
- `data.raw_eval.yaml`: train on `train/images`, but evaluate on `valid_raw/images` and `test/images`
- `analysis/source_stats.{csv,json}`: raw split statistics at the configured `imgsz`
- `analysis/generated_stats.{csv,json}`: generated sample metadata and final combined counts
- `analysis/rejections.csv`: rejected generation attempts and reasons
- `analysis/review_manifest.csv`: review images exported per generated split/bucket/strategy group
- `train_balanced.txt`, `valid_balanced.txt`, `valid_raw.txt`, `test_raw.txt`: relative-path audit manifests inside the derived root

Train against the balanced validation split:

```bash
python train_pose.py \
  --data data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.scalebalance_moderate_v2.yolov8/data.yaml
```

Train the same balanced train list while keeping evaluation on raw `valid/test` for direct comparison with older runs:

```bash
python train_pose.py \
  --data data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.scalebalance_moderate_v2.yolov8/data.raw_eval.yaml
```
