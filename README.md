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

## Offline pose augmentation builder

Build a self-contained offline-augmented derivative of the `v8-add-blue-real-marker` pose dataset:

```bash
conda run -n tech_core_yolo python scripts/build_pose_augmented_dataset.py \
  --config configs/offline_pose_aug_medium.yaml
```

Preview the augmentation pipeline on a small subset and export review images without writing the full derived dataset:

```bash
conda run -n tech_core_yolo python scripts/build_pose_augmented_dataset.py \
  --config configs/offline_pose_aug_medium.yaml \
  --dry-run \
  --limit 20 \
  --visualize
```

Provided presets:

- `configs/offline_pose_aug_conservative.yaml`
- `configs/offline_pose_aug_medium.yaml`
- `configs/offline_pose_aug_aggressive.yaml`

Default medium output root:

`data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.offlineaug_medium_v1.yolov8`

Key outputs inside the derived root:

- `train/images` and `train/labels`: original train samples plus accepted offline augmentations
- `valid/images`, `valid_raw/images`, and `test/images`: copied raw evaluation splits
- `analysis/augment_log.{csv,json}`: accepted augmentation records with transform parameters
- `analysis/rejections.{csv,json}`: rejected augmentation attempts and reasons
- `analysis/review/**`: rendered bbox/keypoint review images
- `train_augmented.txt`, `valid_raw.txt`, `test_raw.txt`: relative-path audit manifests
- `data.yaml` and `data.raw_eval.yaml`: train-ready dataset entrypoints

Train against the derived dataset exactly like any other YOLO pose dataset:

```bash
conda run -n tech_core_yolo python train_pose.py \
  --data data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.offlineaug_medium_v1.yolov8/data.raw_eval.yaml
```

## Real-Only Dataset

Extract the filename-defined real-domain subset into a standalone dataset:

```bash
python scripts/build_real_only_pose_dataset.py \
  --dataset-root data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.yolov8
```

Defaults:

- source split scan: `train`, `valid`, `test`
- real-domain filename patterns: `frame_*.jpg`, `1_*.jpg`, `3_*.jpg`
- default output root:
  `data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.realonly_v1.yolov8`

Outputs:

- `data.yaml` with local `train/images`, `valid/images`, `test/images`
- `analysis/selection_manifest.csv`
- `analysis/selection_summary.json`

## Repartition Dataset

Repartition a self-contained YOLO pose dataset into a new shuffled `train/valid/test` split:

```bash
conda run -n tech_core_yolo python scripts/repartition_pose_dataset.py \
  --dataset-root data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.yolov8 \
  --train-ratio 0.7 \
  --valid-ratio 0.2 \
  --test-ratio 0.1
```

Preview the planned split counts without writing files:

```bash
conda run -n tech_core_yolo python scripts/repartition_pose_dataset.py \
  --dataset-root data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.yolov8 \
  --train-ratio 0.7 \
  --valid-ratio 0.2 \
  --test-ratio 0.1 \
  --dry-run
```

The repartition tool supports both:

- self-contained `train/valid/test` dataset roots
- list-file exports shaped like `dataset.yaml + splits/*.txt + images/** + labels/**`

Defaults:

- output root:
  `data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.resplit_v1.yolov8`
- all source `train/valid/test` samples are combined, shuffled with the provided seed, then repartitioned
- output filenames are prefixed with the original source split to avoid collisions

Outputs:

- `data.yaml` with local `train/images`, `valid/images`, `test/images`
- `analysis/repartition_manifest.csv`
- `analysis/repartition_summary.json`

Example for the 2026-04-08 real export:

```bash
conda run -n tech_core_yolo python scripts/repartition_pose_dataset.py \
  --dataset-root data/real_20260408T075927Z \
  --output-root data/real_20260408T075927Z.resplit_rand811_v1.yolov8 \
  --train-ratio 0.8 \
  --valid-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 52
```

Train the conservative S14-style transfer recipe on that split:

```bash
conda run -n tech_core_yolo python train_pose.py \
  --config configs/energy_core_pose_real_20260408_gray_transfer_from_s14_v1.yaml
```

## Dataset Merge

Merge multiple self-contained YOLO pose datasets by per-split sampling ratios:

```bash
python scripts/merge_pose_datasets.py \
  --config configs/dataset_merge_example.yaml
```

Sampling behavior:

- ratios are applied separately to `train`, `valid`, and `test`
- sample count is `floor(split_count * ratio)` for each input dataset and split
- sampling is deterministic by `sampling.seed`
- sampled files are copied into a new standalone dataset
- copied filenames are prefixed with `<input_name>__`

Example merge config:

```yaml
output:
  root: data/Energy_Core_Position_Estimate.merge_example_v1.yolov8

inputs:
  - name: v7
    root: data/Energy_Core_Position_Estimate.v7i.yolov8
    ratios:
      train: 0.10
      valid: 0.20
      test: 0.20
  - name: v8
    root: data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.yolov8
    ratios:
      train: 0.10
      valid: 0.20
      test: 0.20

sampling:
  seed: 52

merge:
  rename_mode: prefix_input_name

postprocess:
  scale_balance:
    enabled: false
```

Optional scale-balance postprocessing:

- set `postprocess.scale_balance.enabled=true`
- add `postprocess.scale_balance.builder_config=<scale-balance builder yaml>`
- the merge tool will first emit an intermediate merged dataset, then reuse
  `build_scale_balanced_pose_dataset.py` on top of that merged dataset

Current limitation:

- the optional scale-balance post-step is intended for the current Energy Core
  single-instance pose datasets that already match on `kpt_shape`, `nc`, and
  `names`
