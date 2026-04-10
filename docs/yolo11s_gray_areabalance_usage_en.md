# YOLO11s Gray Area-Balance Two-Stage Workflow

This document describes the fully implemented workflow currently available in
the repository:

- train a new `YOLO11s pose gray` base model on the `v9` large dataset
- fine-tune a transfer model on the `2026-04-09` real-domain small dataset
- use explicit offline `bbox area ratio` balancing
- connect W&B through the local `127.0.0.1:10080` proxy during training

The active transfer-data source is now:

- raw export: `data/real_20260409T113532Z`
- repartitioned dataset: `data/real_20260409T113532Z.resplit_rand811_v1.yolov8`
- processed training dataset: `data/real_20260409T113532Z.areabalance_videofit_x10_v1.yolov8`

Do not use the old `real_20260408T075927Z` transfer chain anymore.

## 1. Important Files

Training entrypoint:

- [train_pose.py](/data/home/sim6g/code/tech_core_yolo_train/train_pose.py)

Dataset builders:

- [scripts/build_pose_area_balanced_dataset.py](/data/home/sim6g/code/tech_core_yolo_train/scripts/build_pose_area_balanced_dataset.py)
- [configs/v9_realmix_areabalance_x10_v1.yaml](/data/home/sim6g/code/tech_core_yolo_train/configs/v9_realmix_areabalance_x10_v1.yaml)
- [configs/real_20260409T113532Z_areabalance_videofit_x10_v1.yaml](/data/home/sim6g/code/tech_core_yolo_train/configs/real_20260409T113532Z_areabalance_videofit_x10_v1.yaml)

Training configs:

- base:
  [configs/energy_core_pose_v9_yolo11_pose_gray_s26_realmix_areabalancex10_v1_from_official_e120.yaml](/data/home/sim6g/code/tech_core_yolo_train/configs/energy_core_pose_v9_yolo11_pose_gray_s26_realmix_areabalancex10_v1_from_official_e120.yaml)
- transfer:
  [configs/energy_core_pose_v9_yolo11_pose_gray_s27_real20260409_areabalancevideofitx10_from_s26_e180.yaml](/data/home/sim6g/code/tech_core_yolo_train/configs/energy_core_pose_v9_yolo11_pose_gray_s27_real20260409_areabalancevideofitx10_from_s26_e180.yaml)

Evaluation:

- [test_pose.py](/data/home/sim6g/code/tech_core_yolo_train/test_pose.py)
- [scripts/eval_pose_size_buckets.py](/data/home/sim6g/code/tech_core_yolo_train/scripts/eval_pose_size_buckets.py)

## 2. Datasets Already Built

Base-model dataset:

- `data/Energy_Core_Position_Estimate.v9-960-960-realmix_dataset.areabalance_x10_v1.yolov8`
- `train=27780`
- `valid=564`
- `valid_raw=564`
- `test=371`

Transfer dataset:

- `data/real_20260409T113532Z.areabalance_videofit_x10_v1.yolov8`
- `train=1150`
- `valid=14`
- `valid_raw=14`
- `test=14`

Both datasets are already built and ready to use.

## 3. W&B Proxy

Training configs already include:

```yaml
wandb:
  enabled: true
  proxy:
    enabled: true
    url: http://127.0.0.1:10080
    no_proxy:
      - 127.0.0.1
      - localhost
```

Before importing W&B, `train_pose.py` exports:

- `HTTP_PROXY`
- `HTTPS_PROXY`
- `http_proxy`
- `https_proxy`
- `NO_PROXY`
- `no_proxy`

To override the proxy URL at runtime:

```bash
conda run -n tech_core_yolo python train_pose.py \
  --config <config.yaml> \
  --set wandb.proxy.url=http://127.0.0.1:10080
```

## 4. Base Training

Command:

```bash
conda run -n tech_core_yolo python train_pose.py \
  --config configs/energy_core_pose_v9_yolo11_pose_gray_s26_realmix_areabalancex10_v1_from_official_e120.yaml
```

Key points:

- model init: `yolo11s-pose.pt`
- input mode: `gray 3-channel`
- `imgsz=960`
- run name:
  `v9_yolo11_pose_gray_s26_realmix_areabalancex10_v1_from_official_e120`

Output directory:

- `runs/pose/v9_yolo11_pose_gray_s26_realmix_areabalancex10_v1_from_official_e120`

Important artifacts:

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `post_eval_valid.yaml`
- `post_eval_test.yaml`

## 5. Transfer Training

Command:

```bash
conda run -n tech_core_yolo python train_pose.py \
  --config configs/energy_core_pose_v9_yolo11_pose_gray_s27_real20260409_areabalancevideofitx10_from_s26_e180.yaml
```

Key points:

- model init:
  `runs/pose/v9_yolo11_pose_gray_s26_realmix_areabalancex10_v1_from_official_e120/weights/best.pt`
- dataset:
  `data/real_20260409T113532Z.areabalance_videofit_x10_v1.yolov8/data.raw_eval.yaml`
- run name:
  `v9_yolo11_pose_gray_s27_real20260409_areabalancevideofitx10_from_s26_e180`

Output directory:

- `runs/pose/v9_yolo11_pose_gray_s27_real20260409_areabalancevideofitx10_from_s26_e180`

## 6. Evaluation

Standard validation:

```bash
conda run -n tech_core_yolo python test_pose.py \
  --weights runs/pose/<run_name>/weights/best.pt \
  --data <dataset_root>/data.raw_eval.yaml \
  --split valid \
  --grayscale
```

Standard test:

```bash
conda run -n tech_core_yolo python test_pose.py \
  --weights runs/pose/<run_name>/weights/best.pt \
  --data <dataset_root>/data.raw_eval.yaml \
  --split test \
  --grayscale
```

Size-bucket evaluation:

```bash
conda run -n tech_core_yolo python scripts/eval_pose_size_buckets.py \
  --weights runs/pose/<run_name>/weights/best.pt \
  --data <dataset_root>/data.raw_eval.yaml \
  --split test \
  --imgsz 960 \
  --grayscale \
  --output runs/pose_test/<run_name>_sizebucket_test.json
```

## 7. Dataset Statistics

Each processed dataset root contains:

- `analysis/source_stats.csv`
- `analysis/generated_stats.csv`
- `analysis/rejections.csv`
- `analysis/bin_summary.json`
- `analysis/review_manifest.csv`

The most important file is `analysis/bin_summary.json`, which includes:

- `raw_train_count`
- `generated_count`
- `final_train_count`
- `generated_only_hist`
- `full_train_hist`

For both currently built datasets, `generated_only_hist` is exactly uniform.

## 8. Recommended Order

1. Train the base model `s26`
2. Confirm that `best.pt` exists
3. Train the transfer model `s27`
4. Run `test_pose.py`
5. Run `eval_pose_size_buckets.py`

## 9. Common Questions

### 1. W&B cannot connect

Check that the local proxy is alive:

```bash
curl -x http://127.0.0.1:10080 https://api.wandb.ai
```

### 2. I want a different proxy URL

Override it in the training command:

```bash
--set wandb.proxy.url=http://127.0.0.1:10080
```

### 3. Why is `real_20260408T075927Z` no longer used

Because the active transfer source has been switched to
`data/real_20260409T113532Z`. The old `20260408` chain may still exist on disk,
but it is no longer the recommended path.

### 4. Should I use `best.pt` or `last.pt`

This workflow consistently uses `best.pt` as the deliverable model.
