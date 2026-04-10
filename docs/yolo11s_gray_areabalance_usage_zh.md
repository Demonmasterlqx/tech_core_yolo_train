# YOLO11s Gray Area-Balance 两阶段训练使用文档

本文档说明当前仓库里已经实现好的整套流程，目标是：

- 先基于 `v9` 大数据集训练新的 `YOLO11s pose gray` 基模
- 再基于 `2026-04-09` 真实小数据集做迁移训练
- 使用显式的离线 `bbox area ratio` 分布控制
- 训练时通过本地 `127.0.0.1:10080` 代理接入 W&B

当前生效的小数据集数据源是：

- 原始导出：`data/real_20260409T113532Z`
- 重分割后：`data/real_20260409T113532Z.resplit_rand811_v1.yolov8`
- 处理后训练集：`data/real_20260409T113532Z.areabalance_videofit_x10_v1.yolov8`

不要再使用旧的 `real_20260408T075927Z` 迁移链路。

## 1. 关键文件

训练入口：

- [train_pose.py](/data/home/sim6g/code/tech_core_yolo_train/train_pose.py)

数据构造：

- [scripts/build_pose_area_balanced_dataset.py](/data/home/sim6g/code/tech_core_yolo_train/scripts/build_pose_area_balanced_dataset.py)
- [configs/v9_realmix_areabalance_x10_v1.yaml](/data/home/sim6g/code/tech_core_yolo_train/configs/v9_realmix_areabalance_x10_v1.yaml)
- [configs/real_20260409T113532Z_areabalance_videofit_x10_v1.yaml](/data/home/sim6g/code/tech_core_yolo_train/configs/real_20260409T113532Z_areabalance_videofit_x10_v1.yaml)

训练配置：

- 基模：
  [configs/energy_core_pose_v9_yolo11_pose_gray_s26_realmix_areabalancex10_v1_from_official_e120.yaml](/data/home/sim6g/code/tech_core_yolo_train/configs/energy_core_pose_v9_yolo11_pose_gray_s26_realmix_areabalancex10_v1_from_official_e120.yaml)
- 迁移：
  [configs/energy_core_pose_v9_yolo11_pose_gray_s27_real20260409_areabalancevideofitx10_from_s26_e180.yaml](/data/home/sim6g/code/tech_core_yolo_train/configs/energy_core_pose_v9_yolo11_pose_gray_s27_real20260409_areabalancevideofitx10_from_s26_e180.yaml)

评测：

- [test_pose.py](/data/home/sim6g/code/tech_core_yolo_train/test_pose.py)
- [scripts/eval_pose_size_buckets.py](/data/home/sim6g/code/tech_core_yolo_train/scripts/eval_pose_size_buckets.py)

## 2. 当前已经生成好的数据集

基模数据集：

- `data/Energy_Core_Position_Estimate.v9-960-960-realmix_dataset.areabalance_x10_v1.yolov8`
- `train=27780`
- `valid=564`
- `valid_raw=564`
- `test=371`

迁移数据集：

- `data/real_20260409T113532Z.areabalance_videofit_x10_v1.yolov8`
- `train=1150`
- `valid=14`
- `valid_raw=14`
- `test=14`

说明：

- 这两个数据集都已经正式生成完成
- 可以直接拿其中的 `data.raw_eval.yaml` 开始训练

## 3. W&B 代理

训练配置里已经内置：

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

`train_pose.py` 会在导入 W&B 前自动设置：

- `HTTP_PROXY`
- `HTTPS_PROXY`
- `http_proxy`
- `https_proxy`
- `NO_PROXY`
- `no_proxy`

如果你要临时覆盖代理地址：

```bash
conda run -n tech_core_yolo python train_pose.py \
  --config <config.yaml> \
  --set wandb.proxy.url=http://127.0.0.1:10080
```

## 4. 基模训练

命令：

```bash
conda run -n tech_core_yolo python train_pose.py \
  --config configs/energy_core_pose_v9_yolo11_pose_gray_s26_realmix_areabalancex10_v1_from_official_e120.yaml
```

关键信息：

- 模型起点：`yolo11s-pose.pt`
- 输入：`gray 3-channel`
- `imgsz=960`
- run 名称：
  `v9_yolo11_pose_gray_s26_realmix_areabalancex10_v1_from_official_e120`

训练输出：

- `runs/pose/v9_yolo11_pose_gray_s26_realmix_areabalancex10_v1_from_official_e120`

重点文件：

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `post_eval_valid.yaml`
- `post_eval_test.yaml`

## 5. 迁移训练

命令：

```bash
conda run -n tech_core_yolo python train_pose.py \
  --config configs/energy_core_pose_v9_yolo11_pose_gray_s27_real20260409_areabalancevideofitx10_from_s26_e180.yaml
```

关键信息：

- 起点模型：
  `runs/pose/v9_yolo11_pose_gray_s26_realmix_areabalancex10_v1_from_official_e120/weights/best.pt`
- 数据集：
  `data/real_20260409T113532Z.areabalance_videofit_x10_v1.yolov8/data.raw_eval.yaml`
- run 名称：
  `v9_yolo11_pose_gray_s27_real20260409_areabalancevideofitx10_from_s26_e180`

训练输出：

- `runs/pose/v9_yolo11_pose_gray_s27_real20260409_areabalancevideofitx10_from_s26_e180`

## 6. 评测

标准评测：

```bash
conda run -n tech_core_yolo python test_pose.py \
  --weights runs/pose/<run_name>/weights/best.pt \
  --data <dataset_root>/data.raw_eval.yaml \
  --split valid \
  --grayscale
```

测试集：

```bash
conda run -n tech_core_yolo python test_pose.py \
  --weights runs/pose/<run_name>/weights/best.pt \
  --data <dataset_root>/data.raw_eval.yaml \
  --split test \
  --grayscale
```

尺寸分桶评测：

```bash
conda run -n tech_core_yolo python scripts/eval_pose_size_buckets.py \
  --weights runs/pose/<run_name>/weights/best.pt \
  --data <dataset_root>/data.raw_eval.yaml \
  --split test \
  --imgsz 960 \
  --grayscale \
  --output runs/pose_test/<run_name>_sizebucket_test.json
```

## 7. 数据构造统计怎么看

每个处理后数据集根目录下都有：

- `analysis/source_stats.csv`
- `analysis/generated_stats.csv`
- `analysis/rejections.csv`
- `analysis/bin_summary.json`
- `analysis/review_manifest.csv`

最关键的是 `analysis/bin_summary.json`：

- `raw_train_count`
- `generated_count`
- `final_train_count`
- `generated_only_hist`
- `full_train_hist`

当前两份数据集的 `generated_only_hist` 都是严格均匀分布。

## 8. 推荐训练顺序

1. 先训练基模 `s26`
2. 基模完成后确认 `best.pt` 存在
3. 再训练迁移模型 `s27`
4. 最后跑 `test_pose.py` 和 `eval_pose_size_buckets.py`

## 9. 常见问题

### 1. W&B 连不上

先确认本机代理监听：

```bash
curl -x http://127.0.0.1:10080 https://api.wandb.ai
```

### 2. 想切别的代理地址

直接在训练命令里覆盖：

```bash
--set wandb.proxy.url=http://127.0.0.1:10080
```

### 3. 为什么不用 `real_20260408T075927Z`

因为当前迁移源已经改成 `data/real_20260409T113532Z`，旧 `20260408` 链路保留在磁盘上，但不再是当前推荐方案。

### 4. 训练时应该用 `best.pt` 还是 `last.pt`

当前方案统一用 `best.pt` 作为交付模型。
