# V7 真实帧调参总结

本文总结了在 `v7i` 数据集上，对 `YOLO11n` 和 `YOLO11s` 进行独立真实帧调参的过程与结论。

## 调参目标

- 提升真实帧子集 `frame_*.jpg` 上的表现
- 同时保证模型在全量 `valid/test` 上保持稳定
- 允许 `n` 和 `s` 使用不同的训练参数与部署阈值

## 真实帧子集构成

- `train frame_*`: 19 张
- `valid frame_*`: 11 张
- `test frame_*`: 2 张

用户重点关注的两张 `valid` 图片为：

- `frame_000002_jpg.rf.3f72539f8f9a752bd4c55082e6c25afb.jpg`
- `frame_000024_jpg.rf.272f46943a1b332728fdd74658dfaf4c.jpg`

## 主要结论

- 对真实帧做过采样，对 `n` 和 `s` 都明显有效。
- 对 `n` 而言，`frameboost_r3` 配合更低学习率和更高的 `pose` 权重，效果最好。
- 对 `s` 而言，初始主要问题是实拍帧上的多目标误检，`frameboost_r7` 很有效地修复了这个问题。
- 当 `s` 在 `S2` 阶段已经达到“每张真实帧只预测 1 个目标”之后，继续 refine 的 `S3/S4` 虽然还能维持不错的全量指标，但没有超过 `S2` 在真实帧子集上的效果。
- 对最终胜者来说，`conf` 阈值已经不再是主要矛盾，因为在合理阈值范围内，它们都能稳定做到真实帧单目标预测。

## 基线与最终胜者对比

### YOLO11n

- 基线 checkpoint:
  `runs/pose/v7_yolo11n_refine_from_coarse_e30_seed3407/weights/best.pt`
- 基线 `valid` 真实帧指标:
  - `mean_pred_objects = 1.0`
  - `mean_norm_kp_err = 0.025489`
- 最终胜者:
  `runs/pose/v7_yolo11n_realframe_n2_fb3_pose26_e18/weights/best.pt`
- 胜者 `valid` 真实帧指标:
  - `mean_pred_objects = 1.0`
  - `mean_norm_kp_err = 0.006896`
- 胜者 `test` 真实帧指标:
  - `mean_pred_objects = 1.0`
  - `mean_norm_kp_err = 0.006610`
- 胜者全量指标:
  - `val mAP50-95(P) = 0.987368`
  - `test mAP50-95(P) = 0.979808`

### YOLO11s

- 基线 checkpoint:
  `runs/pose/v7_yolo11s_pretrained_full_e200/weights/best.pt`
- 基线 `valid` 真实帧指标:
  - `mean_pred_objects = 2.818182`
  - `max_pred_objects = 4`
  - `mean_norm_kp_err = 0.034511`
- 最终胜者:
  `runs/pose/v7_yolo11s_realframe_s2_fb7_e20/weights/best.pt`
- 胜者 `valid` 真实帧指标:
  - `mean_pred_objects = 1.0`
  - `max_pred_objects = 1`
  - `mean_norm_kp_err = 0.007186`
- 胜者 `test` 真实帧指标:
  - `mean_pred_objects = 1.0`
  - `mean_norm_kp_err = 0.008347`
- 胜者全量指标:
  - `val mAP50-95(P) = 0.991001`
  - `test mAP50-95(P) = 0.980806`

## 候选方案总结

### YOLO11n 候选

- `N0` (`v7_yolo11n_realframe_n0_clean_e12`)
  - 保住了单目标行为，但真实帧关键点误差反而比基线更差。
- `N1` (`v7_yolo11n_realframe_n1_fb3_e12`)
  - 加入真实帧过采样后，效果立刻大幅提升。
- `N2` (`v7_yolo11n_realframe_n2_fb3_pose26_e18`)
  - 是 `n` 路线的最终最优方案。虽然 `mean_best_iou` 略低于 `N1`，但关键点误差更低，而这次调参里关键点精度优先级更高，所以最终选择 `N2`。

### YOLO11s 候选

- `S1` (`v7_yolo11s_realframe_s1_fb3_e20`)
  - 成功解决多目标误检问题，真实帧表现显著提升。
- `S2` (`v7_yolo11s_realframe_s2_fb7_e20`)
  - 是 `s` 路线的最终最优方案，真实帧关键点误差最优，同时全量 `valid` 指标也最强。
- `S3` (`v7_yolo11s_realframe_s3_fb7_refine_e15`)
  - 相比 `S2`，全量指标基本接近，但真实帧子集退步。
- `S4` (`v7_yolo11s_realframe_s4_fb7_pose26_e15`)
  - 同样没有超过 `S2`，真实帧子集表现更差。

## 部署建议

### YOLO11n

- 训练参数文件:
  [energy_core_pose_v7_yolo11n_realframe_best.yaml](/data/home/sim6g/code/tech_core_yolo_train/configs/energy_core_pose_v7_yolo11n_realframe_best.yaml)
- 最佳 checkpoint:
  `runs/pose/v7_yolo11n_realframe_n2_fb3_pose26_e18/weights/best.pt`
- 推荐推理 `conf`:
  `0.25`

### YOLO11s

- 训练参数文件:
  [energy_core_pose_v7_yolo11s_realframe_best.yaml](/data/home/sim6g/code/tech_core_yolo_train/configs/energy_core_pose_v7_yolo11s_realframe_best.yaml)
- 最佳 checkpoint:
  `runs/pose/v7_yolo11s_realframe_s2_fb7_e20/weights/best.pt`
- 推荐推理 `conf`:
  `0.25`

## 经验总结

- 如果真实帧上的主要问题是“多预测了几个框”，那么先做真实帧过采样，往往比单纯调高推理阈值更有效。
- 当模型已经在真实帧子集上稳定做到“每张图只出 1 个目标”后，后续 refine 虽然可能继续改善全量指标，但也可能轻微伤害目标子集。这种情况下，应优先按目标子集选优。
- 对这个项目来说，`n` 并不需要比 `r3` 更强的过采样，真实帧问题已经被充分强调，再继续加权收益不大。
- 对 `s` 而言，`r7` 是值得的，因为它修复了基线里很明显的真实帧域偏差。
