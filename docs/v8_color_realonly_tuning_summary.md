# V8 彩色 Real-Only 调参总结

本文记录了基于 `v8_yolo11s_color_s9_1boost_e500` 的彩色 real-only 迁移训练路线、候选结果和最终胜者。

## 数据与基线

- real-only 训练集: `/data/home/sim6g/code/tech_core_yolo_train/data/Energy_Core_Position_Estimate.v8-add-blue-real-marker.yolov8/train_real_only.txt`
- 真实子集模式: `frame_*.jpg, 1_*.jpg, 3_*.jpg`
- 真实子集计数: `train=45` / `valid=18` / `test=6`
- 基线 checkpoint: `/data/home/sim6g/code/tech_core_yolo_train/runs/pose/v8_yolo11s_color_s9_1boost_e500/weights/best.pt`
- 基线全量指标: `valid mAP50-95(P)=0.968159` / `test mAP50-95(P)=0.968341`
- 基线 real `valid`: `misses=0`, `mean_pred_objects=1.000000`, `mean_norm_kp_err=0.027307`
- 基线 real `test`: `misses=0`, `mean_pred_objects=1.166667`, `mean_norm_kp_err=0.047546`

## 候选结果

- `R0` / `v8_yolo11s_color_r0_realonly_base_e500`: `accepted=False`, `conf=0.25`, `valid_real_misses=0`, `valid_real_mean_pred_objects=1.000000`, `valid_real_mean_norm_kp_err=0.023209`, `test_real_misses=1`, `test_real_mean_pred_objects=0.833333`, `test_real_mean_norm_kp_err=0.016982`, `valid_mAP50-95(P)=0.920432`, `test_mAP50-95(P)=0.926001`
- `R1` / `v8_yolo11s_color_r1_realonly_augopen_e500`: `accepted=False`, `conf=0.25`, `valid_real_misses=0`, `valid_real_mean_pred_objects=1.000000`, `valid_real_mean_norm_kp_err=0.024087`, `test_real_misses=0`, `test_real_mean_pred_objects=1.000000`, `test_real_mean_norm_kp_err=0.051864`, `valid_mAP50-95(P)=0.826095`, `test_mAP50-95(P)=0.837182`

## 最终选择

- 胜者: `BASELINE` (`v8_yolo11s_color_s9_1boost_e500`)
- 最佳 checkpoint: `/data/home/sim6g/code/tech_core_yolo_train/runs/pose/v8_yolo11s_color_s9_1boost_e500/weights/best.pt`
- 推荐 `conf`: `0.25`
- `accepted=False`
- `valid mAP50-95(P)=0.968159`, `test mAP50-95(P)=0.968341`
- real `valid`: `misses=0`, `mean_pred_objects=1.000000`, `max_pred_objects=1`, `mean_norm_kp_err=0.027307`, `mean_best_iou=0.905116`
- real `test`: `misses=0`, `mean_pred_objects=1.166667`, `max_pred_objects=2`, `mean_norm_kp_err=0.047546`, `mean_best_iou=0.807455`

## 必查图片


## 最差样本

- real `valid` 最差 5 张: `[{"frame": "frame_000013_jpg.rf.49bc42ae6cde001c1a10cb9bf81389ad.jpg", "kp_err": 0.049648820198060335, "pred_objects": 1, "best_iou": 0.8433941365660226}, {"frame": "frame_000011_jpg.rf.3a6098dc72e4bdcf7823403aedf4b86e.jpg", "kp_err": 0.045900899595221194, "pred_objects": 1, "best_iou": 0.9474442284444766}, {"frame": "frame_000025_jpg.rf.691d6c9d4ec773a3f0f4e2ffc9fec10c.jpg", "kp_err": 0.04586464584138982, "pred_objects": 1, "best_iou": 0.9687908347138824}, {"frame": "frame_000024_jpg.rf.272f46943a1b332728fdd74658dfaf4c.jpg", "kp_err": 0.04569623764421002, "pred_objects": 1, "best_iou": 0.9744750249833056}, {"frame": "frame_000015_jpg.rf.944f8e37a6411a2ac81aad238b332291.jpg", "kp_err": 0.04500972311588536, "pred_objects": 1, "best_iou": 0.9391498016664185}]`
- real `test` 最差 5 张: `[{"frame": "1_001860_jpg.rf.45a18a99628a5c295b3d19ee4d5b52da.jpg", "kp_err": 0.18144995943170988, "pred_objects": 1, "best_iou": 0.36551297472703376}, {"frame": "frame_000038_jpg.rf.b46c6dc33d081a135ecf043ea9bb2b2c.jpg", "kp_err": 0.042197526629753734, "pred_objects": 1, "best_iou": 0.8795828148536917}, {"frame": "frame_000055_jpg.rf.1d67580f016f922d919d63052ea72a04.jpg", "kp_err": 0.04032702467353826, "pred_objects": 2, "best_iou": 0.7177273042157095}, {"frame": "3_000240_jpg.rf.68cc1702445ad1d825dd1e296654ff89.jpg", "kp_err": 0.008508834011228575, "pred_objects": 1, "best_iou": 0.9826449389704796}, {"frame": "3_000330_jpg.rf.fbaf82226f2b40202abc1169aafdd491.jpg", "kp_err": 0.006877423887714974, "pred_objects": 1, "best_iou": 0.9195916561470633}]`
