# V8 Gray S12 训练经验总结

本文记录 `v8_yolo11s_gray_s12_realboostr7_augopen_e500` 的训练经验，重点回答它为什么能成为当前 `gray` 路线里最稳的 `YOLO11s` 候选。

## 结论

- 当前 `gray` 路线里，`S12` 是整体最均衡的一条：
  - full `valid/test mAP50-95(P)` 约为 `0.9812 / 0.9627`
  - real `valid/test` 都做到 `misses=0`
  - real `valid/test mean_pred_objects` 都是 `1.0`
  - real `valid/test mean_norm_kp_err` 约为 `0.0101 / 0.0140`
- 它不仅保住了全量指标，还把之前最顽固的 `1_001860...` 从 miss 变成了可稳定命中，只是这张图仍然是 `test` 子集里最大的残余误差来源。

## 有效经验

- `YOLO11s` 比 `YOLO11n` 更适合这条灰度路线。
  `s` 线在真实子集上更容易同时做到 “不 miss” 和 “单目标输出”，最后也明显比 `n` 线更稳。

- `realboost_r7` 比 `1boost/frame1boost` 更适合 `s` 线。
  `S12` 的关键不是只盯住某一类 hard case，而是继续保留全量数据，同时让 `frame_* + 1_* + 3_*` 在训练分布里足够强势。对 `s` 模型来说，这种“均衡但偏向 real”比只强化单一域更有效。

- 强一些的“非翻转增强全开”是加分项。
  `S12` 使用了 `randaugment + erasing + mosaic + mixup + copy_paste`，同时保留 `flipud=0.0`、`fliplr=0.0`。这说明在灰度设定下，`s` 模型对较强的光照/尺度/构图扰动仍然能吃进去，而且没有把真实子集行为打坏。

- 长跑到早停是必要的。
  `S12` 不是短训模型，它跑了很长时间才稳定下来。best epoch 在大约 `398`，而不是很早就出现，说明这条路线确实需要足够长的收敛窗口。

- `conf=0.25` 仍然是最合适的部署起点。
  在 `S12` 上，real `valid` 的最佳阈值仍然落在 `0.25`，没有必要为了压误检而再把阈值抬高。

## 反面经验

- 只做 `1boost` 或 `frame1boost`，虽然可能修复单个 hard case，但更容易让 `test` 子集的稳定性变差。
  这在 `n13/s13` 一类实验里表现得很明显：局部样本可能更强，但整体 real `test` 行为更容易飘。

- 对 `s` 线来说，后期最不该牺牲的是 “real test 不 miss”。
  一旦模型已经能稳定命中真实子集，后续更该优先看关键点误差和单目标行为，而不是只盯 full-map 继续堆训练。

## 推荐沿用的配方

- 起点：从成熟灰度 `s` checkpoint 继续训，而不是重新从很早期的灰度模型起跑
- 数据：`data.realboost_r7.yaml`
- 训练骨架：
  - `epochs=500`
  - `patience=80`
  - `imgsz=960`
  - `batch=4`
  - `optimizer=AdamW`
  - `lr0=1e-5`
  - `lrf=0.2`
  - `warmup_epochs=1`
  - `pose=36.0`
  - `preprocess.grayscale=true`
- 增强：
  - `auto_augment=randaugment`
  - `erasing=0.25`
  - `degrees=5.0`
  - `translate=0.1`
  - `scale=0.3`
  - `shear=3.0`
  - `perspective=0.001`
  - `mosaic=0.7`
  - `mixup=0.15`
  - `copy_paste=0.15`
  - `flipud=0.0`
  - `fliplr=0.0`

## 当前建议

- 如果 `gray` 路线需要先选一个可直接部署的 `YOLO11s`，优先用 `S12`。
- 后续如果继续探索 `gray s`，优先在 `S12` 附近做微调，而不是回退到更早的 `1boost` 基线重新找方向。
