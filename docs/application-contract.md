# 联合应用推理契约

J5 默认应用只加载 J4 的 `artifacts/releases/dlcpd25-ip102-joint-v1/`。一张图片只解码一次、只生成一个 RGB bicubic 直缩 `224 x 224` 张量，并调用一次 `forward_joint()`；分类与检测共享同一次 ResNet-50 主干前向。

## Predictor

```python
predictor = JointPredictor.from_bundle(bundle_path, device="auto")
result = predictor.predict(image)
```

`device="auto"` 优先使用 CUDA；CUDA 构建或预热失败时重新构建 CPU 模型。显式 `cuda` 失败则拒绝启动。应用启动前校验模型包全部 checksum、唯一 `.pt` 权重、架构、203/96 类数、taxonomy、IP102 映射、预处理、后处理和 torch/torchvision 版本。

结果 schema v1：

```text
schema_version, model_version, config_sha256, git_commit, device
classification
  class_id, official_name, host_zh, category_zh, detail_name
  confidence, top_k, low_confidence
detections[]
  class_id, official_name, host_zh, category_zh
  score, box_xyxy_original
original_size, inference_ms
```

分类 `class_id` 与检测 `class_id` 均使用冻结的 DLCPD-25 `0-202` 编号。内部 detector label `1-96` 不得暴露。`top_k` 概率相同时按 class ID 升序。

## 图片与坐标

支持 JPG、PNG、WEBP、BMP 和 TIFF。应用校正 EXIF、转换 RGB 并限制为 20 MiB、4000 万像素。损坏、未知扩展名或超限图片只返回稳定中文提示，不泄露堆栈。

联合模型使用完整图片直接缩放到 `224 x 224`，不裁剪、不保持宽高比。检测输出的 224 坐标分别乘以 `原宽/224` 和 `原高/224`，裁剪到原图边界后返回和绘制。

## 能力边界

- 分类覆盖 DLCPD-25 全部 203 类，包括害虫、病害、健康和生理缺陷。
- 检测只定位 IP102 有边界框监督并映射成功的 96 类害虫。
- 不得为其余病害、健康或缺陷伪造检测框。
- 分类置信度低于 `0.55` 时显示不确定提示；检测仅显示分数不低于 `0.5` 的框。
- 无检测框是正常结果，不代表分类失败或图片健康。

历史 `Predictor` 和分类 bundle 仅为 F0 回归兼容，不被默认 J5 服务加载。
