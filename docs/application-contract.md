# 应用推理契约

当前已实现的是下述 schema v1 历史分类契约。J5 将改为只加载 J4 的一个联合模型包，并增加 `classification`、`detections` 和单一 `model_version`；详细目标合同见 `workplans/application-engineer-detection.md`。J5 验收前不得声称当前应用已接入联合模型。

## Predictor

统一入口为：

```python
predictor = Predictor.from_bundle(bundle_path, device="auto")
result = predictor.predict(image)
```

P1 使用 `create_fake_predictor()` 提供固定 203 维 logits 验证应用链路。P2 的 `from_bundle()` 先完成全部文件校验，再按 manifest 构建模型并严格加载 checkpoint。`device="auto"` 优先选择 CUDA 并执行预热，失败时重新构建 CPU 后端；显式 `cuda` 失败则拒绝启动。

结果 schema 版本为 `1`，字段包括：

- `model_version`、`data_version`、`config_sha256`、`git_commit` 和 `device`；
- `class_id`、`official_name`、`host_zh`、`category_zh` 和 `detail_name`；
- `confidence`、`top_k`、`low_confidence` 和 `inference_ms`。

`top_k` 每项包含稳定排名、class ID、官方细类名、宿主、四大类和概率。概率相同时按 class ID 升序。三级结果全部由同一 class ID 经冻结 taxonomy 得到。

`inference_ms` 从图片解码开始计时，覆盖 EXIF/RGB 处理、确定性 transform、后端执行和结果整理，不包含网络上传及页面渲染。

## 模型包

模型包目录不可覆盖，必须包含：

```text
best.pt
manifest.json
resolved-config.yaml
preprocessing.json
taxonomy.json
metrics.json
model-card.md
checksums.sha256
```

`manifest.json` schema 版本为 `1`，必须记录：模型与数据版本、Git commit、架构、`num_classes=203`、taxonomy 和预处理 SHA-256、置信度阈值、RGB、输入和 resize 尺寸、center crop、bicubic 插值、mean/std，以及 torch/torchvision 版本。

`checksums.sha256` 必须覆盖至少七个必需文件；A3 正式模型包当前包含 13 个条目。应用先校验清单内全部 SHA-256，再解析 manifest、preprocessing 和 taxonomy，最后核对 torch/torchvision 版本及 checkpoint；任何文件缺失、hash 不匹配、类别数错误或预处理不兼容都拒绝加载。

## 图片输入

支持 JPG、PNG、WEBP、BMP 和 TIFF。图片在应用侧校正 EXIF 方向并转换为 RGB，再调用算法侧共享的确定性 eval transform。默认限制为 20 MiB 和 4000 万像素；损坏图片、未知扩展名和超限图片返回稳定的用户提示，不向页面泄露堆栈。

J5 联合推理只解码一次图片，直接缩放为一个 `224 x 224` 张量并调用一次联合模型。检测框必须映射回原图坐标，并只暴露 DLCPD-25 公共类别 ID。
