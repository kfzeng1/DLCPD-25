# 应用工程师工作单：T4

## 任务目标

把现有 203 类整图分类与 T3 发布的 96 类害虫检测模型接入同一图片上传流程。页面同时显示分类 Top-5 和目标框，并准确说明两种能力的覆盖边界。

应用工程师可在 T3 前只用假检测器准备接口和页面测试；真实模型联调必须使用 T3 验收后的完整模型包。

## 固定输入

- 分类模型包：`artifacts/releases/dlcpd25-resnet50-weighted-v1/`
- 检测模型包：由 T3 固定，路径写入应用配置
- taxonomy：由两个模型包携带并校验
- 公共编号：DLCPD-25 `class_id 0-202`
- 现有代码：`project/src/dlcpd25_classifier/inference/`、`web/`

## 执行内容

1. 扩展模型包校验，检查权重、结构、预处理、后处理、映射、版本和 checksum。
2. 实现联合 Predictor。一次图片解码后分别执行分类与检测，避免两套不一致的图片方向/RGB处理。
3. 分类结果保留宿主、四大类、具体类别、置信度、Top-5 和低置信提示。
4. 检测结果绘制 `xyxy` 框、类别名和置信度，支持无框、多框、重叠框及阈值过滤。
5. 页面明确展示：“整图分类覆盖 203 类；目标检测只定位 IP102 有框的 96 类害虫。”
6. 保留损坏图、超大图、CPU/CUDA 回退、模型缺失及校验失败处理，并更新启动文档。

## 推理返回合同

```text
schema_version
classification
  class_id, names, host, category, confidence, top_k, low_confidence
detections[]
  box_xyxy, score, class_id, names
classification_model_version
detection_model_version
device
inference_ms
```

检测内部 `1-96` 不得出现在公共结果。无检测框是合法结果，不代表分类失败。分类结果和框类别可能不同，界面不得强行合并为同一个结论。

## 交付与验收

- 联合推理内核、Web 页面、配置和自动化测试；
- 假模型覆盖无框、单框、多框、低分框和非法输出；
- 真实 CPU/CUDA 固定样例结果、推理耗时和显存记录；
- 浏览器桌面/移动视口检查，框的位置、缩放和标签不重叠；
- 模型包缺失、checksum/taxonomy/映射不一致时拒绝启动；
- JPG、PNG、WEBP、BMP、TIFF、EXIF 方向、损坏图和超限图路径通过；
- 分类功能回归、检测定向测试、项目全量测试和 `git diff --check` 通过；
- `docs/application-contract.md` 与 `docs/application-runbook.md` 更新为联合系统。

## 禁止事项

- 修改训练数据、split、mapping、taxonomy、训练代码或模型权重；
- 加载未经 T3 验收的裸 checkpoint；
- 在应用侧重新实现或擅自修改训练预处理、NMS 和阈值；
- 将整图、Grad-CAM 或分类热区伪装为目标框；
- 宣称可以定位没有框标注的病害、健康和缺陷；
- 自行执行 F1、提交或推送。
