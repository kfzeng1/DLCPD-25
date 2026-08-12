# IP102 目标检测扩展

## 系统边界

系统保留现有 DLCPD-25 ResNet-50 的 203 类整图分类能力，并在同一 ResNet-50 主干上增加 FPN、RPN 和 Faster R-CNN ROI 检测头。分类头输出 DLCPD-25 `class_id 0-202`；检测结果也统一输出该编号，但只能定位 IP102 提供边界框的害虫类别。没有边界框标注的病害、缺陷和健康类别仍只能整图分类。

## 类别编号

`metadata/ip102-detection-class-map.json` 是唯一映射合同，由 `scripts/build_ip102_detection_mapping.py` 从官方 IP102 类别表、全部 VOC XML 和冻结的 DLCPD-25 taxonomy 生成。

- IP102 检测标注实际出现 97 个源标签。
- 映射后对应 96 个 DLCPD-25 类别；IP102 的 `legume blister beetle` 与 `blister beetle` 在 DLCPD-25 中合并为 `class_id 97`。
- Faster R-CNN 内部使用连续标签 `1-96`，`0` 保留为背景。这只是损失函数需要的内部编号。
- 数据审计保留原始 IP102 ID；训练目标同时保存内部标签和 DLCPD-25 ID；推理出口只返回 DLCPD-25 `class_id 0-202`。

因此，不允许直接把 IP102 的 `0-101` 写入应用结果，也不允许让检测头输出稀疏的 203 维前景类别。这两种做法分别会造成编号冲突和大量无标注类别参与检测损失。

## 权重复用

`build_shared_detection_model` 从现有发布包 `best.pt` 加载：

- ResNet-50 卷积主干权重；
- 203 类分类头权重；
- 冻结的 BatchNorm 统计。

FPN、RPN 和 ROI 检测头为新增参数。初始训练冻结 ResNet-50 主干和 203 类分类头，仅训练检测分支；稳定后可只解冻 `layer4` 做小学习率微调。分类头不得使用 IP102 检测损失更新。

## 数据位置

```text
data/raw/ip102/downloads/Detection/VOC2007/
├── JPEGImages/
├── Annotations/
└── ImageSets/Main/
    ├── trainval.txt
    └── test.txt
```

官方 `test.txt` 保留到最终评估。验证集应从 `trainval.txt` 中按类别和图片分组一次性生成，训练过程中不得读取官方测试指标调参。
