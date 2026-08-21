# DLCPD-25 农产品病虫害与缺陷分类目标检测系统

本仓库为课程大作业的工程仓库。**最终提交目录为 [`project/`](project/)**，其中包含完整源代码、实验报告、实验数据、冻结数据合同、模型权重、结果截图和支撑材料。

## 课题

基于 DLCPD-25 数据集的农产品病虫害与缺陷分类目标检测系统，采用 Plan-A 双专家架构：

- **分类专家**：ConvNeXt-Tiny @ 384，DLCPD-25 203 类细粒度分类；
- **检测专家**：ConvNeXt-Tiny-FPN + Faster R-CNN @ 640，IP102 96 类害虫目标检测；
- **推理系统**：双模型 Web 应用，输出分类 Top-5、害虫检测框和标注图。

## 提交目录

```text
project/
├── src/            项目源代码
├── configs/        训练、评估与 Web 配置
├── data/           原始实验数据
├── artifacts/      冻结数据合同与划分
├── metadata/       类别层级与检测类别映射
├── models/         最终模型权重
├── results/        测试指标、逐类结果、曲线与截图
└── docs/           实验报告、模型报告、架构 PPT/PDF
```

## GitHub 说明

- 代码仓库：<https://github.com/kfzeng1/DLCPD-25>
- 远程仓库只提供工程代码与配置，以下文件**不会推送**：

- `data/` 原始数据
- `research/` 研究资料
- `docs/` 内部文档与报告
- `project/data`、`project/artifacts`、`project/metadata`
- `project/models`、`project/results`、`project/docs`
- `artifacts/data` 与训练产物

这些内容保留在本地，随 `project/` 目录线下提交。

## 快速复现

```bash
cd project
source /home/zkf/pytorch-env/bin/activate
pip install -e .

# 分类测试评估
python -m dlcpd25_v2.classification.evaluate \
  --checkpoint models/classification_best.pt

# 检测测试评估
python -m dlcpd25_v2.detection.evaluate \
  --checkpoint models/detection_best.pt

# 双模型 Web
python -m dlcpd25_v2.web --config configs/plan-a/app.yaml
```

## 结果摘要

| 任务 | 测试集结果 |
|---|---|
| 分类 Top-1 / Top-5 | 89.2240% / 97.7366% |
| 分类 Macro-F1 | 75.3537% |
| 检测 mAP@0.5:0.95 | 34.3449% |
| 检测 AP50 | 59.6817% |
