# 基于 DLCPD-25 数据集的农产品病虫害与缺陷分类目标检测系统

本目录为课程大作业提交目录，包含完整源代码、实验数据、冻结数据合同、最终模型、实验结果截图和实验报告。

## 目录结构

```text
project/
├── src/                 项目源代码
├── configs/             训练、评估与 Web 配置
├── data/                原始数据（DLCPD-25 + IP102 Detection）
├── artifacts/           冻结数据合同与划分
├── metadata/            类别层级与检测类别映射
├── models/              最终分类与检测模型权重
├── results/             测试指标、逐类结果、训练曲线、运行截图
├── docs/                实验报告、模型报告、架构 PPT/PDF
├── pyproject.toml       Python 工程配置
└── README.md            本说明
```

## 实验报告

- `docs/实验报告.md`
- `docs/实验报告.docx`
- `docs/实验报告.pdf`

## 复现步骤

推荐使用 Python 3.10+，安装与显卡匹配的 PyTorch。已有环境：

```bash
source /home/zkf/pytorch-env/bin/activate
```

### 1. 安装工程

```bash
cd project
pip install -e .
```

### 2. 分类模型评估

```bash
python -m dlcpd25_v2.classification.evaluate \
  --checkpoint models/classification_best.pt
```

### 3. 检测模型评估

```bash
python -m dlcpd25_v2.detection.evaluate \
  --checkpoint models/detection_best.pt
```

### 4. 重新训练

```bash
python -m dlcpd25_v2.classification.train --config configs/plan-a/classification.yaml
python -m dlcpd25_v2.detection.train --config configs/plan-a/detection.yaml
```

### 5. 启动双模型 Web 应用

```bash
python -m dlcpd25_v2.web --config configs/plan-a/app.yaml
```

浏览器打开 `http://127.0.0.1:7860`。

## 结果摘要

| 任务 | 关键指标 |
|---|---|
| DLCPD-25 分类 | Top-1 89.2240%，Top-5 97.7366%，Macro-F1 75.3537% |
| IP102 检测 | mAP50:95 34.3449%，AP50 59.6817%，Precision 88.1562% |

## 说明

- 原始数据和模型权重体积较大，仅保存在本地提交目录中，不推送到公开 GitHub；
- GitHub 仓库地址见项目根目录说明，远程仓库只包含可复现代码与配置。
