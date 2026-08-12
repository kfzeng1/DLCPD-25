# 分类与目标检测开发指南

## 目录

```text
data/       DLCPD-25 与 IP102 原始数据、只读浏览视图
metadata/   203类taxonomy、别名、IP102到DLCPD-25映射
research/   论文、数据来源与翻译
project/    分类、检测、训练、评估、推理与Web代码
scripts/    可复现的数据和映射生成脚本
artifacts/  数据合同、训练run和模型包（Git忽略）
docs/       开发计划、职责、接口、验收和日志
```

原始数据不得移动或改写：DLCPD-25 位于 `data/raw/dlcpd25-203/`；IP102 检测数据位于 `data/raw/ip102/downloads/Detection/VOC2007/`。`data/views/` 只用于浏览。

## 环境

复用 `/home/zkf/pytorch-env`：Python 3.12、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128。RTX 4070 Laptop 可用显存约 7.62 GiB。

```bash
/home/zkf/pytorch-env/bin/pip install -e 'project[app,dev]'
PYTHONPATH=project/src /home/zkf/pytorch-env/bin/python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 分类基线

分类模型保持冻结：ResNet-50、RGB、resize 256、center crop 224、ImageNet normalization，输出 `[N,203]`。模型包为 `artifacts/releases/dlcpd25-resnet50-weighted-v1/`。正式分类 test 已消费，不重新训练或调阈值。

## 检测数据合同

IP102 原始标注为 Pascal VOC XML。官方 `trainval.txt` 用于生成固定 train/val；官方 `test.txt` 只在 T3 使用。数据入口必须保留三套编号：

- `ip102_class_ids`：原始标签，供追溯；
- `labels`：Faster R-CNN 内部连续标签 `1-96`；
- `dlcpd25_class_ids`：系统公共编号 `0-202`。

映射只来自 `metadata/ip102-detection-class-map.json`，不得靠目录排序或模糊字符串在训练时动态推断。

T0 必须冻结派生清洗规则：`IP087000986.xml` 的重复根只计一次；`IP046000898.xml` 的零宽框被过滤但同图有效框保留；正式划分外 5 张 JPEG 排除。不得改写官方 XML 或图片。

## 检测模型

使用共享 ResNet-50 主干、FPN、RPN 和 Faster R-CNN ROI 头。初始化时从现有分类 `best.pt` 加载 ResNet-50 主干和 203 类分类头。检测头为 96 个前景类加背景。

第一阶段冻结 ResNet 主干和分类头，仅训练 FPN/RPN/ROI；稳定后最多解冻 `layer4`。起始 batch 2、AMP，OOM 时降 batch 1 并使用梯度累积。检测输入由 torchvision transform 保持纵横比缩放，不套用分类的 224 center crop。

## 评估与模型包

T2 只报告 val 的 mAP@0.5、mAP@0.5:0.95、Precision、Recall 和逐类 AP。T3 在模型、阈值和预处理冻结后消费一次官方检测 test。

```text
artifacts/training/detection/<run-id>/
artifacts/releases/<detector-version>/
  best.pt
  manifest.json
  resolved-config.yaml
  preprocessing.json
  class-map.json
  metrics.json
  model-card.md
  checksums.sha256
```

版本目录不可覆盖。模型包必须记录分类 checkpoint 来源、映射 SHA-256、96 个检测类、203 类公共编号空间和依赖版本。

## 联合推理

同一上传图片进入分类与检测分支。应用显示整图分类 Top-5，并在原图上绘制检测框。无框是合法结果；低于检测阈值的框不展示；分类低置信度继续显示不确定提示。不得将检测能力扩展描述为病害或缺陷定位。

## 测试

普通任务运行相关测试和 `git diff --check`。T1、T3、T4、F1 运行：

```bash
PYTHONPATH=project/src /home/zkf/pytorch-env/bin/pytest -q project/tests
git diff --check
```

不重复执行 DLCPD-25 的耗时 D2-D4，也不重新运行已消费的分类 A3 test。
