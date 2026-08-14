# 基于 DLCPD-25 的农产品病虫害与缺陷分类目标检测系统

本工程实现农产品图像的联合分类与目标检测。模型以 ResNet-50-FPN 为共享主干，一次前向同时输出：

- DLCPD-25 的 203 类整图分类 Top-5；
- IP102 中与 DLCPD-25 对齐的 96 类害虫检测框。

病害、健康和非生物/生理缺陷只有图像级标签，因此只提供分类结果；系统不会为没有边界框标注的类别生成虚假检测框。

## 模型结构

```text
RGB 图片
  -> EXIF 校正与 RGB 转换
  -> Bicubic 直缩 224 x 224
  -> ResNet-50 共享特征主干
      |-- 全局池化 + Linear(2048, 203) -> 分类 Top-5
      `-- FPN + RPN + ROI Heads         -> 96 类害虫检测框
```

分类与检测使用同一输入张量和同一份主干特征。检测框在推理后从 `224 x 224` 坐标反算至原图。

## 目录结构

```text
project/
  configs/                    # 应用、分类训练和联合训练配置
  src/dlcpd25_classifier/
    data/                     # DLCPD-25 Dataset
    detection/                # IP102 Dataset、映射、模型与 COCO 评估
    inference/                # 模型包校验、图片处理与联合推理
    models/                   # ResNet-50 分类模型构造
    training/                 # 分类训练、交替联合训练与模型发布
    web/                      # Gradio 应用
  tests/                      # 数据、模型、训练、推理和页面测试
  pyproject.toml              # Python 包与依赖
```

## 环境安装

推荐 Python 3.10 及以上版本。CUDA 训练环境需要与显卡驱动匹配的 PyTorch；仅使用 CPU 也可以运行应用，但推理速度较慢。

```bash
cd project
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[app,training,dev]'
```

本项目完成时使用 PyTorch `2.11.0+cu128`、torchvision `0.26.0+cu128` 和 RTX 4070 Laptop GPU。

## 运行应用

默认配置为 `configs/app.yaml`。模型包默认位于工程目录同级的：

```text
artifacts/releases/dlcpd25-ip102-joint-v1/
```

该模型包需包含 `joint-best.pt`、manifest、预处理与后处理配置、taxonomy、IP102 映射、指标和 `checksums.sha256`。模型权重体积较大，不包含在源码提交中。

```bash
cd project
python -m dlcpd25_classifier.web --host 127.0.0.1 --port 7860
```

也可以使用安装后的命令：

```bash
dlcpd25-web --host 127.0.0.1 --port 7860
```

浏览器打开 <http://127.0.0.1:7860>。`device: auto` 优先使用 CUDA，CUDA 初始化失败时自动回退 CPU。

## 配置说明

- `configs/app.yaml`：最终联合推理应用；
- `configs/j3.yaml`：双数据集 `1:1` 交替联合训练；
- `configs/j4.yaml`：冻结模型评估与发布包构建；
- `configs/train.yaml`：203 类分类训练；
- `configs/j1.yaml`、`configs/j2.yaml`：预处理适配和联合训练链路验证配置。

配置中的 `../data`、`../metadata` 和 `../artifacts` 均相对于本工程目录解析。若只提交源码，应将数据集和模型权重作为独立附件，并按配置中的相对路径放置。

## 训练方法

联合训练对两个 DataLoader 进行 `1:1` 交替采样：

```text
DLCPD-25 batch -> 分类交叉熵 -> 更新共享主干与分类头
IP102 batch    -> 检测损失   -> 更新共享主干与检测分支
```

正式配置训练 10 轮，分类步骤使用 AMP FP16，检测步骤使用 FP32。模型先满足分类验证集精度门槛，再按照检测验证集 `mAP@0.5:0.95` 选择最佳 checkpoint。

训练需要同级目录中的 DLCPD-25、IP102、固定 split、taxonomy、类别映射和分类初始化权重。完整路径与 SHA-256 均已固定在训练配置中。

## 测试

不加载正式模型的快速测试：

```bash
cd project
pytest -q tests/test_taxonomy.py tests/test_detection_mapping.py
```

联合推理回归测试会加载正式模型包：

```bash
cd project
pytest -q tests/test_inference_j5.py
```

完整测试依赖同级的数据、metadata、artifacts 和仓库数据脚本：

```bash
cd project
pytest -q
```

## 最终指标

| 任务 | 指标 | 结果 |
|---|---|---:|
| DLCPD-25 分类 | Top-1 | 91.3157% |
| DLCPD-25 分类 | Top-5 | 96.4289% |
| DLCPD-25 分类 | Macro-F1 | 75.4451% |
| IP102 检测 | mAP@0.5:0.95 | 35.8823% |
| IP102 检测 | AP50 | 65.5326% |
| IP102 检测 | Precision | 68.9095% |
| IP102 检测 | Recall | 80.1980% |

统一使用 `224 x 224` 输入能够控制显存和训练时间，但会限制小目标检测能力。系统定位范围仅为具有 IP102 边界框的 96 类害虫，不代表能够定位 DLCPD-25 的全部 203 类。
