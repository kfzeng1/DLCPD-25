# DLCPD-25 分类与目标检测系统工程

本目录是训练、评估和图片推理应用的唯一代码工程。现有 ResNet-50 提供 203 类整图分类；新增 `detection/` 使用 IP102 边界框训练共享 ResNet-50 主干的 Faster R-CNN 检测分支。

## 模块

```text
project/
  configs/                 # 训练和应用配置
  src/dlcpd25_classifier/  # 分类、检测、训练、推理与应用
  tests/                   # 单元和冒烟测试
  pyproject.toml           # Python 依赖和入口
```

默认模型为 ImageNet 预训练 ResNet-50。ConvNeXt-Tiny 只在时间和资源允许时作为对照；当前电脑的 RTX 4070 Laptop 8 GiB 显存不适合在项目周期内从零完成论文级 MAE、SimCLR v2 或 MoCo v3 预训练。

本机已有 `/home/zkf/pytorch-env`：PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128，CUDA 可用。工程直接复用这个环境。从仓库根目录安装开发依赖使用 `pip install -e 'project[dev]'`，应用依赖使用 `pip install -e 'project[app]'`。

开发流程、硬件限制和三位工程师职责见 `../docs/`。历史分类阶段 D0-F0 已全部通过；目标检测扩展按 T0-T4、F1 推进。

```bash
PYTHONPATH=project/src /home/zkf/pytorch-env/bin/python \
  -m dlcpd25_classifier.training.preflight
```

## 图片分类应用

应用默认加载已验收的 `dlcpd25-resnet50-weighted-v1` 冻结模型包。从仓库根目录运行：

```bash
/home/zkf/pytorch-env/bin/pip install -e 'project[app]'
PYTHONPATH=project/src /home/zkf/pytorch-env/bin/python \
  -m dlcpd25_classifier.web --host 127.0.0.1 --port 7860
```

打开 <http://127.0.0.1:7860>。`device: auto` 优先使用 CUDA，CUDA 初始化或预热失败时回退 CPU；显式配置 `cuda` 时失败会拒绝启动。置信度低于冻结阈值 `0.55` 的结果会显示不确定提示。

启动前会校验模型包全部 checksum、taxonomy、预处理、依赖版本和 checkpoint 契约。详细排错及固定演示样例见 `../docs/application-runbook.md`。
