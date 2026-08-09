# DLCPD-25 分类系统工程

本目录是训练、评估和图片推理应用的唯一代码工程。数据、论文和研究资料位于仓库根目录的 `data/`、`research/` 和 `metadata/`，不会复制到 `project/`。

## 模块

```text
project/
  configs/                 # 训练和应用配置
  src/dlcpd25_classifier/  # 类别读取、数据集、模型、训练、推理、应用
  tests/                   # 单元和冒烟测试
  pyproject.toml           # Python 依赖和入口
```

主线模型为 ImageNet 预训练 ConvNeXt-Tiny，ResNet-50 是基线。当前电脑的 RTX 4070 Laptop 8 GiB 显存适合微调，不适合在本项目周期内从零完成论文级 MAE、SimCLR v2 或 MoCo v3 预训练。

本机已有 `/home/zkf/pytorch-env`：PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128，CUDA 可用。工程直接复用这个环境。数据和算法阶段使用 `pip install -e '.[dev]'`；到 P2 应用阶段再使用 `pip install -e '.[app]'` 安装 Gradio，避免提前引入不需要的应用依赖。

开发流程、硬件限制和三位工程师职责见 `../docs/`。用户按 `workflow.md` 逐阶段派工，总负责人按 `acceptance-checklist.md` 复验后才能推进。
