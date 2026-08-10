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

默认模型为 ImageNet 预训练 ResNet-50。ConvNeXt-Tiny 只在时间和资源允许时作为对照；当前电脑的 RTX 4070 Laptop 8 GiB 显存不适合在项目周期内从零完成论文级 MAE、SimCLR v2 或 MoCo v3 预训练。

本机已有 `/home/zkf/pytorch-env`：PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128，CUDA 可用。工程直接复用这个环境。数据和算法阶段使用 `pip install -e '.[dev]'`；到 P1 应用阶段再使用 `pip install -e '.[app]'` 安装 Gradio，避免提前引入不需要的应用依赖。

开发流程、硬件限制和三位工程师职责见 `../docs/`。数据 D0-D5 已冻结；后续只执行 A1-A3、P1-P2 和 F0。A1 的快速数据检查入口为：

```bash
PYTHONPATH=project/src /home/zkf/pytorch-env/bin/python \
  -m dlcpd25_classifier.training.preflight
```

## P1 假模型应用

P1 页面使用固定 logits 验证图片处理、层级映射和界面，不代表真实模型结果。从仓库根目录运行：

```bash
/home/zkf/pytorch-env/bin/pip install -e 'project[app]'
PYTHONPATH=project/src /home/zkf/pytorch-env/bin/python \
  -m dlcpd25_classifier.web --host 127.0.0.1 --port 7860
```

打开 <http://127.0.0.1:7860>。真实冻结模型只能在 A3 通过后于 P2 接入。
