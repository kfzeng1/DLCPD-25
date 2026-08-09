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

开发流程、硬件限制、三位工程师职责和验收标准见 `../docs/project-plan.md`、`../docs/development-guide.md` 和 `../docs/team-responsibilities.md`。
