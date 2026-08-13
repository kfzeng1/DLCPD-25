# DLCPD-25 分类与目标检测系统工程

本目录是训练、评估和图片推理应用的唯一代码工程。最终模型使用 DLCPD-25 与 IP102 交替联合训练：一个 ResNet-50 共享主干、一个 203 类分类头和一个 Faster R-CNN 检测分支，最终只发布一份联合权重。

## 模块

```text
project/
  configs/                 # 训练和应用配置
  src/dlcpd25_classifier/  # 分类、检测、训练、推理与应用
  tests/                   # 单元和冒烟测试
  pyproject.toml           # Python 依赖和入口
```

默认模型为 ImageNet 预训练 ResNet-50。ConvNeXt-Tiny 只在时间和资源允许时作为对照；当前电脑的 RTX 4070 Laptop 8 GiB 显存不适合在项目周期内从零完成论文级 MAE、SimCLR v2 或 MoCo v3 预训练。

本机已有 `/home/zkf/pytorch-env`：PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128，CUDA 可用。工程直接复用这个环境。联合训练及测试使用 `pip install -e 'project[training,dev]'`；应用使用 `pip install -e 'project[app]'`。

开发流程、硬件限制和三位工程师职责见 `../docs/`。历史分类阶段 D0-F0、IP102 T0、联合模型 J1-J5 和最终验收 F1 均已通过；当前工程进入维护与课程演示状态。

训练代码按职责分层：`training/joint.py` 放两个任务共用的优化器、梯度开关、检测 batch 拼接和随机状态；`training/j2.py` 只保留小样本链路验收；`training/j3.py` 只负责完整联合训练与断点续训；`detection/evaluation.py` 只负责 COCO 指标。新训练代码不能跨阶段导入私有函数。

```bash
PYTHONPATH=project/src /home/zkf/pytorch-env/bin/python \
  -m dlcpd25_classifier.training.preflight
```

## 联合分类检测应用

应用默认加载已验收的 `dlcpd25-ip102-joint-v1` 唯一联合模型包。从仓库根目录运行：

```bash
/home/zkf/pytorch-env/bin/pip install -e 'project[app]'
PYTHONPATH=project/src /home/zkf/pytorch-env/bin/python \
  -m dlcpd25_classifier.web --host 127.0.0.1 --port 7860
```

打开 <http://127.0.0.1:7860>。一次上传同时输出 203 类分类 Top-5 和 96 类害虫检测框。`device: auto` 优先使用 CUDA，CUDA 初始化或预热失败时回退 CPU；显式配置 `cuda` 时失败会拒绝启动。分类置信度低于 `0.55` 时显示不确定提示，检测只显示分数不低于 `0.5` 的框。

启动前会校验模型包全部 checksum、唯一权重、taxonomy、IP102 映射、预处理、后处理、依赖版本和 checkpoint 契约。详细排错及固定演示样例见 `../docs/application-runbook.md`。
