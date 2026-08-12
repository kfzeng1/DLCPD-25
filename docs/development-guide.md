# 联合分类与目标检测开发指南

## 目录边界

```text
data/       DLCPD-25与IP102原始数据，只读
metadata/   203类taxonomy和IP102映射
project/    联合模型、训练、评估、推理和Web代码
scripts/    数据合同构建与验证脚本
artifacts/  数据合同、训练run和模型包，Git忽略
docs/       计划、职责、工作单、验收和日志
research/   论文、来源和翻译资料
```

不得移动或改写 `data/raw/`、T0 合同、DLCPD-25 固定 split、taxonomy 和历史分类模型包。

## 环境

```bash
/home/zkf/pytorch-env/bin/pip install -e 'project[training,dev]'
PYTHONPATH=project/src /home/zkf/pytorch-env/bin/python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

环境为 Python 3.12、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128；GPU 为 RTX 4070 Laptop，约 7.62 GiB 可用显存。

## 统一预处理

两个任务必须共享以下确定性输入合同：

1. 校正 EXIF 并转换 RGB；
2. 整张图片直接 bicubic resize 为 `224 x 224`，不 center crop、不保持纵横比；
3. IP102 框同步执行 `x *= 224/原宽`、`y *= 224/原高`；
4. 使用 ImageNet mean/std 归一化；
5. Faster R-CNN 内部不得再次 resize 或 normalization。

历史分类模型的预处理不同，只用于初始化。J1 必须用 DLCPD-25 train/val 适配新合同。

## 联合模型

```text
JointResNet50FasterRCNN
  backbone: ResNet-50
  classification_head: Linear(2048, 203)
  detection_neck: FPN
  detection_head: RPN + ROI Heads，背景+96前景
```

联合推理必须先计算一次 ResNet-50 body 特征，再将同一份特征送入分类池化头和 FPN。禁止用两套输入分别运行两次主干后仍称为共享一次前向。

最终 checkpoint 同时保存主干、分类头、FPN、RPN 和 ROI 参数，以及 optimizer、scheduler、AMP scaler、随机状态、预处理和映射哈希。

## J1 分类适配

从 `artifacts/releases/dlcpd25-resnet50-weighted-v1/best.pt` 初始化。只读取 DLCPD-25 train/val，使用统一 224 直缩预处理进行短周期微调；不读取分类 test。按 val Macro-F1 保存最佳 checkpoint，记录相对历史模型在同一 val 上的变化。

J1 不是从零重新训练分类模型，而是为联合输入合同建立可靠起点。

## J2-J3 交替训练

每轮以固定 `classification_steps:detection_steps = 1:1` 交替训练，两个 DataLoader 独立打乱并循环取样：

- 分类 step：只计算分类损失；共享主干和分类头更新，检测分支不更新。
- 检测 step：只计算 Faster R-CNN 损失；共享主干和检测分支更新，分类头不更新。

J3 固定训练 `10 epochs`，并使用三组学习率：主干 `1e-5`、203 类分类头 `1e-5`、检测头 `1e-4`。检测头需要更快适配 IP102，主干与分类头保持低学习率以防 J1 分类能力骤降。分类 step 使用 AMP FP16，检测 step 固定 FP32 以避免 Faster R-CNN 的数值不稳定；batch size 由 J2 显存实测决定。每个验证周期分别跑 DLCPD-25 val 与 IP102 val，不把两个指标简单相加掩盖退化。

## 评估与模型包

J3 从已验收的 J1 checkpoint 构建共享主干和分类头，并随机初始化检测分支。J2 r8 是小样本链路、显存和恢复能力的验收证据，不作为正式 J3 权重初始化。J3 开始前先执行 `--preflight-only`，它只校验冻结输入并评估该初始模型的分类 val，不创建训练目录、不执行训练 step、不读 test。只有 Top-1 达到 `88.783659%` 门槛才允许正式运行。每个训练周期后若分类 val Top-1 低于 `85%`，自动停止并标记阻塞。J3 checkpoint 先满足分类 val Top-1 相对 J1 下降不超过 2 个百分点，再以检测 val mAP@0.5:0.95 选优。J4 冻结后分别对两个 test 执行一次最终评估。

```bash
PYTHONPATH=project/src /home/zkf/pytorch-env/bin/python \
  -m dlcpd25_classifier.training.j3 \
  --config configs/j3.yaml --preflight-only
```

```text
artifacts/releases/<joint-model-version>/
  joint-best.pt
  manifest.json
  resolved-config.yaml
  preprocessing.json
  postprocessing.json
  taxonomy.json
  ip102-detection-class-map.json
  metrics-classification.json
  metrics-detection.json
  model-card.md
  checksums.sha256
```

发布包只含一份联合权重。IP102 test 缺源类 61，必须标记无支持，不伪造 AP。

## 联合应用

上传图片只生成一个 224 张量并调用一次联合模型。返回分类 Top-5 和检测框；检测框从 224 坐标缩放回原图。无框是合法结果，分类与检测结论允许不同。应用不得加载历史分类模型作为第二个后端。

## 验证

普通阶段运行定向测试与 `git diff --check`；J1、J2、J4、J5、F1 运行项目全量测试。不得重跑 DLCPD-25 D2-D4 或用 test 调参。
