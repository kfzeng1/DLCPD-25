# 算法工程师工作单：J1-J4

## 目标合同

最终只交付一个联合 checkpoint：一个 RGB `224 x 224` 输入、一个 ResNet-50 主干、一次主干前向，同时输出 203 类整图分类和 96 类害虫检测框。

固定输入：

- DLCPD-25：`artifacts/data/v1/d3-r2/` 与 D5 taxonomy
- IP102：`artifacts/data/ip102-detection-v1/`
- 初始化权重：`artifacts/releases/dlcpd25-resnet50-weighted-v1/best.pt`
- 映射：`metadata/ip102-detection-class-map.json`
- 环境：`/home/zkf/pytorch-env`，RTX 4070 Laptop，约 7.62 GiB 显存

## J1：分类预处理适配

从历史分类权重加载 ResNet-50 和 203 类头。将 train/val 预处理统一为 RGB 整图 bicubic 直缩 `224 x 224` 加 ImageNet normalization，不使用 center crop。短周期微调，只按 DLCPD-25 val Macro-F1 选择 checkpoint，不读取分类 test。

必须报告：适配前后在同一分类 val 上的 Top-1、Top-5、Macro-F1、Balanced Accuracy；训练时长和峰值显存；checkpoint 重载与 checksum。J1 最终产物是联合模型的分类初始化，不是最终发布模型。

若新预处理使分类 val Top-1 比历史权重在旧预处理下下降超过 3 个百分点，应停止并分析，不得直接进入 J2。

## J2：交替训练链路与冒烟

实现联合模型和两个独立 DataLoader。固定 `classification_steps:detection_steps = 1:1`，在小规模样本上交替：

- 分类 step：更新共享主干与分类头；检测头参数不变。
- 检测 step：更新共享主干与 FPN/RPN/ROI；分类头参数不变。

必须通过：

- 一个 224 张量只运行一次主干，同时生成 `[N,203]` logits 和检测输出；
- IP102 图片和框同步缩放至 224，Faster R-CNN 不二次预处理；
- 两种 loss 和梯度均有限，任务头更新边界正确；
- 两类小样本 loss 均下降，且分类小样本不会因检测 step 改变分类头；
- AMP、显存、吞吐、CPU/CUDA、checkpoint 中断恢复和固定预测重载通过；
- checkpoint 包含共享主干、两个头、两个 loader 随机状态和训练 step 状态。

J2 根据实测决定 batch 和梯度累积，不改变 224 输入或 1:1 任务比例。

## J3：完整双数据集联合训练

从 J2 验收配置开始，完整使用 DLCPD-25 train 和 IP102 train。共享主干学习率初始为任务头的 `0.1` 倍。每个验证周期同时评估 DLCPD-25 val 和 IP102 val。

checkpoint 选择规则在训练前冻结：

1. 分类 val Top-1 相对 J1 最佳模型下降不超过 2 个百分点；
2. 在满足分类门槛的 checkpoint 中选择检测 val `mAP@0.5:0.95` 最高者；
3. 若无 checkpoint 满足门槛，回退调整主干学习率或任务比例，但仍只使用 val，不访问 test。

交付训练曲线、每任务 loss、两个 val 指标、逐类 AP、长尾/小目标分析、分类混淆摘要、最佳/最终 checkpoint、配置、环境、速度、显存和 checksum。

## J4：冻结评估与唯一模型包

总负责人先冻结权重、统一预处理、分类阈值、检测 score/NMS/max detections 和所有哈希，再授权最终测试。分类 test 和 IP102 test 各运行一次；测试结果不用于重训。

最终模型包：

```text
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

分类报告 Top-1、Top-5、Macro-F1、Balanced Accuracy；检测报告 mAP@0.5:0.95、AP50、Precision、Recall 和逐类 AP。IP102 test 缺源类 61，标记无支持。

## 禁止事项

- 读取 test 调参；修改原始数据、split、taxonomy 或映射；
- 使用两份模型权重、两套输入、两次主干前向；
- 改为 640 输入或只训练检测导致分类遗忘；
- 把内部检测标签暴露给应用；
- 自行进入下一阶段、提交或推送。
