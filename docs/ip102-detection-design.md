# DLCPD-25 与 IP102 联合模型设计

## 模型边界

最终模型由一个 ResNet-50 共享主干、一个 203 类分类头和一个 Faster R-CNN 检测分支组成。输入固定为 RGB 直接缩放 `224 x 224`；主干只计算一次，分类头读取 layer4，FPN/RPN/ROI 读取同一组主干特征。

检测仅覆盖 IP102 有框且能映射到 DLCPD-25 的 96 类害虫。其余病害、健康、缺陷和无框害虫只能整图分类。

## 编号合同

- DLCPD-25 公共 ID：`0-202`，分类和应用统一使用。
- IP102 原始检测标签：97 个，用于追溯。
- Faster R-CNN 内部标签：`1-96`，`0` 为背景。
- IP102 类别 50、51 合并到同一个检测标签和 DLCPD-25 `class_id 97`。

唯一映射为 `metadata/ip102-detection-class-map.json`。不得在训练时按目录或名称动态推断。

## 统一输入

历史分类模型使用 `resize 256 + center crop 224`，不能直接与检测框共用。联合模型改为整图直缩 `224 x 224`：

```text
x_new = x_old * 224 / original_width
y_new = y_old * 224 / original_height
```

随后做 ImageNet normalization。Faster R-CNN 内部 transform 必须配置为固定 224 且 identity normalization，避免二次处理。

## 训练方法

1. 从 ImageNet V2 权重初始化 ResNet-50，在 DLCPD-25 的 203 类分类任务上先训练 25 轮。
2. 在 DLCPD-25 train/val 上继续训练 5 轮，适配统一的 `224 x 224` 整图直缩预处理。
3. 交替执行 DLCPD-25 分类 step 和 IP102 检测 step，共进行 10 轮联合训练。
4. 分类 step 更新主干和分类头；检测 step 更新主干、FPN、RPN、ROI。
5. 分类 step 冻结检测头，检测 step 冻结分类头；主干使用较小学习率持续参与两种任务。
6. 训练配置和后处理参数冻结后，分别进行最终分类和检测测试，发布一个联合 checkpoint。

不得只用 IP102 长时间微调整个主干，否则会造成分类遗忘；也不得部署两套权重规避联合训练目标。

## 数据位置

```text
DLCPD-25 split: artifacts/data/v1/d3-r2/
IP102 contract: artifacts/data/ip102-detection-v1/
IP102 raw:      data/raw/ip102/downloads/Detection/VOC2007/
```

IP102 与 DLCPD-25 的测试集均不用于模型、数据比例、阈值或训练轮数选择。
