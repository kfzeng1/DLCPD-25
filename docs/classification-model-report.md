# DLCPD-25 分类模型报告

- 生成时间：2026-08-18 08:48:15 CST
- 模型阶段：Plan-A 分类专家（最终冻结）
- 权重：`artifacts/training/classification/convnext-tiny-384-plan-a-v1/checkpoints/best.pt`

## 1. 模型概况

| 项目 | 内容 |
|---|---|
| 架构 | ConvNeXt-Tiny |
| 输入尺寸 | 384×384 RGB |
| 参数量 | 28.00 M |
| 预训练 | ImageNet-1K |
| 主分类头 | 203 类 DLCPD-25 |
| 辅助头 | 22 宿主作物 + 4 标签属性（仅训练时使用） |
| 损失函数 | Focal Loss (γ=2) + 类别平衡权重 + 辅助交叉熵 |
| 采样策略 | 根号逆频率（第一训练阶段） |
| 优化器 | AdamW，warmup 3 epochs + cosine |
| AMP / EMA | BF16 / EMA 0.999 |
| 选用权重 | epoch 5 的 EMA 权重 |

## 2. 训练与模型选择

- 训练集：177,021 张；验证集：22,178 张；测试集：22,179 张。
- 训练计划 40 epochs，验证 Macro-F1 在第 5 轮达到最高，第 6~11 轮未突破，触发早停。
- 后续尝试 15 轮低学习率微调（逆频率采样），验证 Macro-F1 为 73.52，未超过第一训练阶段，因此不采用。
- 最终冻结第一阶段 epoch 5 EMA 权重。

## 3. 测试集最终指标

| 指标 | 结果 |
|---|---:|
| Test Top-1 | **89.2240%** |
| Test Top-5 | **97.7366%** |
| Test Macro-F1 | **75.3537%** |
| Test Balanced Accuracy | **77.5782%** |
| Test Loss | 0.373483 |
| 测试样本数 | 22179 |

### 与 v1 基线对比

| 指标 | v1 ResNet-50 联合模型 | 当前 ConvNeXt-Tiny | 变化 |
|---|---:|---:|---:|
| Test Top-1 | 91.3157% | 89.2240% | -2.09 |
| Test Top-5 | 96.4289% | 97.7366% | +1.31 |
| Test Macro-F1 | 75.4451% | 75.3537% | -0.09 |

结论：当前模型在 Top-5 上略优于 v1；Top-1 和 Macro-F1 基本接近 v1，但这是独立分类专家、推理更快，并且后续可与检测专家灵活编排。

## 4. 宿主作物表现

| 宿主 | 测试样本 | 准确率 |
|---|---:|---:|
| 番茄 | 4672 | 99.64% |
| 葡萄 | 2019 | 90.94% |
| 玉米 | 1872 | 91.13% |
| 棉花 | 1748 | 99.71% |
| 柑橘 | 1550 | 94.13% |
| 水稻 | 1449 | 94.82% |
| 苹果 | 1440 | 99.93% |
| 马铃薯 | 1155 | 98.18% |
| 大豆 | 978 | 97.55% |
| 桃 | 813 | 99.51% |
| 芒果 | 589 | 79.46% |
| 苜蓿 | 573 | 78.71% |
| 甜椒 | 537 | 96.09% |
| 草莓 | 526 | 99.05% |
| 小麦 | 475 | 84.21% |
| 樱桃 | 398 | 100.00% |
| 南瓜 | 357 | 40.06% |
| 蓝莓 | 332 | 99.70% |
| 树莓 | 278 | 100.00% |
| 甜菜 | 220 | 68.18% |
| 辣椒 | 170 | 99.41% |
| 大蒜 | 28 | 100.00% |

## 5. 最容易出错的 15 个类别

| class_id | 类别 | 宿主 | 测试样本 | F1 | 召回率 |
|---|---|---|---:|---:|---:|
| 91 | beet fly(beet) | 甜菜 | 7 | 0.00% | 0.00% |
| 132 | paddy stem maggot(rice) | 水稻 | 16 | 11.11% | 12.50% |
| 19 | Coccinellidae(soybean) | 大豆 | 15 | 18.18% | 13.33% |
| 61 | Rhammatocerus schistocercoides(soybean) | 大豆 | 15 | 18.75% | 20.00% |
| 177 | therioaphis maculata Buckton(alfalfa) | 苜蓿 | 16 | 21.43% | 18.75% |
| 44 | Mango flat beak leafhopper(mango) | 芒果 | 6 | 22.22% | 33.33% |
| 37 | Lagria villosa(soybean) | 大豆 | 15 | 24.24% | 26.67% |
| 121 | large cutworm(corn) | 玉米 | 30 | 28.12% | 30.00% |
| 123 | longlegged spider mite(wheat) | 小麦 | 15 | 30.77% | 26.67% |
| 9 | Bactrocera tsuneonis(citru) | 柑橘 | 10 | 31.58% | 30.00% |
| 30 | Euschistus heros ninfa(soybean) | 大豆 | 15 | 31.58% | 40.00% |
| 49 | Nipaecoccus vastalor(citru) | 柑橘 | 6 | 33.33% | 33.33% |
| 152 | rice leaf caterpillar(rice) | 水稻 | 29 | 34.48% | 34.48% |
| 65 | Spodoptera albula(soybean) | 大豆 | 15 | 34.78% | 26.67% |
| 26 | Edessa meditabunda(soybean) | 大豆 | 15 | 35.71% | 33.33% |

## 6. 最易混淆的类别对

| 次数 | 真实类别 | 预测类别 |
|---:|---|---|
| 214 | squash powdery mildew | tomato powdery mildew |
| 161 | apple black rot | apple scab |
| 45 | apple scab | apple black rot |
| 34 | corn（maize） northern leaf blight | corn curvularia leaf spot fungus |
| 33 | rice healthy | rice hispa |
| 32 | apple black rot | apple frogeye spot |
| 26 | Cicadellidae(mango) | Cicadella viridis(vitis) |
| 24 | corn healthy | maize dwarf mosaic virus |
| 21 | soybean healthy | potato healthy |
| 21 | blister beetle(alfalfa) | lytta polita(alfalfa) |
| 20 | Miridae(vitis) | alfalfa plant bug(alfalfa) |
| 19 | Miridae(vitis) | tarnished plant bug(alfalfa) |

## 7. 使用与复现

```bash
/home/zkf/pytorch-env/bin/python -m dlcpd25_v2.classification.evaluate \
  --checkpoint artifacts/training/classification/convnext-tiny-384-plan-a-v1/checkpoints/best.pt
```

测试集仅在配置冻结后评估一次，未参与训练、阈值或模型选择。
