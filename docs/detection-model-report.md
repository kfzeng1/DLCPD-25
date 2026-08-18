# IP102 检测模型报告

- 生成时间：2026-08-18 13:37:30 CST
- 模型阶段：Plan-A 检测专家（最终冻结）
- 权重：`artifacts/training/detection/fasterrcnn-convnext-tiny-640-v1/checkpoints/best.pt`（epoch 8 EMA）

## 1. 模型概况

| 项目 | 内容 |
|---|---|
| 架构 | ConvNeXt-Tiny-FPN + Faster R-CNN |
| 输入 | 最长边 640，保持宽高比 |
| 参数量 | 45.54 M |
| 检测类别 | 96 类（检测器标签 1..96，输出映射 DLCPD-25 class_id 0..202） |
| 骨干初始化 | DLCPD-25 分类专家 best.pt 的 EMA 骨干权重 |
| 采样 | Repeat Factor Sampling |
| 优化器 | AdamW，backbone lr=1e-5，head lr=1e-4 |
| AMP / EMA | BF16 / 0.999 |
| 早停 | val mAP50:95 连续 6 轮不提升 |

## 2. 训练过程

- 训练集：12,142 张；验证集：3,036 张；测试集：3,798 张。
- 计划 40 轮，第 8 轮验证 mAP 最高，第 9~14 轮未突破，触发早停。
- 最终选用 epoch 8 EMA 权重。

## 3. 测试集最终指标

| 指标 | 结果 |
|---|---:|
| Test mAP@0.5:0.95 | **34.3449%** |
| Test AP50 | **59.6817%** |
| Test AP75 | **34.8622%** |
| Test AR@100 | 54.2029% |
| Test AP small | 9.1343% |
| Test AP medium | 34.2585% |
| Test AP large | 35.7267% |
| Precision@0.5 | 88.1562% |
| Recall@0.5 | 76.7102% |
| 测试图片数 | 3798 |

### 与 v1 联合检测基线对比

| 指标 | v1 ResNet-50 联合模型 | 当前检测专家 |
|---|---:|---:|
| mAP@0.5:0.95 | 35.8823% | 34.3449% |
| AP50 | 65.5326% | 59.6817% |
| Precision@0.5 | 68.9095% | 88.1562% |
| Recall@0.5 | 80.1980% | 76.7102% |

结论：当前检测专家 Precision 明显更高、误报更少；Recall 和 AP50 低于 v1，整体 mAP 接近但仍差约 1.5 个百分点。

## 4. 表现最好的 10 类

| label | 类别 | AP | AP50 |
|---:|---|---:|---:|
| 3 | Ampelophaga | 70.48% | 94.31% |
| 14 | Cicadellidae | 67.60% | 97.57% |
| 38 | Sternochetus frigidus | 66.26% | 83.35% |
| 24 | Lycorma delicatula | 66.16% | 99.82% |
| 85 | sericaorient alismots chulsky | 64.88% | 88.10% |
| 33 | Potosiabre vitarsis | 60.85% | 90.45% |
| 44 | Xylotrechus | 60.00% | 94.94% |
| 36 | Salurnis marginella Guerr | 59.84% | 87.66% |
| 46 | alfalfa seed chalcid | 57.20% | 84.25% |
| 80 | rice leaf roller | 57.05% | 85.55% |

## 5. 表现最差的 10 类

| label | 类别 | AP | AP50 |
|---:|---|---:|---:|
| 8 | Brevipoalpus lewisi McGregor | -100.00% | -100.00% |
| 4 | Aphis citricola Vander Goot | 0.00% | 0.00% |
| 88 | therioaphis maculata Buckton | 0.00% | 0.00% |
| 64 | green bug | 1.11% | 3.83% |
| 22 | Limacodidae | 3.11% | 7.82% |
| 5 | Apolygus lucorum | 8.10% | 25.93% |
| 86 | small brown plant hopper | 8.28% | 16.81% |
| 92 | white margined moth | 8.75% | 16.16% |
| 7 | bird cherry-oataphid | 11.83% | 20.54% |
| 31 | Pieris canidia | 11.92% | 23.02% |

## 6. 限制与后续改进方向

- 训练只进行 14 轮即早停，后续可尝试更长训练、更低学习率或 sigmoid focal loss；
- AP small 仅 9.13%，小目标害虫仍是主要短板；
- 可用 IP102 原始分类图片扩充稀有害虫类别；
- 可尝试 YOLOv8/RT-DETR 等单阶段检测器作为对比。

## 7. 复现评估

```bash
/home/zkf/pytorch-env/bin/python -m dlcpd25_v2.detection.evaluate \
  --checkpoint artifacts/training/detection/fasterrcnn-convnext-tiny-640-v1/checkpoints/best.pt
```
