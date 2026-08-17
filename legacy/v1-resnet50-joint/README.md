# v1 ResNet-50 联合模型归档

旧版本使用一份共享 ResNet-50 权重，在 224×224 输入下同时完成：

- DLCPD-25 203 类整图分类；
- IP102 96 类害虫 Faster R-CNN 目标检测。

## 最终指标

| 任务 | 指标 | 结果 |
|---|---|---:|
| DLCPD-25 分类 | Top-1 | 91.3157% |
| DLCPD-25 分类 | Top-5 | 96.4289% |
| DLCPD-25 分类 | Macro-F1 | 75.4451% |
| IP102 检测 | mAP@0.5:0.95 | 35.8823% |
| IP102 检测 | AP50 | 65.5326% |

## 目录

- `docs/`：实验报告、提交说明、架构 PPT 和截图；
- `assets/model/`：旧 checkpoint（本地保留，不进入 Git）；
- 完整旧代码见 Git 分支 `legacy/v1-resnet50-joint` 和标签 `legacy-v1`。

旧模型指标同时复制在 `baselines/v1-resnet50-joint/`，作为 Plan-A 的对比基线。
