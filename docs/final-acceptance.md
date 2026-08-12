# F0 历史分类基线验收报告

本报告只证明 203 类整图分类基线已经完成，不代表后续联合模型已经完成。后续状态以 `acceptance-checklist.md` 中的 T0、J1-J5、F1 为准。

验收日期：2026-08-11
验收输入：P2 commit `e9f1463481b4f7e201a40f1b5052840f540ddcdc`

## 版本矩阵

| 层级 | 冻结版本 | 关键校验 |
|---|---|---|
| 数据 | `data-v1-d5-r1` | taxonomy `5cfa1a261b1a9fbb80adf24f299bca0883a42dd523914a70234f31dbf748bd31` |
| train/val/test | 177,021 / 22,178 / 22,178 | `af457fcd...d778` / `a5db4559...0833` / `23897e0a...1dc8` |
| 模型 | `dlcpd25-resnet50-weighted-v1` | `best.pt` `68fc44f1b4acfe321e5590b5f27dead65b735a777798c141c6528c510e11eabd` |
| 模型包 | A3 ResNet-50 release | checksum 清单 `b5b970ebe0f4cae436115fd7449e43f4f49ee6f361724e81b7bb7e4c4128af6a` |
| 应用 | P2 `e9f1463` | `app.yaml` `06cb8c992642e3fb972e60de3f9fd15da562af0e504a5e62306c6b87b39b287c` |

模型包记录的 Git commit 是冻结前输入 `1d34280`，A3 验收提交是 `1099300`，P2 应用提交是 `e9f1463`。这些提交表示顺序依赖链，不要求三个字段相同。

## 最终结果

Test Top-1 `88.5517%`、Top-5 `95.7796%`、Macro-F1 `71.2177%`、Balanced Accuracy `71.2654%`。测试集只在 A3 冻结后评估一次，消费凭据为 `metadata/a3-test-evaluation.json`。

应用使用 RGB、resize 256、center crop 224、bicubic 和 ImageNet normalization。页面输出宿主、四大类、具体类别、置信度、Top-5、模型/数据版本、设备和耗时。低于 `0.55` 时显示不确定提示。

## 验收证据

- 模型包 13 项 checksum 与 A3 评估目录 9 项 checksum 全部通过。
- 三张固定 val 样例在 CPU/CUDA 的 class ID 和 Top-5 顺序一致。
- 正常、低置信度、损坏图片三条路径通过。
- 全量测试 `73 passed`；P2 范围 ruff 与 `git diff --check` 通过。
- Git 未追踪原图、模型权重、artifacts、checkpoint 或缓存。

## 使用边界

F0 验收对象是 203 类图像分类系统，当时尚未提供目标检测。宿主和四大类由预测 class ID 查询 taxonomy 得到，不是三个独立模型。后续目标检测只覆盖 IP102 有框的 96 类害虫。结果不能替代农业专家诊断；长尾类别、相似类别、域外图片和低质量图片仍可能误判。
