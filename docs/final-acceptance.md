# F1 联合分类检测系统最终验收报告

验收日期：2026-08-13

验收输入：J5 commit `586ca12`
课题：基于DLCPD-25数据集的农产品病虫害与缺陷分类目标检测系统

## 最终结论

项目通过 F1 验收。最终交付是一份联合模型权重、一个 `224 x 224` 输入和一次共享 ResNet-50 主干前向：

```text
RGB 图片 -> bicubic 直缩 224x224 -> 共享 ResNet-50
                                  ├─ 203 类整图分类头
                                  └─ FPN/RPN/Faster R-CNN 96 类害虫检测头
```

应用同时显示宿主、四大类、具体分类 Top-5 与检测框。检测内部标签不会暴露，外部分类和检测统一使用 DLCPD-25 `class_id 0-202`。

## 版本矩阵

| 层级 | 冻结版本 | 关键合同 |
|---|---|---|
| DLCPD-25 数据 | `data-v1-d5-r1` | 203 类，train/val/test `177,021/22,178/22,178` |
| IP102 数据 | `ip102-detection-v1` | train/val/test `12,142/3,036/3,798`，97 源标签映射为 96 检测类 |
| 联合训练 | `j3-joint-full-e67e96e-r2` | 10 epochs，最佳 epoch 10，两个 test 未读 |
| 联合模型 | `dlcpd25-ip102-joint-v1` | `joint-best.pt` SHA-256 `5ec0f4f7891b729ddf26a51cd70d5c56a69825b2dd587c7f6af55854d3c06c49` |
| J4 冻结评估 | commit `75a787c` | 两个 test 各单遍评估一次，收据状态 `consumed` |
| J5 应用 | commit `586ca12` | 默认 7860 加载唯一联合模型包 |

## 最终指标

DLCPD-25 classification test，22,178 张：

- Top-1：`91.3157%`
- Top-5：`96.4289%`
- Macro-F1：`75.4451%`
- Balanced Accuracy：`74.6621%`
- 阈值 `0.55` 下低置信度率：`71.3004%`

IP102 detection test，3,798 张、4,444 个目标：

- mAP@0.5:0.95：`35.8823%`
- AP50：`65.5326%`
- Precision：`68.9095%`
- Recall：`80.1980%`
- Small AP：`6.1139%`
- 95/96 个检测类有 test 支持；IP102 源类 61 对应 detector label 8 无 test 样本

测试指标没有用于调参、重训或阈值回调。

## 验收证据

- J4 模型包 14 项 checksum 全部通过；目录中只有 `joint-best.pt` 一份权重。
- manifest、taxonomy、IP102 映射、统一 224 直缩、阈值、NMS 和运行依赖校验通过。
- checkpoint 严格重载、CPU/CUDA 联合推理、共享主干 hook 计数 `1` 通过。
- IP102 val 真实样例在 CUDA/CPU 均返回 class 156 和 2 个检测框；CUDA 约 `68.686 ms`，CPU 约 `947.309 ms`。
- 7860 `/analyze` 真实上传返回 5 行分类 Top-5、2 行检测、带框图和低置信度提示。
- Edge CDP 桌面 `1650x785` 与移动 `390x844` 验收通过，无页面级横向溢出或元素重叠。
- F1 项目全量测试 `126 passed, 9 warnings in 251.89s`；warning 均为 Gradio 6.0 弃用提示；Ruff 和 `git diff --check` 通过。
- Git 中无原图、模型权重、缓存、超过 10 MiB 的文件或 `artifacts` 大产物。

浏览器截图位于 `artifacts/audit/j5-browser/`，模型与训练产物按 `.gitignore` 保留在本机，不进入 Git。

## 运行

```bash
PYTHONPATH=project/src /home/zkf/pytorch-env/bin/python \
  -m dlcpd25_classifier.web --host 127.0.0.1 --port 7860
```

打开 <http://127.0.0.1:7860>。演示图片和排错见 `docs/application-runbook.md`。

## 使用边界

- DLCPD-25 只有整图标签，因此 203 类都可分类，但并非 203 类都可画框。
- 检测只覆盖 IP102 有框监督并成功映射的 96 类害虫；病害、健康和生理缺陷不能承诺定位。
- 224 直缩适合本机 8 GiB 显存和课程周期，但小目标性能有限。
- 低置信度、无框和域外图片必须保留不确定性提示；无框不等于健康。
- 本系统是课程项目原型，不替代农业专家诊断。

历史 F0 分类基线保留为阶段证据，不是最终部署模型。
