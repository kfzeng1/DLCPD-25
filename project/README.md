# 基于 DLCPD-25 数据集的农产品病虫害与缺陷分类目标检测系统

本项目提供农产品图片的联合分析功能：

- 对 DLCPD-25 的 203 个细粒度类别输出 Top-5 分类结果；
- 对 IP102 中具有边界框标注、并与 DLCPD-25 对齐的 96 类害虫输出位置框；
- 病害、健康状态和生理缺陷只具备图像级标签，因此仅输出分类，不生成虚假检测框。

## 模型

输入图片先进行 EXIF 校正、RGB 转换、Bicubic 直缩和 ImageNet 归一化，统一为 `224 x 224`。一份联合权重使用共享 ResNet-50 主干：分类分支输出 203 类 logits，FPN、RPN 与 ROI Heads 组成的 Faster R-CNN 分支输出 96 类害虫框。检测框会按原图尺寸还原坐标。

## 目录

```text
project/
  assets/       模型、指标、样例图片和数据说明
  configs/      应用配置
  docs/         实验报告、架构材料和运行截图
  src/          数据、模型、推理和 Web 应用源码
  pyproject.toml
```

## 安装与运行

推荐 Python 3.10 及以上版本，并安装与显卡驱动匹配的 PyTorch。CPU 可以运行应用，但推理较慢。

```bash
cd project
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[app]'
python3 -m dlcpd25_classifier.web --host 127.0.0.1 --port 7860
```

浏览器打开 <http://127.0.0.1:7860>。默认模型包位于 `assets/model/`，启动前会验证模型清单及 SHA-256 校验值。

## 最终结果

| 任务 | 指标 | 结果 |
|---|---|---:|
| DLCPD-25 分类 | Top-1 | 91.3157% |
| DLCPD-25 分类 | Top-5 | 96.4289% |
| DLCPD-25 分类 | Macro-F1 | 75.4451% |
| IP102 检测 | mAP@0.5:0.95 | 35.8823% |
| IP102 检测 | AP50 | 65.5326% |
| IP102 检测 | Precision | 68.9095% |
| IP102 检测 | Recall | 80.1980% |

完整实验报告、截图、模型架构材料、逐类结果和数据说明均在本目录内。
