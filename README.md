# DLCPD-25 农产品病虫害与缺陷分析系统（Plan A）

本仓库基于两份数据集构建“分类 + 目标检测”系统：

| 数据集 | 任务 | 数据量 | 当前状态 |
|---|---|---|---|
| DLCPD-25 | 203 类细粒度图像分类 | 221,396 张图，22 个宿主作物 | 数据工程已整理，待训练 |
| IP102 Detection | 96 类农业害虫目标检测 | 18,976 个已标注样本 | 数据工程已整理，待训练 |

系统采用 **Plan A：双专家模型** 架构：

- **分类专家**：`ConvNeXt-Tiny` 或 `EfficientNetV2-S`，输入 384×384，输出 DLCPD-25 的 203 类；
- **检测专家**：`ConvNeXt-Tiny-FPN + Faster R-CNN` 或 `YOLOv8s`，输入最长边 640，输出 96 类 IP102 害虫框；
- **推理编排**：检测结果映射回 DLCPD-25 公共 `class_id 0..202`，与整图 Top-5 分类结果同时返回。

旧的 `ResNet-50 + Faster R-CNN` 联合模型已从当前工程删除。
## 目录结构

```text
data/
  raw/dlcpd25/              DLCPD-25 分类原图（203 个类别目录）
  raw/ip102/VOC2007/        IP102 目标检测 VOC 数据（JPEGImages/Annotations/ImageSets）
  raw/ip102/classification-labels/  IP102 源类别表（仅名称参考，图片未保留）
  views/by-host/            按宿主→类别属性→具体标签生成的浏览软链接
metadata/
  dlcpd25/                  DLCPD-25 官方类名、目录别名、分类层级
  ip102/                    IP102→DLCPD-25 检测类别映射
artifacts/data/
  dlcpd25/                  DLCPD-25 图像清单与 train/val/test 固定划分
  ip102/                    IP102 检测合同（划分、标注、审计）
scripts/
  dlcpd25/                  分类数据工程脚本
  ip102/                    检测数据工程脚本
configs/plan-a/             双模型训练与推理配置
project/                    后续模型训练、推理和 Web 应用代码位置
baselines/                  模型指标基线（新模型训练后写入）
```

## 数据工程复现

```bash
# DLCPD-25：审计 203 类目录、重建分类层级和浏览视图
python3 scripts/dlcpd25/audit_dataset.py
python3 scripts/dlcpd25/build_taxonomy.py

# DLCPD-25：生成图像清单与固定 train/val/test 划分（全量遍历约 5-10 分钟）
python3 scripts/dlcpd25/build_manifest_splits.py --workers 8

# IP102：构建冻结检测合同并独立校验
python3 scripts/ip102/build_detection_contract.py
python3 scripts/ip102/verify_detection_contract.py
```

## 当前进度

- [x] 删除旧训练代码与旧版归档
- [x] 两份数据集物理目录与数据合同整理清晰
- [x] DLCPD-25 分类层级、清单、固定划分
- [x] IP102 检测合同重建
- [x] Plan-A 分类训练代码、checkpoint 与进度网页
- [x] Plan-A 分类模型训练、早停、微调对比与测试集评估
- [x] Plan-A 检测模型训练、早停与测试集评估
- [ ] 双模型 Web 应用与推理编排

## 标签与类别合同

- DLCPD-25 公共类别：`class_id 0..202`，与 `metadata/dlcpd25/class-taxonomy.json` 绑定；
- IP102 检测源标签：97 个；内部检测标签：`1..96`，`0` 为背景；
- 检测输出必须映射回 DLCPD-25 的公共 `class_id`；
- IP102 源类别 50、51 合并到 DLCPD-25 `class_id 97`；
- 官方 IP102 测试划分保持原样，只用于最终评估。
