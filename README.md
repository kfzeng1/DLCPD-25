# 基于DLCPD-25数据集的农产品病虫害与缺陷分类目标检测系统

本项目围绕 DLCPD-25（Dataset of Large-scale Crop Pests and Diseases, 2025）建设农产品病虫害与缺陷分类及目标检测系统。DLCPD-25 提供 203 类整图分类数据，IP102 提供其中 96 个公共害虫类别的边界框标注。

## 当前交付

203 类分类基线 D0-F0 和双数据集联合模型 J1-J4 已完成。J4 最终只发布一份联合权重：一个 RGB `224 x 224` 直缩输入、一个共享 ResNet-50-FPN 主干、一个 203 类分类头和一个 96 类害虫检测头。

联合模型在 DLCPD-25 test 上的 Top-1 为 `91.3157%`、Top-5 为 `96.4289%`、Macro-F1 为 `75.4451%`；在 IP102 test 上的 mAP@0.5:0.95 为 `35.8823%`、AP50 为 `65.5326%`、Precision 为 `68.9095%`、Recall 为 `80.1980%`。分类置信度低于冻结阈值 `0.55` 时，应用必须明确提示结果不确定。

唯一模型包位于 `artifacts/releases/dlcpd25-ip102-joint-v1/`。当前 `7860` 页面仍加载历史分类基线；下一阶段 J5 将改为加载联合模型包，一次推理同时显示分类 Top-5 和支持类别的检测框。

## 快速启动

本机复用 `/home/zkf/pytorch-env`。从仓库根目录运行：

```bash
/home/zkf/pytorch-env/bin/pip install -e 'project[app,dev]'
PYTHONPATH=project/src /home/zkf/pytorch-env/bin/python \
  -m dlcpd25_classifier.web --host 127.0.0.1 --port 7860
```

打开 <http://127.0.0.1:7860>。默认配置会加载 `artifacts/releases/dlcpd25-resnet50-weighted-v1/`；`device: auto` 优先使用 CUDA，失败时回退 CPU。

固定 val 样例、Top-5 一致性和损坏图片处理可用以下命令复验，命令不会访问正式 test：

```bash
PYTHONPATH=project/src /home/zkf/pytorch-env/bin/python \
  -m dlcpd25_classifier.inference.smoke \
  --output-dir /tmp/dlcpd25-p2-smoke
```

若该临时目录已经存在，请换一个新目录；验证命令拒绝覆盖旧证据。完整演示和排错见 [`docs/application-runbook.md`](docs/application-runbook.md)。

## 当前开发入口

历史分类训练与 F0 验收已经冻结，不再复跑 A2/A3。IP102 T0 和联合模型 J1-J4 已通过。两个冻结 test 均已在 J4 各评估一次，不得重跑或用于调参；当前代码进入 J5，只接入唯一联合模型包，不再训练模型。

J5 完成后执行 F1 最终验收。完整规则见：

- [`docs/project-plan.md`](docs/project-plan.md)
- [`docs/development-guide.md`](docs/development-guide.md)
- [`模型架构PPT`](docs/presentations/dlcpd25-joint-model-architecture.pptx)
- [`模型架构PDF`](docs/presentations/dlcpd25-joint-model-architecture.pdf)
- [`docs/workplans/algorithm-engineer-detection.md`](docs/workplans/algorithm-engineer-detection.md)
- [`docs/acceptance-checklist.md`](docs/acceptance-checklist.md)

## 关键结论

- DLCPD-25 是图像级分类数据集，不含边界框或分割标注；当前分类任务不需要人工画框。
- 论文摘要和官方云盘当前以 203 类为准；论文正文表 1 又合计为 210 类，属于论文内部不一致。
- 本地子集包含全部 203 类、221,396 个文件，约 17 GB；不要求补足论文声称的 221,943 张。
- 项目采用“宿主作物 → 四大标签属性 → 具体标签”的层级：当前 22 个宿主下再分农业有害生物、植物病害、健康、非生物/生理缺陷四类。
- 官方数据仓库没有 `LICENSE` 文件；论文的 CC BY 4.0 不应自动推定为数据文件许可。

## 官方入口

- 论文：<https://doi.org/10.3390/s25227098>
- PubMed Central：<https://pmc.ncbi.nlm.nih.gov/articles/PMC12656478/>
- 官方仓库：<https://github.com/hwzhanng/DLCPD-25-Dataset>
- 官方百度网盘：<https://pan.baidu.com/s/1KWLVESB1InGPl-M6Mq8MBw?pwd=gnp5>，提取码 `gnp5`

## 目录结构

```text
data/
  README.md                    # 数据目录说明
  raw/dlcpd25-203/             # 唯一原图，203 个类别目录
  raw/ip102/                   # IP102 原始检测图片与 VOC XML
  views/by-host/               # 宿主/四大类/具体标签软链接视图
docs/
  dataset-taxonomy.md          # 本地分组定义与边界
  project-plan.md              # 20 天实施计划和硬件决策
  development-guide.md         # 完整开发、训练和验收文档
  team-responsibilities.md     # 总负责人和三位 AI 工程师职责
  workflow.md                  # 用户逐阶段调用和工程师汇报流程
  acceptance-checklist.md      # 总负责人维护的动态验收状态
  prompts/                     # 三位 AI 工程师固定启动提示词
  workplans/                   # T0、J1-J5 各工程师执行工作单
  worklogs/                    # 工程实施和总负责人验收日志
project/
  pyproject.toml               # 分类工程依赖和打包配置
  configs/                     # 训练和应用配置
  src/                         # 分类、检测、训练、推理和Web代码
research/
  dataset-card.md              # 数据卡
  source-audit.md              # 来源检索与论文勘误
  citations.md                 # 引用记录
  paper-category-tables-zh.pdf # 论文类别表翻译及 203 类对照
metadata/
  official-class-names.txt     # 官方 203 类原名
  class-directory-aliases.json # 官方名到本地目录名
  class-taxonomy.json          # 203 类结构化上位分类
  class-taxonomy.csv           # 便于表格软件查看的版本
scripts/
  audit_dataset.py             # 完整性和数量审计
  build_dataset_taxonomy.py    # 重建分类元数据和视图
  build_research_pdf.py        # 重建论文类别表中文 PDF
artifacts/                     # 可重新生成的运行产物
```

## 数据验收

```bash
python3 scripts/audit_dataset.py \
  --output artifacts/audit/dataset-summary.json
python3 scripts/build_dataset_taxonomy.py
```

审计退出码为 `0` 表示 203 个官方类别全部存在、非空且没有额外类别。训练时以 203 个细粒度类别为最终标签，宿主和四大类可用于级联分类、界面筛选及分层评估。

## 使用边界

DLCPD-25 的分类 split 已按重复组隔离。目标检测使用 IP102 的对象级框标注，不能把 DLCPD-25 整幅分类图片自动当作目标框。
