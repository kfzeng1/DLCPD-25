# 基于 DLCPD-25 数据集的农产品病虫害与缺陷分类系统

本项目围绕 DLCPD-25（Dataset of Large-scale Crop Pests and Diseases, 2025）建设 203 类农产品病虫害与缺陷图像分类系统。公开资料、官方类别清单、本地数据审计和项目级上位分类已经整理完成。

## 关键结论

- DLCPD-25 是图像级分类数据集，不含边界框或分割标注；当前分类任务不需要人工画框。
- 论文摘要和官方云盘当前以 203 类为准；论文正文表 1 又合计为 210 类，属于论文内部不一致。
- 本地子集包含全部 203 类、221,396 个文件，约 17 GB；不要求补足论文声称的 221,943 张。
- 项目将 203 类归入农业有害生物、植物病害、健康、非生物/生理缺陷、混合歧义五个上位组。该分组是项目元数据，不冒充论文官方层级。
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
  views/by-category/           # 五个上位组的软链接视图
docs/
  dataset-taxonomy.md          # 本地分组定义与边界
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

审计退出码为 `0` 表示 203 个官方类别全部存在、非空且没有额外类别。训练时以 203 个细粒度类别为真实标签，五个上位组主要用于统计、筛选和层级分类实验。

## 使用边界

训练集、验证集和测试集应先按内容哈希或近重复组切分，再进行数据增强，避免同源图片泄漏。若以后扩展为目标检测，仍然需要对象级框标注，不能把整幅图自动当作目标框。
