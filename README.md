# 基于DLCPD-25数据集的农产品病虫害与缺陷分类目标检测系统

本项目目标是在 20 天内完成“基于 DLCPD-25 的农产品病虫害与缺陷分类目标检测系统”。DLCPD-25（Dataset of Large-scale Crop Pests and Diseases, 2025）的公开资料、官方 203 类目录清单和本地数据审计工具已整理完成。项目采用“类别完整优先”的子集口径：不要求补齐论文宣称的全部图片，只要求本地 203 个类别均存在，并如实报告本地样本数。

## 关键结论

- DLCPD-25 是图像级分类数据集，不含 bounding box 或分割标注，不能直接当作目标检测数据集。
- 论文摘要和实验章节以 203 类为主；官方百度网盘当前也可枚举出 203 个唯一类别目录。
- 论文内部同时出现 203 类、210 类、221,943 张和无法复算的汇总表。详见 [资料勘误](docs/research/source-audit.md)。
- 官方 GitHub 仓库没有 `LICENSE` 文件，论文的 CC BY 4.0 只明确覆盖论文，不应自动推定为数据文件许可。公开、再分发或商业使用前应向作者确认。
- 当前本地子集位于 `data/`：203 个有效类别、221,396 个文件，约 17 GB。图片数量无需补到论文版本。

## 官方入口

- 论文：<https://doi.org/10.3390/s25227098>
- PubMed Central 全文：<https://pmc.ncbi.nlm.nih.gov/articles/PMC12656478/>
- 官方数据仓库：<https://github.com/hwzhanng/DLCPD-25-Dataset>
- 官方百度网盘：<https://pan.baidu.com/s/1KWLVESB1InGPl-M6Mq8MBw?pwd=gnp5>，提取码 `gnp5`

以上网盘链接和提取码已于 2026-08-08 验证，分享根目录为 `DLCPD-25`，含 203 个类别目录。

## 目录结构

```text
data/
  README.md                  # 数据放置说明；图像不进入 Git
  <203 个类别目录>/          # 当前已下载子集；目录名含中文别名
docs/
  research/
    dataset-card.md          # 数据集用途、事实和限制
    source-audit.md          # 来源检索、统计勘误、许可风险
    citations.md             # 截至检索日的引用与相关项目
  plans/
    detection-system.md      # 从分类数据扩建检测系统的方案
metadata/
  official-class-names.txt   # 官方云端实际枚举的 203 个目录名
  class-directory-aliases.json # 官方原名到本地目录名的一一映射
scripts/
  audit_dataset.py           # 本地类别完整性和文件数量审计
```

## 本地验收

```bash
python3 scripts/audit_dataset.py data
```

退出码为 `0` 表示 203 个官方类别全部存在、每类至少有一个文件且没有额外类别。图片数量可以少于论文版本；报告中的 `total_files` 和 `per_class` 才是当前训练子集的真实统计。

如需保存机器可读报告：

```bash
python3 scripts/audit_dataset.py data \
  --output artifacts/audit/dataset-summary.json
```

## 使用边界

训练 203 类分类器时，应先按内容哈希或近重复组切分，再进行数据增强，避免同源图片泄漏到验证集。若要做目标检测，必须另建对象级标注；不能把类别目录名转换成整图框。20 天内采用“203 类全部分类、5–20 个高质量类别真正检测”的 MVP，详细日程见 [20 天实施方案](docs/plans/detection-system.md)。
