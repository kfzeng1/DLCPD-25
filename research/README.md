# 研究资料

本目录保存数据集论文、来源审计、引用记录和论文表格翻译。原始论文 PDF 放在被 Git 忽略的 `papers/`，项目生成的中文翻译 PDF 保留在本目录。

- `dataset-card.md`：数据集事实、适用范围和限制；
- `source-audit.md`：论文、官方云盘和本地数据之间的差异；
- `citations.md`：来源与相关研究；
- `paper-category-tables-zh.md` / `.pdf`：论文表 1 中文翻译、本地 203 类分组及逐类中英对照。

PDF 重建命令：

```bash
python3 scripts/build_research_pdf.py
```
