# 本地数据目录

数据图像不提交到 Git。当前 203 个类别目录已使用“英文 + 中文”本地名称，工具通过 `metadata/class-directory-aliases.json` 将其映射回官方原始名称：

```text
data/
  Adristyrannus 枯叶夜蛾(柑橘)/
  ...
  yellow rice borer 三化螟(水稻)/
```

本项目只要求 [官方 203 类清单](../metadata/official-class-names.txt) 均有一个非空的本地目录，不要求每类图片数与论文一致。现有名称保持不动；后续规范名、学名和译名修订应放在元数据字段中，避免再次批量改目录。

放置完成后运行：

```bash
python3 scripts/audit_dataset.py data
```
