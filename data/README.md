# 本地数据目录

数据图像不提交到 Git。原始图片只有一份，位于 `raw/dlcpd25-203/`。203 个类别目录使用“英文 + 中文”本地名称，并通过 `metadata/class-directory-aliases.json` 映射回官方原名。

```text
data/
  raw/dlcpd25-203/
    Adristyrannus 枯叶夜蛾(柑橘)/
    ...
    yellow rice borer 三化螟(水稻)/
  views/by-category/
    01_pest_农业有害生物/
    02_disease_植物病害/
    03_healthy_健康/
    04_disorder_非生物或生理缺陷/
    05_mixed_混合或歧义/
```

`views/` 只包含指向原图类别目录的软链接，用于浏览，不复制图片，也不改变 203 类训练标签。五个上位组是项目复核结果，不是论文官方发布的类别层级。

本项目要求官方 203 类均有一个非空目录，但不要求每类图片数与论文一致。现有目录名保持不动；规范名和译名修订只能放在元数据字段中。

```bash
python3 scripts/audit_dataset.py
python3 scripts/build_dataset_taxonomy.py
```
