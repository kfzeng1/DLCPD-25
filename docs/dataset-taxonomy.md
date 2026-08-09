# DLCPD-25 本地 203 类分组规范

本项目保留 DLCPD-25 的 203 个细粒度类别，并增加一个仅用于数据分析和界面筛选的上位分组。上位分组不会替代 203 类训练标签，也不是作者发布的官方层级。

| 分组 | 含义 | 类别数 |
| --- | --- | ---: |
| `pest` | 昆虫、螨类、腹足类等农业有害生物 | 126 |
| `disease` | 真菌、细菌、病毒、卵菌等植物病害或以病害命名的症状 | 56 |
| `healthy` | 明确标为健康的作物图片 | 17 |
| `disorder` | 药害、红叶、花叶等非生物或生理异常 | 3 |
| `mixed` | 同一目录同时包含病虫害、无法形成单一语义 | 1 |

完整逐类结果见 `metadata/class-taxonomy.json` 和 `metadata/class-taxonomy.csv`。`class_id` 按 `metadata/official-class-names.txt` 的固定顺序从 0 编号。

## 重要边界

- `pest` 是农业数据语义，不是严格动物分类学层级；其中包含昆虫、螨和软体动物。
- `disease` 与 `disorder` 的划分根据官方英文名和当前中文释义完成，不能替代植保专家诊断。
- `garlic pest and diseases` 同时指病害和虫害，保留为 `mixed`，不能强行归入单一类别。
- 旧分析曾把 `beet spot flies`、`Diabrotica speciosa`、`Protaetia brevitarsis` 和 `tomato two-spotted spider mite` 误分为病害，本版已更正为有害生物。

## 重建命令

```bash
python3 scripts/build_dataset_taxonomy.py
python3 scripts/audit_dataset.py
```

脚本会检查 203 类完整性，重写结构化分类清单，并在 `data/views/by-category/` 下生成不占用额外图片空间的软链接视图。
