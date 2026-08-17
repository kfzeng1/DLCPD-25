# 数据合同

训练、评估和 Web 应用必须只读取以下冻结产物。

## DLCPD-25 分类合同

| 文件 | 内容 |
|---|---|
| `metadata/dlcpd25/official-class-names.txt` | 203 个官方类名，顺序即 class_id |
| `metadata/dlcpd25/class-directory-aliases.json` | 官方类名 → 本地目录名 |
| `metadata/dlcpd25/class-taxonomy.json` | 宿主、类别属性、逐类图片数 |
| `artifacts/data/dlcpd25/manifest.csv.gz` | 每张图的相对路径、class_id、sha256、宽高、解码状态、split |
| `artifacts/data/dlcpd25/manifest.csv` | 未压缩完整清单（本地生成，不提交 Git） |
| `artifacts/data/dlcpd25/split-summary.json` | train/val/test 逐类统计 |
| `artifacts/data/dlcpd25/excluded-images.csv` | 无法解码、不进入划分的图片 |

划分规则：

- sha256 相同图片归为同一重复组，重复组不会跨 train/val/test；
- 类别至少有 3 个独立重复组时，train/val/test 都覆盖；
- 只有 2 个独立组时覆盖 train/test；
- 只有 1 个组时只进入 train；
- 坏图不进入任何 split。

## IP102 检测合同

| 文件 | 内容 |
|---|---|
| `metadata/ip102/detection-class-map.json` | 97 源标签 → 检测标签 1..96 → DLCPD-25 class_id |
| `artifacts/data/ip102/train.txt` | train 12,142 张 |
| `artifacts/data/ip102/val.txt` | val 3,036 张 |
| `artifacts/data/ip102/test.txt` | 官方 test 3,798 张，原样保留 |
| `artifacts/data/ip102/annotations.jsonl` | 每张图的尺寸、路径、sha256、有效框 |
| `artifacts/data/ip102/audit-summary.json` | 原始数据审计 |
| `artifacts/data/ip102/exceptions.json` | 重复 XML、过滤框、官方划分外图片 |

规则：

- 原始 VOC 文件只读；
- `IP087000986.xml` 去重解析，`IP046000898.xml` 的无效框只过滤框、保留图片；
- 官方 test 与派生 train/val 完全隔离；
- IP102 源类别 61 在官方 test 中无支持，不伪造指标。
