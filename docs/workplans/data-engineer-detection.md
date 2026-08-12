# 数据工程师工作单：T0

## 任务目标

把 IP102 官方 Pascal VOC 检测数据整理成可复验的数据合同，供算法工程师直接训练。只生成派生索引和统计，不移动、删除或改写官方图片/XML。

## 已知事实

- 原始目录：`data/raw/ip102/downloads/Detection/VOC2007/`
- 图片 18,981 张，XML 18,976 个；正式划分 18,976 张
- 官方 `trainval/test`：15,178 / 3,798
- 去重后的原始框 22,284 个，清洗后有效框 22,283 个
- 97 个 IP102 源标签映射为 96 个 DLCPD-25 公共检测类别
- `IP087000986.xml` 含重复根结构，解析时去重
- `IP046000898.xml` 含一个零宽框，同图有效框必须保留
- 5 张 JPEG 不属于官方划分，只排除，不删除
- 官方 test 缺 IP102 类别 61，最终仅覆盖 95/96 个公共类别

## 执行内容

1. 全量审计图片、XML、官方 split、尺寸、框和类别映射。
2. 从官方 `trainval` 生成固定、可复现的 train/val，保持图片级隔离；采用多标签分层，稀有类尽量在两侧保留支持。
3. 建立派生标注 manifest。记录原框和清洗原因，只过滤退化框，不修改 XML。
4. 冻结三套编号：IP102 原标签、检测内部 `1-96`、DLCPD-25 `class_id 0-202`。
5. 全量遍历 Dataset，分别对 train/val 做 DataLoader 冒烟；官方 test 只做完整性冻结，不用作验证。
6. 输出配置、统计、异常清单、checksum 和交接说明。

## 目录合同

```text
scripts/ip102/                         # 构建与验证脚本
artifacts/data/ip102-detection-v1/
  train.txt
  val.txt
  test.txt
  annotations.jsonl
  class-map.json
  audit-summary.json
  split-summary.json
  exceptions.json
  build-config.json
  data-handoff.md
  checksums.sha256
```

实际文件名可在实现前小幅调整，但发布目录和字段一旦冻结不得覆盖。路径应相对仓库或 VOC 根目录，不写临时绝对路径。

## 验收标准

- train/val 非空、无交集，并集严格等于官方 trainval；test 保持官方 3,798 张；
- 正式 18,976 张均有图片和可解析 XML，5 张额外 JPEG 不进入任何 split；
- 重复根只计一次，22,284 个原框可追溯，唯一退化框被明确过滤，得到 22,283 个有效框；
- 所有有效框有限、在图像边界内且满足 `xmin < xmax`、`ymin < ymax`；
- 97 个源标签全部可映射，内部标签严格为 `1-96`，公共 ID 严格为 `0-202`；
- Dataset 全量遍历和 DataLoader 冒烟通过，checksum 可独立复验；
- 统计明确说明 test 缺失类别，不生成虚假样本或 AP；
- 定向测试与 `git diff --check` 通过。

## 禁止事项

- 删除、移动或改写 IP102 原图/XML 和官方 split；
- 修改 DLCPD-25 taxonomy、冻结 split、分类模型包或检测映射语义；
- 使用官方 test 选随机种子、划分比例、清洗策略或训练参数；
- 重跑旧 DLCPD-25 D2-D4 全量流程；
- 自行开始 T1、提交或推送。
