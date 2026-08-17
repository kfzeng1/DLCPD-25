# 数据目录

原始图片不提交到 Git，只保留目录结构和数据合同脚本。

```text
data/
  raw/
    dlcpd25/                              DLCPD-25 分类数据集
      <英文 + 中文类别目录>/                203 个类别目录，只含图像级标签
    ip102/
      VOC2007/                            IP102 官方 Detection 子集，VOC 格式
        JPEGImages/                       18,981 张 JPEG
        Annotations/                      18,976 个 VOC XML
        ImageSets/Main/trainval.txt       15,178
        ImageSets/Main/test.txt           3,798
      classification-labels/              IP102 源类别名称（classes.txt/docx）
  views/
    by-host/                              按“宿主作物 → 四大属性 → 具体标签”生成的软链接视图
```

## DLCPD-25

- 原始图片只有一份：`data/raw/dlcpd25/`。
- 203 个类别目录使用“英文 + 中文”本地名称，通过
  `metadata/dlcpd25/class-directory-aliases.json` 映射回官方原名。
- 图片只有**整图分类标签**，没有框；病害、健康、生理缺陷类别不做检测。
- 分类层级固定为：宿主作物（22 个）→ 标签属性（pest/disease/healthy/disorder）→ 具体类别。
- 层级规范与逐类信息见 `metadata/dlcpd25/class-taxonomy.json`。
- 清单和 train/val/test 划分见 `artifacts/data/dlcpd25/`。

重建浏览视图：

```bash
python3 scripts/dlcpd25/build_taxonomy.py
```

## IP102 Detection

- 原始检测数据唯一来源：`data/raw/ip102/VOC2007/`，原始 XML 和划分文件**不得修改**。
- 数据加载器需兼容 `IP087000986.xml` 的重复根节点，并过滤 `IP046000898.xml` 的无效框。
- 官方 trainval 再按 80/20 迭代多层分层得到 `train.txt` / `val.txt`，官方 `test.txt` 原样保留。
- IP102 源标签 97 个映射为检测标签 `1..96`；检测输出再映射到 DLCPD-25 `class_id 0..202`。
- 映射合同：`metadata/ip102/detection-class-map.json`。
- 派生标注与审计结果：`artifacts/data/ip102/`。

重建检测合同：

```bash
python3 scripts/ip102/build_detection_contract.py
python3 scripts/ip102/verify_detection_contract.py
```

## 使用规则

- 训练代码不得重新扫描目录或动态推断类别，必须读取 `metadata/` 和 `artifacts/data/` 下的冻结合同。
- IP102 官方测试集和 DLCPD-25 测试划分只能用于冻结配置后的最终评估。
- 原始图片只读；所有派生产物写入 `artifacts/data/`。
