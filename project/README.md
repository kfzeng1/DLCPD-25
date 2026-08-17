# Plan-A 训练/推理工程

这是删除 v1 联合训练代码后的新工程位置。当前先完成数据工程，下一步按 `configs/plan-a/` 实现：

```text
project/
  configs/                # 继承根目录 configs/plan-a 的配置
  src/dlcpd25_v2/
    data/                 # 读取 data/raw 与 artifacts/data 的数据加载器
    classification/       # DLCPD-25 分类专家（ConvNeXt-Tiny @384）
    detection/            # IP102 检测专家（Faster R-CNN/ConvNeXt-Tiny-FPN @640）
    serving/              # 双模型推理编排
    web/                  # Web 应用
  tests/
```

## 数据合同

所有数据访问必须使用：

- `metadata/dlcpd25/class-taxonomy.json`
- `metadata/ip102/detection-class-map.json`
- `artifacts/data/dlcpd25/manifest.csv`
- `artifacts/data/ip102/*.txt` + `annotations.jsonl`

不得在模型代码中重新扫描原始目录或硬编码类别名称。
