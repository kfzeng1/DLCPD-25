# Plan-A 配置

- `classification.yaml`：DLCPD-25 203 类分类专家。
- `detection.yaml`：IP102 96 类检测专家。

两份配置是训练代码的目标合同。正式训练脚本将读取这些配置、`metadata/` 和 `artifacts/data/`，禁止在训练代码内硬编码路径或类别映射。
