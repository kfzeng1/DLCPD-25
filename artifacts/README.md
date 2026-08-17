# 运行产物

`artifacts/data/` 下的数据合同提交到 Git，其余产物（权重、日志、训练输出）本地保留。

```text
artifacts/
  data/
    dlcpd25/    DLCPD-25 图像清单、固定 train/val/test 划分
    ip102/      IP102 检测 train/val/test、派生标注、审计
```

所有派生产物都应通过 `scripts/` 重建，不得手改。
