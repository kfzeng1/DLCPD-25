# 脚本

所有脚本只做两件事：审计原始数据、生成冻结数据合同。训练和推理代码后续放在 `project/`。

```text
scripts/
  dlcpd25/
    audit_dataset.py             检查 203 个类别目录是否完整
    build_taxonomy.py            重建分类层级 JSON/CSV 和 by-host 软链接视图
    build_manifest_splits.py     生成图像清单、sha256 清单和固定 train/val/test 划分
  ip102/
    build_detection_contract.py  生成 IP102 检测划分、派生标注和审计合同
    verify_detection_contract.py 独立遍历全部划分，校验图片/XML/框数量
```

## 运行顺序

```bash
python3 scripts/dlcpd25/audit_dataset.py
python3 scripts/dlcpd25/build_taxonomy.py
python3 scripts/dlcpd25/build_manifest_splits.py --workers 8

python3 scripts/ip102/build_detection_contract.py
python3 scripts/ip102/verify_detection_contract.py
```

原始图片和 XML 只读；输出只写入 `artifacts/data/`。
