# 运行产物目录

`artifacts/` 不进入 Git。原始数据不在此处，所有文件均可由受控代码和冻结输入重新生成。

## 保留规则

- `data/v1/`：已验收的 DLCPD-25 数据冻结链，只读使用。
- `data/ip102-detection-v1/`：已验收的 IP102 T0 数据合同，只读使用。
- `releases/`：已验收模型或应用发布包；J5 只能加载下述唯一联合模型包。
- `training/j1-direct-resize-ed09c0f/`：J1 分类初始化来源。
- `training/detection/j2-alternating-smoke-95b24b9-r8/`：唯一正式 J2 冒烟产物，仅用于证明联合链路、显存和 checkpoint 恢复能力，不作为 J3 权重初始化。
- `training/detection/j3-joint-full-e67e96e-r2/`：已验收的 10 轮联合训练正式 run，J4 的唯一权重来源。
- `training/detection/j4-test-dlcpd25-ip102-joint-v1/`：已验收的一次性双 test 评估证据。
- `releases/dlcpd25-ip102-joint-v1/`：J4 冻结的唯一联合模型包，J5 只允许加载此包。

未验收、失败、被撤销或调试用的 run 不保留在本目录。确认无进程使用后，将其移入系统回收站；不要覆盖或复用同一个 run ID。

J3/J4 已完成，以上正式产物不得覆盖。J5 只能新增应用交付，不得重新评估 test 或生成第二份联合权重。
