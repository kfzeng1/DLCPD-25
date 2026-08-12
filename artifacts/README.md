# 运行产物目录

`artifacts/` 不进入 Git。原始数据不在此处，所有文件均可由受控代码和冻结输入重新生成。

## 保留规则

- `data/v1/`：已验收的 DLCPD-25 数据冻结链，只读使用。
- `data/ip102-detection-v1/`：已验收的 IP102 T0 数据合同，只读使用。
- `releases/`：已验收模型或应用发布包；当前仅历史分类发布包，不能作为联合应用的第二模型。
- `training/j1-direct-resize-ed09c0f/`：J1 分类初始化来源。
- `training/detection/j2-alternating-smoke-95b24b9-r8/`：唯一正式 J2 冒烟产物，仅用于证明联合链路、显存和 checkpoint 恢复能力，不作为 J3 权重初始化。

未验收、失败、被撤销或调试用的 run 不保留在本目录。确认无进程使用后，将其移入系统回收站；不要覆盖或复用同一个 run ID。

J3 只允许新建 `training/detection/j3-joint-full-<commit>-rN/`。完成并通过验收后，由总负责人把唯一正式 run 记录到本文件和验收清单。
