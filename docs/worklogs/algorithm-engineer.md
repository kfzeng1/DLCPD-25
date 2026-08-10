# AI 算法工程师工作日志

当前状态：A0 已取消并合并到 A1，等待执行 A1。

后续记录必须按 `README.md` 模板追加，不得覆盖历史记录。

## 2026-08-10 A0 data-v1 准入

- 状态：待验收
- 指挥者指令：数据工程全部完成，正式进入算法工程 A0；只执行 A0。
- 前置版本：Git commit `f4ee10fb6bddabdd3b708a0679b85ec0d518cf57`；data-v1 `D5-R1`；固定 split `D3-R2`；taxonomy SHA-256 `5cfa1a261b1a9fbb80adf24f299bca0883a42dd523914a70234f31dbf748bd31`。
- 允许范围：算法训练模块、算法测试、`artifacts/training/` 和本日志。
- 实际修改：新增 `project/src/dlcpd25_classifier/training/admission.py` 和 `project/tests/test_training_admission.py`；更新本日志；未修改数据、taxonomy、split、数据脚本、其他日志或验收清单。
- 生成产物：`artifacts/training/a0-data-v1-f4ee10f/`；`admission-report.json` SHA-256 `58d5151160b10a2e5092fdfa9fb87b552ee02b2efd886be54e50cb46618feb2e`；`resolved-config.json` SHA-256 `118915c5df78b212a2839b71a1e56c9b3e685e5d52d8e9c17d7168d94979267f`；`checksums.sha256` SHA-256 `c00daf6a8e764e1033153b56a424bbb9601882129701120357e6d266a469cb50`。
- 执行命令：定向测试 `/home/zkf/pytorch-env/bin/pytest -q project/tests/test_training_admission.py`，退出 0，`4 passed in 10.62s`；准入 CLI 与 `--verify-only` 各退出 0；全量测试在 `project/` 执行 `/home/zkf/pytorch-env/bin/pytest -q`，退出 0，`35 passed in 229.65s`；`git diff --check` 退出 0。
- 验收证据：D5-R1 checksum 清单 39 项全部匹配；taxonomy 为 203 类、ID 0-202、22 个宿主和四大类；train/val/test 的文件 SHA-256、schema、数量和 class coverage 均匹配冻结发布；路径重叠、duplicate group 泄漏和同 SHA 分组冲突均为 0；不可覆盖产物经重新计算后与保存报告一致。
- 关键统计：train/val/test 为 177,021/22,178/22,178；总计 221,377 条唯一可用路径；检查 221,377 个源文件均存在且非符号链接；203 类在三个 split 中全部覆盖；156,852 个唯一 duplicate group；193,333 个唯一内容 SHA-256；19 个长尾类别。
- 偏差、风险和阻塞：无阻塞。A0 验证冻结索引哈希、split 内容和所有源文件存在性，但不重新计算 17 GB 原图内容哈希或全量解码；这些证据继承已验收 D2/D4。数据许可、跨类别标签冲突、保守近重复召回和长尾方差仍为 data-v1 已知限制。
- Git 状态：`main` 相对 `origin/main` ahead 16；本日志和两个新增算法文件未提交；`artifacts/` 被忽略；未提交或推送。
- 下一步建议：由总负责人独立复验 A0；只有 A0 标记通过且用户明确下令后才执行 A1。
- 边界声明：未执行下一阶段。

## 2026-08-10 流程调整

- 独立 A0 已取消；有效的数据契约检查改为无状态 preflight，并入 A1。
- 后续算法阶段压缩为 A1-A3，按 `docs/workflow.md` 的 7 行模板记录。
