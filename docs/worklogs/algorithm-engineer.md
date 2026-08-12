# AI 算法工程师工作日志

当前状态：历史分类算法 A1-A3、分类应用 P2 和分类基线 F0 已通过；目标检测 T1-T3 未开始，等待 T0。

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

## 2026-08-10 A1 训练链路

- 阶段：A1，状态待验收；输入 commit `4143fc80d03bfb220ce2b2deb4417f2a7ce8eb50`、data-v1 D5-R1、taxonomy SHA-256 `5cfa1a261b1a9fbb80adf24f299bca0883a42dd523914a70234f31dbf748bd31`。
- 修改文件：新增 `models/classifier.py`、`training/{transforms,checkpoint,train}.py` 和 `test_training_a1.py`，更新模型导出、训练配置与本日志；生成 `artifacts/training/a1-smoke-4143fc8/`，其 checksum 清单 SHA-256 为 `fb3ecb81654402fc2b75015abb7112afa00ebce75976b1effec1384b7cc6e0dd`。
- 运行命令：运行无状态 preflight、A1 定向 pytest、`training.train --a1-smoke`、产物 `sha256sum -c`、独立 checkpoint 重载、项目全量 pytest 和 `git diff --check`，均退出 0。
- 测试结果：定向测试 `8 passed in 13.45s`；全量测试 `39 passed in 209.61s`；A1 产物 6 项 checksum 全部 OK；CPU checkpoint 严格重载通过。
- 关键指标：ImageNet V2 预训练 ResNet-50，固定 train 子集 32 张（8 类各 4 张），224 输入、batch 16、AMP；初始/final loss `5.2343/0.1697`，初始/final accuracy `6.25%/96.875%`，3 epoch、14.33 秒，峰值显存 `1,098,122,240` bytes；CUDA `[16,203]`、CPU `[2,203]` logits 有限。
- 遗留问题：无 A1 阻塞；小样本使用确定性 eval transform 诊断链路，不代表泛化指标；未读取 val/test 指标，A2 完整训练尚未执行；改动与忽略的 artifacts 尚未提交或推送。
- 是否进入下一阶段：否。

## 2026-08-10 A2 完整训练

- 阶段：A2，状态进行中；输入 commit `cc7e776`、data-v1 D5-R1 和 A1 已验收训练链路。
- 边界：只训练和验证 ResNet-50 的普通 CE 与 class-weighted CE，不读取 test 指标，不执行 A3。

2026-08-11 A2 运行记录

阶段：A2，进行中；普通 CE 已完成，class-weighted CE 已启动，未读取 test 指标。
修改文件：本次仅追加本日志；生成 `artifacts/training/a2-resnet50-weighted-ce-cc7e776/` 及独立训练面板运行状态，未修改数据、taxonomy 或 split。
运行命令：CE 使用固定 split/seed、ResNet-50 ImageNet V2、224、AMP、batch 128、workers 6、AdamW、LR `3e-4`、warmup 2、cosine、25 轮上限；weighted CE 使用完全相同参数并改用 clipped inverse-frequency class weights；面板由 `dlcpd25-a2-dashboard.service` 托管。
测试结果：CE `metrics.json` 状态为 `completed_pending_project_lead_acceptance`，25 轮完成、最佳 checkpoint 重载通过；CE 产物 `sha256sum -c checksums.sha256` 全部 OK；A2 定向测试 `6 passed`，`git diff --check` 通过。
关键指标：CE 最佳 epoch 23，Val Top-1 `88.2135%`、Top-5 `96.5732%`、Macro-F1 `71.8358%`、Balanced Accuracy `71.0727%`，峰值显存 `6,213,218,816` bytes，耗时 `23961.31s`；weighted CE 尚无完整指标。
遗留问题：等待 weighted CE 完成后执行仅基于 val Macro-F1 的两组对照；A2 comparison、项目负责人验收和 A3 尚未执行；训练服务 unit `dlcpd25-a2-weighted.service` 当前进行中。
是否进入下一阶段：否

## 2026-08-11 A2 完整训练完成

阶段：A2，待验收；输入 commit `71ee1dcc732f039ffcb42cab80928b6d71da8ed1`、data-v1 D5-R1、固定 split D3-R2，普通 CE 与 class-weighted CE 均完成且未读取 test 指标。
修改文件：新增 `training/{a2,compare,dashboard,metrics,progress}.py` 和 `test_training_a2.py`，更新 `training/{checkpoint,train}.py` 与本日志；产物为 `artifacts/training/a2-resnet50-{ce,weighted-ce}-cc7e776/` 和正式对照 `artifacts/training/a2-resnet50-comparison-r2-cc7e776/`。
运行命令：两组均使用 ImageNet V2 ResNet-50、224、AMP、batch 128、workers 6、AdamW、LR `3e-4`、warmup 2、cosine、25 轮上限和固定 seed/split，仅切换 `ce`/`weighted_ce`；运行两组 checksum、A2 定向 pytest、comparison CLI、正式 comparison checksum 和 `git diff --check`。
测试结果：A2 定向测试 `7 passed in 7.05s`，CE 与 weighted CE 各 11 项 checksum 全部 OK，正式 comparison checksum 通过且清单 SHA-256 为 `e9cd45eaf5dad9fb514441404019b5d9e552566b97a62041dc3bf3db967c31d2`，`git diff --check` 通过，best checkpoint 均严格重载通过。
关键指标：普通 CE 最佳 epoch 23，Val Top-1/Top-5/Macro-F1/Balanced Accuracy 为 `88.2135%/96.5732%/71.8358%/71.0727%`；weighted CE 最佳 epoch 21，为 `88.3398%/95.7886%/72.5674%/72.4703%`；按 Val Macro-F1 选择 weighted CE，Macro-F1 `+0.7316` pp、Balanced Accuracy `+1.3976` pp、Top-1 `+0.1263` pp、Top-5 `-0.7846` pp。
遗留问题：A2 等待总负责人验收；初版 comparison 由旧 duration 汇总生成，正式结果以 R2 为准，R2 从完整 history 统计 CE/weighted 训练加验证耗时 `26118.87/26604.22s`；原始 CE `metrics.json` 的 duration 仅覆盖断点恢复后进程但逐轮时间完整；未提交或推送，未执行 A3。
是否进入下一阶段：否

## 2026-08-11 A3 最终测试与模型包

阶段：A3，待验收；输入 commit `1d34280ea30fe54a15031240e502274e07a555bb`、A2 已验收 weighted CE epoch 21、data-v1 D5-R1；模型/预处理/阈值 `0.55` 先冻结，固定 test 只评估一次且未调参。
修改文件：新增 `project/src/dlcpd25_classifier/training/a3.py`、`project/tests/test_training_a3.py` 并更新本日志；生成 `artifacts/training/a3-test-dlcpd25-resnet50-weighted-v1/` 和 `artifacts/releases/dlcpd25-resnet50-weighted-v1/`，未修改原图、taxonomy、split、应用代码或其他日志。
运行命令：运行 A3 合成定向 pytest、`a3 --verify-inputs-only`、唯一一次正式 A3 CLI、训练/发布 checksum、bundle loader、源权重与包内权重 `cmp`、固定 val 样例重复推理、项目全量 pytest、只读错误摘要和 `git diff --check`。
测试结果：A3 定向测试 `3 passed in 3.89s`；项目全量测试 `65 passed, 4 warnings in 216.85s`，warning 均为既有 Gradio 6.0 弃用提示；训练评估与发布 checksum 全部 OK，三张固定 val 样例重复 logits 及源/包 logits 逐位一致，taxonomy/权重 hash 篡改拒绝测试通过。
关键指标：test 22,178 张，loss `1.220233`、Top-1 `88.5517%`、Top-5 `95.7796%`、Macro-F1 `71.2177%`、Balanced Accuracy `71.2654%`，109.96 秒、201.68 img/s、峰值显存 `1,520,168,960` bytes；发布 checksum 清单 SHA-256 `b5b970ebe0f4cae436115fd7449e43f4f49ee6f361724e81b7bb7e4c4128af6a`。
遗留问题：阈值 `0.55` 为 test 前沿用 P1 已验收配置，test 低置信度率 `72.7342%`，P2 必须如实显示不确定提示且不得用 test 回调；class 162 仅 1 张且 F1 为 0，长尾与相似类混淆仍是风险；等待总负责人验收，未提交或推送，未执行 P2。
是否进入下一阶段：否
