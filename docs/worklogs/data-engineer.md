# AI 数据工程师工作日志

当前状态：D0-D5 已全部通过并收敛为单版本正式链；D3-R2 直接读取 D2-R2；未执行 A0。

后续记录必须按 `README.md` 模板追加，不得覆盖历史记录。

## 2026-08-10 D5-R1 冻结并交接 data-v1

- 状态：待验收
- 指挥者指令：以 D2-R2、D3-R2、D4-R1 为正式链路冻结 data-v1，生成最终 release、taxonomy 快照、数据交接文档和校验清单；完成后停止，不启动 A0。
- 前置版本：Git commit `e8b2d639d5c5540f48c760248a9fb9b658468d18`；D0-D4 已由总负责人验收通过；正式 taxonomy SHA-256 为 `5cfa1a261b1a9fbb80adf24f299bca0883a42dd523914a70234f31dbf748bd31`。
- 允许范围：新增 D5-R1 冻结脚本、对应测试、`artifacts/data/v1/d5-r1/` 和本日志。D0-D4 代码/产物、原图、taxonomy、其他角色日志和验收清单保持只读；不得执行 A0、训练或应用开发。
- 实际修改：新增 `scripts/freeze_data_v1_d5_r1.py` 和 `project/tests/test_handoff_d5_r1.py`；更新本日志。旧退回的 D5 脚本、测试和产物未修改。D5-R1 只引用正式 `d2-r2/`、`d3-r2/`、`d4-r1/` 路径，并把本阶段脚本和测试纳入源码校验闭包。
- 生成产物：在 `artifacts/data/v1/d5-r1/` 新增 `d5-r1-config.json`、`data-v1-release.json`、`data-handoff-v1.md`、`taxonomy-v1.json`、`d5-r1-summary.json` 和 `checksums.sha256`，共 6 个文件、约 120 KiB。配置 SHA-256 为 `135d6bed198c4cc04a42ed90d39b8ff9f9dbacca65ae844759e3e7f2ff2ed2bf`，release 为 `a4a9e865429c8d60d67321971342395feedc0378cbc8b92b74d8832cc7eade8f`，交接文档为 `5ea4037cf7d78cb58af53ee678c0d74a2da6d59bbd61e6bf6e8214250e466d05`，taxonomy 快照为 `5cfa1a261b1a9fbb80adf24f299bca0883a42dd523914a70234f31dbf748bd31`，摘要为 `addacd2ce697fb66eb63d37fc2ef607419771908d86a57182a6bae998845fd57`，checksum 文件自身为 `d996e4ecdb75f24c469d3412d9392ab9e86da187fa21b0bb913a8e0a2775ad71`。
- 执行命令：`py_compile` 退出 0；D5-R1 轻量测试退出 0，`2 passed, 1 deselected`；D0/D1/D2-R2/D3-R2/D4-R1 五条上游 checksum 预检均退出 0；`/home/zkf/pytorch-env/bin/python scripts/freeze_data_v1_d5_r1.py` 退出 0、耗时 18.17 秒；同脚本 `--verify-only` 退出 0、耗时 12.56 秒；`sha256sum -c artifacts/data/v1/d5-r1/checksums.sha256` 退出 0，39 项全部 OK；taxonomy 源文件与快照 `cmp` 退出 0；项目全量测试退出 0，`50 passed in 497.13s`；正式 D2-R2/D3-R2/D4-R1 verifier 分别退出 0、耗时 272.03/13.92/72.28 秒；`git diff --check` 退出 0。
- 验收证据：release 固定正式链 `D0 -> D1 -> D2-R2 -> D3-R2 -> D4-R1 -> D5-R1`，索引 34 个关键上游产物、数据加载源码和受控阶段脚本；固定合同直接记录 manifest、重复组、train/val/test 和 taxonomy snapshot 的路径、大小与 SHA-256。交接文档明确算法工程师只读取 `d3-r2` 固定 CSV 的相对路径、`class_id`、SHA-256、`duplicate_group_id` 和 D5 taxonomy 快照，不得重新扫描、推断标签、重分组或重切分。文档采用总负责人修正措辞：D4 hardlink 兼容视图“由 D3 只读使用”，不声称文件权限本身只读；D3 的 D2-R1 名称仅为字节兼容引用，正式 release 来源为 D2-R2。
- 关键统计：原始文件 221,396，可用图片 221,377，坏图 19；完全重复 SHA 组 26,705，近重复组 32,731，最终非单例组 47,900，跨类别组 10,714，非单例文件 112,425，最大组 21。train/val/test 为 177,021/22,178/22,178，均覆盖 203 类且重复组泄漏和路径重叠为 0；可用图片少于 100 的长尾类别为 19 类。release 状态为 `frozen_pending_project_lead_acceptance`，`a0_executed=false`。
- 偏差、风险和阻塞：无 D5-R1 阻塞。已知限制完整写入交接文档：19 张坏图、702 张扩展名/编码错配、complete-link 漏召回风险、10,714 个跨类别组、长尾、D3-R1 兼容命名、D4 只实际解码 9 张以及数据许可未明确。D5-R1 通过 release 索引引用大文件，不复制 manifest 或 split；taxonomy 另存字节一致快照。release 在总负责人验收前保持 pending 状态。
- Git 状态：`main` 相对 `origin/main` ahead 14；本阶段新增 D5-R1 脚本和测试并修改本日志，D5-R1 artifacts 被 Git 忽略。进入本阶段前已有的旧 D2-R0、D3-R0/R1、D4-R0、D5-R0 未提交文件保持原样。未提交或推送。
- 下一步建议：由总负责人独立复验 D5-R1 verifier、39 项 checksum、34 项关键索引、taxonomy 快照、正式 split SHA 和交接文档；验收通过后再由用户决定是否调用算法工程师执行 A0。
- 边界声明：未执行 A0、训练或应用开发

## 2026-08-10 D4-R1 两轮独立复现与 Dataset 加载验证（解除阻塞后重启）

- 状态：待验收
- 指挥者指令：D2-R2 验收通过后重新执行 D4-R1；使用 D2-R2 与 D3-R2 完成两轮独立重建，并验证 train/val/test 均可由项目 Dataset 正确加载。D3-R2 无需重建版本，其 D2-R1 配置名称仅作为字节兼容引用。
- 前置版本：Git commit `b43e46e67f162a3da5c0fcdffdb0e6f989bd6cac`；D2-R2、D3-R2 已由总负责人验收通过；D2-R2 与 D2-R1 的 16 个兼容核心产物字节一致；taxonomy SHA-256 为 `5cfa1a261b1a9fbb80adf24f299bca0883a42dd523914a70234f31dbf748bd31`。
- 允许范围：D4-R1 复现脚本、Dataset 加载器及包导出、对应测试、`artifacts/data/v1/d4-r1/` 和本日志。D0-D3 代码/产物、原图、taxonomy、其他角色日志和验收清单保持只读；不得进入 D5/A0。
- 实际修改：新增 `scripts/verify_data_d4_r1.py` 和 `project/tests/test_data_d4_r1.py`；正式纳入 `project/src/dlcpd25_classifier/data/dataset.py` 及 `data/__init__.py` 的 Dataset 导出；更新本日志。旧退回的 `scripts/verify_data_d4.py`、旧测试及其他未提交文件未修改。D4-R1 只调用已受控的 `build_duplicates_d2_r2.py` 与 `build_splits_d3_r2.py`。
- 生成产物：在 `artifacts/data/v1/d4-r1/` 新增两套独立的 `repro-run-{1,2}/d2-r2/`、`d2-r1-compatible/`、`d3-r2/`，以及顶层 `d4-r1-config.json`、`reproduction-summary.json`、`load-smoke.json`、`d4-r1-summary.json` 和 `checksums.sha256`，共 67 个文件、约 504 MiB。配置 SHA-256 为 `46c47130e0ce2a41245b832a64de1153eed45f6b3fd86d3acc92705904f0103d`，复现摘要为 `5a2b7a52e0bf09b7ad5083eef95a3ef7e463eb5d3aaa9060655bcaebd508fcbe`，加载冒烟为 `c352bf3306a8d9eed5b94f8756946399f287eb574978e32145c1995407dcb343`，D4 摘要为 `07643433d2eceafbb839a2ee2fe22333d09087c9fb09455c5a5b4f1d35f409da`，checksum 文件自身为 `4ded1d6bb6f1caba02465887eb08a43bd9b4601aeddd17fe74868f42e6212cbc`。
- 执行命令：`py_compile` 退出 0；D4-R1 轻量预检退出 0，`5 passed, 1 deselected`；`/home/zkf/pytorch-env/bin/python scripts/verify_data_d4_r1.py --workers 6` 退出 0，总耗时 3905.79 秒；其正式输入 D2/D3 verifier 分别退出 0、耗时 184.250/9.444 秒；run 1 的 D2/D3 退出 0、耗时 1529.801/100.421 秒，run 2 为 1816.904/136.836 秒；同脚本 `--verify-only` 退出 0、耗时 70.85 秒；`sha256sum -c artifacts/data/v1/d4-r1/checksums.sha256` 退出 0，101 项全部 OK；独立 `cmp` 对两轮 D2 核心完成 38/38、D3 核心完成 10/10；项目全量测试退出 0，`47 passed in 493.57s`；`git diff --check` 退出 0。
- 验收证据：每轮均从独立空目录调用 D2-R2，从 D1 与原图重新计算 221,396 个 SHA-256 和 221,377 个 dHash/pHash；D2 内置 verifier 与 R1 字节兼容门均通过。每轮随后为该轮 R2 manifest 创建单独的 D2-R1 兼容视图，视图 manifest 与该轮 R2 manifest 为同一 hardlink、由 D3 只读使用且 SHA-256 相同，再由未修改的 D3-R2 生成 split；这落实了“D3 配置中的 D2-R1 仅为字节兼容引用”。正式 D2-R2、run 1、run 2 的 19 个 D2 核心文件三方 SHA 全部一致；正式 D3-R2、run 1、run 2 的 5 个数据核心文件三方 SHA 全部一致。
- 关键统计：复现运行 2/2；D2 核心匹配 19/19，D3 核心匹配 5/5。Dataset 初始化对 train 177,021、val 22,178、test 22,178 共 221,377 条记录检查相对路径、CSV schema、class_id 0-202、taxonomy 目录映射和文件存在性；各 split 解码首/中/末 3 张，共 9 张，均得到有限 `torch.float32 [3,224,224]` 张量，target 与记录一致，抽样覆盖 class 0、131、202。
- 偏差、风险和阻塞：无 D4-R1 阻塞。D3-R2 源码和生成配置仍使用 `D2-R1` 字段/参数名，这是总负责人明确批准的兼容契约，不表示 D4 使用旧 D2 实现；D4 配置和报告明确正式实现为 D2-R2。兼容视图使用同文件系统 hardlink 以避免每轮额外复制约 184 MiB manifest，顶层 checksum 同时覆盖视图和源文件。Dataset 初始化检查全量路径和标签，但只实际解码固定 9 张，图片全量解码结论仍继承 D1；加载时不逐张重算 SHA-256，数据内容完整性由两轮 D2 重算和 D4 checksum 保证。
- Git 状态：`main` 相对 `origin/main` ahead 13；本阶段新增 D4-R1 脚本、专用测试和 Dataset 文件，修改数据包 `__init__.py` 及本日志；D4-R1 artifacts 被 Git 忽略。进入本阶段前已有的旧 D2-R0、D3-R0/R1、D4-R0、D5 代码/测试保持原样。未提交或推送。
- 下一步建议：由总负责人独立复验 D4-R1 的 101 项 checksum、19/5 核心三方匹配、两个兼容视图来源及三个 split Dataset 加载；验收通过前不执行 D5-R1。
- 边界声明：未执行 D5-R1 或 A0

## 2026-08-10 D2-R2 独立重复组实现返工

- 状态：待验收
- 指挥者指令：消除 D2-R1 对未跟踪 D2-R0 脚本的依赖，保持 D2-R1 核心产物字节一致，完成后停止交验。
- 前置版本：Git commit `0524a1f088513318dc1b82e62f89fed6f6d7448d`；D0-D1 已通过；D2-R1 分组内容通过但因隐式源码依赖被退回；taxonomy SHA-256 为 `5cfa1a261b1a9fbb80adf24f299bca0883a42dd523914a70234f31dbf748bd31`。
- 允许范围：新增 D2-R2 独立脚本、对应测试、`artifacts/data/v1/d2-r2/` 和本日志。D2-R0/R1、D3-R2、原图、taxonomy、其他角色日志和验收清单保持只读。
- 实际修改：新增 `scripts/build_duplicates_d2_r2.py` 和 `project/tests/test_duplicates_d2_r2.py`；更新本日志。R2 为不导入任何仓库模块的单文件运行时实现，直接从 D1 manifest 和原图重算 SHA-256、dHash、pHash，并在同一文件内重建仅用于追溯的旧 D2-R0 group ID；未修改 D2-R1 代码或产物。
- 生成产物：在 `artifacts/data/v1/d2-r2/` 新增 manifest、重复组、回归报告、抽查索引与 12 张抽查页，以及 `d2-r2-config.json`、`d2-r2-summary.json`、`d2-r2-compatibility.json` 和 `checksums.sha256`。manifest SHA-256 为 `177e785b0cffd53ad0de7eb5aa3f2a2899127ca77558a774297929c2e2b80828`，重复组为 `58c50dcbe3bf40a21c58cd193c7bff08e2eefee7777ecdce95dc0ff7db910c0a`，配置为 `b9c576962f3a4159b61b7f19d5f8e049af88c5f5ee709a46442cb3b0ff706ef3`，摘要为 `480626519207b9803f200d0ce7aceb34a97bb380fff6b97aacd1173856e6b0af`，兼容报告为 `6f86c9617ce26e74e885574486475a68423662298ad1610cdfd31967cb491a2e`，checksum 文件自身为 `a2399e808e533f5899d82e6a168f908e9bdae83afaefaa84196f91881ebe5f5b`。
- 执行命令：`py_compile` 退出 0；首次专用预检因合成测试第三个 dHash 实际只距前项 1-2 位而错误预期为不同组，结果 `1 failed, 8 passed, 1 deselected`，确认实现符合旧传递闭包规则后将夹具改为距离大于 5 的 `0xFFFF`，重跑为 `9 passed, 1 deselected`、退出 0；`/home/zkf/pytorch-env/bin/python scripts/build_duplicates_d2_r2.py --workers 6` 退出 0，耗时 1538.83 秒；同脚本 `--verify-only` 退出 0，耗时 187.26 秒；`sha256sum -c artifacts/data/v1/d2-r2/checksums.sha256` 退出 0，25 项全部 OK；逐项 `cmp` D2-R1/R2 的 16 个核心文件，全部退出 0；项目全量测试退出 0，`41 passed in 321.56s`；`git diff --check` 退出 0。
- 验收证据：隔离测试只复制 D2-R2 脚本到临时 `scripts/`，明确不存在 `build_duplicates_d2.py`，仍可导入并执行 legacy group 和 R2 group 算法。R2 checksum 同时覆盖唯一运行时脚本和专用测试，其 SHA-256 分别为 `fb47db4b13ecfb51f46d9e0e2ecc972e0f84d42f597108fc3a45b6fc721886cd`、`5578f472663e368e09df8038c7d4a41564a56d10f7f7caed39aaf741e89391b5`，且不包含旧 D2-R0 脚本或 D2-R0 产物。脚本内固化 16 个 R1 核心文件预期 SHA-256，构建与 verifier 均强制逐项一致；manifest、重复组、抽查 JSON、回归 JSON 和 12 张抽查页全部逐字节相同。R1/R2 摘要 16 个核心统计字段差异为 0。
- 关键统计：总文件 221,396；SHA-256 221,396；dHash/pHash 各 221,377；坏图无感知哈希 19。最终 group 156,871，其中单例 108,971、非单例 47,900，非单例文件 112,425；完全重复 SHA 组 26,705，近重复组 32,731，跨类别组 10,714，最大组 21；最大 dHash/pHash 组内直径 5/8；被退回旧组回归 2/2 通过；R1 字节兼容核心产物 16/16。
- 偏差、风险和阻塞：无 D2-R2 阻塞。预检的一次失败属于测试夹具错误，已在正式生成前修正并全量回归。保守 complete-link 规则仍可能漏召回变化较大的真实近重复，10,714 个跨类别组仍提示潜在标签冲突，本阶段未改标签。D3-R2 当前配置仍以 D2-R1 路径和阶段名作为来源；虽然 R2 核心 manifest 字节一致，后续由总负责人决定是否仅更新 D4-R1 输入，或要求 D3 再生成新的来源元数据，本阶段不越权修改。
- Git 状态：`main` 相对 `origin/main` ahead 12；本阶段新增 D2-R2 脚本和测试并修改本日志，D2-R2 artifacts 被 Git 忽略。进入本阶段前已有的旧 D2-R0、D3-R0/R1、D4-D5 与 Dataset 未提交文件保持原样。未提交或推送。
- 下一步建议：由总负责人从干净受控源码独立复验 D2-R2 的导入、构建入口、25 项 checksum 和 16 项字节兼容门；通过前不重启 D4-R1。
- 边界声明：未执行 D3 返工、D4-R1、D5 或 A0

## 2026-08-10 D4-R1 两轮复现与 Dataset 加载验证

- 状态：阻塞
- 指挥者指令：基于 D2-R1 和 D3-R2 完成两轮独立复现，并验证 train/val/test 均可由项目 Dataset 正确加载；完成后停止交验。
- 前置版本：Git commit `7f4205cfe94442b44734f195790c5f59c6d34349`；总负责人已验收通过 D2-R1 和 D3-R2。
- 允许范围：D4-R1 复现脚本、Dataset 加载器、对应测试、`artifacts/data/v1/d4-r1/` 和本日志。不得返工未授权阶段或进入 D5/A0。
- 实际修改：仅更新本日志。未新增或修改 D4-R1 代码和产物，未修改原图、taxonomy、上游产物、其他角色日志或验收清单。
- 生成产物：无；在任何 D4-R1 正式产物生成前识别到上游复现入口阻断并停止。
- 执行命令：检查 Git 基线、受控文件和 D2-R1/D3-R2 入口，均为只读；从 commit `c4b56211319a42cf65aba11cc1163f82c9f13841` 使用 `git archive` 创建系统临时目录干净快照，确认其中不存在 `scripts/build_duplicates_d2.py`，再通过 Python `importlib` 导入快照内 `scripts/build_duplicates_d2_r1.py`，退出码 1，报 `FileNotFoundError: .../scripts/build_duplicates_d2.py`。
- 验收证据：`scripts/build_duplicates_d2_r1.py` 第 29-35 行在模块加载时运行时导入 `scripts/build_duplicates_d2.py`；后者不在 `git ls-files`，也不在已通过 D2-R1 的 commit `c4b5621` 中。当前工作树存在该未跟踪旧脚本，所以就地运行可能成功；干净 Git 快照无法导入，更无法进行两轮独立 D2-R1 重建。该问题与 D3-R1 曾被退回的隐式源码依赖性质相同。
- 关键统计：两轮正式复现 0/2，Dataset 正式加载验证 0/3；这是前置复现入口失败后的主动停止，不是数据或加载测试失败。D2-R1、D3-R2 既有 checksum 未改。
- 偏差、风险和阻塞：阻断。若绕过 D2-R1 重建、只把既有 D2-R1 manifest 当输入重跑 D3-R2，两轮结果只能证明 split 算法可复现，不能满足 D4 原定的 D2/D3 核心结果两轮独立重建。建议先执行 D2-R2：将 D2-R1 所需 SHA/dHash/EXIF 等辅助实现收进独立受控脚本或正式共享模块，并将全部运行时源码纳入 checksum；或者由总负责人明确缩减 D4-R1 范围为仅重建 D3-R2。未经决策不采用任一方案。
- Git 状态：`main` 相对 `origin/main` ahead 11；进入 D4-R1 前已有的旧 D2-R0、D3-R0/R1、D4-D5 未提交文件保持原样；本阶段只修改数据工程师日志。未提交或推送。
- 下一步建议：总负责人决定执行 D2-R2 依赖收敛返工，或明确批准缩减 D4-R1 的复现范围；解除阻断后重新执行 D4-R1。
- 边界声明：未执行 D4-R1 正式生成、D5 或 A0

## 2026-08-10 D3-R2 独立可复现 split 返工

- 状态：待验收
- 指挥者指令：消除 D3-R1 对未跟踪旧 D3-R0 脚本的运行时依赖，将全部依赖源码纳入 checksum，重新生成并验证 split，然后停止交验。
- 前置版本：Git commit `23db3b3eef49f83f73d963204dbb29698ab56132`；D2-R1 已通过；D3-R1 数据质量门通过但因不可独立复现被退回。
- 允许范围：新增 D3-R2 独立脚本、对应测试、`artifacts/data/v1/d3-r2/` 和本日志。D2-R1、D3-R1、旧 D3 及原图保持只读。
- 实际修改：新增 `scripts/build_splits_d3_r2.py` 和 `project/tests/test_splits_d3_r2.py`；更新本日志。未修改 D2-R1、D3-R1、旧 D3 产物、原图、taxonomy、其他角色日志或验收清单。
- 生成产物：在 `artifacts/data/v1/d3-r2/` 新增 `train.csv`、`val.csv`、`test.csv`、`group-assignments.csv`、`excluded-bad-images.csv`、`d3-r2-config.json`、`d3-r2-summary.json` 和 `checksums.sha256`。train/val/test SHA-256 分别为 `af457fcd9c49af93b9929585175aa68f973113d43895e5a035db61bbe7f7d778`、`a5db45590dd3dd97e46564046fc32c0223dab144e826854ea2a0e5aa3aec0833`、`23897e0a1a1b2209d1390845c6261ee48a1f94d935fc0173f3a71d18facc1dc8`；group assignments 为 `db68c397b52bf4c789f4c2b679d575cf4954c799d6f1ba628b6db936b074aa2e`；摘要为 `ba2a5f17ea0dded81c7f0b15e661c549b34765c3451271f6dce44f71f44b933d`；配置为 `1cca0fd31e82c34628617914a9d7807d822930d5760aef6675d1f94905c211c9`；checksums 文件自身为 `91bd054b73088e777efdbb04b4fdc351151a554fbb8ba2d877640b6d2c4f68bd`。
- 执行命令：`/home/zkf/pytorch-env/bin/python -m py_compile scripts/build_splits_d3_r2.py`，退出码 0；`cd project && /home/zkf/pytorch-env/bin/python -m pytest tests/test_splits_d3_r2.py -q -k 'not full_d3_r2_artifact and not checksum_has'`，退出码 0，`2 passed, 2 deselected`；`rg` 静态检查旧 D3 导入，退出码 0，仅命中 verifier 的禁止依赖文本；`/home/zkf/pytorch-env/bin/python scripts/build_splits_d3_r2.py`，退出码 0，耗时 134.74 秒；同脚本 `--verify-only`，退出码 0，耗时 12.56 秒；`sha256sum -c artifacts/data/v1/d3-r2/checksums.sha256`，退出码 0，13 项均为 `OK`；`cmp` 比较 D3-R1/R2 的 train、val、test、group assignments 和坏图清单，五项均退出 0；`cd project && /home/zkf/pytorch-env/bin/python -m pytest -q`，退出码 0，`31 passed in 176.86s`；`git diff --check`，退出码 0。
- 验收证据：D3-R2 是仅使用 Python 标准库的单文件实现，源码中不导入任何仓库内模块。隔离回归测试只把 D3-R2 脚本复制到临时 `scripts/`，明确确认旧 `build_splits_d3.py` 和 `build_splits_d3_r1.py` 不存在，再独立导入脚本并执行分组算法，退出码 0。配置固化唯一运行时源码路径及 SHA-256；13 项 checksum 同时覆盖 D2-R1 manifest/摘要/checksum、taxonomy、D3-R2 脚本、D3-R2 测试及全部 R2 输出，不包含被退回的 D3-R0/R1 脚本。正式 verifier 仍对 221,377 条 split 路径与 D2-R1 做集合和逐字段比对，并重算坏图、group assignment、class coverage 和泄漏。
- 关键统计：train 177,021（79.9636%）、val 22,178（10.0182%）、test 22,178（10.0182%）；group 数分别为 125,404、15,725、15,723；路径重叠 0，duplicate group 泄漏 0，坏图排除 19，三个 split 均覆盖 203 类。class 162 为唯一少于 10 个 group 的类别，5 张/5 组按 3/1/1 覆盖。D3-R1 与 D3-R2 的五个核心数据文件逐字节一致，证明依赖收敛未改变已通过的数据划分。
- 偏差、风险和阻塞：无阻塞。D3-R2 消除了 D3-R1 的隐式运行时依赖；其源码和测试仍需由总负责人纳入后续 Git 提交，当前 checksum 已固定二者内容。D2-R1 保守分组可能漏召回变化较大的真实近重复，因此零 group 泄漏只针对已识别的 R1 group。跨类别 group 被整组切分，未修改标签。
- Git 状态：`main` 相对 `origin/main` ahead 10；旧 D2-R0、D3-R0/R1、D4-D5 未提交代码仍保留，本阶段新增 D3-R2 脚本和测试并更新本日志；`artifacts/data/v1/d3-r2/` 被 Git 忽略。未提交或推送。
- 下一步建议：完成后交总负责人独立验收；通过前不得执行 D4-R1。
- 边界声明：未执行 D4-R1 或 A0

## 2026-08-10 D3-R1 基于 D2-R1 重建固定 split

- 状态：待验收
- 指挥者指令：只基于 `artifacts/data/v1/d2-r1/` 重建固定 train/val/test split；完成 D3-R1 自验和日志后停止，不得进入 D4-R1 或 A0。
- 前置版本：Git commit `c4b56211319a42cf65aba11cc1163f82c9f13841`；总负责人已验收通过 D2-R1；输入 manifest SHA-256 为 `177e785b0cffd53ad0de7eb5aa3f2a2899127ca77558a774297929c2e2b80828`。
- 允许范围：新增 D3-R1 数据脚本、对应测试、`artifacts/data/v1/d3-r1/` 和本日志。D2-R1、旧 D3 及原图保持只读。
- 实际修改：新增 `scripts/build_splits_d3_r1.py` 和 `project/tests/test_splits_d3_r1.py`；更新本日志。未修改 D2-R1、旧 D3 产物、原图、taxonomy、其他角色日志或验收清单。
- 生成产物：在 `artifacts/data/v1/d3-r1/` 新增 `train.csv`、`val.csv`、`test.csv`、`group-assignments.csv`、`excluded-bad-images.csv`、`d3-r1-config.json`、`d3-r1-summary.json` 和 `checksums.sha256`。train/val/test SHA-256 分别为 `af457fcd9c49af93b9929585175aa68f973113d43895e5a035db61bbe7f7d778`、`a5db45590dd3dd97e46564046fc32c0223dab144e826854ea2a0e5aa3aec0833`、`23897e0a1a1b2209d1390845c6261ee48a1f94d935fc0173f3a71d18facc1dc8`；group assignments 为 `db68c397b52bf4c789f4c2b679d575cf4954c799d6f1ba628b6db936b074aa2e`；摘要为 `f577e49aea6e056b0cb47c65bcf33d8898a63d56c41e585f0ea4fe4b250e46d5`；配置为 `6b89b1837699f610652b580ad82c7709045f8eb427cd0698244daa09ae4c8295`；checksums 文件自身为 `f5c275cc43168529ba66376afb178c5d1cd31f15635878c0a909433aad51813b`。
- 执行命令：`/home/zkf/pytorch-env/bin/python -m py_compile scripts/build_splits_d3_r1.py`，退出码 0；`cd project && /home/zkf/pytorch-env/bin/python -m pytest tests/test_splits_d3_r1.py -q -k 'not full_d3_r1_artifact'`，退出码 0，`2 passed, 1 deselected`；`/home/zkf/pytorch-env/bin/python scripts/build_splits_d3_r1.py`，退出码 0，耗时 133.79 秒；同脚本 `--verify-only`，退出码 0，耗时 12.65 秒；`sha256sum -c artifacts/data/v1/d3-r1/checksums.sha256`，退出码 0，12 项均为 `OK`；`cd project && /home/zkf/pytorch-env/bin/python -m pytest -q`，退出码 0，`27 passed in 157.57s`；`git diff --check`，退出码 0。
- 验收证据：输入加载器强制要求 D2-R1 的 pHash、旧 group 追溯字段和 `dg-r1-` group ID，并先验证 D2-R1 checksum 链。正式 verifier 将三个 split 的全部 221,377 条路径与 D2-R1 manifest 做集合和逐字段比对，class_id、SHA-256 和 duplicate group ID 差异均为 0；19 张坏图与 D2-R1 排除集合及字段完全一致。路径只出现一次、绝对路径和 `data/views/` 引用均为 0；156,852 个可用 group 全部只分配到一个 split，group assignments 的 split 和 size 与 CSV 重算一致；三个 split 均覆盖 203 类。
- 关键统计：train 177,021（79.9636%）、val 22,178（10.0182%）、test 22,178（10.0182%）；group 数分别为 125,404、15,725、15,723；路径重叠 0，duplicate group 泄漏 0，坏图排除 19。唯一少于 10 个独立 group 的类别为 class 162，共 5 张/5 组，按 train/val/test 3/1/1 覆盖。
- 偏差、风险和阻塞：无阻塞。固定沿用已记录的 `sparse-group-stratified-greedy-v1`、seed `20260809` 和 80/10/10 目标比例；duplicate group 完整性优先于精确比例。D2-R1 的保守分组仍可能漏召回变化较大的真实近重复，因此零 group 泄漏只针对已识别的 R1 group，不等价于证明不存在所有视觉近重复泄漏。跨类别 group 被整组切分，未修改标签。
- Git 状态：`main` 相对 `origin/main` ahead 9；旧 D2-R0、D3-D5 未提交代码仍保留，本阶段新增 D3-R1 脚本和测试并更新本日志；`artifacts/data/v1/d3-r1/` 被 Git 忽略。未提交或推送。
- 下一步建议：完成后交总负责人独立验收；通过前不得执行 D4-R1。
- 边界声明：未执行 D4-R1 或 A0

## 2026-08-10 D2-R1 重复组返工

- 状态：待验收
- 指挥者指令：总负责人基于 commit `80ef142` 退回 D2-D5，用户要求重新处理；按验收记录只执行 D2-R1，修复 dHash 无约束传递闭包造成的误合并并停止，不得进入 D3-R1 或 A0。
- 前置版本：Git commit `80ef142f8700bcfca10ad9159197730f3df6c1c9`；D0-D1 已通过；D2-R0 的 SHA-256 和 dHash 结果保留为已校验输入，但其重复组及 D3-D5 下游均已退回。
- 允许范围：新增 D2-R1 数据脚本、对应测试、`artifacts/data/v1/d2-r1/` 和本日志。旧 `artifacts/data/v1/d2/` 及原图保持只读。
- 实际修改：新增 `scripts/build_duplicates_d2_r1.py` 和 `project/tests/test_duplicates_d2_r1.py`；更新本日志。未修改 D2-R0 脚本、测试和产物，未修改原图、taxonomy、其他角色日志或验收清单。
- 生成产物：在 `artifacts/data/v1/d2-r1/` 新增 `manifest-hashed.jsonl`、`duplicate-groups.jsonl`、`d2-r1-config.json`、`d2-r1-summary.json`、`rejected-groups-regression.json`、`audit-samples.json`、12 张分层人工抽查页和 `checksums.sha256`。manifest SHA-256 为 `177e785b0cffd53ad0de7eb5aa3f2a2899127ca77558a774297929c2e2b80828`；重复组为 `58c50dcbe3bf40a21c58cd193c7bff08e2eefee7777ecdce95dc0ff7db910c0a`；配置为 `17f6cde9895bb17ee7fa3148f31f66a43aceb98c3a5893696903a9891ec47cbc`；摘要为 `2254efd0a8dcc19e99d02c3d6f397e025311e6ee2871638989da1c24ced04c27`；回归报告为 `6cde091b37121fea4a9b122fae7fb135e71a3d13c7573c46f4d41d06bf9c97ec`；抽查索引为 `b8ceda3c9f9c6c221641f771921a6a360ef2626775ca09044e01d08290d23bd6`；checksums 文件自身为 `b20e07fa6c7ed3f7f1f40775e83a893912e19979aec8ec983c530d0a1285f5fe`。
- 执行命令：只读诊断两个退回组的 pHash 分布，退出码 0；`/home/zkf/pytorch-env/bin/python -m py_compile scripts/build_duplicates_d2_r1.py`，退出码 0；`cd project && /home/zkf/pytorch-env/bin/python -m pytest tests/test_duplicates_d2_r1.py -q -k 'not full_d2_r1_artifact'`，退出码 0，`5 passed, 1 deselected`；`/home/zkf/pytorch-env/bin/python scripts/build_duplicates_d2_r1.py --workers 6`，退出码 0，耗时 1,243.88 秒；同脚本 `--verify-only`，退出码 0，耗时 148.60 秒；`sha256sum -c artifacts/data/v1/d2-r1/checksums.sha256`，退出码 0，25 项均为 `OK`；逐页人工查看 12 张抽查图，工具调用均成功；`cd project && /home/zkf/pytorch-env/bin/python -m pytest -q`，退出码 0，`24 passed in 147.06s`；`git diff --check`，退出码 0。
- 验收证据：D2-R0 checksum 链先验通过，221,396 个 SHA-256 和 221,377 个 dHash 原值保留；19 张坏图仍无感知哈希；为 221,377 张可解码图片新增 pHash。dHash 只召回候选，4,724 个候选指纹对被 pHash 拒绝；1,667 次会破坏 complete-link 的合并被拒绝。最终每组所有可解码指纹两两 dHash 距离不超过 5、pHash 距离不超过 8，全局实测最大直径正好为 5/8。精确 SHA 组仍为 26,705，未拆散任何精确重复。被点名的 `dg-001396` 从 182 张拆为 114 组、最大替代组 3 张；`dg-004019` 从 132 张拆为 101 组、最大替代组 3 张；两者所有替代组均通过直径门。
- 关键统计：共 156,871 个 group，其中单例 108,971、非单例 47,900，112,425 个文件位于非单例组；完全重复 SHA 组 26,705，涉及 54,749 个文件；包含不同 SHA 的近重复组 32,731；跨类别重复组 10,714；最大组 21。候选 dHash value 对 27,458，确认后的指纹边 28,533。人工抽查固定为最大组 20、跨类别组 20、随机组 20（seed `20260810`），明显误合并分别为 0/20、0/20、0/20；跨类别样例观察到相同主体被赋予不同 class_id 的标签冲突，但未见主体或构图无关的链式拼接。
- 偏差、风险和阻塞：无阻塞。返工采用 64-bit pHash 阈值 8 和确定性贪心 complete-link clique partition，消除了无约束传递闭包；更保守的规则可能把真实但变化较大的近重复拆成不同组，后续 D3 仍存在未召回近重复造成泄漏的剩余风险。60 组人工抽查能发现明显误合并但不能证明全量视觉正确；跨类别组仍需总负责人判断是否另立标签质量治理任务，本阶段未改标签。D2-R0 及其 D3-D5 下游仍保持退回状态，未覆盖或删除。
- Git 状态：`main` 相对 `origin/main` ahead 8；D2-R0 至 D5 的未提交代码仍保留，本阶段新增 D2-R1 脚本和测试并更新本日志；`artifacts/data/v1/d2-r1/` 被 Git 忽略。未提交或推送。
- 下一步建议：完成 D2-R1 产物、自动化质量门和人工抽查后提交总负责人复验；验收通过前不得执行 D3-R1。
- 边界声明：未执行 D3-R1 或 A0

## 2026-08-09 D0 冻结数据根与 taxonomy v1

- 状态：待验收
- 指挥者指令：执行 D0，冻结数据根目录、203 类类别口径和 taxonomy v1；不得进入 D1，不得提交或推送 Git。
- 前置版本：Git commit `d5d02c1e245edd4dc947fdd78d3dc30222619453`；原始数据基线为 203 类、221,396 个文件；taxonomy 为当前 `metadata/class-taxonomy.json`。
- 允许范围：`scripts/` 数据脚本、经批准的 `metadata/`、`project/src/dlcpd25_classifier/data/`、数据测试、`artifacts/data/` 和本日志。
- 实际修改：新增 `metadata/d0-freeze-config-v1.json`、`scripts/freeze_data_d0.py`、`project/tests/test_data_freeze_d0.py`；更新本日志。未修改原始图片、既有 taxonomy、其他角色日志或验收清单。
- 生成产物：`artifacts/data/v1/d0/` 下生成 `d0-freeze.json`、`checksums.sha256`、taxonomy JSON/CSV、aliases、官方类别清单和 D0 配置的 v1 快照。`d0-freeze.json` SHA-256 为 `ff2a614578e3aca2caa9557249ee8eee1fa293ae42af624a56add876e75d06e1`；`checksums.sha256` 自身 SHA-256 为 `c54b957b30163027aafb46723731ba17228b9256cfc5b203f65cf9b20b4bb807`。
- 执行命令：`python3 -m py_compile scripts/freeze_data_d0.py`，退出码 0；`cd project && /home/zkf/pytorch-env/bin/python -m pytest -q`，退出码 0，`4 passed`；`python3 scripts/freeze_data_d0.py` 连续运行两次，退出码均为 0 且摘要一致；`sha256sum -c artifacts/data/v1/d0/checksums.sha256`，退出码 0，13 项均为 `OK`；调用 `scripts/audit_dataset.py` 的审计函数复核当前目录，退出码 0；`git diff --check`，退出码 0。
- 验收证据：`artifacts/data/v1/d0/d0-freeze.json` 证明 class ID 来源为 taxonomy `classes` 数组且连续为 0-202；官方类别、aliases、taxonomy JSON/CSV 和本地目录均为 203 项且一一对应；独立目录审计结果为完整、无缺失、无额外、无空类别；`checksums.sha256` 固化数据配置、aliases、taxonomy、生成脚本和快照校验和；报告固化 22 个宿主、四大属性类别数和中文语义。
- 关键统计：203 类，221,396 个文件，总字节数 16,714,602,771；22 个宿主；`pest=126`、`disease=57`、`healthy=17`、`disorder=3`；扩展名统计为 BMP 92、GIF 7、JFIF 513、JPEG 78、JPG 220,428、PNG 277、WEBP 1；隐藏根条目 0；路径与文件大小目录指纹 SHA-256 为 `df51896d7b0ce2cbf61dd8d473ccf8a62bd71875cda967d78f0ace36dfda3ec9`。
- 偏差、风险和阻塞：无阻塞。工作树存在用户已有的文档和工程骨架改动，均已保留。D0 目录指纹只覆盖相对路径和文件大小，不能发现同路径、同大小的内容替换；逐文件内容 SHA-256 属于 D2。D0 未解码图片，坏图状态属于 D1。`target_data_version=data-v1` 当前仅为 D0 冻结目标，尚未完成 D5 交接。
- Git 状态：`main` 相对 `origin/main` ahead 5。本阶段新增 3 个未跟踪代码/配置文件并更新未跟踪的本日志；`artifacts/data/v1/d0/` 被 Git 忽略。其他已修改和未跟踪项为进入 D0 前已有改动，未清理、覆盖或回退。
- 下一步建议：由总负责人独立复验 D0 并给出通过、退回或阻塞结论；未通过前不得执行 D1。
- 边界声明：未执行下一阶段

## 2026-08-09 D2 内容哈希、dHash 与重复组

- 状态：待验收
- 指挥者指令：基于 commit `4903e91` 连续执行 D2-D5；当前先完成 D2 全量 SHA-256、可解码图片 dHash、完全重复与近重复组，完成阶段自验和日志后方可进入 D3。
- 前置版本：Git commit `4903e91910251ebd0288a5660066c4b709e2824c`；D1 已由总负责人标记通过；D1 manifest SHA-256 为 `53c853a9f307d503edfda7f06548da57be15146351b6a3a81088f54d9374dbb2`。
- 允许范围：`scripts/` 数据脚本、数据测试、`artifacts/data/` 和本日志。
- 实际修改：新增 `scripts/build_duplicates_d2.py` 和 `project/tests/test_duplicates_d2.py`；更新本日志。未修改原图、taxonomy、其他角色日志或验收清单。
- 生成产物：`artifacts/data/v1/d2/manifest-hashed.jsonl`、`duplicate-groups.jsonl`、`duplicate-samples.json`、`d2-config.json`、`d2-summary.json` 和 `checksums.sha256`。manifest SHA-256 为 `4a098ea1b792de9b50a43f643d25195465082925a373181a9825076b1cc275b4`；重复组清单为 `e1887322baf13e0692e1c132dcfa54dc7ca3b77aabb792950e1ad005ad543d18`；摘要为 `26f229e9e980a5eb63435c96fe4e81507318fafab6e713abdede135d69ee4c64`；checksums 文件自身为 `16dcd9749482ed6981df721fc1669bcff41205d0c5e469a63000babfce12fcd4`。
- 执行命令：`python3 -m py_compile scripts/build_duplicates_d2.py`，退出码 0；`cd project && /home/zkf/pytorch-env/bin/python -m pytest -q`，退出码 0，`12 passed`；`/home/zkf/pytorch-env/bin/python scripts/build_duplicates_d2.py --workers 6`，稳定版本退出码 0，耗时 1,132.02 秒；同脚本 `--verify-only`，退出码 0，耗时 129.99 秒；`sha256sum -c artifacts/data/v1/d2/checksums.sha256`，退出码 0，10 项均为 `OK`；`git diff --check`，退出码 0。
- 验收证据：全部 221,396 个文件具有 SHA-256；D1 的 221,377 张可解码图片具有 16 位十六进制 dHash；19 张坏图保留 SHA-256 且 dHash 为空。独立 verifier 从 dHash 重新构建 group ID，与 manifest 逐行一致；所有相同 SHA-256 文件均只属于一个组。配置固化 dHash v1、阈值 5、六段候选索引、传递闭包和确定性 group ID 规则；抽查文件优先列出跨类别及最大组。
- 关键统计：共 153,409 个 duplicate group，其中单例 105,599、重复组 47,810，115,797 个文件处于非单例组；完全重复 SHA 组 26,705，涉及 54,749 个文件；包含不同 SHA 的近重复组 32,945；跨类别重复组 11,099；最大组 182。唯一 dHash 176,727；直接近邻边 27,458，距离 1/2/3/4/5 分别为 10,920/5,684/3,409/3,203/4,242。EXIF 状态为 applied 2,871、identity 5,893、invalid_ignored 17、missing 212,596、坏图 not_applicable 19。
- 偏差、风险和阻塞：首次全量运行因 17 张图片含非法字符串/范围 EXIF Orientation，Pillow 通用转置报错并在发布产物前退出；修复为仅对整数 2-8 应用标准变换、非法值明确记为 `invalid_ignored` 后完整重跑。阈值 5 加传递闭包是防泄漏优先的保守策略，人工样例确认部分大组存在链式端点差异，并发现真实跨标签完全相同 SHA；因此 group 不等价于人工确认的同一语义，D3 必须整组切分且量化分层偏差。未删除任何重复或坏图。
- Git 状态：`main` 相对 `origin/main` ahead 7；本阶段修改本日志，新增未跟踪的 D2 脚本和测试；D2 artifacts 被 Git 忽略。开始 D2 时工作树干净。
- 下一步建议：D2 自验已通过，按用户本次连续授权进入 D3；总负责人仍需逐阶段独立验收。
- 边界声明：尚未执行 D3-D5 或 A0

## 2026-08-09 D3 固定 train/val/test split

- 状态：待验收
- 指挥者指令：在 D2 自验通过后继续 D3；排除 19 张坏图，按 duplicate group 整组切分，固定 seed 和比例，泄漏必须为 0 才能进入 D4。
- 前置版本：Git commit `4903e91910251ebd0288a5660066c4b709e2824c`；D2 manifest SHA-256 为 `4a098ea1b792de9b50a43f643d25195465082925a373181a9825076b1cc275b4`；D2 已完成阶段自验。
- 允许范围：`scripts/` 数据脚本、数据测试、`artifacts/data/` 和本日志。
- 实际修改：新增 `scripts/build_splits_d3.py` 和 `project/tests/test_splits_d3.py`；更新本日志。未修改 D0-D2 产物、原图、taxonomy、其他角色日志或验收清单。
- 生成产物：`artifacts/data/v1/d3/train.csv`、`val.csv`、`test.csv`、`group-assignments.csv`、`excluded-bad-images.csv`、`d3-config.json`、`d3-summary.json` 和 `checksums.sha256`。train/val/test SHA-256 分别为 `ad23207863130c81927938e2a467e1bdbe40c2d4c0a8bce878ddddc5aa961e14`、`20f2cfc4ac8f1d53b1f7ee2cb2c2c9938a3daedc60c0a74b4850c42238b25786`、`82e03836c50882a9f7f4aea706ded47febc169d442842b700308c1888cdda658`；摘要为 `8248e665d7fe2f9707451735fa3152ebcdb225ff39ecae8746834b0c138e83db`；checksums 文件自身为 `c7bcdc1fec3dfaeb84b226106d2450afbd2f85a06e7bd18dd8dbd37e4be235d2`。
- 执行命令：`python3 -m py_compile scripts/build_splits_d3.py`，退出码 0；`cd project && /home/zkf/pytorch-env/bin/python -m pytest -q`，退出码 0，`15 passed`；`/home/zkf/pytorch-env/bin/python scripts/build_splits_d3.py`，退出码 0，耗时 91.48 秒；同脚本 `--verify-only`，退出码 0；`sha256sum -c artifacts/data/v1/d3/checksums.sha256`，退出码 0，12 项均为 `OK`；`git diff --check`，退出码 0。
- 验收证据：19 张 D1 坏图全部进入独立排除清单且未进入三个 split；train/val/test 共 221,377 条路径、路径重叠 0；153,390 个可用 duplicate group 仅分配到一个 split，跨 split group 数为 0；三个 split 均覆盖 203 类；CSV 只包含相对路径、固定 class ID、SHA-256、duplicate group 和 split，schema 已固化。
- 关键统计：train 177,012（79.9595%）、val 22,183（10.0205%）、test 22,182（10.0200%）；group 数分别为 122,710、15,313、15,367。独立 group 少于 10 的类别为 class 162（5 组、3/1/1 张）、167（5 组、16/2/4 张）、169（9 组、17/2/2 张）和 170（9 组、49/6/7 张），均通过覆盖预留进入三个 split。
- 偏差、风险和阻塞：无阻塞。初始只读 SGKF 原型运行 13 分钟仍未完成且警告最少类别仅 5 个 group，已终止、未写产物；正式采用 `sparse-group-stratified-greedy-v1`，固定 seed `20260809`，先满足少组类别覆盖，再按全局和逐类目标缺口分配。首次 D3 测试因动态导入未注册 `sys.modules` 在收集阶段失败，修复测试加载器后全部通过；未带问题进入正式生成。保守 D2 大组会限制精确分层，实际比例偏差已量化且很小。
- Git 状态：`main` 相对 `origin/main` ahead 7；D2 修改仍未提交，本阶段新增 D3 脚本和测试并更新本日志；D3 artifacts 被 Git 忽略。
- 下一步建议：D3 自验已通过，按用户连续授权进入 D4；总负责人仍需逐阶段独立验收。
- 边界声明：尚未执行 D4-D5 或 A0

## 2026-08-09 D4 复现与 Dataset 加载验证

- 状态：待验收
- 指挥者指令：在 D3 零泄漏自验通过后继续 D4；从固定配置连续两次重建 D2/D3 核心结果，实现项目 Dataset，并对三个 split 做加载和预处理冒烟。
- 前置版本：Git commit `4903e91910251ebd0288a5660066c4b709e2824c`；D2/D3 已完成阶段自验；固定 split hashes 为 train `ad232078...61e14`、val `20f2cfc4...25786`、test `82e03836...da658`。
- 允许范围：`scripts/` 数据脚本、`project/src/dlcpd25_classifier/data/`、数据测试、`artifacts/data/` 和本日志。
- 实际修改：新增 `project/src/dlcpd25_classifier/data/dataset.py`、`project/tests/test_dataset_d4.py` 和 `scripts/verify_data_d4.py`；更新数据包 `__init__.py` 和本日志。未修改原图、taxonomy、正式 D2/D3 产物、其他角色日志或验收清单。
- 生成产物：`artifacts/data/v1/d4/repro-run-1/`、`repro-run-2/` 两套独立 D2/D3 重建，及顶层 `d4-config.json`、`reproduction-summary.json`、`load-smoke.json`、`d4-summary.json` 和 `checksums.sha256`。D4 摘要 SHA-256 为 `cab09eaadcac35089f127718c57709b1b0b3833d68e688e79ace33c832d13ff3`；复现摘要为 `37ae00e191804b99b28b33a878e96cb328af2ac9d39a24e9e3cef2d58046bbf2`；加载冒烟为 `29721e4c6ac4869125e970da4d7cbfb32b42b016d5d3db8e2d1efe4c77eb0521`；checksums 文件自身为 `6c1419bf73c8d6cac288d3785d150084a7bd5ac3d5ec6bb1f06c7dfca6e3d043`。
- 执行命令：`python3 -m py_compile scripts/verify_data_d4.py project/src/dlcpd25_classifier/data/dataset.py`，退出码 0；`cd project && /home/zkf/pytorch-env/bin/python -m pytest -q`，退出码 0，`17 passed`；`/home/zkf/pytorch-env/bin/python scripts/verify_data_d4.py --workers 6`，退出码 0，总耗时 2,420.85 秒；同脚本 `--verify-only`，退出码 0；`sha256sum -c artifacts/data/v1/d4/checksums.sha256`，退出码 0，18 项均为 `OK`；`git diff --check`，退出码 0。
- 验收证据：两次从 D1 独立重建 D2，再分别从各自 D2 重建 D3。正式版本、run 1、run 2 的 D2 核心 `manifest-hashed.jsonl`、重复组、抽查样例、配置和摘要 hashes 全部一致；D3 核心 train/val/test、group assignments 和坏图排除清单 hashes 全部一致。Dataset 初始化时校验相对路径、CSV schema、class ID、taxonomy 目录和源文件；train/val/test 各加载首/中/末 3 个样本，均得到有限 `torch.float32 [3,224,224]` tensor 且 target 与记录一致。
- 关键统计：run 1 D2/D3 耗时 1,126.762/68.731 秒，run 2 为 1,084.300/74.664 秒；加载长度 train 177,012、val 22,183、test 22,182；每 split 冒烟 3 张，共 9 张。
- 偏差、风险和阻塞：无阻塞。两次复现会在 D4 下保留约两套 D2/D3 可再生产物，占用额外磁盘但未复制原始图片；这是 D4 的复现证据。Dataset 只验证 CSV 中已冻结 SHA 字段格式，加载时不重复计算内容 SHA；数据准入时由 checksums 和 D4 复现负责完整性。加载器对非法 EXIF Orientation 采用与 D2 相同的忽略策略并统一转 RGB。
- Git 状态：`main` 相对 `origin/main` ahead 7；D2-D3 修改仍未提交，本阶段新增 D4 脚本、加载器和测试并更新数据包导出及本日志；D4 artifacts 被 Git 忽略。
- 下一步建议：D4 自验已通过，按用户连续授权进入 D5；总负责人仍需逐阶段独立验收。
- 边界声明：尚未执行 D5 或 A0

## 2026-08-10 D5 冻结并交接 data-v1

- 状态：待验收
- 指挥者指令：在 D4 复现和加载全部通过后继续 D5；冻结最终 manifest、重复组和 split 版本，生成 `data-handoff-v1.md`、版本清单和全部关键 SHA-256；不得启动 A0。
- 前置版本：Git commit `4903e91910251ebd0288a5660066c4b709e2824c`；D2-D4 均已完成阶段自验；D4 两轮复现 hashes 全部一致。
- 允许范围：`scripts/` 数据脚本、数据测试、`artifacts/data/` 和本日志。
- 实际修改：新增 `scripts/freeze_data_v1_d5.py` 和 `project/tests/test_handoff_d5.py`；更新本日志。未修改原图、taxonomy、D0-D4 产物、其他角色日志或验收清单。
- 生成产物：`artifacts/data/v1/d5/taxonomy-v1.json`、`data-v1-release.json`、`data-handoff-v1.md`、`d5-summary.json` 和 `checksums.sha256`。release SHA-256 为 `9122cfb7af50bbb27449f895025c441f252aaad24838cda96195d5f4d3669912`；交接文档为 `815b3bce4ada9fcd77a96f10b3364d4ed98e6c50a40df600c4dbc5d7a87d4776`；taxonomy 快照为 `5cfa1a261b1a9fbb80adf24f299bca0883a42dd523914a70234f31dbf748bd31`；摘要为 `a52b78b876b666bda7d3ee02419e3d67440276e9e5bcbe0ba198affd866780a2`；checksums 文件自身为 `4253d6089ae444836595813676367047f2b4a9a14806bffad79beb318da0c99e`。
- 执行命令：`/home/zkf/pytorch-env/bin/python -m py_compile scripts/freeze_data_v1_d5.py`，退出码 0；`cd project && /home/zkf/pytorch-env/bin/python -m pytest tests/test_handoff_d5.py -q`，退出码 0，`1 passed`；`/home/zkf/pytorch-env/bin/python scripts/freeze_data_v1_d5.py`，退出码 0，耗时 15.26 秒；同脚本 `--verify-only`，两次退出码均为 0，最终耗时 12.24 秒；`sha256sum -c artifacts/data/v1/d5/checksums.sha256`，退出码 0，25 项均为 `OK`；D2/D3/D4 最终 `--verify-only` 分别退出码 0；`cd project && /home/zkf/pytorch-env/bin/python -m pytest -q`，退出码 0，`18 passed`；逐项执行 D0-D5 全部 `checksums.sha256`，退出码 0；`git diff --check`，退出码 0。
- 验收证据：D5 生成前重验 D0-D4 checksum 链；release index 固化 20 个关键上游文件的仓库相对路径、大小和 SHA-256，并保存与源 taxonomy 字节一致的 v1 快照。交接文档明确任务边界、固定数据契约、三个 split、长尾类别、已知限制、加载器入口、生成命令、环境和关键 hashes；release 状态为 `frozen_pending_project_lead_acceptance`，摘要明确 `a0_executed=false`。最终 D2 verifier 重建全部重复关系并逐行一致；D3 泄漏和路径重叠均为 0；D4 两次复现与加载结果保持通过。
- 关键统计：原始文件 221,396，可用图片 221,377，坏图 19；完全重复 SHA 组 26,705，近重复组 32,945，最终非单例重复组 47,810，跨类别组 11,099；train/val/test 为 177,012/22,183/22,182，均覆盖 203 类；以可用图片少于 100 定义的长尾类别共 19 类。运行环境为 Python 3.12.3、Pillow 12.2.0、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128。
- 偏差、风险和阻塞：无阻塞。首次组合语法检查命令误在 `project/` 工作目录引用根目录相对脚本路径，退出码 1，未开始语法检查或测试、未写产物；随后在正确工作目录重跑并通过。data-v1 采用 release manifest 引用并校验 D0-D4 固定大文件，不在 D5 重复复制 manifest/split；D5 自身保存 taxonomy 快照和交接文档。已知限制包括 19 张坏图、702 张扩展名/编码错配、保守 dHash 传递闭包、跨类别重复组、少样本类别方差和数据许可未明确；均已写入交接文档。
- Git 状态：`main` 相对 `origin/main` ahead 7；D2-D4 修改仍在工作树中，本阶段新增 D5 脚本和测试并更新本日志；D0-D5 artifacts 被 Git 忽略。未提交或推送。
- 下一步建议：D5 自验完成后停止，等待总负责人逐阶段验收；不启动 A0。
- 边界声明：未执行 A0

## 2026-08-09 D1 Manifest 与图片解码审计

- 状态：待验收
- 指挥者指令：基于 commit `cc99801` 执行 D1，生成完整图片 manifest，完成图片解码与坏图审计；产物放入 `artifacts/data/v1/d1/`，不得进入 D2，不得提交或推送 Git。
- 前置版本：Git commit `cc99801be041d1756c5e4a6bcf8a07eac4e3f40b`；D0 已由总负责人标记通过；`d0-taxonomy-v1` 报告 SHA-256 为 `ff2a614578e3aca2caa9557249ee8eee1fa293ae42af624a56add876e75d06e1`。
- 允许范围：`scripts/` 数据脚本、`project/src/dlcpd25_classifier/data/`、数据测试、`artifacts/data/` 和本日志。
- 实际修改：新增 `scripts/build_manifest_d1.py` 和 `project/tests/test_manifest_d1.py`；更新本日志。未修改原始图片、taxonomy、其他角色日志或验收清单。
- 生成产物：`artifacts/data/v1/d1/manifest.jsonl`、`bad-images.jsonl`、`d1-config.json`、`d1-summary.json`、`sampled-successful-images.json` 和 `checksums.sha256`。manifest SHA-256 为 `53c853a9f307d503edfda7f06548da57be15146351b6a3a81088f54d9374dbb2`；坏图清单为 `fac9297a125963200cc24e9158d93dd4fbd9928fbe6b6a8640286a716fea266a`；摘要为 `7c48fb94291136897d3f37c7a11e9222f9c01d0f0dafa544dc3d229b5d4412a0`；checksums 文件自身为 `2235bcbf3d5b0dde7084a4f4314e22e9d8cbdc132782a3a41aa307bc7e56bead`。
- 执行命令：`python3 -m py_compile scripts/build_manifest_d1.py`，退出码 0；`cd project && /home/zkf/pytorch-env/bin/python -m pytest -q`，退出码 0，`8 passed`；`/home/zkf/pytorch-env/bin/python scripts/build_manifest_d1.py --workers 6`，退出码 0，稳定版本耗时 691.79 秒；同脚本 `--verify-only`，退出码 0；`sha256sum -c artifacts/data/v1/d1/checksums.sha256`，退出码 0，10 项均为 `OK`；只读调用 D0 扫描函数重算目录指纹，退出码 0；`git diff --check`，退出码 0。
- 验收证据：manifest 共 221,396 行且相对路径 221,396 个、全部唯一，不含绝对数据根或 `data/views/`；每行 schema 固定并包含 class ID、宿主、四大类、扩展名、Pillow 格式、尺寸、模式、通道、帧数和解码状态；203 类逐类计数与 taxonomy 一致且每类至少一张成功解码；固定种子 `20260809` 为每类抽取 1 张成功样本；坏图清单 19 行，与 manifest 的全部 `bad` 记录顺序一致；D0 目录指纹重算仍为 `df51896d7b0ce2cbf61dd8d473ccf8a62bd71875cda967d78f0ace36dfda3ec9`。
- 关键统计：总文件 221,396，成功 221,377，坏图 19，坏图分布在 16 类；错误类型为 `UnidentifiedImageError=18`、`OSError=1`。实际格式为 JPEG 220,394、PNG 883、BMP 92、GIF 7、WEBP 1；模式为 RGB 221,104、RGBA 259、P 14；通道数为 3 通道 221,104、4 通道 259、1 通道 14；宽 63-8,113，高 52-6,000。发现 702 个可成功解码的扩展名/编码错配：`.jpg->PNG` 654、`.png->JPEG` 48。
- 偏差、风险和阻塞：无阻塞。首轮正式产物中 Pillow 错误文本包含仓库绝对路径，会破坏跨目录复现；已在验收前删除该轮 6 个可再生产物并完整重建，稳定版使用 `<source>` 脱敏，原图未受影响。一次直接重跑 D0 冻结脚本因当前 commit 与不可覆盖的 D0 报告不同而按设计退出 1，未覆盖 D0 产物；随后用只读扫描函数确认目录指纹、文件数和总字节数均未变化。19 张坏图只标记不删除；702 个格式错配说明下游不能只按扩展名判断解码器。D1 未计算图片内容 SHA-256、dHash、重复组或 split，这些属于 D2-D3。
- Git 状态：`main` 相对 `origin/main` ahead 6；本阶段修改 `docs/worklogs/data-engineer.md`，新增未跟踪的 `scripts/build_manifest_d1.py` 和 `project/tests/test_manifest_d1.py`；D1 artifacts 被 Git 忽略。开始 D1 时工作树干净，没有覆盖用户改动。
- 下一步建议：由总负责人独立复验 D1 并给出通过、退回或阻塞结论；D1 未通过前不得执行 D2。
- 边界声明：未执行下一阶段
