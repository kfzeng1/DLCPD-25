# AI 数据工程师工作日志

当前状态：D2-D5 已退回；D2-R1 已完成自验并待总负责人验收；未执行 D3-R1 或 A0。

后续记录必须按 `README.md` 模板追加，不得覆盖历史记录。

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
