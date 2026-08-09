# AI 数据工程师工作日志

当前状态：D1 待验收；D0 已由总负责人验收通过，未开始 D2。

后续记录必须按 `README.md` 模板追加，不得覆盖历史记录。

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
