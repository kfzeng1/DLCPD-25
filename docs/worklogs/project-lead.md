# 总负责人工作与验收日志

当前状态：历史分类基线 F0 已通过；目标检测扩展按 T0-T4、F1 推进，T0 尚未开始。

## 2026-08-12 目标检测扩展治理准备

- 结论：通过
- 工作分配：数据工程师负责 T0；算法工程师依次负责 T1、T2、T3；应用工程师负责 T4；总负责人负责逐阶段复验、提交和 F1。
- 完成内容：统一项目计划、职责、流程、验收表和三份启动提示词；新增三份工程师工作单；更新 README、应用契约和历史 F0 边界说明；建立 `scripts/ip102/` 入口。
- 目录处理：保留 `data/raw/` 原始数据和 `artifacts/` 已验收产物，清理 Python 与 pytest 缓存，不移动现有受控路径。
- 独立复验：Markdown 本地链接检查无断链；检测映射、Dataset、共享模型定向测试 `7 passed`；`git diff --check` 通过。
- 下一阶段：用户调用数据工程师执行 T0；T0 未验收前，算法工程师不得启动 T1。

## 2026-08-09 G0 项目治理准备

- 结论：通过
- 验收输入：现有目录、项目文档、工程模块骨架和本机 PyTorch 环境
- 独立复验：数据目录审计为 203 类、221,396 个文件；现有测试为 2 passed
- 完成内容：建立阶段编号、职责边界、调用流程、验收清单、固定提示词和长期工作日志
- 当前限制：D0-D5、A0-A5、P0-P4 均未开始，不能把治理准备视为数据、算法或应用交付
- 建议下一阶段：由用户调用 AI 数据工程师执行 D0
- 用户决定：已执行 D0

## 2026-08-09 D0 验收

- 结论：通过
- 验收输入：`d0-taxonomy-v1`、基线 commit `d5d02c1e245edd4dc947fdd78d3dc30222619453`、数据工程师 D0 报告和 `artifacts/data/v1/d0/`
- 独立复验命令与结果：项目测试 `4 passed`；数据审计为 203 类、221,396 个文件、无缺失/额外/空类；`sha256sum -c` 的 13 项全部为 `OK`；在全新临时目录重建并用 `cmp` 对比，冻结报告和 taxonomy/aliases 快照完全一致
- 文件和范围检查：新增 `metadata/d0-freeze-config-v1.json`、`scripts/freeze_data_d0.py` 和 `project/tests/test_data_freeze_d0.py`；未修改原图、既有 taxonomy、其他工程师日志或验收状态；未执行 D1
- 关键证据：class ID 连续为 0-202；22 个宿主；四大类为 pest 126、disease 57、healthy 17、disorder 3；目录指纹为 `df51896d7b0ce2cbf61dd8d473ccf8a62bd71875cda967d78f0ace36dfda3ec9`；冻结报告 SHA-256 为 `ff2a614578e3aca2caa9557249ee8eee1fa293ae42af624a56add876e75d06e1`
- 发现的问题：无阻断问题。D0 指纹按相对路径和文件大小生成，不覆盖文件内容；这是已记录的阶段边界，图片解码属于 D1，逐文件 SHA-256 和 dHash 属于 D2
- 验收清单更新：D0 标记为通过；D1 和 P0 的前置条件现已满足
- 建议下一阶段：先提交当前治理和 D0 改动形成可追溯基线，再调用 AI 数据工程师执行 D1；P0 可并行准备，但不是当前关键路径
- 用户决定：已提交 D0，并执行 D1

## 2026-08-09 D1 验收

- 结论：通过
- 验收输入：commit `cc99801be041d1756c5e4a6bcf8a07eac4e3f40b`、D1 manifest、坏图清单、配置、摘要、抽样清单、校验和及数据工程师工作日志
- 独立复验命令与结果：`--verify-only` 返回 221,396 行、19 张坏图、203 类均有可用图片；项目测试 `8 passed`；D1 校验清单 10 项全部为 `OK`；独立遍历 manifest 确认 221,396 个相对路径唯一、文件存在且大小一致；重新解码每类 1 张固定样本，共 203 张全部成功
- 文件和范围检查：新增 `scripts/build_manifest_d1.py` 和 `project/tests/test_manifest_d1.py`，仅更新数据工程师自己的日志；未修改原图、taxonomy、其他角色日志或验收状态；未执行 D2
- 关键证据：manifest SHA-256 为 `53c853a9f307d503edfda7f06548da57be15146351b6a3a81088f54d9374dbb2`；221,377 张可解码，19 张坏图分布在 16 类；实际编码错配 702 张，其中 `.jpg->PNG` 654 张、`.png->JPEG` 48 张
- 发现的问题：19 张坏图不能进入训练、验证或测试；702 张扩展名与实际编码不一致，后续加载必须依赖 Pillow 实际解码，不能只按扩展名判断格式
- 数据处理决策：D2 对全部 221,396 个文件计算 SHA-256；只对 221,377 张可解码图片计算 dHash 和近重复组。D3 生成 split 时排除 19 张坏图，但保留在主 manifest 和坏图审计中，不删除原文件
- 验收清单更新：D1 标记为通过；D2 前置条件已满足
- 建议下一阶段：调用 AI 数据工程师执行 D2，完成内容哈希、感知哈希和重复组识别
- 用户决定：已连续执行 D2-D5，并提交总负责人验收

## 2026-08-10 D2-D5 验收

- 结论：D2 退回；D3、D4、D5 因依赖 D2 一并退回
- 验收输入：commit `4903e91910251ebd0288a5660066c4b709e2824c`、D2-D5 脚本、测试、阶段日志和 `artifacts/data/v1/d2/` 至 `d5/`
- 自动复验结果：D2、D3、D4、D5 verifier 均退出 0；D0-D5 全部校验链通过；项目测试 `18 passed`；D2 manifest 与三个 split 的 221,377 条可用记录全量关联一致，19 张坏图全部排除，duplicate group 跨 split 为 0
- 阻断问题：D2 使用“dHash 距离不超过 5 的边 + 无约束传递闭包”形成近重复组，链式连接将明显无关图片合并。`dg-001396` 包含 182 张、横跨 34 类，人工抽查同时出现棉花叶片、蓝底芒果虫瘿缩略图和健康棉花叶，抽样成员两两 dHash 距离最高 21；`dg-004019` 包含 132 张、横跨 34 类，同时出现白粉虱、柑橘实蝇幼虫和麦蚜，抽样距离最高 15
- 判断依据：D2 的目标是识别精确和近重复图片。可重复地产生错误分组不等于分组正确；当前 47,810 个非单例组涉及 115,797 个文件，并有 11,099 个跨类别组，错误合并会扭曲分层 split 和重复统计
- 返工要求：保留已经正确的全文件 SHA-256；dHash 只能作为候选召回，不得直接通过无约束传递闭包定义最终近重复组。增加更强的二次相似度确认，并限制最终组内直径或相对代表样本的距离；将上述两个错误组加入回归测试；分别抽查最大组、跨类别组和固定随机组，报告明显误合并率
- 下游处理：D2 修订后必须重新生成 D2 产物；D3 必须基于新组重新切分并复验零泄漏；D4 重新完成两轮复现；D5 重新冻结。当前 D3-D5 产物不得交给算法工程师，A0 仍未满足前置条件
- 验收清单更新：D2-D5 均标记为退回，未勾选相应验收项
- 建议下一阶段：让 AI 数据工程师执行 `D2-R1`，只完成 D2 返工并停止，交给总负责人重新验收
- 用户决定：已执行 D2-R1，并提交总负责人验收

## 2026-08-10 D2-R1 验收

- 结论：通过；D2 状态由退回改为通过，D3-D5 仍保持退回
- 验收输入：commit `80ef142f8700bcfca10ad9159197730f3df6c1c9`、D2-R1 脚本与测试、数据工程师返工日志及 `artifacts/data/v1/d2-r1/`
- 独立自动复验：`build_duplicates_d2_r1.py --verify-only` 退出 0；项目测试 `24 passed in 153.57s`；D2-R1 校验清单 25 项全部为 `OK`；`git diff --check` 通过
- 输入继承检查：逐行比较 D2-R0 与 D2-R1 manifest，共 221,396 行，SHA-256 差异 0、dHash 差异 0；221,377 张可解码图片均具有 dHash 和 pHash，19 张坏图均不含感知哈希
- 分组正确性检查：验证器从指纹重新构建全部 group ID，确认同 SHA 不拆组，并计算每组所有可解码成员的两两距离；全局最大 dHash 直径为 5、最大 pHash 直径为 8，均满足冻结阈值
- 错误组回归：旧 `dg-001396` 的 182 张拆为 114 组、最大替代组 3；旧 `dg-004019` 的 132 张拆为 101 组、最大替代组 3；未再形成大型链式误合并组
- 人工视觉复验：独立查看 12 张审计页，覆盖最大组 20、跨类别组 20、固定随机组 20；60 组未见主体、场景或构图明显无关的图片被合并。跨类别组主要是同图或同一构图被赋予不同 class ID，属于后续标签质量问题，不构成本阶段分组误合并
- 关键统计：156,871 个组，单例 108,971、非单例 47,900；112,425 个文件位于非单例组；完全重复 SHA 组 26,705，近重复组 32,731，跨类别组 10,714，最大组 21；pHash 拒绝 4,724 个候选指纹对，complete-link 拒绝 1,667 次合并
- 关键校验和：R1 manifest 为 `177e785b0cffd53ad0de7eb5aa3f2a2899127ca77558a774297929c2e2b80828`；duplicate groups 为 `58c50dcbe3bf40a21c58cd193c7bff08e2eefee7777ecdce95dc0ff7db910c0a`；配置为 `17f6cde9895bb17ee7fa3148f31f66a43aceb98c3a5893696903a9891ec47cbc`
- 范围检查：工程师停止在 D2-R1，未执行 D3-R1 或 A0；未修改原图和 taxonomy。工作区中的旧 D2-R0、D3-D5 代码及 `data/__init__.py` 改动不属于本次通过范围，不纳入本次提交
- 剩余风险：complete-link 规则偏保守，可能拆开变化较大的真实近重复；60 组抽查不能证明全量视觉正确。该风险低于 R0 错误合并导致 split 污染的风险，允许进入 D3-R1，并要求继续以整个 R1 group 为单位切分
- 验收清单更新：D2-R1 标记为通过并勾选 D2 四项；D3-D5 保持退回，A0 仍等待 D5
- 建议下一阶段：调用 AI 数据工程师执行 `D3-R1`，只基于 `artifacts/data/v1/d2-r1/` 重建固定 split，复验路径互斥、203 类覆盖和 duplicate group 跨 split 为 0 后停止验收

## 2026-08-10 D3-R1 验收

- 结论：退回；split 内容质量门通过，但交付无法从 Git 记录独立复现
- 验收输入：commit `c4b56211319a42cf65aba11cc1163f82c9f13841`、D3-R1 脚本与测试、数据工程师日志及 `artifacts/data/v1/d3-r1/`
- 独立自动复验：`build_splits_d3_r1.py --verify-only` 退出 0；项目测试 `27 passed in 160.16s`；D3-R1 校验清单 12 项全部为 `OK`；`git diff --check` 通过
- 已通过的数据检查：三个 split 合计且仅包含 221,377 条可用路径，路径重叠 0；19 张坏图完整排除；156,852 个可用 duplicate group 跨 split 数为 0；train/val/test 均覆盖 203 类；数量为 177,021/22,178/22,178
- 阻断问题：`scripts/build_splits_d3_r1.py` 在模块加载时直接导入 `scripts/build_splits_d3.py`，并复用其常量、数据结构、分组算法及读写验证函数；但该旧 D3-R0 脚本未被 Git 跟踪，也未列入 D3-R1 的 `checksums.sha256`。当前测试和 verifier 只因工作区残留了被退回的旧文件而通过，从本次可提交文件恢复时会在导入阶段失败
- 判断依据：阶段交付必须能从受控源码和记录依赖复现。D3-R1 的校验链只校验自身脚本，不能证明实际执行的分组算法源码未变化；把未验收的 D3-R0 工作树文件作为隐式依赖不满足通用检查中的“无隐式环境”和“关键版本可追溯”要求
- 返工要求：执行 `D3-R2`，将所需 split 实现收进 D3-R2 脚本或已提交的正式共享模块，不得运行时导入未跟踪的 D3-R0 脚本；将全部运行时源码纳入 checksum；增加缺少旧 D3-R0 文件时仍可导入和执行的回归测试；重新生成产物并保持现有数据质量门
- 范围处理：D3-R1 脚本、测试和数据工程师日志均不纳入本次验收提交；旧 D2-R0、D3-D5 文件继续保留在工作区但仍未验收。D2-R1 保持通过，D4-D5 保持退回，A0 不得开始
- 验收清单更新：D3 继续标记为退回，并将摘要更新为 D3-R1 的实际阻断原因
- 建议下一阶段：让 AI 数据工程师只执行 `D3-R2` 依赖收敛返工，完成后停止并重新交验

## 2026-08-10 D3-R2 验收

- 结论：通过；D3 状态由退回改为通过，D4-D5 仍保持退回
- 验收输入：commit `23db3b3eef49f83f73d963204dbb29698ab56132`、独立 D3-R2 脚本与测试、数据工程师日志及 `artifacts/data/v1/d3-r2/`
- 返工闭环：D3-R2 为单文件 Python 标准库实现，不再导入任何仓库内模块；配置声明并校验唯一运行时源码，checksum 同时覆盖 R2 脚本和测试，且明确排除被退回的 D3-R0/R1 脚本
- 独立自动复验：`build_splits_d3_r2.py --verify-only` 退出 0；项目测试 `31 passed in 181.76s`；D3-R2 校验清单 13 项全部为 `OK`；`git diff --check` 通过
- 独立重建复验：在新系统临时目录从已通过的 D2-R1 输入完整执行一次 D3-R2，生成后将 train、val、test、group assignments 和坏图清单与正式产物逐字节比较，五项全部一致
- 数据完整性：三个 split 合计且仅包含 221,377 条可用相对路径，路径重叠 0；19 张坏图完整排除；156,852 个可用 duplicate group 全部只属于一个 split，跨 split group 数为 0；CSV schema、class ID、SHA-256 和 group ID 均与 D2-R1 逐行匹配
- 关键统计：train 177,021（79.9636%）、val 22,178（10.0182%）、test 22,178（10.0182%）；group 数为 125,404/15,725/15,723；三个 split 均覆盖 203 类；class 162 仅 5 张/5 组，按 3/1/1 覆盖
- 关键校验和：train 为 `af457fcd9c49af93b9929585175aa68f973113d43895e5a035db61bbe7f7d778`；val 为 `a5db45590dd3dd97e46564046fc32c0223dab144e826854ea2a0e5aa3aec0833`；test 为 `23897e0a1a1b2209d1390845c6261ee48a1f94d935fc0173f3a71d18facc1dc8`；group assignments 为 `db68c397b52bf4c789f4c2b679d575cf4954c799d6f1ba628b6db936b074aa2e`
- 范围检查：工程师停止在 D3-R2，未执行 D4-R1 或 A0；未修改 D2-R1、原图或 taxonomy。旧 D2-R0、D3-R0/R1、D4-D5 工作区文件不属于本次通过范围，不纳入提交
- 剩余风险：零泄漏只覆盖 D2-R1 已识别的重复组；变化较大的未召回近重复仍可能跨 split。该限制已记录，不阻断 D4-R1
- 验收清单更新：D3-R2 标记为通过并勾选 D3 四项；D4-D5 保持退回，A0 仍等待 D5
- 建议下一阶段：调用 AI 数据工程师执行 `D4-R1`，基于 D2-R1 与 D3-R2 完成两轮独立复现和三个 split 的 Dataset 加载验证，完成后停止交验

## 2026-08-10 D4-R1 阻塞验收

- 结论：阻塞成立；撤销 D2-R1 的通过状态并退回 D2，D3-R2 的 split 内容与独立实现保留通过，D4-R1 标记为阻塞
- 验收输入：commit `7f4205cfe94442b44734f195790c5f59c6d34349`、数据工程师 D4-R1 阻塞报告、D2-R1 受控源码和校验清单
- 独立复核：`scripts/build_duplicates_d2_r1.py` 第 29-35 行在导入时加载 `scripts/build_duplicates_d2.py`；`git ls-files --error-unmatch` 确认该依赖未被跟踪；`git cat-file` 确认 commit `c4b5621` 不含该文件；D2-R1 checksum 只包含 R1 脚本，不包含实际复用的 R0 源码
- 阻断影响：干净 Git 快照无法导入 D2-R1，更无法从 D1 完成两轮 D2 重建。当前工作区恰好残留未受控脚本，不构成复现证据；两轮复现为 0/2，三个 split 的 Dataset 正式加载验证为 0/3，D4-R1 未生成任何代码或产物
- 责任修正：此前 D2-R1 验收验证了现成产物、分组边界、旧错误组回归和视觉样本，但遗漏了运行时源码闭包检查。D4 发现该问题后，不能继续维持 D2 的正式通过状态，也不能通过缩减 D4 范围绕过
- 阶段判断：D2-R1 的现有分组内容仍保留为 R2 的回归基准；D3-R2 本身已是独立实现，且其 split 内容质量门已通过，因此暂不撤销 D3。但 D2-R2 必须重建出与 D2-R1 字节一致的核心 manifest 和 duplicate groups，否则 D3 必须随上游变化重新生成和复验
- 返工授权：执行 `D2-R2`，将 SHA-256、dHash、pHash、EXIF、候选召回、complete-link 分组、verifier 和审计所需实现收敛到受控源码；不得运行时导入未跟踪的 D2-R0；将全部运行时源码及测试纳入 checksum；增加旧脚本不存在时仍可导入和执行的隔离回归测试
- D2-R2 验收门：保持 221,396/221,377/19 数量，SHA-256/dHash 与已校验输入一致，组内 dHash/pHash 最大直径不超过 5/8，同 SHA 不拆组，两个旧错误组回归通过；核心 manifest 和 duplicate groups 应与 D2-R1 字节一致，否则必须解释并触发 D3 重建
- 后续顺序：D2-R2 验收通过后解除 D4 阻塞，重新执行完整 D4-R1；不得先执行 D5 或 A0
- 验收清单更新：D2 改为退回并取消运行时源码追溯项，D3 保留通过且注明条件，D4-R1 改为阻塞，D5 保持退回

## 2026-08-10 D2-R2 验收

- 结论：通过；D2 恢复通过，D4-R1 复现阻塞解除
- 验收输入：commit `0524a1f088513318dc1b82e62f89fed6f6d7448d`、D2-R2 独立脚本与测试、数据工程师日志及 `artifacts/data/v1/d2-r2/`
- 依赖闭环：R2 为单文件运行时实现，不导入仓库内模块；SHA-256、dHash、pHash、EXIF、旧 lineage、候选召回、complete-link、审计和 verifier 均收敛到受控脚本；配置与 checksum 覆盖唯一运行时源码和专用测试，不包含旧 D2-R0 脚本或产物
- 独立自动复验：`build_duplicates_d2_r2.py --verify-only` 退出 0；项目测试 `41 passed in 338.39s`；D2-R2 校验清单 25 项全部为 `OK`；独立隔离导入测试通过；`git diff --check` 通过
- 数据质量：共 221,396 个 SHA-256，221,377 个 dHash/pHash，19 张坏图不含感知哈希；156,871 个 group，非单例 47,900，最大组 21；组内最大 dHash/pHash 直径 5/8；同 SHA 不拆组；两个旧错误组回归 2/2 通过
- 兼容性复验：独立逐项 `cmp` R1/R2 的 manifest、duplicate groups、回归报告、审计索引和 12 张审计页，16/16 字节一致；manifest SHA-256 为 `177e785b0cffd53ad0de7eb5aa3f2a2899127ca77558a774297929c2e2b80828`，duplicate groups 为 `58c50dcbe3bf40a21c58cd193c7bff08e2eefee7777ecdce95dc0ff7db910c0a`
- 范围检查：工程师只执行 D2-R2，未执行 D3 返工、D4-R1、D5 或 A0；未修改原图、taxonomy 和既有上游产物。旧 D2-R0、D3-R0/R1、D4-D5 与 Dataset 工作区文件不属于本次通过范围
- 测试偏差判断：正式生成前一次合成测试失败源于测试夹具的 dHash 距离预期错误；修正夹具后专用测试和全量测试均通过，未影响正式产物，不构成阻断
- 剩余风险：complete-link 仍可能漏召回变化较大的真实近重复；10,714 个跨类别组仍可能包含标签冲突。本阶段未改标签，风险继续进入数据交接说明
- D3 来源决策：D2-R2 与 D2-R1 的 manifest 和 duplicate groups 字节一致，因此 D3-R2 的既有 split 与算法验收继续有效，不要求 D3-R3。D4-R1 必须使用 D2-R2 作为正式 D2 实现，并在报告中注明 D3-R2 配置中的 D2-R1 是字节兼容引用；D5 最终 release 必须以 D2-R2 为正式来源
- 验收清单更新：D2-R2 标记为通过并恢复 D2 源码追溯项；D3 保持通过；D4 从阻塞改为未开始；D5 保持退回，A0 仍等待 D5
- 建议下一阶段：重新调用 AI 数据工程师执行完整 `D4-R1`，使用 D2-R2 和 D3-R2 完成两轮独立重建及三个 split 的 Dataset 加载验证，完成后停止交验

## 2026-08-10 D4-R1 验收

- 结论：通过；D4 由未开始改为通过，D5-R1 可以开始，A0 仍等待 D5
- 验收输入：commit `b43e46e67f162a3da5c0fcdffdb0e6f989bd6cac`、D4-R1 复现脚本、Dataset 与测试、数据工程师日志及 `artifacts/data/v1/d4-r1/`
- 独立自动复验：`verify_data_d4_r1.py --verify-only` 退出 0；项目测试 `47 passed in 498.78s`；D4-R1 校验清单 101 项全部为 `OK`；`git diff --check` 通过
- 两轮复现证据：每轮均从 D1 和原图独立执行 D2-R2，再以该轮 manifest 运行 D3-R2；正式产物、run 1、run 2 的 D2 核心 19/19 和 D3 核心 5/5 三方 SHA 全部一致；独立 `cmp` 复核 D2 为 38/38、D3 为 10/10
- Dataset 验证：初始化时全量检查 train 177,021、val 22,178、test 22,178 条记录的 schema、相对路径、class ID、taxonomy 目录映射和文件存在性；每个 split 解码首/中/末 3 张，共 9 张，均得到有限 `torch.float32 [3,224,224]` 张量且 target 与记录一致
- 兼容视图判断：每轮 D3 输入 manifest 与对应 D2-R2 manifest 共享 inode，SHA-256 相同，运行后仍由 checksum 和三方比较复核。工程师日志和交付摘要称其为“只读 hardlink”，但实际权限为 `0600`，并非文件系统只读；准确表述应为“D3 只读使用的 hardlink 兼容视图”。该措辞偏差不影响已验证的内容完整性和复现结果
- 关键产物：D4 配置 SHA-256 为 `46c47130e0ce2a41245b832a64de1153eed45f6b3fd86d3acc92705904f0103d`；复现摘要为 `5a2b7a52e0bf09b7ad5083eef95a3ef7e463eb5d3aaa9060655bcaebd508fcbe`；加载冒烟为 `c352bf3306a8d9eed5b94f8756946399f287eb574978e32145c1995407dcb343`
- 范围检查：工程师停止在 D4-R1，未执行 D5-R1 或 A0；未修改原图、taxonomy 和 D0-D3 产物。旧 D2-R0、D3-R0/R1、D4-R0、D5 工作区文件不属于本次通过范围
- 剩余风险：Dataset 只实际解码固定 9 张，全量解码结论继承 D1；加载时不逐张重算 SHA-256，内容完整性由两轮 D2 重算和 D4 checksum 保证。D4 约 504 MiB 复现产物被 Git 忽略，属于可再生产物
- 验收清单更新：D4-R1 标记为通过，并勾选两轮复现和三个 split Dataset 加载两项；D5 保持退回，A0 仍未开始
- 建议下一阶段：调用 AI 数据工程师执行 `D5-R1`，以 D2-R2、D3-R2、D4-R1 为正式链路冻结 data-v1，生成最终 release、taxonomy 快照、交接文档和校验清单，完成后停止交验

## 2026-08-10 D5-R1 验收

- 结论：通过；D0-D5 数据工程链全部完成，data-v1 正式冻结，A0 数据前置条件解除但尚未执行
- 验收输入：commit `e8b2d639d5c5540f48c760248a9fb9b658468d18`、D5-R1 冻结脚本与测试、数据工程师日志及 `artifacts/data/v1/d5-r1/`
- 独立自动复验：`freeze_data_v1_d5_r1.py --verify-only` 退出 0；项目测试 `50 passed in 493.35s`；D5-R1 校验清单 39 项全部为 `OK`；`git diff --check` 通过
- Release 完整性：正式链固定为 `D0 -> D1 -> D2-R2 -> D3-R2 -> D4-R1 -> D5-R1`；独立重算 34 项关键索引的路径、文件大小和 SHA-256，差异 0；固定 manifest、duplicate groups、train/val/test 和 taxonomy snapshot 均指向已通过版本
- taxonomy 与契约：`taxonomy-v1.json` 与 `metadata/class-taxonomy.json` 逐字节一致，SHA-256 为 `5cfa1a261b1a9fbb80adf24f299bca0883a42dd523914a70234f31dbf748bd31`；算法工程师入口固定使用 D3-R2 CSV 和 D5 taxonomy 快照，不得重扫目录、推断标签、重分组或重切分
- 交接文档：明确这是 203 类图像分类而非目标检测；记录 221,396 个原始文件、221,377 张可用图片、19 张坏图、47,900 个非单例组、10,714 个跨类别组、177,021/22,178/22,178 split、19 个长尾类别和九项已知限制
- 关键校验和：release 为 `a4a9e865429c8d60d67321971342395feedc0378cbc8b92b74d8832cc7eade8f`；交接文档为 `5ea4037cf7d78cb58af53ee678c0d74a2da6d59bbd61e6bf6e8214250e466d05`；配置为 `135d6bed198c4cc04a42ed90d39b8ff9f9dbacca65ae844759e3e7f2ff2ed2bf`
- 冻结状态解释：产物中的 `frozen_pending_project_lead_acceptance` 是生成时不可变状态；本验收记录、验收清单和对应 Git 提交共同构成项目负责人正式接受证据，不回写已冻结产物
- 范围检查：工程师只执行 D5-R1，未执行 A0、训练或应用开发；未修改原图、taxonomy 和 D0-D4 产物。旧 D2-R0、D3-R0/R1、D4-R0、D5-R0 工作区文件不属于本次通过范围
- 剩余风险：数据许可尚不明确；complete-link 可能漏召回变化较大的近重复；跨类别组可能存在标签冲突；长尾类别指标方差较大；D4 只实际解码 9 张。这些限制均已进入正式交接文档
- 验收清单更新：D5-R1 标记为通过并勾选 D5 两项；A0 保持未开始，但前置条件已满足
- 建议下一阶段：由用户调用算法工程师执行 `A0` 数据准入，只读取 D5-R1 交接文档、D3-R2 固定 split、D5 taxonomy 快照和受控 Dataset；完成后停止交验

## 2026-08-10 单版本数据链整理验收

- 结论：通过；核心数据结果未变，D0-D5 由多修订工作区收敛为每阶段一个正式目录，A0 仍未执行
- 问题闭环：清理退回产物后发现 D3-R2 的全量 verifier 仍通过 D2-R1 checksum 间接读取 D2-R0；此前 `50 passed` 依赖工作区旧文件，不能作为干净目录复现证据
- 依赖修复：D3-R2 改为直接读取并校验 D2-R2；D4-R1 删除 D2-R1 hardlink 兼容视图，两套 D3 均直接读取对应的独立 D2-R2 重建产物；D5-R1 同步冻结新契约
- 清理范围：删除 D2-R0、D2-R1、D3-R0/R1、D4-R0、D5-R0 代码、测试和产物，删除 Python/pytest 缓存；保留 `d0`、`d1`、`d2-r2`、`d3-r2`、`d4-r1`、`d5-r1` 六个正式目录
- 数据不变证明：train/val/test SHA-256 仍为 `af457fcd...d778`、`a5db4559...0833`、`23897e0a...658`；样本数仍为 177,021/22,178/22,178；203 类覆盖、路径重叠和 duplicate group 泄漏检查均通过
- 复现与加载：D4 两套独立 D2 核心与正式 D2 三方一致，直接生成的 D3 五个核心文件三方一致；三个 split 均可由 Dataset 加载，固定 9 张张量冒烟通过
- 最终验证：D2 verifier、D3 重建、D4 重建、D5 冻结均退出 0；单版本全量测试 `31 passed in 206.67s`；D2/D3/D4/D5 checksum 分别为 25/13/95/39 项
- 版本边界：测试数从 50 减少到 31 是因为移除了退回版本测试；当前测试集只覆盖唯一正式链。历史日志保留，用于说明返工过程，不代表旧版文件仍存在
- 下一阶段：由用户调用算法工程师执行 A0，只读取 D3-R2 固定 split、D5-R1 taxonomy 快照和受控 Dataset

## 2026-08-10 A0 data-v1 准入验收

- 结论：退回；数据准入结果正确，但冻结产物无法跨验收提交稳定复验，A1 不得开始
- 验收输入：基线 commit `f4ee10fb6bddabdd3b708a0679b85ec0d518cf57`、`training/admission.py`、专用测试、算法工程师日志和 `artifacts/training/a0-data-v1-f4ee10f/`
- 通过项：A0 `--verify-only` 退出 0；专用测试 `4 passed in 11.31s`；全量测试 `35 passed in 215.46s`；D5 的 39 项 checksum、203 类 taxonomy、177,021/22,178/22,178 split、221,377 条唯一路径、零路径重叠和零 duplicate group 泄漏均通过
- 阻断问题：`run_admission()` 每次用当前 `git rev-parse HEAD` 生成 `a0_git_commit`，`verify_artifacts()` 又要求新报告与已保存报告完全相等。A0 产物保存的是 `f4ee10f`；按项目规则提交验收后 HEAD 必然变化，随后 `--verify-only` 会稳定失败
- 独立复现：保持其他输入不变，仅模拟 HEAD 变为另一 40 位 commit，`verify_artifacts()` 抛出 `AdmissionError: stored A0 report differs from a fresh admission check`
- 返工要求：把“生成 A0 产物时的受控基线 commit”作为冻结输入从 config/report 读取或显式传入；验证时校验源码、测试和输入数据 SHA-256，不得用当前 HEAD 覆盖冻结字段；增加“HEAD 变化后 verify-only 仍通过”的自动化回归测试
- 范围检查：算法工程师只执行 A0，未执行 A1、训练或应用开发；数据、taxonomy 和固定 split 未修改。总负责人本次仅记录验收结论，未修改算法交付
- 下一阶段：算法工程师执行 A0-R1 返工，完成后停止交验；A0-R1 通过前不得开始 A1

## 2026-08-10 A1 训练链路验收

- 结论：通过；A1 完成，允许进入 A2，未读取 val/test 指标。
- 范围：新增 ResNet-50、共享 transform、checkpoint、A1 冒烟 CLI 与测试；只更新训练配置和算法日志，未修改冻结数据、taxonomy 或 split。
- 复验：8 项定向测试通过；独立加载 A1 checkpoint 并在 RTX 4070 Laptop 上得到有限 `[2,203]` logits；6 项 artifact checksum 全部通过；工程师全量测试为 `39 passed in 209.61s`。
- 指标：固定 train 子集 32 张、8 类，3 epoch 后 accuracy 从 6.25% 升至 96.875%，loss 从 5.2343 降至 0.1697，峰值显存约 1.02 GiB。
- 修正：checkpoint 加载改为 `weights_only=True`，现有 A1 checkpoint 兼容，降低后续加载不可信 pickle 的风险。
- 已知限制：A1 artifact 的 commit 字段记录开发前基线 `4143fc8`，不是本次实现提交；该产物仅作可再生链路冒烟，不作为 A3 冻结模型包。
- 下一阶段：算法工程师执行 A2 完整训练；应用工程师可并行执行 P1。

## 2026-08-10 P1 应用骨架验收

- 结论：通过；P1 完成，P2 继续等待 A3 冻结模型包。
- 范围：Predictor、模型包校验、图片输入、假 logits、Gradio 页面、应用配置、测试和契约文档；未接入真实权重，未纳入并行 A2 改动。
- 复验：P1 定向测试 `15 passed`，P1 与相关契约测试共 24 项通过，ruff 与 `git diff --check` 通过；真实数据图片经浏览器上传并返回三级结果和 Top-5。
- 浏览器：桌面端 1846 px 正常；首次 390 px 检查发现 1180 px 横向溢出，修复后 `scrollWidth == clientWidth == 390`，移动端上传与结果展示通过。
- 修正：Gradio 根容器改为响应式宽度；上传组件改为 `filepath`，确保 20 MiB 文件限制在解码前真实执行，并增加组件契约测试。
- 限制：当前是固定假模型，只验证应用链路；Gradio 5.50 的 6.0 弃用警告不影响当前 `<6` 依赖范围。
- 下一阶段：等待 A2、A3；A3 通过后执行 P2 真实模型集成。

## 2026-08-11 A2 完整训练验收

- 结论：通过；A2 完成，允许进入 A3，未执行或读取 test 指标。
- 范围：ResNet-50 普通 CE 与 clipped inverse-frequency weighted CE 两组完整训练、验证指标、断点恢复、对照脚本和测试；数据、taxonomy 与 split 未修改。
- 复验：A2/A1 定向测试 `12 passed`，ruff、`git diff --check`、两组各 11 项 checksum 和 comparison checksum 通过；两组各 25 epoch、history 与最佳 checkpoint 指标一致。
- CE：最佳 epoch 23，val Top-1 88.2135%、Top-5 96.5732%、Macro-F1 71.8358%、Balanced Accuracy 71.0727%。
- weighted CE：最佳 epoch 21，val Top-1 88.3398%、Top-5 95.7886%、Macro-F1 72.5674%、Balanced Accuracy 72.4703%；按既定 Macro-F1 选择为 A3 候选。
- 修正与清理：对照门禁改为除 loss strategy/权重公式外全部 actual 配置一致；删除错误时长的旧 comparison 和临时 systemd 日志，只保留 R2 正式对照。
- 限制：两组开发基线 commit 不同，但差异提交仅为 P1，训练配置与实现一致；正式耗时采用 history 汇总的 26118.87/26604.22 秒。单次种子下 +0.7316 pp 不代表统计显著。
- 下一阶段：A3 冻结 weighted CE 的 best checkpoint 后执行一次 test，生成评估报告和模型包；不得再基于 test 调参。

## 2026-08-11 A3 最终评估与模型包验收

- 结论：通过；算法工程 A1-A3 全部完成，允许应用工程进入 P2；验收过程未再次执行正式 test 推理。
- 冻结与隔离：输入 commit `1d34280ea30fe54a15031240e502274e07a555bb`，按 val Macro-F1 选择 weighted CE epoch 21；checkpoint、224 输入预处理和 `0.55` 阈值在读取 test 前冻结，test 只消费一次且未参与调参。
- 指标：test 共 22,178 张，Top-1 `88.5517%`、Top-5 `95.7796%`、Macro-F1 `71.2177%`、Balanced Accuracy `71.2654%`；独立从 `203 x 203` 混淆矩阵重算 Top-1、Macro-F1 和 Balanced Accuracy，与发布指标一致。
- 模型包：`artifacts/releases/dlcpd25-resnet50-weighted-v1/` 的 13 项 checksum 全部通过；包内 `best.pt` 与 A2 源 checkpoint 逐字节一致，SHA-256 为 `68fc44f1b4acfe321e5590b5f27dead65b735a777798c141c6528c510e11eabd`；训练评估目录 9 项 checksum 全部通过。
- 推理复验：bundle loader 通过；三张固定 val 样例的源模型/包内模型 logits 逐位一致且重复推理稳定；taxonomy 或权重被篡改时由测试确认拒绝加载。
- 验收修正：原实现的一次性门禁只绑定可改名的 evaluation/model 目录；新增受 Git 管理的 `metadata/a3-test-evaluation.json` 仓库级消费凭据和原子占用逻辑，换名也无法二次读取同一 test，并新增回归测试。正式指标和冻结模型包均未改写。
- 测试：A3 定向测试 `4 passed`；项目全量测试 `66 passed, 4 warnings in 274.61s`；ruff、`git diff --check` 均通过。4 条 warning 均为既有 Gradio 6.0 弃用提示。
- 风险：阈值 `0.55` 下低置信度率为 `72.7342%`，P2 必须明确显示“不确定”，不得用 test 重新选择阈值；长尾和相似类别混淆仍存在，本系统是图像分类而非目标检测。
- 下一阶段：应用工程师执行 P2，校验并接入 `dlcpd25-resnet50-weighted-v1`，完成 CPU/CUDA、固定样例一致性、低置信度提示、异常输入和发布说明后停止交验。

## 2026-08-11 P2 真实模型集成验收

- 结论：通过；P1 假模型已由 A3 冻结的真实 ResNet-50 替换，允许进入 F0。
- 模型契约：A3 模型包 13 项 checksum 全部通过；应用从 bundle 读取 `203` 类 taxonomy、224 输入预处理、阈值 `0.55`、模型和数据版本，依赖或 checkpoint 契约不一致时拒绝启动。
- 推理一致性：独立 CPU smoke 的三张固定 val 样例 Top-5 与 A3 参考完全一致，其中 1 张低置信度；损坏图片稳定返回 `decode_failed`。未访问或重新评估正式 test。
- 设备路径：真实 `auto` 路径在本机选择 CUDA；合成测试确认 CUDA 预热失败后回退 CPU，显式 CUDA 不可用时返回 `device_unavailable`；CPU 路径可独立运行。
- 页面与 API：`http://127.0.0.1:7860` 返回 HTTP 200；真实 `/classify` API 的正常样例返回 class 0、`90.62%`，低置信度样例返回 class 131、`20.27%` 及“不确定”提示，版本栏显示真实模型、data-v1 和 CUDA。
- 测试：应用定向测试 `22 passed`；全量测试 `73 passed, 4 warnings in 264.65s`；ruff、`git diff --check`、P2 evidence checksum 均通过。4 条 warning 是既有 Gradio 6.0 弃用提示。
- 浏览器限制：按 `browser-harness` 技能检查时 Chrome 未开放 CDP 授权，因此未绕过授权生成新截图；沿用 P1 已通过的响应式组件测试，并以真实 HTTP、Gradio API 和组件构建测试完成 P2 验收。
- 下一阶段：总负责人执行 F0，冻结 P2 Git 基线，核对数据/模型/应用版本链、根 README、演示路径和 Git 追踪范围，完成后提交最终验收。

## 2026-08-11 F0 最终验收

- 结论：通过；D0-F0 全部完成，项目达到可安装、可复验、可演示和可答辩状态。
- 版本链：数据为 `data-v1-d5-r1`，taxonomy SHA-256 `5cfa1a26...8bd31`；模型为 `dlcpd25-resnet50-weighted-v1`，权重 SHA-256 `68fc44f1...eabd`，A3 验收提交 `1099300`；应用 P2 基线提交 `e9f1463`，配置 SHA-256 `06cb8c99...287c`。
- 复现边界：根 README 已提供安装、preflight、CE/weighted CE 训练、冻结评估 checksum、全量测试、真实应用和 fixed-val smoke 命令；A3 test 消费凭据 SHA-256 为 `8f89ec47...12d9`，不得再次执行正式 test。
- Git 范围：F0 暂存后 `git ls-files` 共 89 个工程与文档文件；仅追踪 `data/README.md`，没有原图、模型权重、checkpoint、Python 缓存或 artifacts。大文件均由 `.gitignore` 排除。
- 演示：真实 Gradio API 正常样例 class 0、`90.62%`；低置信度样例 class 131、`20.27%` 并显示“不确定”；损坏图片返回稳定中文错误。页面运行于 `http://127.0.0.1:7860`。
- 回归：P2/F0 共同全量回归 `73 passed, 4 warnings in 264.65s`；A3/P2 checksum、CPU smoke、CUDA API、P2 范围 ruff 和 `git diff --check` 均通过。warning 仅为 Gradio 6.0 未来弃用提示。
- 答辩口径：系统对整张图片进行 203 类分类，并由 taxonomy 映射宿主和四大类；不输出边界框，不得称为目标检测或专业农业诊断。
- 浏览器限制：Chrome 未授权 CDP，未生成新的浏览器截图；P1 响应式验收、P2 真实 HTTP/API 和组件测试共同覆盖现有页面，后续展示可直接打开正在运行的 7860 服务。
