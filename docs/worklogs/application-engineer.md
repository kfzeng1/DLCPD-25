# AI 应用工程师工作日志

当前状态：J5 已通过总负责人验收；等待 F1 最终验收。

## 2026-08-13 J5 单模型联合应用

- 状态：总负责人验收通过；未执行 F1、未推送。
- 输入：J4 唯一模型包 `artifacts/releases/dlcpd25-ip102-joint-v1/`；默认配置不再加载历史分类模型。
- 实现：新增 `JointPredictor` 和无初始化权重的联合架构构造器；严格加载唯一 `joint-best.pt`，一张图只解码/直缩一次并调用一次 `forward_joint()`；分类输出 203 类 Top-5，检测内部标签映射为 DLCPD-25 `class_id 0-202`，224 框反算至原图并绘制。
- 页面：默认 `7860` 显示上传图、带框图、分类三级结果/置信度/Top-5、检测明细、设备与耗时，并明确分类覆盖 203 类、检测只定位 96 类害虫。
- 验证：J5 定向 `7 passed`，J5/历史应用兼容回归 `35 passed`，项目全量 `124 passed`；Ruff 与 `git diff --check` 通过。无框、多框、低分过滤、低置信度、损坏图、非方图坐标、CUDA 失败回退 CPU 和显式 CUDA 不可用均有测试。
- 真实联调：IP102 val `IP000000378.jpg` 在 CUDA 上共享主干 hook 计数 `1`，返回 class 156 和 2 个框；直接 Python 推理约 `68.686 ms`。7860 `/analyze` 返回 11 项、5 行分类和 2 行检测；CPU 返回相同 class 156 与 2 个框，约 `947.309 ms`。
- 边界：未读取或重新评估任一正式 test；病害、健康和缺陷只分类，不伪造框。检测阈值保持 `0.5`，分类阈值保持 `0.55`。
- 浏览器验收：用户在 Edge 开放 9222 CDP 后，`browser-harness` 连接现有本地 7860 标签。桌面 `1650x785` 与移动 `390x844` 页面级 `scrollWidth == clientWidth`；真实 val 图片上传后出现低置信度提示、5 行 Top-5、2 行检测和带框图。标题换行、双图纵向重排、字段 `2+2+1` 重排均正常，无控件或文字重叠；检测表在窄屏内使用自身横向滚动。
- 截图：`artifacts/audit/j5-browser/desktop-top.png`、`desktop-after-upload.png`、`mobile-top.png`、`mobile-bottom.png`。

后续记录必须按 `README.md` 模板追加，不得覆盖历史记录。

## 2026-08-10 P1 假模型推理内核与页面

- 状态：待验收
- 指挥者指令：项目已经重构，开始 P1。
- 前置版本：Git `cc7e7762ebce664d174160bfb82c291491567e6a`；验收清单标记 D0-D5 与 A1 通过；taxonomy 使用 `metadata/class-taxonomy.json`；P2 仍等待 A3。
- 允许范围：`inference/`、`web/`、`project/configs/app.yaml`、应用测试、应用文档和本日志。
- 实际修改：实现应用配置解析、模型包 manifest/checksum/preprocessing/taxonomy 校验、受限图片解码与 EXIF/RGB 处理、固定 203 logits Predictor、稳定 Top-5 与三级映射、低置信度标志、Gradio 页面和根目录启动入口；更新 `project/README.md` 并新增 `docs/application-contract.md`。
- 生成产物：无冻结 artifacts；P1 服务运行于 `http://127.0.0.1:7860`。关键 SHA-256：`app.yaml` `acdaed292cb0b98dd2a47b194a35d884cab3f3646dfa6dbb63b24208c253566c`；`bundle.py` `1b7d5a4a2162219224082c167b18738305161c9fcecc66fca079a0ab07db8e21`；`predictor.py` `c2d8dda562ba3c82fb41ca458843bfd5719bde61d19cf0b16078efdf37c16d46`；`web/app.py` `08e54218ced719ba08fb8eb79759d94f929edfc307b3bd76db24d5c0c115e6a7`；契约文档 `5857e3e89c30d0ccc94cbdd52083023ff1abd5ddc98366daf5f4485dfd795810`。
- 执行命令：`pip install -e 'project[app]'` 首次因下载速度过慢人工取消（退出 1），改用清华 PyPI 镜像后安装成功（退出 0，Gradio 5.50.0）；P1 定向 pytest 最终 `17 passed`（退出 0）；P1 范围 `ruff check` 通过（退出 0）；`git diff --check` 通过（退出 0）；HTTP 首页返回 200；Gradio `/classify` 使用数据集图片调用成功。
- 验收证据：灰度、RGBA、EXIF、损坏内容、文件/像素超限、未知扩展名、缺失模型包、checksum 篡改、预处理与 manifest 不一致、非法输出形状、NaN、低置信度、稳定 Top-5 和页面字段均有测试；API 返回宿主“番茄”、属性“植物病害”、细类 `tomato bacterial spot`、置信度 93.65%、5 行 Top-5、模型/数据版本和耗时，并明确标注固定假模型。
- 关键统计：203 类；Top-k=5；默认阈值 0.55；文件上限 20 MiB；像素上限 4000 万；API 冒烟推理耗时约 38 ms（固定假 logits，仅代表应用链路）。
- 偏差、风险和阻塞：`browser-harness --doctor` 因 Chrome 未开放 CDP/daemon 连接而退出 1，按技能安全规则未自行启动独立浏览器，因此没有浏览器截图；以 Gradio 组件测试、HTTP 和真实上传 API 冒烟替代。Gradio 5.50 对未来 6.0 API 给出弃用警告，当前依赖明确限制 `<6`。安装 Gradio 将 Pillow 12.2.0 降至 11.3.0、Pydantic 2.13.4 降至 2.12.3，并报告环境中既有 `pyrender`/PyOpenGL 版本冲突；P1 定向测试未受影响。真实模型加载和 CPU/CUDA 选择未实现，严格留待 P2。
- Git 状态：`main` 相对 `origin/main` ahead 19；P1 修改/新增 `project/README.md`、`project/configs/app.yaml`、`inference/`、`web/`、`docs/application-contract.md`、`project/tests/test_inference_p1.py`、`project/tests/test_web_p1.py` 和本日志。工作区另有算法工程师的 `algorithm-engineer.md`、`training/` 与 `test_training_a2.py` 改动，未修改或回退。
- 下一步建议：总负责人独立复验 P1；P1 通过后仍等待 A3，再由用户明确下令执行 P2。
- 边界声明：未执行下一阶段。

## 2026-08-11 P2 真实模型集成与发布

- 状态：待验收
- 指挥者指令：A3 验收通过，下一阶段 P2；接入真实模型并明确展示低置信度提示，禁止使用 test 调整阈值。
- 前置版本：Git `10993001fbeaaea26052ce4f15f34bb02be99567`；P1、A3 均通过；模型包 `artifacts/releases/dlcpd25-resnet50-weighted-v1/`；模型包 checksum 清单 SHA-256 `b5b970ebe0f4cae436115fd7449e43f4f49ee6f361724e81b7bb7e4c4128af6a`；冻结阈值 `0.55`。
- 允许范围：`inference/`、`web/`、`project/configs/app.yaml`、应用测试、应用文档、应用验证 artifacts 和本日志。
- 实际修改：实现真实 ResNet-50 checkpoint 加载、torch/torchvision 契约核对、bundle taxonomy/预处理/阈值消费、CPU/CUDA 设备路径、`auto` CUDA warmup 失败回退 CPU、启动错误转换和 localhost 代理绕过；应用切换真实 bundle 并强化低置信度提示；新增真实模型集成测试、P2 smoke CLI、三张 fixed-val 演示与启动排错说明。
- 生成产物：`artifacts/releases/application-p2-v1/validation-summary.json`，SHA-256 `a9f69c2c976e8fd4a3b2eac50e7396a9421c8db3d8fa0796a6b96b544e0fdcc8`；产物 checksum 文件 SHA-256 `aeb2fc2716ad0c9f284d8881d83a775bbfaa9464178bfa3f06ba9dbed02270de`。真实应用运行于 `http://127.0.0.1:7860`。
- 执行命令：CPU 最小固定样例推理；P2 定向 pytest；真实 Gradio `/classify` 正常与低置信度 API 冒烟；`python -m dlcpd25_classifier.inference.smoke`；最终全量 `pytest project/tests -q`；P2 范围 `ruff check`；`git diff --check`；HTTP 健康检查。未执行 `training.a3`，未读取正式 test split 复算指标。
- 验收证据：A3 13 项 checksum 全部验证；CPU 三张 fixed-val Top-5 顺序与算法参考完全一致；`auto` 实机选择 CUDA，模拟 CUDA warmup 失败后成功回退 CPU，显式 CUDA 不可用返回稳定错误；API 正常样例显示 class 0、90.62%、真实模型/数据版本和 CUDA，低置信度样例显示 class 131、20.27% 及“低置信度：结果不确定”；损坏图片返回稳定解码错误。
- 关键统计：全量测试 `73 passed`、4 条 Gradio 6 未来弃用警告、耗时 242.91 秒；P2 evidence 3 张 fixed-val、Top-5 一致、1 张低置信度、CUDA 中位推理耗时 22.152 ms；冻结阈值保持 0.55。A3 提供的 test 低置信度率 72.7342% 仅用于展示口径，未据此调阈值。
- 偏差、风险和阻塞：CPU 与 A3 CPU 参考概率最大浮点差约 `1.2e-7`；CUDA 概率最大差约 `7.4e-4`，class ID 与 Top-5 顺序一致。`browser-harness` daemon 可短暂启动但 Chrome 未授权活动 CDP 连接，最小检查后 daemon 退出，因此未生成 P2 浏览器截图；已用响应式 Gradio 组件测试、HTTP 200 和真实上传 API 冒烟替代。Gradio 5.50 对 6.0 API 有弃用警告，当前项目依赖明确为 `<6`。
- Git 状态：开始 P2 时工作区干净；当前 `main` 相对 `origin/main` ahead 22，仅有本阶段应用代码、配置、测试、文档和本日志改动；未修改模型包、训练代码、taxonomy、split、test 凭据或验收清单。
- 下一步建议：总负责人独立复验 P2；通过后由总负责人执行 F0，应用工程师不自行进入下一阶段。
- 边界声明：未执行下一阶段。
