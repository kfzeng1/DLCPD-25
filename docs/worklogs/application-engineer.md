# AI 应用工程师工作日志

当前状态：P1 已通过；P2 等待 A3 模型包。

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
