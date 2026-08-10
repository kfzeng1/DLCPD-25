# 三位 AI 工程师职责

## 总负责人

总负责人维护阶段范围、接口、验收状态和 Git 提交；派工前检查前置条件，交付后重跑必要测试。用户决定是否启动下一阶段或推送远端。

## 数据工程师

数据工程已完成，当前为维护角色。正式链为 `D0 -> D1 -> D2-R2 -> D3-R2 -> D4-R1 -> D5-R1`，冻结入口是 `artifacts/data/v1/d5-r1/`。

允许维护 `scripts/` 数据脚本、`project/src/dlcpd25_classifier/data/`、数据测试和 `artifacts/data/`。禁止修改或删除原图，禁止从 `data/views/` 训练，禁止静默调整 class ID、taxonomy 或 split。除非总负责人确认数据契约变化，否则不重跑耗时的哈希、近重复和复现流程。

## 算法工程师

负责 `models/`、`training/`、算法测试、`project/configs/train.yaml`、`artifacts/training/` 和 `artifacts/releases/`。

| 阶段 | 工作 | 通过条件 |
|---|---|---|
| A1 | 运行快速 preflight；完成 Dataset、transform、ResNet-50、训练 CLI 和 checkpoint 冒烟；固定小样本过拟合 | 输出 `[N,203]`，loss 有限，checkpoint 可重载，CPU/CUDA 冒烟通过，小样本明显过拟合 |
| A2 | 使用 ImageNet 预训练 ResNet-50 完整训练；比较普通 CE 与一种不均衡策略 | 固定 split/seed，只用 val 选型，记录时长、显存、Macro-F1；ConvNeXt-Tiny 仅作可选对照 |
| A3 | 冻结配置和阈值后执行一次 test；生成指标、错误分析和模型包 | 权重、预处理、taxonomy、配置、指标、模型卡和 checksum 完整 |

禁止修改原图、taxonomy 和 split，禁止使用 test 调参，禁止覆盖旧 run。MAE、SimCLR v2、MoCo v3 不属于当前必做范围。

## 应用工程师

负责 `inference/`、`web/`、`project/configs/app.yaml`、应用测试和演示资料。

| 阶段 | 工作 | 通过条件 |
|---|---|---|
| P1 | 定义模型包与 Predictor 契约；用假 logits 实现图片预处理、Top-k、三级映射和 Gradio 页面 | 处理 RGB/灰度/RGBA/EXIF、损坏图和模型缺失；页面显示三级结果、置信度、Top-5、版本和耗时 |
| P2 | 校验并接入 A3 模型包；完成 CPU/CUDA 路径、异常处理、演示和发布说明 | 算法与应用结果一致，全量测试通过，可从仓库根目录启动 |

禁止修改训练数据和权重，禁止伪造检测框，禁止把低置信度结果包装为确定诊断。

## 交接关系

```text
data-v1 -> A1 -> A2 -> A3 -----> P2 -> F0
                    P1 ---------/
```

以 D5 数据契约和 A3 模型包 manifest 为唯一交接依据。发现冲突时停止集成并报告，不在下游临时修改上游产物。
