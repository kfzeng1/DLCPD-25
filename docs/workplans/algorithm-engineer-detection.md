# 算法工程师工作单：T1-T3

## 任务目标

在已发布的 203 类 ResNet-50 分类模型上增加 Faster R-CNN + FPN 检测分支。检测使用 IP102 边界框，覆盖映射后的 96 个 DLCPD-25 害虫类别；分类仍覆盖全部 203 类。

统一编号合同：IP102 原标签用于追溯，检测器内部使用 `1-96`（`0` 为背景），所有外部接口使用 DLCPD-25 `class_id 0-202`。IP102 类别 50、51 均映射到 DLCPD-25 `class_id 97`。

## 固定输入

- T0 发布的数据合同：`artifacts/data/ip102-detection-v1/`
- 原始数据：`data/raw/ip102/downloads/Detection/VOC2007/`
- 类别映射：`metadata/ip102-detection-class-map.json`
- 分类模型包：`artifacts/releases/dlcpd25-resnet50-weighted-v1/`
- 检测骨架：`project/src/dlcpd25_classifier/detection/`
- 环境：`/home/zkf/pytorch-env`，RTX 4070 Laptop，约 7.62 GiB 可用显存

T0 未通过时不得开始 T1。官方 `test.txt` 在 T3 获总负责人放行前不得用于统计、抽样、可视化、选模型或调参。

## T1：训练链路与硬件冒烟

实现检测训练 CLI、评估器、配置和 checkpoint 恢复。先从 train 固定抽取 16-32 张含框图片完成小样本过拟合，不读取 val/test 作为训练样本。

初始配置为 AMP、短边 640、长边上限 1024、batch 1；显存有至少 0.8 GiB 余量后才尝试 batch 2。第一版只使用同步变换框的安全增强。

交付：

- `project/configs/detection.yaml`
- 检测训练、评估和 checkpoint 代码及定向测试
- resolved config、固定样本清单、loss、显存和吞吐记录
- `best.pt`、`last.pt`、重载一致性及 checksum

验收：

- 前向、反向、优化和 AMP 无 NaN/Inf；总 loss 明显下降；
- ROI 分类器为背景加 96 个前景类，对外结果转换为 DLCPD-25 ID；
- 分类头无梯度，冻结方案下 ResNet-50 主干哈希不变，检测分支参数确实更新；
- checkpoint 恢复 optimizer、scheduler、scaler、epoch 和随机状态；
- 固定输入重载前后预测一致；全量测试和 `git diff --check` 通过。

若小样本不能稳定过拟合，先检查坐标、标签、transform 和评估实现，不得直接开始完整训练。

## T2：完整训练、验证选型与冻结

只使用 T0 固定 train/val。先冻结 ResNet-50 主干和分类头，训练 FPN/RPN/ROI，建立基线。若验证曲线仍有明确提升空间，可从最佳基线仅解冻 `layer4`，主干学习率不高于检测头的 0.1 倍；stem、layer1-3 和分类头继续冻结。

默认按 val `mAP@0.5:0.95` 选 checkpoint。只有 layer4 方案相对冻结基线提升至少 `0.005`，且固定分类回归集 Top-1 下降不超过 2 个百分点，才采用共享微调版本；否则交付更稳定的冻结主干版本。训练轮数由 T1 实测速率和 val 曲线决定，不使用 test 决定。

交付：

- 不可覆盖的训练 run，含配置、代码 commit、best/last checkpoint 和 checksum；
- 每 epoch loss、学习率、val mAP、AP50、AP75、AR、Precision、Recall；
- 96 类支持数和逐类 AP、长尾分析、错误案例、速度与峰值显存；
- 冻结候选包，含权重、预处理、后处理、映射、taxonomy、验证指标和模型卡。

验收：全程无 NaN/OOM，checkpoint 可恢复；参数更新边界正确；固定分类功能回归通过；模型选择只引用 val；候选包可在 CPU/CUDA 加载且校验和完整。低样本类 AP 为 0 时必须如实报告，不得删类或改为五类。

## T3：一次官方 test 与算法交接

总负责人先验收 T2 并冻结权重、预处理、score threshold、NMS、最大框数和映射哈希，再授权唯一一次官方检测 test。test 结果不得反向影响模型或参数。

交付：

- 一次性评估记录和未变更的冻结包哈希；
- test `mAP@0.5:0.95`、AP50、AP75、AR、Precision、Recall；
- 逐类支持数/AP、错误案例、速度、显存和最终模型卡；
- 应用可加载的检测模型包及统一推理接口说明。

官方 test 缺 IP102 类别 61，因此只覆盖 95/96 个公共检测类。该类必须标记为“test 无支持”，不得伪造 AP，也不得据此重新划分 test。

## 应用交接合同

模型包至少包含：

```text
best.pt
manifest.json
resolved-config.yaml
preprocessing.json
postprocessing.json
ip102-detection-class-map.json
taxonomy.json
metrics-validation.json
metrics-test.json
model-card.md
checksums.sha256
```

每个预测框对外返回 `box_xyxy`、`score`、DLCPD-25 `class_id`、中英文类别名和 `model_version`，不得返回内部 detector label。分类头与检测头的版本关系、共享或双权重模式必须在 manifest 中写明。

## 禁止事项

- 修改原始图片/XML、官方 split、taxonomy、类别映射或已发布分类模型包；
- 用 IP102 检测损失更新 203 类分类头；
- 在 T3 前读取 test 内容或指标；
- 宣称检测覆盖全部 203 类，或伪造病害、健康和缺陷框；
- 自行进入下一阶段、提交或推送。
