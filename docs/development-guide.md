# 分类系统开发指南

## 目录与环境

```text
data/       唯一原图和只读浏览视图
metadata/   官方类别、别名和 taxonomy
research/   论文、来源审计和翻译
project/    训练、评估、推理与 Web 工程
scripts/    数据与文档生成脚本
artifacts/  本地数据、训练和发布产物（Git 忽略）
docs/       计划、职责、验收和日志
```

原图只位于 `data/raw/dlcpd25-203/`。`data/views/by-host/` 是软链接浏览视图，不能用于训练。

复用 `/home/zkf/pytorch-env`：Python 3.12、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128。当前 RTX 4070 Laptop 可用显存约 7.62 GiB。

```bash
cd project
/home/zkf/pytorch-env/bin/pip install -e '.[dev]'
/home/zkf/pytorch-env/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

P1 需要 Gradio 时再安装 `.[app]`。

## 数据契约

算法只读取 `artifacts/data/v1/d3-r2/{train,val,test}.csv` 和 D5 taxonomy 快照。CSV 字段为 `relative_path,class_id,sha256,duplicate_group_id,split`。class ID 固定为 0-202，不从目录排序推断。

A1 开始时运行：

```bash
PYTHONPATH=project/src /home/zkf/pytorch-env/bin/python \
  -m dlcpd25_classifier.training.preflight
```

该命令只快速验证冻结契约，不生成新阶段产物，也不重复 D2-D4。

## 训练规范

默认模型为 ImageNet 预训练 ResNet-50，输入 224 x 224，AMP，batch size 从 16 开始。OOM 时依次降至 8、4，再使用梯度累积。固定 seed 和 split，最佳 checkpoint 按 val Macro-F1 保存。

增强限于随机裁剪、水平翻转、轻度颜色扰动和轻度模糊。至少比较普通 CE 与一种不平衡策略，不同时叠加多个补偿方法。ConvNeXt-Tiny 是可选对照，不影响主线交付。

模型只输出 203 类 logits；宿主和四大类由 taxonomy 映射，避免多个分类头产生矛盾。test 只在 A3 配置冻结后执行一次。

## 产物规范

```text
artifacts/training/<run-id>/       单次训练配置、日志、checkpoint、val 指标
artifacts/releases/<model-version>/
  best.pt
  manifest.json
  resolved-config.yaml
  preprocessing.json
  taxonomy.json
  metrics.json
  model-card.md
  checksums.sha256
```

版本目录不可覆盖。manifest 至少记录模型/数据版本、架构、`num_classes=203`、taxonomy SHA-256、输入尺寸、RGB、resize/crop、mean/std、依赖版本和置信度阈值。

## 推理与页面

统一接口为：

```python
predictor = Predictor.from_bundle(bundle_path, device="auto")
result = predictor.predict(image)
```

结果至少包含模型与数据版本、class ID、宿主、四大类、具体类别、置信度、Top-k、低置信度标志和耗时。应用层调用同一预处理实现，不复制训练逻辑；模型包 hash 或输出维度错误时拒绝加载。

Grad-CAM 只能称为热力图，不能称为检测框。低置信度和域外图片必须提示不确定性。

## 测试规则

普通任务只跑相关测试与 `git diff --check`。A1、A3、P2、F0 运行：

```bash
PYTHONPATH=project/src /home/zkf/pytorch-env/bin/python -m pytest project/tests -q
git diff --check
```

冻结数据的哈希与复现结论直接继承 D5，不在算法和应用阶段重复计算。
