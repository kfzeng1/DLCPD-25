# 分类系统开发文档

## 1. 目录职责

```text
data/       原图、视图和数据说明，不放代码
metadata/   官方类别、别名和宿主/四大类映射
research/   论文、来源审计、翻译和研究资料
project/    训练、评估、推理应用代码和工程配置
scripts/    数据审计、元数据和研究 PDF 生成工具
artifacts/  被忽略的审计、split、训练和模型产物
docs/       项目计划、职责和本开发文档
```

原始图片只允许存在于 `data/raw/dlcpd25-203/`。`data/views/by-host/` 是软链接浏览视图，不能作为第二份数据复制或修改。

## 2. 环境

本机已有专用环境 `/home/zkf/pytorch-env`，当前为 Python 3.12.3、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128。项目直接复用该环境，不重复创建或安装 PyTorch：

```bash
cd project
/home/zkf/pytorch-env/bin/pip install -e '.[dev,app]'
/home/zkf/pytorch-env/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

已实测 CUDA 可用并识别到 RTX 4070 Laptop GPU，显存为 7.62 GiB。当前环境缺少 pytest、scikit-learn 和 Gradio；安装项目 extras 时补齐，不改变现有 PyTorch/CUDA 版本。训练使用 AMP 和最多 6 个数据加载 worker。算法工程师完成环境后必须导出实际依赖版本，避免只依赖本机隐式状态。

## 3. 数据契约

算法代码只读取 `metadata/class-taxonomy.json` 的 `classes` 数组，不从目录排序推断 ID。每条记录包含 class ID、原始类别、宿主组、宿主、四大类属性、目录名和图片数量。

数据工程师还必须交付固定的 `train.csv`、`val.csv`、`test.csv`，每行包含图片相对路径、class ID、内容哈希、duplicate group 和 split。所有 split 必须保存 SHA-256 和生成配置。

## 4. 训练规范

主线模型为 ImageNet 预训练 ConvNeXt-Tiny，ResNet-50 为基线。建议初始设置：224 输入、batch 16、AMP、AdamW、学习率 3e-4、weight decay 1e-4、冻结骨干 2 个 epoch 后全量微调 15–25 个 epoch，使用 early stopping 和保存最佳 Macro-F1 权重。

增强只使用不会破坏病虫害语义的操作：随机裁剪、水平翻转、轻度颜色扰动、轻度模糊。不得把强裁剪导致目标完全消失的图片当作正常增强样本。

类别不均衡至少比较普通交叉熵与 class-weighted 交叉熵；采样器和权重不能同时过度补偿。最终选择依据 Macro-F1、Balanced Accuracy 和少样本类别表现，而不是只看 Accuracy。

## 5. 层级输出

模型输出 203 类 logits。取最高概率的具体类别后，通过 taxonomy 映射得到宿主和四大类，保证三个输出始终一致：

```text
logits -> class_id 178 -> tomato bacterial spot
                    -> 番茄
                    -> 植物病害
```

第一版不要训练三个互相独立的分类头，避免出现宿主、属性和细类互相矛盾。多任务头可以作为后续实验，但最终细类仍以 203 类头为准。

## 6. 评估与应用

算法工程师交付 JSON/CSV 指标、混淆矩阵和至少 30 张错误案例。应用工程师必须处理：非 RGB 图片、超大图片、损坏图片、低置信度、模型文件缺失和未知扩展名。

应用页面至少显示：原图、宿主作物、四大类、详细类别、置信度、Top-5 列表和模型版本。Grad-CAM 只能作为可解释性热力图，不得标记为检测框。

## 7. 版本与验收

数据版本、模型版本、配置 SHA-256 和代码 Git commit 必须写入推理结果。冻结后不可覆盖旧模型；修订使用新版本目录。最终执行：

```bash
python3 scripts/audit_dataset.py
python3 scripts/build_dataset_taxonomy.py
cd project && /home/zkf/pytorch-env/bin/pytest
```
