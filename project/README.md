# Plan-A 训练/推理工程

当前已实现 DLCPD-25 分类专家训练与训练进度网页。

## 已实现

```text
project/src/dlcpd25_v2/
  common.py                         仓库根目录定位
  data/classification_dataset.py    读取冻结 manifest.csv.gz 的分类数据集
  classification/
    model.py                        ConvNeXt-Tiny + 主头/宿主头/属性头
    losses.py                       Focal Loss + 类别平衡权重 + 辅助 CE
    metrics.py                      Top-1/Top-5/Macro-F1/Balanced Accuracy
    transforms.py                   训练 RandAugment / 验证 Resize+CenterCrop
    trainer.py                      训练循环、AMP、EMA、早停、checkpoint、进度文件
    train.py                        CLI 入口
    evaluate.py                    测试集最终评估与报告产物
  web/progress.py                   训练进度网页
```

## 训练命令

```bash
cd /home/zkf/DLCPD-25
source /home/zkf/pytorch-env/bin/activate

# 训练 1 轮（默认配置为 40 轮，这里显式覆盖为 1）
python -m dlcpd25_v2.classification.train \
  --config configs/plan-a/classification.yaml \
  --run-id convnext-tiny-384-plan-a-v1 \
  --epochs 1

# 后续从 last.pt 继续训练
python -m dlcpd25_v2.classification.train \
  --config configs/plan-a/classification.yaml \
  --run-id convnext-tiny-384-plan-a-v1 \
  --epochs 40 \
  --resume artifacts/training/classification/convnext-tiny-384-plan-a-v1/checkpoints/last.pt
```

输出：

```text
artifacts/training/classification/<run_id>/
  state.json        实时进度（网页读取）
  history.json      每轮 train/val 指标
  checkpoints/
    last.pt         每轮结束后的最新 checkpoint
    best.pt         按 val_macro_f1 选择的最佳 checkpoint
```

## 进度网页

训练过程中随时查看：

```bash
uvicorn dlcpd25_v2.web.progress:app --host 0.0.0.0 --port 8765
```

浏览器打开 <http://127.0.0.1:8765>。页面每 3 秒自动刷新，显示：

- 当前轮次、batch、全局进度；
- 最近 loss、本轮平均 loss；
- 学习率、GPU 显存、预计剩余时间；
- train/val 指标曲线和每轮历史表。

## 测试集评估

```bash
/home/zkf/pytorch-env/bin/python -m dlcpd25_v2.classification.evaluate   --checkpoint artifacts/training/classification/convnext-tiny-384-plan-a-v1/checkpoints/best.pt
```

最终分类模型报告见 `docs/classification-model-report.md`。

## 数据合同

训练代码只读取：

- `artifacts/data/dlcpd25/manifest.csv.gz`
- `metadata/dlcpd25/class-taxonomy.json`

不会重新扫描 `data/raw/`，也不会在代码中硬编码类别名。
