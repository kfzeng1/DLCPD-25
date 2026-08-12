# 分类基线应用启动与演示

本运行手册对应已经通过 F0 的 203 类分类应用。目标检测尚未接入当前 `7860` 服务；T4 完成后将在本文件中补充双模型配置、画框演示和联合排错步骤。

## 启动

从仓库根目录运行：

```bash
/home/zkf/pytorch-env/bin/pip install -e 'project[app]'
PYTHONPATH=project/src /home/zkf/pytorch-env/bin/python \
  -m dlcpd25_classifier.web --host 127.0.0.1 --port 7860
```

页面地址为 <http://127.0.0.1:7860>。默认配置位于 `project/configs/app.yaml`，冻结模型包为 `artifacts/releases/dlcpd25-resnet50-weighted-v1/`。

应用验证证据使用 A3 fixed-val 样例生成，不访问正式 test split：

```bash
PYTHONPATH=project/src /home/zkf/pytorch-env/bin/python \
  -m dlcpd25_classifier.inference.smoke
```

## 固定演示样例

以下三张图片来自 A3 固定 val 样例，不读取或重复评估 test：

| 路径 | 预期 Top-1 | 参考置信度 | 页面状态 |
|---|---|---:|---|
| `data/raw/dlcpd25-203/Adristyrannus 枯叶夜蛾(柑橘)/64003.jpg` | class 0 | 90.6372% | 正常结果 |
| `data/raw/dlcpd25-203/orange huanglongbing 柑橘黄龙病(黄龙病)/7d7b6586-62a6-49b3-ae46-917b89ba9b07___CREC_HLB 3977.JPG` | class 131 | 20.2720% | 低置信度、不确定提示 |
| `data/raw/dlcpd25-203/yellow rice borer 三化螟(水稻)/03414.jpg` | class 202 | 60.6341% | 正常结果 |

损坏文件用于错误路径演示，页面应显示“图片已损坏或无法识别”，不得出现 Python 堆栈。

冻结阈值为 `0.55`。A3 已记录 test 低置信度率为 72.7342%，这说明不确定提示是常见正常状态；不得使用该 test 统计重新调整阈值。

## 排错

| 状态 | 处理 |
|---|---|
| 模型包不存在或缺少文件 | 检查 `model_bundle` 是否指向已验收的版本目录，不复制或补写模型文件 |
| checksum、taxonomy 或预处理不匹配 | 停止启动，重新取得完整冻结模型包，不在应用侧修改 |
| torch/torchvision 版本不匹配 | 使用项目指定的 `/home/zkf/pytorch-env` 环境 |
| `auto` 回退 CPU | 检查 CUDA 可用性、驱动和显存；结果栏会显示实际设备 |
| 显式 `cuda` 启动失败 | 改为 `auto` 或 `cpu`，保留错误日志用于排查 |
| 端口被占用 | 使用 `--port` 选择其他空闲端口 |

当前已发布版本只提供 203 类图像分类结果，不输出检测框。T4 目标版本将增加 IP102 96 类害虫检测，但不会定位其余病害、健康和缺陷。低置信度或域外图片只显示不确定结果，不构成生产诊断。
