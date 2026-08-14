# 联合分类检测应用运行手册

## 启动

从仓库根目录运行：

```bash
/home/zkf/pytorch-env/bin/pip install -e 'project[app]'
PYTHONPATH=project/src /home/zkf/pytorch-env/bin/python \
  -m dlcpd25_classifier.web --host 127.0.0.1 --port 7860
```

打开 <http://127.0.0.1:7860>。默认配置是 `project/configs/app.yaml`，只加载 `project/assets/model/` 中的一份 `joint-best.pt`。页面上传一次图片后同时显示：

- 带害虫框的原图；
- 宿主、四大类、203 类细标签、置信度和 Top-5；
- 可检测害虫的 DLCPD-25 Class ID、名称、置信度和原图坐标。

## 演示

检测演示可使用随作业包提供的样例：

```text
project/assets/samples/ip102-pest-demo.jpg
```

该样例可同时返回分类和检测结果。模型输出可能随运行环境有轻微浮点差异，应重点检查结果结构、标签映射、框边界和数值有效性。

分类无框演示可使用 `project/assets/samples/dlcpd25-tomato-bacterial-spot.jpg`。页面应正常给出分类结果，并显示“未发现置信度达到阈值的可检测害虫”。这不是错误，也不能解释为图片健康。

## 排错

| 状态 | 处理 |
|---|---|
| 模型包不存在或缺文件 | 检查 `model_bundle` 是否指向 `assets/model`，并恢复完整模型包 |
| checksum、taxonomy、mapping 或预处理不匹配 | 停止启动，恢复完整模型包，不在应用侧修改 |
| torch/torchvision 版本不匹配 | 使用 `/home/zkf/pytorch-env` |
| `auto` 回退 CPU | 检查 CUDA、驱动和显存；页面会显示实际设备 |
| CUDA 显存不足 | 关闭占用 GPU 的程序后重启；或将配置设备改为 `cpu` |
| 图片无框 | 可能是非 96 类、目标太小或分数低于 `0.5`，不属于服务故障 |
| 分类低置信度 | 保留不确定提示，不使用测试集重新调整阈值 |
| 端口占用 | 使用 `--port` 选择其他空闲端口 |

该系统是课程项目原型，不构成农业生产诊断。224 直缩对小目标能力有限，冻结测试集上的小目标 AP 为 `6.1139%`。
