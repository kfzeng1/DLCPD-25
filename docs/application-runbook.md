# 联合分类检测应用运行手册

## 启动

从仓库根目录运行：

```bash
/home/zkf/pytorch-env/bin/pip install -e 'project[app]'
PYTHONPATH=project/src /home/zkf/pytorch-env/bin/python \
  -m dlcpd25_classifier.web --host 127.0.0.1 --port 7860
```

打开 <http://127.0.0.1:7860>。默认配置是 `project/configs/app.yaml`，只加载 `artifacts/releases/dlcpd25-ip102-joint-v1/` 的一份 `joint-best.pt`。页面上传一次图片后同时显示：

- 带害虫框的原图；
- 宿主、四大类、203 类细标签、置信度和 Top-5；
- 可检测害虫的 DLCPD-25 Class ID、名称、置信度和原图坐标。

## 演示

检测演示可使用冻结 IP102 `val`，不要重新运行两个正式 test：

```text
data/raw/ip102/downloads/Detection/VOC2007/JPEGImages/IP000000378.jpg
```

该样例在 J5 CUDA 联调中完成一次共享主干前向，能同时返回分类和检测结果。模型输出可能随运行环境有轻微浮点差异，演示验收以结构、标签映射、框边界和有限数值为准，不以固定小数位为门禁。

分类无框演示可从 `data/raw/dlcpd25-203/` 选择病害或健康图片。页面应正常给出分类结果，并显示“未发现置信度达到阈值的可检测害虫”。这不是错误，也不能解释为图片健康。

## 排错

| 状态 | 处理 |
|---|---|
| 模型包不存在或缺文件 | 检查 `model_bundle` 是否指向 J4 唯一联合包，不复制旧分类权重 |
| checksum、taxonomy、mapping 或预处理不匹配 | 停止启动，恢复完整 J4 模型包，不在应用侧修改 |
| torch/torchvision 版本不匹配 | 使用 `/home/zkf/pytorch-env` |
| `auto` 回退 CPU | 检查 CUDA、驱动和显存；页面会显示实际设备 |
| CUDA 显存不足 | 关闭占用 GPU 的程序后重启；或将配置设备改为 `cpu` |
| 图片无框 | 可能是非 96 类、目标太小或分数低于 `0.5`，不属于服务故障 |
| 分类低置信度 | 保留不确定提示，不根据 J4 test 重新调整阈值 |
| 端口占用 | 使用 `--port` 选择其他空闲端口 |

该系统是课程项目原型，不构成农业生产诊断。224 直缩对小目标能力有限，J4 test 小目标 AP 为 `6.1139%`。
