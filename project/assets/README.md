# 作业支撑材料

本目录保存课程作业运行、展示和结果说明所需的轻量资料。完整原始数据集体积较大且数据许可未明确，因此不重复复制原图；本地提交包保留模型权重和演示样本。

```text
assets/
  model/      最终联合模型包及其 checksum
  results/    分类、检测与最终测试结果
  samples/    两张可直接上传到 Web 页面的演示图片
  data/       IP102 数据审计摘要和数据获取说明
```

## 最终模型包

`model/joint-best.pt` 是唯一联合权重，SHA-256 为：

```text
5ec0f4f7891b729ddf26a51cd70d5c56a69825b2dd587c7f6af55854d3c06c49
```

模型包内的 `checksums.sha256` 用于校验发布文件。模型权重约 484 MiB，提交时需保留 `assets/model/joint-best.pt`，即可使用默认配置启动应用。

## 演示样本

| 文件 | 来源 | SHA-256 | 用途 |
|---|---|---|---|
| `samples/ip102-pest-demo.jpg` | IP102 检测图片 `IP000000378.jpg` | `f254961a216b40fe454e2bcaf2c942cda8b2f2c265b740885c27dc358ee401fe` | 害虫检测演示 |
| `samples/dlcpd25-tomato-bacterial-spot.jpg` | DLCPD-25 番茄细菌性斑点病类别 | `5f181076dd7e4aa79676842ad50ec4433cdca6a91c4400b120973adb272a7d5d` | 病害分类演示 |

使用样例时应说明：检测分支只对 IP102 中具有边界框监督的 96 类害虫产生框；病害图片只输出分类，不应期待病斑定位框。
