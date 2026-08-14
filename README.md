# DLCPD-25 农产品病虫害与缺陷分析项目

本目录保存 DLCPD-25 与 IP102 的本地数据、研究资料、元数据、可复现实验代码和最终课程作业包。

最终提交内容位于 [`project/`](project/)，其中包含联合模型、实验报告、结果截图、架构材料和 Web 应用源码。

```text
data/       DLCPD-25 与 IP102 本地数据
metadata/   类别名称、层级与检测类别映射
scripts/    数据审计、元数据生成与辅助脚本
artifacts/  可重新生成的运行产物索引
research/   论文、翻译和数据集资料
docs/       数据、检测设计与应用说明
project/    最终课程作业包
```

系统使用一份联合权重，对 DLCPD-25 的 203 个细粒度类别进行整图分类，并对 IP102 中已映射的 96 类害虫进行目标检测。输入统一为 RGB `224 x 224`，共享 ResNet-50-FPN 主干。

启动与使用说明见 [project/README.md](project/README.md)。
