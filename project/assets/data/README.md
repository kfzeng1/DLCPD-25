# 数据说明

## DLCPD-25 分类数据

- 数据集：Dataset of Large-scale Crop Pests and Diseases, 2025（DLCPD-25）；
- 本项目本地版本：203 类、221,396 个原始文件，其中 221,377 张可用图片；
- 固定 train/val/test：177,021 / 22,178 / 22,178；
- 标注形式：整图单标签分类，不含边界框。

官方论文：<https://doi.org/10.3390/s25227098>

官方入口：<https://github.com/hwzhanng/DLCPD-25-Dataset>

## IP102 检测数据

- 标注形式：VOC XML 边界框；
- 本项目正式图片数：18,976；有效框数：22,283；
- 固定 train/val/test：12,142 / 3,036 / 3,798；
- 97 个 IP102 源标签映射为 96 个可检测的 DLCPD-25 公共类别。

IP102 论文：X. Wu, C. Zhan, Y.-K. Lai, M.-M. Cheng, J. Yang. *IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition*. CVPR, 2019。

本目录的 `ip102-detection-audit.json` 为本地检测数据审计摘要。完整原图和 XML 不随代码仓库分发；复现实验需按项目配置放置原始数据、固定 split、taxonomy 和类别映射。

## 使用边界

DLCPD-25 论文和作者数据入口对数据文件没有提供明确的独立许可证。本作业仅用于课程学习和复现实验，不应擅自将原始图片重新发布或用于商业用途。
