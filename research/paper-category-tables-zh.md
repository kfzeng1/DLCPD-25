---
title: "DLCPD-25 论文类别表中文翻译"
subtitle: "层级结构、原文数字核算与本地 203 类分组对照"
author: "DLCPD-25 项目整理"
date: "2026-08-09"
documentclass: ctexart
classoption:
  - UTF8
geometry: "a4paper,margin=2.1cm"
fontsize: 10.5pt
linestretch: 1.2
colorlinks: true
linkcolor: black
urlcolor: blue
header-includes:
  - |
    \usepackage{booktabs}
    \usepackage{longtable}
    \usepackage{array}
    \usepackage{xcolor}
    \definecolor{tablehead}{HTML}{E8EEF2}
---

# 翻译说明

下表翻译自 Zhang 等发表于 *Sensors* 的论文 *DLCPD-25: A Large-Scale and Diverse Dataset for Crop Disease and Pest Recognition* 中的表 1 “Hierarchical structure of the DLCPD-25 dataset”。翻译保留论文原始数字；EC 译为经济作物，FC 译为粮食作物，`Vitis` 按葡萄属处理。

> 本文是项目整理的中文翻译，不是论文作者发布的中文版。原论文采用 CC BY 4.0；引用信息见文末。

# 表 1 中文翻译

| 类型 | 作物名称 | 类别数 | 图片数 |
|:---:|---|---:|---:|
| 经济作物（EC） | 柑橘（Citrus） | 21 | 15,342 |
| 经济作物（EC） | 番茄（Tomato） | 20 | 46,201 |
| 经济作物（EC） | 葡萄属（Vitis） | 21 | 20,134 |
| 经济作物（EC） | 苹果（Apple） | 5 | 14,390 |
| 经济作物（EC） | 大豆（Soybean） | 23 | 9,613 |
| 经济作物（EC） | 桃（Peach） | 2 | 8,133 |
| 经济作物（EC） | 芒果（Mango） | 10 | 5,840 |
| 经济作物（EC） | 苜蓿（Alfalfa） | 11 | 5,703 |
| 经济作物（EC） | 甜椒（Bell Pepper） | 2 | 5,379 |
| 经济作物（EC） | 草莓（Strawberry） | 2 | 5,264 |
| 经济作物（EC） | 樱桃（Cherry） | 2 | 3,972 |
| 经济作物（EC） | 棉花（Cotton） | 11 | 3,794 |
| 经济作物（EC） | 南瓜（Squash） | 1 | 3,571 |
| 经济作物（EC） | 蓝莓（Blueberry） | 1 | 3,318 |
| 经济作物（EC） | 树莓（Raspberry） | 1 | 2,781 |
| 经济作物（EC） | 黄瓜（Cucumber） | 7 | 2,384 |
| 经济作物（EC） | 甜菜（Beet） | 7 | 2,176 |
| 经济作物（EC） | 辣椒（Pepper） | 2 | 1,689 |
| 经济作物（EC） | 大蒜（Garlic） | 1 | 279 |
| 粮食作物（FC） | 玉米（Corn） | 20 | 18,677 |
| 粮食作物（FC） | 水稻（Rice） | 21 | 14,450 |
| 粮食作物（FC） | 马铃薯（Potato） | 4 | 11,553 |
| 粮食作物（FC） | 小麦（Wheat） | 15 | 4,522 |
| **合计（按表内数字复算）** | **23 种作物** | **210** | **209,165** |

# 原表数字核查

论文不同位置存在不能同时成立的统计口径，使用时应分别引用，不能混为同一个版本。

| 来源位置 | 作物数 | 类别数 | 图片数 |
|---|---:|---:|---:|
| 论文摘要及多数正文表述 | 23 | 203 | 221,943 |
| 论文表 1 逐行复算 | 23 | 210 | 209,165 |
| 表 1 后正文叙述（EC 150 + FC 60） | 23 | 210 | 未给合计 |
| 2026-08-08 官方云盘目录枚举 | 未形成可靠结构化字段 | 203 | 未以表 1 口径公布 |
| 本地已下载并审计的子集 | 目录名可归入 22 个作物组 | 203 | 221,396 |

主要差异包括：论文表 1 有 7 个黄瓜类别，而当前官方 203 目录清单没有黄瓜类别；论文表 1 的棉花为 3,794 张，本地棉花相关目录合计显著更多。因此，表 1 适合说明作者设计的“作物层级”，不应直接作为当前本地训练数据的类别清单或数量真值。

# 本地 203 类项目分组

为了便于分类系统筛选和统计，本项目在不改变 203 个细粒度标签的前提下增加五个上位组。

| 项目分组 | 类别数 | 定义 |
|---|---:|---|
| 农业有害生物 | 126 | 昆虫、螨类、腹足类等农业有害生物 |
| 植物病害 | 56 | 真菌、细菌、病毒、卵菌等病害或以病害命名的症状 |
| 健康 | 17 | 官方类别名明确标为健康的作物图片 |
| 非生物或生理缺陷 | 3 | 药害、红叶、花叶等异常 |
| 混合或歧义 | 1 | `garlic pest and diseases`，无法归入单一病害或虫害组 |
| **合计** | **203** | 逐类结果见 `metadata/class-taxonomy.json` |

这五组是项目级辅助元数据，不是论文作者发布的官方分类。模型的最终分类标签仍应是 203 个原始细粒度类别；上位组可用于界面筛选、层级评估或两阶段分类。

# 引用

Zhang, H.-W.; Wang, R.-F.; Wang, Z.; Su, W.-H. DLCPD-25: A Large-Scale and Diverse Dataset for Crop Disease and Pest Recognition. *Sensors* **2025**, *25*, 7098. DOI: [10.3390/s25227098](https://doi.org/10.3390/s25227098).

原论文：Copyright © 2025 by the authors，按 Creative Commons Attribution 4.0 International（CC BY 4.0）许可发布。本翻译表注明了来源，并保留对原始统计矛盾的核查说明。
