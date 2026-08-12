# 应用工程师工作单：J5

## 任务目标

只加载 J4 发布的一个联合模型包。上传图片后执行一次 RGB `224 x 224` 统一预处理和一次模型调用，同时显示 203 类分类 Top-5 与 96 类害虫检测框。

## 固定输入

- J4 唯一联合模型包
- 包内 taxonomy、IP102 映射、预处理和后处理合同
- 公共类别编号 DLCPD-25 `class_id 0-202`

不得继续加载历史分类模型或额外检测 checkpoint。

## 执行内容

1. 校验联合权重、结构、预处理、后处理、映射、版本和 checksum。
2. 图片只解码一次，直接缩放到 224 并调用一次联合 Predictor。
3. 分类显示宿主、四大类、具体类别、置信度、Top-5 和不确定提示。
4. 检测框从 224 坐标映射回原图，显示类别与置信度。
5. 支持无框、多框、重叠框、低分框、损坏图、超大图和 CPU/CUDA 回退。
6. 页面明确：“分类覆盖203类；检测只定位IP102有框的96类害虫。”

## 返回合同

```text
schema_version
model_version
classification
  class_id, names, host, category, confidence, top_k, low_confidence
detections[]
  box_xyxy_original, score, class_id, names
device
inference_ms
```

## 验收

- 测试证明只加载一个权重、只构建一个联合模型；
- 分类与检测来源于同一次模型调用；
- 224 框反算原图坐标准确；
- checksum、taxonomy、mapping 或预处理不一致时拒绝启动；
- CPU/CUDA 固定样例、项目全量测试和浏览器桌面/移动视口通过；
- 更新 `docs/application-contract.md` 与 `docs/application-runbook.md`。

禁止伪造病害/缺陷框，禁止回退为双模型，禁止修改模型参数或训练合同，禁止自行执行 F1、提交或推送。
