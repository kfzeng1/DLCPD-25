# IP102 数据脚本

本目录只存放 T0 开始后新增的 IP102 审计、固定划分、派生标注和数据合同验证脚本。

脚本必须以 `data/raw/ip102/downloads/Detection/VOC2007/` 为只读输入，输出到新的 `artifacts/data/ip102-detection-v1/`。不得在原始目录内写入修复后的 XML、重命名图片或删除正式划分外文件。

T0 详细输入、输出和验收标准见 `../../docs/workplans/data-engineer-detection.md`。
