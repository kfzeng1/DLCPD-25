# IP102 数据脚本

本目录存放 IP102 检测数据审计、固定划分、派生标注和数据合同验证脚本。

脚本必须以 `data/raw/ip102/downloads/Detection/VOC2007/` 为只读输入，输出到新的 `artifacts/data/ip102-detection-v1/`。不得在原始目录内写入修复后的 XML、重命名图片或删除正式划分外文件。

- `build_detection_data_t0.py`：读取 VOC 图片与 XML，生成固定数据清单和派生标注；
- `verify_detection_dataset_t0.py`：独立检查数据统计、类别映射、文件完整性和校验值。

类别映射和模型使用边界见 `../../docs/ip102-detection-design.md`。
