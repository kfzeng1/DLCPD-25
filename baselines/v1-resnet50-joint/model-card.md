# DLCPD-25 + IP102 Joint Model v1

## Model

One `224 x 224` RGB tensor is processed by one shared ResNet-50 body. The classification head predicts all 203 DLCPD-25 image classes. The FPN/RPN/ROI detection branch localizes 96 IP102 pest classes. The bundle contains one joint checkpoint.

## Final Test

- Classification Top-1: 91.315718%
- Classification Top-5: 96.428893%
- Classification Macro-F1: 75.445058%
- Detection mAP@0.5:0.95: 35.882310%
- Detection AP50: 65.532560%
- Detection Precision: 68.909513%
- Detection Recall: 80.198020%

最终测试在模型权重、预处理、类别映射、分数阈值、NMS 和最大检测数固定后执行一次；测试结果未参与训练或调参。

## Limitations

The detection branch only has box supervision for 96 mapped IP102 pest classes. Diseases, healthy classes and physiological defects can be classified but cannot be promised bounding boxes. IP102 source class 61 has no test support, corresponding to detector label 8. Small-object performance is limited by direct resize to 224.
