# 元数据

所有类别、映射和层级合同都放在这里；训练与推理代码必须读取这些文件，不得自行推断。

```text
metadata/
  dlcpd25/
    official-class-names.txt          官方 203 类名称，顺序即 class_id 0..202
    class-directory-aliases.json      官方类名 → 本地目录名
    class-taxonomy.json               宿主、属性、类别层级的机器可读合同
    class-taxonomy.csv                层级合同表格
  ip102/
    detection-class-map.json           IP102 97 个源标签 → 检测标签 1..96 → DLCPD-25 class_id
```

## 编号合同

- DLCPD-25：`class_id 0..202`，按 `official-class-names.txt` 顺序固定。
- IP102 检测源标签：97 个。
- 检测器内部标签：`1..96`，`0` 为背景。
- 检测输出：必须转换为 DLCPD-25 公共 `class_id`。
- IP102 类别 50、51 映射到同一个检测标签和 DLCPD-25 `class_id 97`。

## 再生成

```bash
python3 scripts/dlcpd25/build_taxonomy.py
```
