# 元数据说明

`official-class-names.txt` 是 2026-08-08 从作者 GitHub 所指向的百度网盘公开分享根目录只读枚举得到的 203 个唯一类别目录名，并按 Unicode 码点排序。

这些名称是发布版本标识，不等于经过分类学规范化的学名。文件没有 class ID；训练前应显式生成并固化映射。不要依赖操作系统或框架临时返回的目录顺序。

当前文件 SHA-256：`6f9c4f1920b2e51131a105315177b8ad7147a1b4ebe9088622e69ee4a0c51225`。

来源：<https://github.com/hwzhanng/DLCPD-25-Dataset>。

`class-directory-aliases.json` 保存官方原始目录名到当前本地“英文 + 中文”目录名的一一映射。它只用于兼容当前文件布局；其中中文译名属于项目元数据，尚未经过植保和昆虫分类专家逐项确认。

`class-taxonomy.json` 和 `class-taxonomy.csv` 保存 203 类的固定 ID、宿主作物、经济/粮食作物组、四大标签属性和本地图片数量。训练与推理必须读取这个文件，不要根据目录遍历顺序临时生成类别 ID。

`a3-test-evaluation.json` 是受版本控制的一次性 test 消费凭据。A3 在读取固定 test split 前原子创建该文件，完成后写入模型包和指标哈希；文件存在时禁止再次执行 test 评估。
