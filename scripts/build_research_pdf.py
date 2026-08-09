#!/usr/bin/env python3
"""Build the translated paper tables PDF with the reviewed 203-class appendix."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "research" / "paper-category-tables-zh.md"
DEFAULT_TAXONOMY = REPO_ROOT / "metadata" / "class-taxonomy.json"
DEFAULT_OUTPUT = REPO_ROOT / "research" / "paper-category-tables-zh.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def escape_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def taxonomy_appendix(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    classes = payload.get("classes")
    if not isinstance(classes, list) or len(classes) != 203:
        raise ValueError("taxonomy must contain exactly 203 classes")
    lines = [
        "\\newpage",
        "",
        "# 当前官方 203 类中英对照与项目分组",
        "",
        "论文附录 A1 自称是完整类别表，但经提取只有 204 行、19 个作物组，且名称和数量与论文表 1、当前官方云盘目录均不一致。因此下表不冒充附录 A1 的逐字翻译，而是以当前官方云盘实际枚举的 203 个目录为基准，给出可用于本项目训练的宿主、四大类属性和中英对照标签。官方原始拼写完整保存在随项目提供的 JSON 和 CSV 中。",
        "",
        "| ID | 宿主 / 四大类 | 中英对照类别名 | 图片数 |",
        "|---:|---|---|---:|",
    ]
    for item in classes:
        lines.append(
            "| {class_id} | {host_zh} / {category_zh} | {local_directory} | {image_count:,} |".format(
                class_id=int(item["class_id"]),
                host_zh=escape_cell(item["host_zh"]),
                category_zh=escape_cell(item["category_zh"]),
                local_directory=escape_cell(item["local_directory"]),
                image_count=int(item["image_count"]),
            )
        )
    lines.extend(
        [
            "",
            "表中合计为 203 类、221,396 个本地文件。类别 ID 以 `metadata/official-class-names.txt` 的顺序固定，不依赖文件系统排序。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        source = args.source.read_text(encoding="utf-8")
        combined = source.rstrip() + "\n\n" + taxonomy_appendix(args.taxonomy)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="dlcpd25-paper-") as temp_dir:
            combined_path = Path(temp_dir) / "combined.md"
            combined_path.write_text(combined, encoding="utf-8")
            subprocess.run(
                [
                    "pandoc",
                    str(combined_path),
                    "--from",
                    "markdown+yaml_metadata_block",
                    "--pdf-engine=xelatex",
                    "--output",
                    str(args.output),
                ],
                check=True,
            )
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}")
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
