#!/usr/bin/env python3
"""Freeze the D2-R2/D3-R2/D4-R1 data-v1 release and handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import PIL
import numpy as np
import scipy
import torch
import torchvision


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ARTIFACTS = REPO_ROOT / "artifacts" / "data" / "v1"
DEFAULT_OUTPUT = DATA_ARTIFACTS / "d5-r1"
TAXONOMY = REPO_ROOT / "metadata" / "class-taxonomy.json"
INPUT_GIT_COMMIT = "e8b2d639d5c5540f48c760248a9fb9b658468d18"
TEST_SOURCE = REPO_ROOT / "project" / "tests" / "test_handoff_d5_r1.py"

CRITICAL_PATHS = {
    "d0_freeze_report": DATA_ARTIFACTS / "d0" / "d0-freeze.json",
    "d0_checksums": DATA_ARTIFACTS / "d0" / "checksums.sha256",
    "taxonomy_source": TAXONOMY,
    "d1_manifest": DATA_ARTIFACTS / "d1" / "manifest.jsonl",
    "d1_bad_images": DATA_ARTIFACTS / "d1" / "bad-images.jsonl",
    "d1_config": DATA_ARTIFACTS / "d1" / "d1-config.json",
    "d1_summary": DATA_ARTIFACTS / "d1" / "d1-summary.json",
    "d1_checksums": DATA_ARTIFACTS / "d1" / "checksums.sha256",
    "d2_r2_manifest_hashed": DATA_ARTIFACTS / "d2-r2" / "manifest-hashed.jsonl",
    "d2_r2_duplicate_groups": DATA_ARTIFACTS / "d2-r2" / "duplicate-groups.jsonl",
    "d2_r2_config": DATA_ARTIFACTS / "d2-r2" / "d2-r2-config.json",
    "d2_r2_summary": DATA_ARTIFACTS / "d2-r2" / "d2-r2-summary.json",
    "d2_r2_compatibility": DATA_ARTIFACTS / "d2-r2" / "d2-r2-compatibility.json",
    "d2_r2_checksums": DATA_ARTIFACTS / "d2-r2" / "checksums.sha256",
    "d3_r2_train": DATA_ARTIFACTS / "d3-r2" / "train.csv",
    "d3_r2_val": DATA_ARTIFACTS / "d3-r2" / "val.csv",
    "d3_r2_test": DATA_ARTIFACTS / "d3-r2" / "test.csv",
    "d3_r2_group_assignments": DATA_ARTIFACTS / "d3-r2" / "group-assignments.csv",
    "d3_r2_excluded_bad": DATA_ARTIFACTS / "d3-r2" / "excluded-bad-images.csv",
    "d3_r2_config": DATA_ARTIFACTS / "d3-r2" / "d3-r2-config.json",
    "d3_r2_summary": DATA_ARTIFACTS / "d3-r2" / "d3-r2-summary.json",
    "d3_r2_checksums": DATA_ARTIFACTS / "d3-r2" / "checksums.sha256",
    "d4_r1_config": DATA_ARTIFACTS / "d4-r1" / "d4-r1-config.json",
    "d4_r1_reproduction_summary": DATA_ARTIFACTS / "d4-r1" / "reproduction-summary.json",
    "d4_r1_load_smoke": DATA_ARTIFACTS / "d4-r1" / "load-smoke.json",
    "d4_r1_summary": DATA_ARTIFACTS / "d4-r1" / "d4-r1-summary.json",
    "d4_r1_checksums": DATA_ARTIFACTS / "d4-r1" / "checksums.sha256",
    "dataset_loader": REPO_ROOT / "project" / "src" / "dlcpd25_classifier" / "data" / "dataset.py",
    "dataset_package": REPO_ROOT / "project" / "src" / "dlcpd25_classifier" / "data" / "__init__.py",
    "d2_r2_script": REPO_ROOT / "scripts" / "build_duplicates_d2_r2.py",
    "d3_r2_script": REPO_ROOT / "scripts" / "build_splits_d3_r2.py",
    "d4_r1_script": REPO_ROOT / "scripts" / "verify_data_d4_r1.py",
    "d5_r1_script": Path(__file__).resolve(),
    "d5_r1_test": TEST_SOURCE,
}

UPSTREAM_CHECKSUMS = (
    DATA_ARTIFACTS / "d0" / "checksums.sha256",
    DATA_ARTIFACTS / "d1" / "checksums.sha256",
    DATA_ARTIFACTS / "d2-r2" / "checksums.sha256",
    DATA_ARTIFACTS / "d3-r2" / "checksums.sha256",
    DATA_ARTIFACTS / "d4-r1" / "checksums.sha256",
)


class HandoffError(ValueError):
    """Raised when data-v1 cannot be frozen or verified."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_bytes_idempotent(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise HandoffError(f"refusing to overwrite a different artifact: {path}")
        return
    path.write_bytes(payload)


def verify_checksum_file(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        source = REPO_ROOT / name
        if separator != "  " or not source.is_file() or sha256_file(source) != digest:
            raise HandoffError(f"upstream checksum mismatch: {name}")


def critical_index() -> dict[str, dict[str, Any]]:
    result = {}
    for name, path in CRITICAL_PATHS.items():
        if not path.is_file():
            raise HandoffError(f"critical artifact is missing: {path}")
        result[name] = {
            "path": repo_relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def render_handoff(release: dict[str, Any]) -> str:
    stats = release["statistics"]
    splits = stats["splits"]
    duplicates = stats["duplicates"]
    lines = [
        "# DLCPD-25 data-v1 数据交接",
        "",
        f"- 状态：{release['release_status']}",
        f"- 冻结阶段：`{release['stage']}`",
        f"- Git 基线：`{release['input_git_commit']}`",
        f"- taxonomy SHA-256：`{release['taxonomy_sha256']}`",
        "- 任务：203 类图像分类，不是目标检测。",
        "- 正式数据链：`D0 -> D1 -> D2-R2 -> D3-R2 -> D4-R1 -> D5-R1`。",
        "",
        "## 固定数据契约",
        "",
        "- 原图根目录固定为 `data/raw/dlcpd25-203/`，不得使用 `data/views/`。",
        "- class ID、宿主和四大属性只读取 `metadata/class-taxonomy.json`。",
        "- 正式哈希 manifest 和重复组来自 `artifacts/data/v1/d2-r2/`。",
        "- 正式 split 来自 `artifacts/data/v1/d3-r2/`。",
        "- 算法工程师只读取固定 split CSV 中的相对路径、`class_id`、SHA-256 和 `duplicate_group_id`。",
        "- 禁止重新扫描目录、按目录排序推断标签、重新随机切分或让 duplicate group 跨 split。",
        "- D3-R2 直接读取 D2-R2，不依赖任何 D2-R0/D2-R1 目录或脚本。",
        "",
        "## 数据统计",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
        f"| 原始文件 | {stats['total_files']:,} |",
        f"| 可用图片 | {stats['usable_files']:,} |",
        f"| 坏图（保留原文件、排除 split） | {stats['bad_files']:,} |",
        f"| 完全重复 SHA-256 组 | {duplicates['exact_sha256_groups']:,} |",
        f"| 近重复组 | {duplicates['near_duplicate_groups']:,} |",
        f"| 最终非单例重复组 | {duplicates['duplicate_groups']:,} |",
        f"| 跨类别重复组 | {duplicates['cross_class_duplicate_groups']:,} |",
        "",
        "## 固定 Split",
        "",
        "| Split | 样本 | 比例 | SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for split in ("train", "val", "test"):
        entry = splits[split]
        lines.append(f"| {split} | {entry['count']:,} | {entry['ratio']:.6%} | `{entry['sha256']}` |")
    lines.extend(
        [
            "",
            "三个 split 均覆盖 203 类；路径重叠和 duplicate group 跨 split 均为 0。",
            "",
            "## 长尾类别",
            "",
            "以下类别可用图片少于 100 张：",
            "",
            "| class_id | 官方类别 | 可用图片 | 独立 group | train/val/test |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for item in release["long_tail_classes"]:
        lines.append(
            f"| {item['class_id']} | {item['official_name']} | {item['total']} | {item['group_count']} | "
            f"{item['train']}/{item['val']}/{item['test']} |"
        )
    lines.extend(
        [
            "",
            "## 已知限制",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in release["known_limitations"])
    lines.extend(
        [
            "",
            "## 算法工程师入口",
            "",
            "```python",
            "from dlcpd25_classifier.data import DLCPD25Dataset",
            "",
            "dataset = DLCPD25Dataset(",
            "    data_root=\"data/raw/dlcpd25-203\",",
            "    split_csv=\"artifacts/data/v1/d3-r2/train.csv\",",
            "    taxonomy_path=\"artifacts/data/v1/d5-r1/taxonomy-v1.json\",",
            "    transform=transform,",
            ")",
            "```",
            "",
            "算法工程师不得改写这些 CSV、重新分组或从目录名推断 class ID。D5-R1 经总负责人验收通过后，从 A0 数据准入开始；本交接未执行 A0。",
            "",
            "## 复现命令",
            "",
        ]
    )
    lines.extend(f"- `{command}`" for command in release["commands"])
    lines.extend(
        [
            "",
            "## 关键文件 SHA-256",
            "",
            "| 名称 | 路径 | SHA-256 |",
            "|---|---|---|",
        ]
    )
    for name, item in release["critical_artifacts"].items():
        lines.append(f"| {name} | `{item['path']}` | `{item['sha256']}` |")
    lines.extend(
        [
            "",
            "## 环境",
            "",
        ]
    )
    lines.extend(f"- {name}: `{value}`" for name, value in release["environment"].items())
    return "\n".join(lines) + "\n"


def verify_upstream_chain() -> None:
    for checksum in UPSTREAM_CHECKSUMS:
        verify_checksum_file(checksum)


def build_d5_r1(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    verify_upstream_chain()
    d1 = load_json(DATA_ARTIFACTS / "d1" / "d1-summary.json")
    d2 = load_json(DATA_ARTIFACTS / "d2-r2" / "d2-r2-summary.json")
    d3 = load_json(DATA_ARTIFACTS / "d3-r2" / "d3-r2-summary.json")
    d4 = load_json(DATA_ARTIFACTS / "d4-r1" / "d4-r1-summary.json")
    if d2["stage"] != "D2-R2" or d3["stage"] != "D3-R2" or d4["stage"] != "D4-R1":
        raise HandoffError("formal data chain stage identity differs")
    if d3["duplicate_group_leakage_count"] != 0 or d3["path_overlap_count"] != 0:
        raise HandoffError("D3 leakage gate is not zero")
    if (
        not d4["d2_core_all_match"]
        or not d4["d3_core_all_match"]
        or d4["reproduction_runs"] != 2
        or d4["loaded_split_lengths"] != d3["split_counts"]
    ):
        raise HandoffError("D4 reproduction gate failed")
    taxonomy = load_json(TAXONOMY)
    taxonomy_by_id = {int(item["class_id"]): item for item in taxonomy["classes"]}
    long_tail = []
    for item in d3["per_class"]:
        if int(item["total"]) < 100:
            long_tail.append({**item, "official_name": taxonomy_by_id[int(item["class_id"])]["official_name"]})
    long_tail.sort(key=lambda item: (item["total"], item["class_id"]))

    artifacts = critical_index()
    config = {
        "schema_version": 2,
        "stage": "D5-R1",
        "data_version": "data-v1",
        "input_git_commit": INPUT_GIT_COMMIT,
        "formal_chain": ["D0", "D1", "D2-R2", "D3-R2", "D4-R1", "D5-R1"],
        "formal_directories": {
            "manifest_and_duplicates": "artifacts/data/v1/d2-r2",
            "splits": "artifacts/data/v1/d3-r2",
            "reproduction": "artifacts/data/v1/d4-r1",
            "release": "artifacts/data/v1/d5-r1",
        },
        "dependency_contract": "D3-R2 directly reads the frozen D2-R2 release",
        "upstream_checksums": [repo_relative(path) for path in UPSTREAM_CHECKSUMS],
        "runtime_sources": [
            {"path": repo_relative(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())}
        ],
        "test_sources": [
            {"path": repo_relative(TEST_SOURCE), "sha256": sha256_file(TEST_SOURCE)}
        ],
    }
    write_bytes_idempotent(output_dir / "d5-r1-config.json", canonical_json(config))
    release = {
        "schema_version": 2,
        "stage": "D5-R1",
        "data_version": "data-v1",
        "release_status": "frozen_pending_project_lead_acceptance",
        "input_git_commit": INPUT_GIT_COMMIT,
        "formal_chain": config["formal_chain"],
        "taxonomy_sha256": sha256_file(TAXONOMY),
        "fixed_contract": {
            "manifest": artifacts["d2_r2_manifest_hashed"],
            "duplicate_groups": artifacts["d2_r2_duplicate_groups"],
            "splits": {
                split: artifacts[f"d3_r2_{split}"] for split in ("train", "val", "test")
            },
            "taxonomy_snapshot": {
                "path": "artifacts/data/v1/d5-r1/taxonomy-v1.json",
                "sha256": sha256_file(TAXONOMY),
            },
            "required_fields": ["relative_path", "class_id", "sha256", "duplicate_group_id", "split"],
        },
        "statistics": {
            "total_files": d1["total_files"],
            "usable_files": d1["ok_files"],
            "bad_files": d1["bad_files"],
            "duplicates": {
                key: d2[key]
                for key in (
                    "exact_sha256_groups",
                    "near_duplicate_groups",
                    "duplicate_groups",
                    "cross_class_duplicate_groups",
                    "files_in_duplicate_groups",
                    "largest_group_size",
                )
            },
            "splits": {
                split: {
                    "count": d3["split_counts"][split],
                    "ratio": d3["split_ratios"][split],
                    "sha256": d3[f"{split}_sha256"],
                }
                for split in ("train", "val", "test")
            },
        },
        "long_tail_definition": "usable image count < 100",
        "long_tail_classes": long_tail,
        "known_limitations": [
            "19 张坏图保留在原始目录和主 manifest 中，但不进入 train/val/test。",
            "702 张图片扩展名与实际编码不一致，加载必须依赖 Pillow 内容解码。",
            "D2-R2 使用 dHash<=5、pHash<=8 和 complete-link 直径门；保守规则可能漏召回变化较大的真实近重复。",
            "存在跨类别完全重复和近重复组，可能反映标签冲突；split 保持整组，未擅自修订标签。",
            "类别与独立 group 数量长尾明显，少样本类别指标方差会较大。",
            "D3-R2 直接读取 D2-R2；旧 D2-R0/D2-R1 不属于正式链，也不是运行时依赖。",
            "D4-R1 的 Dataset 冒烟实际解码固定 9 张；全量图片解码结论继承 D1。",
            "数据仓库没有明确 LICENSE 文件，论文 CC BY 4.0 不自动等同于数据文件许可。",
        ],
        "commands": [
            "python3 scripts/freeze_data_d0.py",
            "/home/zkf/pytorch-env/bin/python scripts/build_manifest_d1.py --workers 6",
            "/home/zkf/pytorch-env/bin/python scripts/build_duplicates_d2_r2.py --workers 6",
            "/home/zkf/pytorch-env/bin/python scripts/build_splits_d3_r2.py",
            "/home/zkf/pytorch-env/bin/python scripts/verify_data_d4_r1.py --workers 6",
            "/home/zkf/pytorch-env/bin/python scripts/freeze_data_v1_d5_r1.py",
            "sha256sum -c artifacts/data/v1/d5-r1/checksums.sha256",
        ],
        "environment": {
            "python": platform.python_version(),
            "pillow": PIL.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "critical_artifacts": artifacts,
    }
    write_bytes_idempotent(output_dir / "taxonomy-v1.json", TAXONOMY.read_bytes())
    write_bytes_idempotent(output_dir / "data-v1-release.json", canonical_json(release))
    write_bytes_idempotent(output_dir / "data-handoff-v1.md", render_handoff(release).encode("utf-8"))
    summary = {
        "schema_version": 2,
        "stage": "D5-R1",
        "data_version": "data-v1",
        "release_status": release["release_status"],
        "critical_artifact_count": len(artifacts),
        "long_tail_class_count": len(long_tail),
        "total_files": d1["total_files"],
        "usable_files": d1["ok_files"],
        "bad_files": d1["bad_files"],
        "release_sha256": sha256_file(output_dir / "data-v1-release.json"),
        "handoff_sha256": sha256_file(output_dir / "data-handoff-v1.md"),
        "taxonomy_snapshot_sha256": sha256_file(output_dir / "taxonomy-v1.json"),
        "config_sha256": sha256_file(output_dir / "d5-r1-config.json"),
        "formal_d2_stage": d2["stage"],
        "formal_d3_stage": d3["stage"],
        "formal_d4_stage": d4["stage"],
        "a0_executed": False,
    }
    write_bytes_idempotent(output_dir / "d5-r1-summary.json", canonical_json(summary))
    checksum_paths = list(CRITICAL_PATHS.values()) + [
        output_dir / "taxonomy-v1.json",
        output_dir / "d5-r1-config.json",
        output_dir / "data-v1-release.json",
        output_dir / "data-handoff-v1.md",
        output_dir / "d5-r1-summary.json",
    ]
    lines = [f"{sha256_file(path)}  {repo_relative(path)}" for path in sorted(set(checksum_paths))]
    write_bytes_idempotent(output_dir / "checksums.sha256", ("\n".join(lines) + "\n").encode("utf-8"))
    verify_d5_r1(output_dir)
    return summary


def verify_d5_r1(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    verify_upstream_chain()
    release = load_json(output_dir / "data-v1-release.json")
    summary = load_json(output_dir / "d5-r1-summary.json")
    config = load_json(output_dir / "d5-r1-config.json")
    if release["release_status"] != "frozen_pending_project_lead_acceptance":
        raise HandoffError("unexpected data-v1 release status")
    if release["stage"] != "D5-R1" or release["formal_chain"] != config["formal_chain"]:
        raise HandoffError("D5-R1 stage or formal chain differs")
    if release["taxonomy_sha256"] != sha256_file(TAXONOMY):
        raise HandoffError("release taxonomy SHA-256 differs")
    if (output_dir / "taxonomy-v1.json").read_bytes() != TAXONOMY.read_bytes():
        raise HandoffError("taxonomy snapshot differs from source")
    for name, item in release["critical_artifacts"].items():
        path = REPO_ROOT / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise HandoffError(f"critical artifact differs: {name}")
    handoff = (output_dir / "data-handoff-v1.md").read_text(encoding="utf-8")
    for required in (
        "正式数据链",
        "固定数据契约",
        "固定 Split",
        "长尾类别",
        "已知限制",
        "算法工程师入口",
        "未执行 A0",
    ):
        if required not in handoff:
            raise HandoffError(f"handoff section is missing: {required}")
    for required_path in ("artifacts/data/v1/d2-r2/", "artifacts/data/v1/d3-r2/", "artifacts/data/v1/d5-r1/"):
        if required_path not in handoff:
            raise HandoffError(f"handoff formal path is missing: {required_path}")
    if "scripts/build_duplicates_d2.py" in handoff or "scripts/build_splits_d3.py" in handoff:
        raise HandoffError("handoff contains a rejected R0 command")
    if summary["a0_executed"] or summary["usable_files"] != 221377 or summary["bad_files"] != 19:
        raise HandoffError("D5 summary boundary or counts differ")
    if summary["formal_d2_stage"] != "D2-R2" or summary["formal_d3_stage"] != "D3-R2" or summary["formal_d4_stage"] != "D4-R1":
        raise HandoffError("D5-R1 formal stage summary differs")
    expected_runtime = [{"path": repo_relative(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())}]
    expected_tests = [{"path": repo_relative(TEST_SOURCE), "sha256": sha256_file(TEST_SOURCE)}]
    if config["runtime_sources"] != expected_runtime or config["test_sources"] != expected_tests:
        raise HandoffError("D5-R1 source index differs")
    verify_checksum_file(output_dir / "checksums.sha256")
    return {
        "data_version": release["data_version"],
        "release_status": release["release_status"],
        "critical_artifacts": len(release["critical_artifacts"]),
        "long_tail_classes": len(release["long_tail_classes"]),
        "formal_chain": release["formal_chain"],
        "a0_executed": False,
    }


def main() -> int:
    args = parse_args()
    try:
        result = verify_d5_r1(args.output_dir) if args.verify_only else build_d5_r1(args.output_dir)
    except (HandoffError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
