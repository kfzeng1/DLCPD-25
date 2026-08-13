"""Validation contract for the frozen joint classification/detection bundle."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dlcpd25_classifier.detection import DetectionClassMapping
from dlcpd25_classifier.taxonomy import Taxonomy

from .errors import BundleValidationError

JOINT_BUNDLE_SCHEMA_VERSION = 1
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_JOINT_FILES = frozenset(
    {
        "joint-best.pt",
        "manifest.json",
        "resolved-config.yaml",
        "preprocessing.json",
        "postprocessing.json",
        "taxonomy.json",
        "ip102-detection-class-map.json",
        "metrics-classification.json",
        "metrics-detection.json",
        "model-card.md",
    }
)


@dataclass(frozen=True)
class JointBundleManifest:
    schema_version: int
    model_version: str
    git_commit: str
    architecture: str
    image_size: int
    classification_classes: int
    detection_classes: int
    shared_body_forwards_per_joint_call: int
    checkpoint_sha256: str
    taxonomy_sha256: str
    detection_mapping_sha256: str
    preprocessing_sha256: str
    postprocessing_sha256: str
    torch_version: str
    torchvision_version: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error(code: str, message: str) -> BundleValidationError:
    return BundleValidationError(code, message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _error("manifest_invalid", f"无法读取联合模型包文件：{path.name}。") from exc
    if not isinstance(payload, dict):
        raise _error("manifest_invalid", f"联合模型包文件必须是 JSON 对象：{path.name}。")
    return payload


def _load_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise _error("checksum_invalid", "联合模型包校验清单无法读取。") from exc
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise _error("checksum_invalid", "联合模型包校验清单格式无效。")
        digest, raw_name = parts
        name = raw_name.removeprefix("*")
        candidate = Path(name)
        if (
            not SHA256_PATTERN.fullmatch(digest)
            or candidate.is_absolute()
            or ".." in candidate.parts
            or name in checksums
        ):
            raise _error("checksum_invalid", "联合模型包校验清单包含无效条目。")
        checksums[name] = digest
    return checksums


def load_joint_bundle_manifest(bundle_path: str | Path) -> JointBundleManifest:
    bundle = Path(bundle_path)
    if not bundle.is_dir():
        raise _error("bundle_missing", "联合模型包目录不存在。")
    required = REQUIRED_JOINT_FILES | {"checksums.sha256"}
    missing = sorted(name for name in required if not (bundle / name).is_file())
    if missing:
        raise _error("bundle_incomplete", f"联合模型包缺少文件：{', '.join(missing)}。")
    checksums = _load_checksums(bundle / "checksums.sha256")
    missing_checksums = sorted(REQUIRED_JOINT_FILES - checksums.keys())
    if missing_checksums:
        raise _error(
            "checksum_incomplete",
            f"联合模型包校验清单缺少文件：{', '.join(missing_checksums)}。",
        )
    for name, expected in checksums.items():
        target = bundle / name
        if not target.is_file() or not target.resolve().is_relative_to(bundle.resolve()):
            raise _error("checksum_invalid", f"联合模型包校验路径无效：{name}。")
        if sha256_file(target) != expected:
            raise _error("checksum_mismatch", f"联合模型包文件校验失败：{name}。")
    weight_files = sorted(path.name for path in bundle.glob("*.pt") if path.is_file())
    if weight_files != ["joint-best.pt"]:
        raise _error("weight_contract_invalid", "联合模型包必须且只能包含一个权重文件。")

    payload = _load_json(bundle / "manifest.json")
    try:
        manifest = JointBundleManifest(
            schema_version=int(payload["schema_version"]),
            model_version=str(payload["model_version"]),
            git_commit=str(payload["git_commit"]),
            architecture=str(payload["architecture"]),
            image_size=int(payload["image_size"]),
            classification_classes=int(payload["classification_classes"]),
            detection_classes=int(payload["detection_classes"]),
            shared_body_forwards_per_joint_call=int(
                payload["shared_body_forwards_per_joint_call"]
            ),
            checkpoint_sha256=str(payload["checkpoint_sha256"]),
            taxonomy_sha256=str(payload["taxonomy_sha256"]),
            detection_mapping_sha256=str(payload["detection_mapping_sha256"]),
            preprocessing_sha256=str(payload["preprocessing_sha256"]),
            postprocessing_sha256=str(payload["postprocessing_sha256"]),
            torch_version=str(payload["torch_version"]),
            torchvision_version=str(payload["torchvision_version"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _error("manifest_invalid", "联合模型包 manifest 字段无效。") from exc
    if (
        manifest.schema_version != JOINT_BUNDLE_SCHEMA_VERSION
        or manifest.architecture != "joint-resnet50-fasterrcnn"
        or manifest.image_size != 224
        or manifest.classification_classes != 203
        or manifest.detection_classes != 96
        or manifest.shared_body_forwards_per_joint_call != 1
    ):
        raise _error("contract_mismatch", "联合模型包架构契约不受支持。")
    hash_contract = {
        "joint-best.pt": manifest.checkpoint_sha256,
        "taxonomy.json": manifest.taxonomy_sha256,
        "ip102-detection-class-map.json": manifest.detection_mapping_sha256,
        "preprocessing.json": manifest.preprocessing_sha256,
        "postprocessing.json": manifest.postprocessing_sha256,
    }
    if any(
        not SHA256_PATTERN.fullmatch(expected)
        or sha256_file(bundle / name) != expected
        for name, expected in hash_contract.items()
    ):
        raise _error("contract_hash_mismatch", "联合模型包关键文件哈希与 manifest 不一致。")
    preprocessing = _load_json(bundle / "preprocessing.json")
    if preprocessing.get("resize") != [224, 224] or preprocessing.get("crop") != "none":
        raise _error("preprocessing_mismatch", "联合模型包不是统一 224 直缩预处理。")
    postprocessing = _load_json(bundle / "postprocessing.json")
    expected_postprocessing = {
        "classification_confidence_threshold": 0.55,
        "detection_score_threshold": 0.5,
        "detection_ap_score_threshold": 0.05,
        "detection_nms_iou_threshold": 0.5,
        "detection_max_detections_per_image": 100,
    }
    if any(
        postprocessing.get(name) != value
        for name, value in expected_postprocessing.items()
    ):
        raise _error("postprocessing_invalid", "联合模型包后处理合同已改变。")
    try:
        Taxonomy(bundle / "taxonomy.json")
        mapping = DetectionClassMapping(bundle / "ip102-detection-class-map.json")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _error("metadata_invalid", "联合模型包类别元数据无效。") from exc
    if mapping.num_detector_classes != 96:
        raise _error("mapping_invalid", "联合模型包检测映射不是 96 类。")
    return manifest
