"""Frozen model bundle manifest and checksum validation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dlcpd25_classifier.taxonomy import Taxonomy

from .errors import BundleValidationError

BUNDLE_SCHEMA_VERSION = 1
REQUIRED_FILES = frozenset(
    {
        "best.pt",
        "manifest.json",
        "resolved-config.yaml",
        "preprocessing.json",
        "taxonomy.json",
        "metrics.json",
        "model-card.md",
    }
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class BundleManifest:
    schema_version: int
    model_version: str
    data_version: str
    git_commit: str
    architecture: str
    num_classes: int
    taxonomy_sha256: str
    preprocessing_sha256: str
    confidence_threshold: float
    image_size: int
    color_mode: str
    resize: int
    crop: str
    interpolation: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    torch_version: str
    torchvision_version: str


def _bundle_error(code: str, message: str) -> BundleValidationError:
    return BundleValidationError(code, message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(payload: dict[str, Any], name: str) -> Any:
    if name not in payload:
        raise _bundle_error("manifest_invalid", f"模型包清单缺少字段：{name}。")
    return payload[name]


def _triplet(payload: dict[str, Any], name: str) -> tuple[float, float, float]:
    value = _required(payload, name)
    if not isinstance(value, list) or len(value) != 3:
        raise _bundle_error("manifest_invalid", f"模型包字段 {name} 必须包含三个数值。")
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise _bundle_error("manifest_invalid", f"模型包字段 {name} 包含无效数值。") from exc


def _parse_manifest(path: Path) -> BundleManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _bundle_error("manifest_invalid", "模型包 manifest.json 无法读取或格式无效。") from exc
    if not isinstance(payload, dict):
        raise _bundle_error("manifest_invalid", "模型包 manifest.json 必须是 JSON 对象。")
    try:
        manifest = BundleManifest(
            schema_version=int(_required(payload, "schema_version")),
            model_version=str(_required(payload, "model_version")),
            data_version=str(_required(payload, "data_version")),
            git_commit=str(_required(payload, "git_commit")),
            architecture=str(_required(payload, "architecture")),
            num_classes=int(_required(payload, "num_classes")),
            taxonomy_sha256=str(_required(payload, "taxonomy_sha256")),
            preprocessing_sha256=str(_required(payload, "preprocessing_sha256")),
            confidence_threshold=float(_required(payload, "confidence_threshold")),
            image_size=int(_required(payload, "image_size")),
            color_mode=str(_required(payload, "color_mode")),
            resize=int(_required(payload, "resize")),
            crop=str(_required(payload, "crop")),
            interpolation=str(_required(payload, "interpolation")),
            mean=_triplet(payload, "mean"),
            std=_triplet(payload, "std"),
            torch_version=str(_required(payload, "torch_version")),
            torchvision_version=str(_required(payload, "torchvision_version")),
        )
    except (TypeError, ValueError) as exc:
        raise _bundle_error("manifest_invalid", "模型包 manifest.json 字段类型无效。") from exc
    if manifest.schema_version != BUNDLE_SCHEMA_VERSION:
        raise _bundle_error("schema_mismatch", "模型包 schema 版本不受支持。")
    if manifest.num_classes != 203:
        raise _bundle_error("class_count_mismatch", "模型包类别数量不是 203，已拒绝加载。")
    if manifest.color_mode != "RGB":
        raise _bundle_error("preprocessing_mismatch", "模型包颜色模式不是 RGB，已拒绝加载。")
    if manifest.image_size <= 0 or manifest.resize < manifest.image_size:
        raise _bundle_error("preprocessing_mismatch", "模型包图像尺寸配置无效。")
    if manifest.crop != "center" or manifest.interpolation != "bicubic":
        raise _bundle_error("preprocessing_mismatch", "模型包预处理裁剪或插值配置不受支持。")
    if not 0.0 <= manifest.confidence_threshold <= 1.0:
        raise _bundle_error("threshold_invalid", "模型包置信度阈值无效。")
    if not SHA256_PATTERN.fullmatch(manifest.taxonomy_sha256):
        raise _bundle_error("manifest_invalid", "模型包 taxonomy SHA-256 格式无效。")
    if not SHA256_PATTERN.fullmatch(manifest.preprocessing_sha256):
        raise _bundle_error("manifest_invalid", "模型包 preprocessing SHA-256 格式无效。")
    for name in ("model_version", "data_version", "git_commit", "architecture"):
        if not getattr(manifest, name).strip():
            raise _bundle_error("manifest_invalid", f"模型包字段 {name} 不能为空。")
    return manifest


def _parse_checksums(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise _bundle_error("checksum_invalid", "模型包 checksums.sha256 无法读取。") from exc
    checksums: dict[str, str] = {}
    for line in lines:
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise _bundle_error("checksum_invalid", "模型包 checksums.sha256 格式无效。")
        digest, raw_name = parts
        name = raw_name.removeprefix("*")
        candidate = Path(name)
        if (
            not SHA256_PATTERN.fullmatch(digest)
            or candidate.is_absolute()
            or ".." in candidate.parts
            or name in checksums
        ):
            raise _bundle_error("checksum_invalid", "模型包 checksums.sha256 包含无效条目。")
        checksums[name] = digest
    return checksums


def _validate_preprocessing(path: Path, manifest: BundleManifest) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _bundle_error(
            "preprocessing_invalid", "模型包 preprocessing.json 无法读取或格式无效。"
        ) from exc
    if not isinstance(payload, dict):
        raise _bundle_error("preprocessing_invalid", "模型包 preprocessing.json 必须是 JSON 对象。")
    expected: dict[str, object] = {
        "color_mode": manifest.color_mode,
        "image_size": manifest.image_size,
        "eval_resize": manifest.resize,
        "eval_crop": manifest.crop,
        "interpolation": manifest.interpolation,
        "mean": list(manifest.mean),
        "std": list(manifest.std),
    }
    for name, expected_value in expected.items():
        if payload.get(name) != expected_value:
            raise _bundle_error(
                "preprocessing_mismatch",
                f"模型包预处理字段与 manifest 不一致：{name}。",
            )


def load_bundle_manifest(bundle_path: str | Path) -> BundleManifest:
    """Validate every frozen bundle file before any model deserialization."""
    bundle = Path(bundle_path)
    if not bundle.is_dir():
        raise _bundle_error("bundle_missing", "模型包目录不存在，请检查应用配置。")
    missing = sorted(name for name in REQUIRED_FILES | {"checksums.sha256"} if not (bundle / name).is_file())
    if missing:
        raise _bundle_error("bundle_incomplete", f"模型包不完整，缺少文件：{', '.join(missing)}。")

    checksums = _parse_checksums(bundle / "checksums.sha256")
    expected_entries = REQUIRED_FILES
    missing_checksums = sorted(expected_entries - checksums.keys())
    if missing_checksums:
        raise _bundle_error(
            "checksum_incomplete", f"模型包校验清单缺少文件：{', '.join(missing_checksums)}。"
        )
    for name, expected_digest in sorted(checksums.items()):
        target = bundle / name
        if not target.is_file() or not target.resolve().is_relative_to(bundle.resolve()):
            raise _bundle_error("checksum_invalid", f"模型包校验条目指向无效文件：{name}。")
        if _sha256(target) != expected_digest:
            raise _bundle_error("checksum_mismatch", f"模型包文件校验失败：{name}。")

    manifest = _parse_manifest(bundle / "manifest.json")
    if _sha256(bundle / "taxonomy.json") != manifest.taxonomy_sha256:
        raise _bundle_error("taxonomy_mismatch", "模型包 taxonomy 校验失败，已拒绝加载。")
    if _sha256(bundle / "preprocessing.json") != manifest.preprocessing_sha256:
        raise _bundle_error("preprocessing_mismatch", "模型包预处理配置校验失败，已拒绝加载。")
    _validate_preprocessing(bundle / "preprocessing.json", manifest)
    try:
        Taxonomy(bundle / "taxonomy.json")
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise _bundle_error("taxonomy_invalid", "模型包 taxonomy 不是有效的 203 类映射。") from exc
    return manifest
