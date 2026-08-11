from __future__ import annotations

import hashlib
import io
import json
import shutil
from pathlib import Path

import pytest
import torch
from dlcpd25_classifier.inference import (
    AppSettings,
    BundleValidationError,
    FixedLogitsBackend,
    ImageLimits,
    ImageValidationError,
    PredictionError,
    Predictor,
    create_fake_predictor,
    load_bundle_manifest,
    load_rgb_image,
)
from dlcpd25_classifier.taxonomy import Taxonomy
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = ROOT / "metadata" / "class-taxonomy.json"
APP_CONFIG = ROOT / "project" / "configs" / "app.yaml"


def _image_bytes(mode: str = "RGB", size: tuple[int, int] = (32, 20), image_format: str = "PNG") -> bytes:
    color = 128 if mode == "L" else (30, 120, 60, 140) if mode == "RGBA" else (30, 120, 60)
    image = Image.new(mode, size, color)
    payload = io.BytesIO()
    image.save(payload, format=image_format)
    return payload.getvalue()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "model-v1"
    bundle.mkdir()
    shutil.copyfile(TAXONOMY_PATH, bundle / "taxonomy.json")
    (bundle / "best.pt").write_bytes(b"p1-placeholder")
    (bundle / "resolved-config.yaml").write_text("architecture: resnet50\n", encoding="utf-8")
    preprocessing = {
        "color_mode": "RGB",
        "image_size": 224,
        "eval_resize": 256,
        "eval_crop": "center",
        "interpolation": "bicubic",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    (bundle / "preprocessing.json").write_text(
        json.dumps(preprocessing, sort_keys=True), encoding="utf-8"
    )
    (bundle / "metrics.json").write_text("{}\n", encoding="utf-8")
    (bundle / "model-card.md").write_text("# Test model\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "model_version": "model-v1",
        "data_version": "data-v1-d5-r1",
        "git_commit": "a" * 40,
        "architecture": "resnet50",
        "num_classes": 203,
        "taxonomy_sha256": _sha256(bundle / "taxonomy.json"),
        "preprocessing_sha256": _sha256(bundle / "preprocessing.json"),
        "confidence_threshold": 0.55,
        "image_size": 224,
        "color_mode": "RGB",
        "resize": 256,
        "crop": "center",
        "interpolation": "bicubic",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "torch_version": "2.11.0",
        "torchvision_version": "0.26.0",
    }
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    checked_files = [
        "best.pt",
        "manifest.json",
        "resolved-config.yaml",
        "preprocessing.json",
        "taxonomy.json",
        "metrics.json",
        "model-card.md",
    ]
    (bundle / "checksums.sha256").write_text(
        "".join(f"{_sha256(bundle / name)}  {name}\n" for name in checked_files),
        encoding="utf-8",
    )
    return bundle


def _rewrite_checksums(bundle: Path) -> None:
    checked_files = [
        "best.pt",
        "manifest.json",
        "resolved-config.yaml",
        "preprocessing.json",
        "taxonomy.json",
        "metrics.json",
        "model-card.md",
    ]
    (bundle / "checksums.sha256").write_text(
        "".join(f"{_sha256(bundle / name)}  {name}\n" for name in checked_files),
        encoding="utf-8",
    )


def test_app_settings_resolve_repository_paths_and_limits() -> None:
    settings = AppSettings.from_yaml(APP_CONFIG)
    assert settings.mode == "bundle"
    assert settings.model_bundle == (
        ROOT / "artifacts" / "releases" / "dlcpd25-resnet50-weighted-v1"
    )
    assert settings.taxonomy_path is None
    assert settings.image_size == 224
    assert settings.top_k == 5
    assert len(settings.config_sha256) == 64


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_grayscale_and_rgba_are_converted_to_rgb(mode: str) -> None:
    image = load_rgb_image(_image_bytes(mode))
    assert image.mode == "RGB"
    assert image.size == (32, 20)


def test_exif_orientation_is_applied() -> None:
    image = Image.new("RGB", (40, 20), (20, 80, 140))
    exif = Image.Exif()
    exif[274] = 6
    payload = io.BytesIO()
    image.save(payload, format="JPEG", exif=exif)
    oriented = load_rgb_image(payload.getvalue())
    assert oriented.mode == "RGB"
    assert oriented.size == (20, 40)


def test_invalid_oversized_and_unknown_images_have_stable_errors(tmp_path: Path) -> None:
    with pytest.raises(ImageValidationError) as corrupted:
        load_rgb_image(b"not an image")
    assert corrupted.value.code == "decode_failed"

    with pytest.raises(ImageValidationError) as too_many_bytes:
        load_rgb_image(_image_bytes(), ImageLimits(max_upload_bytes=10))
    assert too_many_bytes.value.code == "file_too_large"

    with pytest.raises(ImageValidationError) as too_many_pixels:
        load_rgb_image(_image_bytes(size=(12, 12)), ImageLimits(max_image_pixels=100))
    assert too_many_pixels.value.code == "pixel_limit_exceeded"

    unknown = tmp_path / "image.unknown"
    unknown.write_bytes(_image_bytes())
    with pytest.raises(ImageValidationError) as extension:
        load_rgb_image(unknown)
    assert extension.value.code == "unsupported_extension"


def test_fake_predictor_returns_stable_hierarchical_top_five() -> None:
    predictor = create_fake_predictor(TAXONOMY_PATH, class_id=178)
    result = predictor.predict(_image_bytes(mode="L"))
    assert result.schema_version == 1
    assert result.class_id == 178
    assert result.official_name == "tomato bacterial spot"
    assert result.host_zh == "番茄"
    assert result.category_zh == "植物病害"
    assert result.low_confidence is False
    assert len(result.top_k) == 5
    assert [item.rank for item in result.top_k] == [1, 2, 3, 4, 5]
    assert result.top_k[0].class_id == result.class_id
    assert result.inference_ms >= 0.0


def test_low_confidence_and_equal_logit_order_are_deterministic() -> None:
    predictor = Predictor(
        taxonomy=Taxonomy(TAXONOMY_PATH),
        backend=FixedLogitsBackend([0.0] * 203),
        model_version="test",
        data_version="data-v1",
        config_sha256="a" * 64,
        git_commit="b" * 40,
    )
    result = predictor.predict(_image_bytes())
    assert result.low_confidence is True
    assert [item.class_id for item in result.top_k] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "output,code",
    [
        (torch.zeros(1, 202), "invalid_output_shape"),
        (torch.full((1, 203), float("nan")), "invalid_output_values"),
    ],
)
def test_predictor_rejects_invalid_backend_outputs(output: torch.Tensor, code: str) -> None:
    predictor = Predictor(
        taxonomy=Taxonomy(TAXONOMY_PATH),
        backend=lambda batch: output,
        model_version="test",
        data_version="data-v1",
        config_sha256="a" * 64,
        git_commit="b" * 40,
    )
    with pytest.raises(PredictionError) as error:
        predictor.predict(_image_bytes())
    assert error.value.code == code


def test_bundle_contract_validates_files_taxonomy_and_checksums(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    manifest = load_bundle_manifest(bundle)
    assert manifest.num_classes == 203
    assert manifest.color_mode == "RGB"

    (bundle / "metrics.json").write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(BundleValidationError) as checksum:
        load_bundle_manifest(bundle)
    assert checksum.value.code == "checksum_mismatch"


def test_bundle_rejects_preprocessing_that_disagrees_with_manifest(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    preprocessing_path = bundle / "preprocessing.json"
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    preprocessing["image_size"] = 299
    preprocessing_path.write_text(json.dumps(preprocessing, sort_keys=True), encoding="utf-8")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["preprocessing_sha256"] = _sha256(preprocessing_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    _rewrite_checksums(bundle)
    with pytest.raises(BundleValidationError) as mismatch:
        load_bundle_manifest(bundle)
    assert mismatch.value.code == "preprocessing_mismatch"


def test_missing_model_bundle_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(BundleValidationError) as missing:
        Predictor.from_bundle(tmp_path / "missing")
    assert missing.value.code == "bundle_missing"
