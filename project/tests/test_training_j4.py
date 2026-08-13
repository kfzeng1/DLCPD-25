from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torchvision
from dlcpd25_classifier.inference import (
    BundleValidationError,
    load_joint_bundle_manifest,
)
from dlcpd25_classifier.training.j4 import (
    claim_test_once,
    complete_receipt,
    evaluate_classification,
    sha256_file,
    validate_config,
    write_checksums,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY = ROOT / "artifacts/data/v1/d5-r1/taxonomy-v1.json"
MAPPING = ROOT / "metadata/ip102-detection-class-map.json"


class CountingClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward_classification(self, images: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        logits = torch.zeros((images.shape[0], 203), device=images.device)
        logits[:, 0] = 4.0
        return logits


def _postprocessing() -> dict[str, float | int]:
    return {
        "classification_confidence_threshold": 0.55,
        "detection_score_threshold": 0.5,
        "detection_ap_score_threshold": 0.05,
        "detection_nms_iou_threshold": 0.5,
        "detection_max_detections_per_image": 100,
    }


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "joint-best.pt").write_bytes(b"one joint checkpoint")
    (bundle / "resolved-config.yaml").write_text("stage: J4\n", encoding="utf-8")
    (bundle / "taxonomy.json").write_bytes(TAXONOMY.read_bytes())
    (bundle / "ip102-detection-class-map.json").write_bytes(MAPPING.read_bytes())
    preprocessing = {
        "color_mode": "RGB",
        "image_size": 224,
        "resize": [224, 224],
        "crop": "none",
        "preserve_aspect_ratio": False,
        "interpolation": "bicubic",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    for name, payload in (
        ("preprocessing.json", preprocessing),
        ("postprocessing.json", _postprocessing()),
        ("metrics-classification.json", {"accuracy": 0.9}),
        ("metrics-detection.json", {"map": 0.4}),
    ):
        (bundle / name).write_text(json.dumps(payload), encoding="utf-8")
    (bundle / "model-card.md").write_text("# Joint model\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "model_version": "joint-v1",
        "git_commit": "a" * 40,
        "architecture": "joint-resnet50-fasterrcnn",
        "image_size": 224,
        "classification_classes": 203,
        "detection_classes": 96,
        "shared_body_forwards_per_joint_call": 1,
        "checkpoint_sha256": sha256_file(bundle / "joint-best.pt"),
        "taxonomy_sha256": sha256_file(bundle / "taxonomy.json"),
        "detection_mapping_sha256": sha256_file(
            bundle / "ip102-detection-class-map.json"
        ),
        "preprocessing_sha256": sha256_file(bundle / "preprocessing.json"),
        "postprocessing_sha256": sha256_file(bundle / "postprocessing.json"),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_checksums(bundle)
    return bundle


def _refresh_contract_hash(bundle: Path, field: str, filename: str) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest[field] = sha256_file(bundle / filename)
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_checksums(bundle)


def test_classification_test_is_one_inference_pass() -> None:
    images = torch.zeros((6, 3, 8, 8))
    targets = torch.zeros(6, dtype=torch.int64)
    loader = DataLoader(TensorDataset(images, targets), batch_size=2)
    model = CountingClassifier()

    metrics, per_class, confusion = evaluate_classification(
        model, loader, torch.device("cpu"), 0.55
    )

    assert model.calls == len(loader)
    assert metrics["inference_passes"] == 1
    assert metrics["samples"] == 6
    assert metrics["accuracy"] == 1.0
    assert len(per_class) == 203
    assert confusion[0][0] == 6


def test_receipt_is_atomic_and_rejects_second_test_read(tmp_path: Path) -> None:
    receipt_path = tmp_path / "metadata/j4-joint-test-evaluation.json"
    frozen = {
        "model_version": "joint-v1",
        "git_commit": "a" * 40,
        "input_sha256": {
            "classification_test_csv": "b" * 64,
            "detection_test_split": "c" * 64,
            "selected_checkpoint": "d" * 64,
        },
    }
    receipt = claim_test_once(receipt_path, frozen, "e" * 64)
    complete_receipt(
        receipt_path,
        receipt,
        checksums_sha256="f" * 64,
        classification_metrics_sha256="1" * 64,
        detection_metrics_sha256="2" * 64,
    )
    consumed = receipt_path.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to read J4 test again"):
        claim_test_once(receipt_path, frozen, "e" * 64)

    assert receipt_path.read_bytes() == consumed


def test_config_contract_is_frozen() -> None:
    config = {
        "model": {
            "architecture": "joint-resnet50-fasterrcnn",
            "image_size": 224,
            "classification_classes": 203,
            "detection_classes": 96,
        },
        "evaluation": {
            **_postprocessing(),
            "detection_score_threshold_for_ap": 0.05,
            "detection_score_threshold_for_precision_recall": 0.5,
            "test_evaluation_budget": 1,
        },
    }
    config["evaluation"].pop("detection_score_threshold")
    config["evaluation"].pop("detection_ap_score_threshold")
    validate_config(config)
    config["model"]["image_size"] = 640
    with pytest.raises(ValueError, match="224"):
        validate_config(config)


def test_joint_bundle_has_one_weight_and_fixed_contract(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    manifest = load_joint_bundle_manifest(bundle)
    assert manifest.image_size == 224
    assert manifest.classification_classes == 203
    assert manifest.detection_classes == 96

    (bundle / "second.pt").write_bytes(b"forbidden")
    with pytest.raises(BundleValidationError) as error:
        load_joint_bundle_manifest(bundle)
    assert error.value.code == "weight_contract_invalid"


@pytest.mark.parametrize(
    ("filename", "field", "payload", "error_code"),
    [
        (
            "postprocessing.json",
            "postprocessing_sha256",
            {**_postprocessing(), "detection_score_threshold": 0.9},
            "postprocessing_invalid",
        ),
        ("taxonomy.json", "taxonomy_sha256", {"classes": []}, "metadata_invalid"),
        (
            "ip102-detection-class-map.json",
            "detection_mapping_sha256",
            {"classes": []},
            "metadata_invalid",
        ),
    ],
)
def test_joint_bundle_rejects_changed_contract_files(
    tmp_path: Path,
    filename: str,
    field: str,
    payload: dict[str, object],
    error_code: str,
) -> None:
    bundle = _bundle(tmp_path)
    (bundle / filename).write_text(json.dumps(payload), encoding="utf-8")
    _refresh_contract_hash(bundle, field, filename)

    with pytest.raises(BundleValidationError) as error:
        load_joint_bundle_manifest(bundle)

    assert error.value.code == error_code


def test_joint_bundle_rejects_tampered_checkpoint(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "joint-best.pt").write_bytes(b"tampered")

    with pytest.raises(BundleValidationError) as error:
        load_joint_bundle_manifest(bundle)

    assert error.value.code == "checksum_mismatch"
