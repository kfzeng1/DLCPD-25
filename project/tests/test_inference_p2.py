from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dlcpd25_classifier.inference.predictor as predictor_module
import pytest
import torch
from dlcpd25_classifier.inference import AppSettings, BundleValidationError, Predictor
from dlcpd25_classifier.inference.bundle import load_bundle_manifest
from dlcpd25_classifier.web import classify_image

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "artifacts" / "releases" / "dlcpd25-resnet50-weighted-v1"
DATA_ROOT = ROOT / "data" / "raw" / "dlcpd25-203"
REFERENCE_PATH = BUNDLE / "fixed-sample-predictions.json"
APP_CONFIG = ROOT / "project" / "configs" / "app.yaml"


@pytest.fixture(scope="session")
def reference_predictions() -> list[dict[str, Any]]:
    payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and len(payload) == 3
    return payload


@pytest.fixture(scope="session")
def cpu_predictor() -> Predictor:
    return Predictor.from_bundle(BUNDLE, device="cpu")


@pytest.fixture(scope="session")
def cpu_results(
    cpu_predictor: Predictor, reference_predictions: list[dict[str, Any]]
) -> list[Any]:
    return [
        cpu_predictor.predict(DATA_ROOT / reference["relative_path"])
        for reference in reference_predictions
    ]


def test_accepted_bundle_has_all_checksums_and_frozen_runtime_contract() -> None:
    manifest = load_bundle_manifest(BUNDLE)
    checksum_lines = (BUNDLE / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    settings = AppSettings.from_yaml(APP_CONFIG)
    assert len(checksum_lines) == 13
    assert manifest.model_version == "dlcpd25-resnet50-weighted-v1"
    assert manifest.data_version == "data-v1-d5-r1"
    assert manifest.num_classes == 203
    assert manifest.confidence_threshold == settings.confidence_threshold == 0.55
    assert manifest.image_size == settings.image_size == 224


def test_cpu_predictor_matches_all_fixed_validation_samples(
    cpu_predictor: Predictor,
    cpu_results: list[Any],
    reference_predictions: list[dict[str, Any]],
) -> None:
    assert cpu_predictor.device == "cpu"
    assert cpu_predictor.model_version == "dlcpd25-resnet50-weighted-v1"
    for result, reference in zip(cpu_results, reference_predictions, strict=True):
        assert [item.class_id for item in result.top_k] == reference["top5_class_ids"]
        assert [item.confidence for item in result.top_k] == pytest.approx(
            reference["top5_confidences"], abs=2e-7
        )


def test_real_low_confidence_result_is_explicit_in_web_output(
    cpu_predictor: Predictor, cpu_results: list[Any]
) -> None:
    result = cpu_results[1]
    assert result.class_id == 131
    assert result.confidence == pytest.approx(0.20271974802017212, abs=1e-7)
    assert result.low_confidence is True
    output = classify_image(
        DATA_ROOT
        / "orange huanglongbing 柑橘黄龙病(黄龙病)"
        / "7d7b6586-62a6-49b3-ae46-917b89ba9b07___CREC_HLB 3977.JPG",
        cpu_predictor,
    )
    assert "低置信度" in output[0]
    assert "结果不确定" in output[0]
def test_auto_device_runs_one_reference_sample(reference_predictions: list[dict[str, Any]]) -> None:
    predictor = Predictor.from_bundle(BUNDLE, device="auto")
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert predictor.device == expected_device
    reference = reference_predictions[2]
    result = predictor.predict(DATA_ROOT / reference["relative_path"])
    assert [item.class_id for item in result.top_k] == reference["top5_class_ids"]
    assert [item.confidence for item in result.top_k] == pytest.approx(
        reference["top5_confidences"], abs=1e-3
    )


def test_auto_device_falls_back_to_cpu_after_cuda_warmup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_bundle_manifest(BUNDLE)
    calls: list[str] = []
    backend = object()

    def fake_loader(_bundle: Path, _manifest: Any, device: torch.device) -> object:
        calls.append(device.type)
        if device.type == "cuda":
            raise RuntimeError("simulated CUDA failure")
        return backend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(predictor_module, "_load_torch_backend", fake_loader)
    selected, loaded = predictor_module._select_backend(BUNDLE, manifest, "auto")
    assert selected == "cpu"
    assert loaded is backend
    assert calls == ["cuda", "cpu"]


def test_explicit_cuda_has_clear_error_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = load_bundle_manifest(BUNDLE)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(BundleValidationError) as error:
        predictor_module._select_backend(BUNDLE, manifest, "cuda")
    assert error.value.code == "device_unavailable"
