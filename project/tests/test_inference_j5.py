from __future__ import annotations

from pathlib import Path
from typing import Any

import dlcpd25_classifier.inference.joint_predictor as joint_predictor_module
import pytest
import torch
from dlcpd25_classifier.detection import DetectionClassMapping
from dlcpd25_classifier.inference import (
    BundleValidationError,
    JointPredictor,
    PredictionError,
    load_joint_bundle_manifest,
)
from dlcpd25_classifier.taxonomy import Taxonomy
from dlcpd25_classifier.web import analyze_image, build_app
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY = ROOT / "artifacts/data/v1/d5-r1/taxonomy-v1.json"
MAPPING = ROOT / "metadata/ip102-detection-class-map.json"
BUNDLE = ROOT / "artifacts/releases/dlcpd25-ip102-joint-v1"


class FixedJointBackend:
    def __init__(
        self,
        *,
        winner: int = 0,
        boxes: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        scores: torch.Tensor | None = None,
    ) -> None:
        self.winner = winner
        self.boxes = boxes if boxes is not None else torch.empty((0, 4))
        self.labels = labels if labels is not None else torch.empty(0, dtype=torch.int64)
        self.scores = scores if scores is not None else torch.empty(0)
        self.calls = 0

    def __call__(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        self.calls += 1
        logits = torch.zeros((1, 203))
        logits[0, self.winner] = 8.0
        return logits, [
            {"boxes": self.boxes, "labels": self.labels, "scores": self.scores}
        ]


def _predictor(backend: FixedJointBackend) -> JointPredictor:
    return JointPredictor(
        taxonomy=Taxonomy(TAXONOMY),
        mapping=DetectionClassMapping(MAPPING),
        backend=backend,
        model_version="joint-test",
        config_sha256="a" * 64,
        git_commit="b" * 40,
        device="cpu",
    )


def test_one_backend_call_returns_classification_and_original_boxes() -> None:
    backend = FixedJointBackend(
        winner=178,
        boxes=torch.tensor([[22.4, 44.8, 112.0, 179.2]]),
        labels=torch.tensor([1]),
        scores=torch.tensor([0.8]),
    )
    predictor = _predictor(backend)

    result = predictor.predict(Image.new("RGB", (400, 200), "white"))

    mapped = predictor.mapping.from_detector(1)
    assert backend.calls == 1
    assert result.class_id == 178
    assert len(result.top_k) == 5
    assert len(result.detections) == 1
    assert result.detections[0].class_id == mapped.dlcpd25_class_id
    assert result.detections[0].box_xyxy_original == pytest.approx(
        (40.0, 40.0, 200.0, 160.0)
    )
    assert result.annotated_image.size == (400, 200)
    assert result.to_dict()["classification"]["class_id"] == 178


def test_low_score_detection_is_not_exposed() -> None:
    backend = FixedJointBackend(
        boxes=torch.tensor([[0.0, 0.0, 224.0, 224.0]]),
        labels=torch.tensor([1]),
        scores=torch.tensor([0.49]),
    )
    result = _predictor(backend).predict(Image.new("RGB", (80, 60)))
    assert result.detections == ()


def test_invalid_joint_outputs_are_public_prediction_errors() -> None:
    backend = FixedJointBackend()
    backend.boxes = torch.zeros((1, 5))
    with pytest.raises(PredictionError) as error:
        _predictor(backend).predict(Image.new("RGB", (80, 60)))
    assert error.value.code == "backend_failed"


def test_joint_web_adapter_and_components() -> None:
    predictor = _predictor(FixedJointBackend(winner=178))
    output = analyze_image(Image.new("RGB", (80, 60)), predictor)
    assert len(output) == 11
    assert output[0].size == (80, 60)
    assert output[2:5] == ("番茄", "植物病害", "tomato bacterial spot")
    assert "未发现" in output[1]
    assert len(output[6]) == 5

    app = build_app(predictor)
    config = app.get_config_file()
    labels = {component.get("props", {}).get("label") for component in config["components"]}
    assert {"待分析图片", "害虫检测结果", "分类 Top-5", "检测明细"} <= labels


def test_joint_web_returns_stable_error_for_broken_image() -> None:
    output = analyze_image(b"broken", _predictor(FixedJointBackend()))
    assert output[1] == "图片已损坏或无法识别，请重新选择有效图片。"
    assert output[0] is None


def test_auto_device_falls_back_to_cpu_after_cuda_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_joint_bundle_manifest(BUNDLE)
    mapping = DetectionClassMapping(MAPPING)
    calls: list[str] = []
    backend = object()

    def fake_loader(
        _bundle: Path,
        _manifest: Any,
        _mapping: Any,
        device: torch.device,
        _score: float,
        _nms: float,
        _maximum: int,
    ) -> object:
        calls.append(device.type)
        if device.type == "cuda":
            raise RuntimeError("simulated CUDA failure")
        return backend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(joint_predictor_module, "_load_backend", fake_loader)
    selected, loaded = joint_predictor_module._select_backend(
        BUNDLE, manifest, mapping, "auto", 0.5, 0.5, 100
    )
    assert selected == "cpu"
    assert loaded is backend
    assert calls == ["cuda", "cpu"]


def test_explicit_cuda_unavailable_has_stable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_joint_bundle_manifest(BUNDLE)
    mapping = DetectionClassMapping(MAPPING)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(BundleValidationError) as error:
        joint_predictor_module._select_backend(
            BUNDLE, manifest, mapping, "cuda", 0.5, 0.5, 100
        )
    assert error.value.code == "device_unavailable"
