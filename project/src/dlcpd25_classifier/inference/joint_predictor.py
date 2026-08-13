"""Single-call classification and detection inference for the J4 joint bundle."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Protocol

import torch
import torchvision
from PIL import Image, ImageDraw
from torch import Tensor, nn

from dlcpd25_classifier.detection import (
    DetectionClassMapping,
    build_empty_shared_detection_model,
)
from dlcpd25_classifier.taxonomy import Taxonomy
from dlcpd25_classifier.training.transforms import build_direct_resize_eval_transform

from .errors import BundleValidationError, PredictionError
from .images import ImageLimits, ImageSource, load_rgb_image
from .joint_bundle import JointBundleManifest, load_joint_bundle_manifest
from .predictor import TopKResult

JOINT_PREDICTION_SCHEMA_VERSION = 1


class JointBackend(Protocol):
    def __call__(self, batch: Tensor) -> tuple[Tensor, list[dict[str, Tensor]]]: ...


class TorchJointBackend:
    """Execute one frozen joint model on one selected device."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model.eval()
        self.device = device

    def warmup(self, image_size: int) -> None:
        with torch.inference_mode():
            logits, detections = self.model.forward_joint(
                torch.zeros(1, 3, image_size, image_size, device=self.device)
            )
        _validate_outputs(logits, detections)

    def __call__(self, batch: Tensor) -> tuple[Tensor, list[dict[str, Tensor]]]:
        logits, detections = self.model.forward_joint(
            batch.to(self.device, non_blocking=self.device.type == "cuda")
        )
        return logits.cpu(), [
            {name: value.cpu() for name, value in prediction.items()}
            for prediction in detections
        ]


@dataclass(frozen=True)
class DetectionResult:
    class_id: int
    official_name: str
    host_zh: str
    category_zh: str
    score: float
    box_xyxy_original: tuple[float, float, float, float]


@dataclass(frozen=True)
class JointPredictionResult:
    schema_version: int
    model_version: str
    config_sha256: str
    git_commit: str
    device: str
    class_id: int
    official_name: str
    host_zh: str
    category_zh: str
    detail_name: str
    confidence: float
    top_k: tuple[TopKResult, ...]
    low_confidence: bool
    detections: tuple[DetectionResult, ...]
    inference_ms: float
    original_size: tuple[int, int]
    annotated_image: Image.Image = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("annotated_image")
        payload["classification"] = {
            key: payload.pop(key)
            for key in (
                "class_id",
                "official_name",
                "host_zh",
                "category_zh",
                "detail_name",
                "confidence",
                "top_k",
                "low_confidence",
            )
        }
        return payload


def _validate_outputs(
    logits: Tensor, detections: list[dict[str, Tensor]]
) -> None:
    if logits.shape != (1, 203) or not torch.isfinite(logits).all():
        raise RuntimeError("joint classification output is invalid")
    if len(detections) != 1 or set(detections[0]) != {"boxes", "labels", "scores"}:
        raise RuntimeError("joint detection output is invalid")
    prediction = detections[0]
    count = len(prediction["boxes"])
    if (
        prediction["boxes"].shape != (count, 4)
        or prediction["labels"].shape != (count,)
        or prediction["scores"].shape != (count,)
        or not all(torch.isfinite(value).all() for value in prediction.values())
    ):
        raise RuntimeError("joint detection tensors are invalid")


def _runtime_contract(manifest: JointBundleManifest, image_size: int) -> None:
    if image_size != manifest.image_size:
        raise BundleValidationError(
            "preprocessing_mismatch", "应用配置与联合模型输入尺寸不一致。"
        )
    if manifest.torch_version != torch.__version__:
        raise BundleValidationError(
            "dependency_mismatch",
            f"PyTorch 版本不匹配：模型包要求 {manifest.torch_version}。",
        )
    if manifest.torchvision_version != torchvision.__version__:
        raise BundleValidationError(
            "dependency_mismatch",
            f"torchvision 版本不匹配：模型包要求 {manifest.torchvision_version}。",
        )


def _load_backend(
    bundle: Path,
    manifest: JointBundleManifest,
    mapping: DetectionClassMapping,
    device: torch.device,
    score_threshold: float,
    nms_threshold: float,
    max_detections: int,
) -> TorchJointBackend:
    model, _ = build_empty_shared_detection_model(mapping)
    checkpoint = torch.load(
        bundle / "joint-best.pt", map_location="cpu", weights_only=False
    )
    expected = {
        "architecture": "joint-resnet50-fasterrcnn",
        "classification_classes": 203,
        "detection_classes": 96,
        "image_size": 224,
    }
    if any(checkpoint.get(name) != value for name, value in expected.items()):
        raise ValueError("joint checkpoint architecture metadata is invalid")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.detector.roi_heads.score_thresh = score_threshold
    model.detector.roi_heads.nms_thresh = nms_threshold
    model.detector.roi_heads.detections_per_img = max_detections
    model.to(device)
    backend = TorchJointBackend(model, device)
    backend.warmup(manifest.image_size)
    return backend


def _select_backend(
    bundle: Path,
    manifest: JointBundleManifest,
    mapping: DetectionClassMapping,
    requested_device: str,
    score_threshold: float,
    nms_threshold: float,
    max_detections: int,
) -> tuple[str, TorchJointBackend]:
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise BundleValidationError("device_invalid", "推理设备必须是 auto、cpu 或 cuda。")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise BundleValidationError("device_unavailable", "CUDA 不可用，无法加载联合模型。")
    candidates = (
        ["cuda", "cpu"]
        if requested_device == "auto" and torch.cuda.is_available()
        else ["cpu" if requested_device == "auto" else requested_device]
    )
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            backend = _load_backend(
                bundle,
                manifest,
                mapping,
                torch.device(candidate),
                score_threshold,
                nms_threshold,
                max_detections,
            )
            return candidate, backend
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            last_error = exc
            if candidate != "cuda" or requested_device != "auto":
                break
            try:
                torch.cuda.empty_cache()
            except RuntimeError as cache_error:
                last_error = cache_error
    raise BundleValidationError(
        "model_load_failed", "联合模型加载或预热失败，请检查模型包和运行环境。"
    ) from last_error


class JointPredictor:
    def __init__(
        self,
        *,
        taxonomy: Taxonomy,
        mapping: DetectionClassMapping,
        backend: JointBackend,
        model_version: str,
        config_sha256: str,
        git_commit: str,
        device: str,
        image_size: int = 224,
        classification_confidence_threshold: float = 0.55,
        detection_score_threshold: float = 0.5,
        top_k: int = 5,
        image_limits: ImageLimits | None = None,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if image_size != 224 or not 1 <= top_k <= 203:
            raise ValueError("joint predictor requires image_size 224 and top_k within 1..203")
        if not 0 <= classification_confidence_threshold <= 1:
            raise ValueError("classification confidence threshold is invalid")
        if not 0 <= detection_score_threshold <= 1:
            raise ValueError("detection score threshold is invalid")
        self.taxonomy = taxonomy
        self.mapping = mapping
        self.backend = backend
        self.model_version = model_version
        self.config_sha256 = config_sha256
        self.git_commit = git_commit
        self.device = device
        self.image_size = image_size
        self.classification_confidence_threshold = classification_confidence_threshold
        self.detection_score_threshold = detection_score_threshold
        self.top_k = top_k
        self.image_limits = image_limits or ImageLimits()
        self._transform = build_direct_resize_eval_transform(image_size)

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path,
        device: str = "auto",
        *,
        top_k: int = 5,
        image_limits: ImageLimits | None = None,
        expected_image_size: int = 224,
        expected_classification_confidence_threshold: float = 0.55,
    ) -> JointPredictor:
        bundle = Path(bundle_path).resolve()
        manifest = load_joint_bundle_manifest(bundle)
        _runtime_contract(manifest, expected_image_size)
        postprocessing = json.loads(
            (bundle / "postprocessing.json").read_text(encoding="utf-8")
        )
        classification_threshold = float(
            postprocessing["classification_confidence_threshold"]
        )
        if classification_threshold != expected_classification_confidence_threshold:
            raise BundleValidationError(
                "threshold_mismatch", "应用配置与联合模型分类阈值不一致。"
            )
        mapping = DetectionClassMapping(bundle / "ip102-detection-class-map.json")
        selected_device, backend = _select_backend(
            bundle,
            manifest,
            mapping,
            device,
            float(postprocessing["detection_score_threshold"]),
            float(postprocessing["detection_nms_iou_threshold"]),
            int(postprocessing["detection_max_detections_per_image"]),
        )
        return cls(
            taxonomy=Taxonomy(bundle / "taxonomy.json"),
            mapping=mapping,
            backend=backend,
            model_version=manifest.model_version,
            config_sha256=sha256((bundle / "resolved-config.yaml").read_bytes()).hexdigest(),
            git_commit=manifest.git_commit,
            device=selected_device,
            image_size=manifest.image_size,
            classification_confidence_threshold=classification_threshold,
            detection_score_threshold=float(
                postprocessing["detection_score_threshold"]
            ),
            top_k=top_k,
            image_limits=image_limits,
        )

    def predict(self, image: ImageSource) -> JointPredictionResult:
        started = time.perf_counter()
        rgb_image = load_rgb_image(image, self.image_limits)
        original_width, original_height = rgb_image.size
        batch = self._transform(rgb_image).unsqueeze(0)
        try:
            with torch.inference_mode():
                logits, predictions = self.backend(batch)
            _validate_outputs(logits, predictions)
        except Exception as exc:
            raise PredictionError("backend_failed", "图片推理失败，请稍后重试。") from exc

        probabilities = logits[0].softmax(dim=0).tolist()
        ranked_ids = sorted(
            range(203), key=lambda class_id: (-probabilities[class_id], class_id)
        )
        top_results: list[TopKResult] = []
        for rank, class_id in enumerate(ranked_ids[: self.top_k], 1):
            record = self.taxonomy.resolve(class_id)
            top_results.append(
                TopKResult(
                    rank=rank,
                    class_id=class_id,
                    official_name=record.official_name,
                    host_zh=record.host_zh,
                    category_zh=record.category_zh,
                    confidence=float(probabilities[class_id]),
                )
            )
        winner = top_results[0]

        scale_x = original_width / self.image_size
        scale_y = original_height / self.image_size
        detection_results: list[DetectionResult] = []
        prediction = predictions[0]
        for box, detector_label, score in zip(
            prediction["boxes"].tolist(),
            prediction["labels"].tolist(),
            prediction["scores"].tolist(),
        ):
            if score < self.detection_score_threshold:
                continue
            mapped = self.mapping.from_detector(int(detector_label))
            record = self.taxonomy.resolve(mapped.dlcpd25_class_id)
            scaled = (
                max(0.0, min(float(original_width), box[0] * scale_x)),
                max(0.0, min(float(original_height), box[1] * scale_y)),
                max(0.0, min(float(original_width), box[2] * scale_x)),
                max(0.0, min(float(original_height), box[3] * scale_y)),
            )
            detection_results.append(
                DetectionResult(
                    class_id=record.class_id,
                    official_name=record.official_name,
                    host_zh=record.host_zh,
                    category_zh=record.category_zh,
                    score=float(score),
                    box_xyxy_original=scaled,
                )
            )
        annotated = rgb_image.copy()
        draw = ImageDraw.Draw(annotated)
        line_width = max(2, round(min(original_width, original_height) / 180))
        for item in detection_results:
            draw.rectangle(item.box_xyxy_original, outline="#e53935", width=line_width)
            x1, y1, _, _ = item.box_xyxy_original
            draw.text(
                (x1 + line_width, y1 + line_width),
                f"ID {item.class_id}  {item.score:.2f}",
                fill="#ffffff",
                stroke_width=max(1, line_width // 2),
                stroke_fill="#111111",
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        if not math.isfinite(elapsed_ms):
            raise PredictionError("invalid_timing", "推理耗时记录无效。")
        return JointPredictionResult(
            schema_version=JOINT_PREDICTION_SCHEMA_VERSION,
            model_version=self.model_version,
            config_sha256=self.config_sha256,
            git_commit=self.git_commit,
            device=self.device,
            class_id=winner.class_id,
            official_name=winner.official_name,
            host_zh=winner.host_zh,
            category_zh=winner.category_zh,
            detail_name=winner.official_name,
            confidence=winner.confidence,
            top_k=tuple(top_results),
            low_confidence=(
                winner.confidence < self.classification_confidence_threshold
            ),
            detections=tuple(detection_results),
            inference_ms=round(elapsed_ms, 3),
            original_size=(original_width, original_height),
            annotated_image=annotated,
        )
