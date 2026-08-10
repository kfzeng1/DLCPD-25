"""Predictor contract shared by the fake P1 backend and future model bundles."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch
from PIL import Image
from torch import Tensor

from dlcpd25_classifier.taxonomy import Taxonomy
from dlcpd25_classifier.training.transforms import build_eval_transform

from .bundle import load_bundle_manifest
from .errors import BundleValidationError, PredictionError
from .images import ImageLimits, ImageSource, load_rgb_image

PREDICTION_SCHEMA_VERSION = 1


class LogitsBackend(Protocol):
    def __call__(self, batch: Tensor) -> Tensor: ...


class FixedLogitsBackend:
    """Deterministic P1 backend; it does not inspect or diagnose the image."""

    def __init__(self, logits: Sequence[float]) -> None:
        tensor = torch.as_tensor(logits, dtype=torch.float32)
        if tensor.shape != (203,) or not torch.isfinite(tensor).all():
            raise ValueError("fixed logits must contain 203 finite values")
        self._logits = tensor

    def __call__(self, batch: Tensor) -> Tensor:
        if batch.ndim != 4 or batch.shape[1] != 3:
            raise ValueError("backend input must have shape [N,3,H,W]")
        return self._logits.to(batch.device).unsqueeze(0).expand(batch.shape[0], -1).clone()


@dataclass(frozen=True)
class TopKResult:
    rank: int
    class_id: int
    official_name: str
    host_zh: str
    category_zh: str
    confidence: float


@dataclass(frozen=True)
class PredictionResult:
    schema_version: int
    model_version: str
    data_version: str
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
    inference_ms: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class Predictor:
    def __init__(
        self,
        *,
        taxonomy: Taxonomy,
        backend: LogitsBackend,
        model_version: str,
        data_version: str,
        config_sha256: str,
        git_commit: str,
        device: str = "cpu",
        image_size: int = 224,
        confidence_threshold: float = 0.55,
        top_k: int = 5,
        image_limits: ImageLimits | None = None,
    ) -> None:
        if device != "cpu":
            raise ValueError("P1 fake Predictor only supports device='cpu'")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between zero and one")
        if not 1 <= top_k <= 203:
            raise ValueError("top_k must be between 1 and 203")
        for name, value in {
            "model_version": model_version,
            "data_version": data_version,
            "config_sha256": config_sha256,
            "git_commit": git_commit,
        }.items():
            if not value:
                raise ValueError(f"{name} must not be empty")
        self.taxonomy = taxonomy
        self.backend = backend
        self.model_version = model_version
        self.data_version = data_version
        self.config_sha256 = config_sha256
        self.git_commit = git_commit
        self.device = device
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.image_limits = image_limits or ImageLimits()
        self._transform = build_eval_transform(image_size)

    @classmethod
    def from_bundle(cls, bundle_path: str | Path, device: str = "auto") -> Predictor:
        load_bundle_manifest(bundle_path)
        raise BundleValidationError(
            "real_backend_unavailable",
            "模型包校验已通过，但真实模型加载需等待 P2 集成。",
        )

    def predict(self, image: ImageSource) -> PredictionResult:
        """Classify one image; inference_ms covers decode, transform, and backend execution."""
        started = time.perf_counter()
        rgb_image: Image.Image = load_rgb_image(image, self.image_limits)
        batch = self._transform(rgb_image).unsqueeze(0)
        try:
            with torch.inference_mode():
                logits = self.backend(batch)
        except Exception as exc:
            raise PredictionError("backend_failed", "图片推理失败，请稍后重试。") from exc
        if logits.shape != (1, 203):
            raise PredictionError("invalid_output_shape", "模型输出维度无效，已拒绝本次结果。")
        if not torch.isfinite(logits).all():
            raise PredictionError("invalid_output_values", "模型输出包含无效数值，已拒绝本次结果。")
        probabilities = logits[0].softmax(dim=0).cpu().tolist()
        ranked_ids = sorted(range(203), key=lambda class_id: (-probabilities[class_id], class_id))
        top_results: list[TopKResult] = []
        for rank, class_id in enumerate(ranked_ids[: self.top_k], start=1):
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
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not math.isfinite(elapsed_ms):
            raise PredictionError("invalid_timing", "推理耗时记录无效。")
        return PredictionResult(
            schema_version=PREDICTION_SCHEMA_VERSION,
            model_version=self.model_version,
            data_version=self.data_version,
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
            low_confidence=winner.confidence < self.confidence_threshold,
            inference_ms=round(elapsed_ms, 3),
        )


def create_fake_predictor(
    taxonomy_path: str | Path,
    *,
    class_id: int = 178,
    model_version: str = "p1-fixed-logits-v1",
    data_version: str = "data-v1-d5-r1",
    config_sha256: str = "development-config",
    git_commit: str = "development-worktree",
    image_size: int = 224,
    confidence_threshold: float = 0.55,
    top_k: int = 5,
    image_limits: ImageLimits | None = None,
) -> Predictor:
    if not 0 <= class_id < 203:
        raise ValueError("fake class_id must be between 0 and 202")
    logits = [0.0] * 203
    logits[class_id] = 8.0
    return Predictor(
        taxonomy=Taxonomy(Path(taxonomy_path)),
        backend=FixedLogitsBackend(logits),
        model_version=model_version,
        data_version=data_version,
        config_sha256=config_sha256,
        git_commit=git_commit,
        image_size=image_size,
        confidence_threshold=confidence_threshold,
        top_k=top_k,
        image_limits=image_limits,
    )
