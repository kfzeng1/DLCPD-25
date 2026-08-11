"""Application configuration loading with repository-stable path semantics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppSettings:
    mode: str
    model_bundle: Path
    taxonomy_path: Path | None
    device: str
    image_size: int
    confidence_threshold: float
    top_k: int
    max_upload_bytes: int
    max_image_pixels: int
    fake_class_id: int | None
    model_version: str | None
    data_version: str | None
    git_commit: str | None
    config_sha256: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppSettings:
        config_path = Path(path).resolve()
        raw = config_path.read_bytes()
        payload = yaml.safe_load(raw)
        if not isinstance(payload, dict):
            raise TypeError("application config must be a mapping")

        project_root = config_path.parent.parent

        def required(name: str) -> Any:
            if name not in payload:
                raise ValueError(f"missing application config field: {name}")
            return payload[name]

        mode = str(required("mode"))
        if mode not in {"fake", "bundle"}:
            raise ValueError("mode must be 'fake' or 'bundle'")
        device = str(required("device"))
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        image_size = int(required("image_size"))
        top_k = int(required("top_k"))
        threshold = float(required("confidence_threshold"))
        max_upload_bytes = int(required("max_upload_bytes"))
        max_image_pixels = int(required("max_image_pixels"))
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 1 <= top_k <= 203:
            raise ValueError("top_k must be between 1 and 203")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence_threshold must be between zero and one")
        if max_upload_bytes <= 0 or max_image_pixels <= 0:
            raise ValueError("image limits must be positive")
        def project_path(name: str) -> Path:
            candidate = Path(str(required(name)))
            return candidate if candidate.is_absolute() else (project_root / candidate).resolve()

        taxonomy_path: Path | None = None
        fake_class_id: int | None = None
        model_version: str | None = None
        data_version: str | None = None
        git_commit: str | None = None
        if mode == "fake":
            taxonomy_path = project_path("taxonomy_path")
            fake_class_id = int(required("fake_class_id"))
            if not 0 <= fake_class_id < 203:
                raise ValueError("fake_class_id must be between 0 and 202")
            model_version = str(required("model_version"))
            data_version = str(required("data_version"))
            git_commit = str(required("git_commit"))

        return cls(
            mode=mode,
            model_bundle=project_path("model_bundle"),
            taxonomy_path=taxonomy_path,
            device=device,
            image_size=image_size,
            confidence_threshold=threshold,
            top_k=top_k,
            max_upload_bytes=max_upload_bytes,
            max_image_pixels=max_image_pixels,
            fake_class_id=fake_class_id,
            model_version=model_version,
            data_version=data_version,
            git_commit=git_commit,
            config_sha256=hashlib.sha256(raw).hexdigest(),
        )
