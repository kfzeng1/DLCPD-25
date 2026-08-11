"""Validate the frozen bundle against A3 fixed validation samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import torch
import torchvision

from .errors import ImageValidationError
from .predictor import Predictor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_smoke(
    bundle_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    bundle = Path(bundle_path).resolve()
    root = Path(data_root).resolve()
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite P2 validation evidence: {target}")
    reference_path = bundle / "fixed-sample-predictions.json"
    references = json.loads(reference_path.read_text(encoding="utf-8"))
    if not isinstance(references, list) or not 3 <= len(references) <= 5:
        raise ValueError("fixed validation reference must contain 3-5 samples")

    predictor = Predictor.from_bundle(bundle, device=device)
    sample_results: list[dict[str, Any]] = []
    for reference in references:
        relative_path = str(reference["relative_path"])
        pure_path = PurePosixPath(relative_path)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise ValueError(f"unsafe fixed sample path: {relative_path}")
        image_path = (root / relative_path).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise FileNotFoundError(f"fixed validation sample is missing: {relative_path}")
        result = predictor.predict(image_path)
        actual_top_ids = [item.class_id for item in result.top_k]
        expected_top_ids = [int(value) for value in reference["top5_class_ids"]]
        if actual_top_ids != expected_top_ids:
            raise RuntimeError(f"fixed sample Top-5 mismatch: {relative_path}")
        sample_results.append(
            {
                "relative_path": relative_path,
                "class_id": result.class_id,
                "official_name": result.official_name,
                "host_zh": result.host_zh,
                "category_zh": result.category_zh,
                "confidence": result.confidence,
                "top5_class_ids": actual_top_ids,
                "low_confidence": result.low_confidence,
                "inference_ms": result.inference_ms,
            }
        )

    try:
        predictor.predict(b"corrupted-image")
    except ImageValidationError as exc:
        invalid_image = {"handled": True, "code": exc.code, "message": exc.user_message}
    else:
        raise RuntimeError("corrupted image did not raise ImageValidationError")

    summary = {
        "schema_version": 1,
        "stage": "P2",
        "status": "completed_pending_project_lead_acceptance",
        "created_at": datetime.now(UTC).isoformat(),
        "model_version": predictor.model_version,
        "data_version": predictor.data_version,
        "bundle_path": str(bundle),
        "bundle_checksums_sha256": _sha256(bundle / "checksums.sha256"),
        "reference_predictions_sha256": _sha256(reference_path),
        "device_requested": device,
        "device_selected": predictor.device,
        "confidence_threshold": predictor.confidence_threshold,
        "sample_source": "A3 fixed validation samples; formal test split was not accessed",
        "sample_count": len(sample_results),
        "low_confidence_count": sum(item["low_confidence"] for item in sample_results),
        "median_inference_ms": statistics.median(
            float(item["inference_ms"]) for item in sample_results
        ),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "invalid_image": invalid_image,
        "samples": sample_results,
    }
    target.mkdir(parents=True)
    summary_path = target / "validation-summary.json"
    _write_json(summary_path, summary)
    (target / "checksums.sha256").write_text(
        f"{_sha256(summary_path)}  validation-summary.json\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=root / "artifacts" / "releases" / "dlcpd25-resnet50-weighted-v1",
    )
    parser.add_argument("--data-root", type=Path, default=root / "data" / "raw" / "dlcpd25-203")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts" / "releases" / "application-p2-v1",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_smoke(args.bundle, args.data_root, args.output_dir, device=args.device)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
