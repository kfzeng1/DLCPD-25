"""Freeze the accepted A2 model, evaluate test once, and build the A3 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from dlcpd25_classifier.data import DLCPD25Dataset
from dlcpd25_classifier.inference.bundle import (
    BUNDLE_SCHEMA_VERSION,
    load_bundle_manifest,
)
from dlcpd25_classifier.models import build_classification_model
from dlcpd25_classifier.taxonomy import Taxonomy
from dlcpd25_classifier.training.checkpoint import load_checkpoint
from dlcpd25_classifier.training.metrics import (
    ClassificationMetrics,
    validate_metric_payload,
)
from dlcpd25_classifier.training.preflight import run_preflight
from dlcpd25_classifier.training.train import (
    environment_versions,
    git_commit,
    resolve_project_path,
    set_seed,
)
from dlcpd25_classifier.training.transforms import (
    build_eval_transform,
    preprocessing_spec,
)

ERROR_CASE_LIMIT = 30
TEST_EVALUATION_RECEIPT = Path("metadata/a3-test-evaluation.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def claim_test_evaluation(
    receipt_path: Path,
    *,
    evaluation_id: str,
    frozen: dict[str, Any],
    frozen_spec_sha256: str,
) -> dict[str, Any]:
    """Atomically consume the repository-wide one-test evaluation budget."""
    receipt = {
        "schema_version": 1,
        "stage": "A3",
        "status": "test_read_started",
        "test_evaluation_number": 1,
        "evaluation_id": evaluation_id,
        "model_version": frozen["model_version"],
        "started_at": utc_now(),
        "git_commit": frozen["git_commit"],
        "test_split_sha256": frozen["test_split_sha256"],
        "selected_checkpoint_sha256": frozen["selected_checkpoint_sha256"],
        "frozen_spec_sha256": frozen_spec_sha256,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to read test again; repository receipt already exists: {receipt_path}"
        ) from exc
    return receipt


def complete_test_evaluation(
    receipt_path: Path,
    claim: dict[str, Any],
    *,
    release_checksums_sha256: str,
    test_metrics_sha256: str,
) -> None:
    stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    if stored != claim or stored.get("status") != "test_read_started":
        raise RuntimeError("A3 test evaluation receipt changed after test access was claimed")
    completed = {
        **claim,
        "status": "consumed",
        "completed_at": utc_now(),
        "release_checksums_sha256": release_checksums_sha256,
        "test_metrics_sha256": test_metrics_sha256,
    }
    write_json(receipt_path, completed)


def verify_checksum_manifest(directory: Path) -> None:
    manifest = directory / "checksums.sha256"
    if not manifest.is_file():
        raise ValueError(f"missing checksum manifest: {manifest}")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        target = directory / name.removeprefix("*")
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"checksum mismatch: {target}")


def validate_selection(selected_run: Path, comparison_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_checksum_manifest(selected_run)
    selected_metrics = json.loads((selected_run / "metrics.json").read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if selected_metrics.get("status") != "completed_pending_project_lead_acceptance":
        raise ValueError("selected A2 run is incomplete")
    if selected_metrics.get("loss_strategy") != "weighted_ce":
        raise ValueError("A3 requires the accepted weighted CE run")
    if selected_metrics.get("test_metrics_read") is not False:
        raise ValueError("selected A2 run does not prove test isolation")
    if comparison.get("selected_run") != selected_metrics.get("run_id"):
        raise ValueError("A2 comparison does not select the requested run")
    if comparison.get("selection_metric") != "val_macro_f1":
        raise ValueError("A2 selection metric is not val Macro-F1")
    if comparison.get("test_metrics_read") is not False:
        raise ValueError("A2 comparison does not prove test isolation")
    return selected_metrics, comparison


def frozen_spec(
    *,
    model_version: str,
    selected_run: Path,
    comparison_path: Path,
    selected_metrics: dict[str, Any],
    repo_commit: str,
    taxonomy_path: Path,
    test_csv: Path,
    confidence_threshold: float,
    preprocessing: dict[str, Any],
) -> dict[str, Any]:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be within [0, 1]")
    checkpoint = selected_run / "best.pt"
    return {
        "schema_version": 1,
        "stage": "A3-freeze",
        "status": "frozen_before_test",
        "frozen_at": utc_now(),
        "model_version": model_version,
        "data_version": "data-v1-d5-r1",
        "git_commit": repo_commit,
        "selected_run": selected_metrics["run_id"],
        "selected_loss_strategy": selected_metrics["loss_strategy"],
        "selected_epoch": selected_metrics["best_epoch"],
        "selection_metric": "val_macro_f1",
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": sha256_file(checkpoint),
        "comparison": str(comparison_path),
        "comparison_sha256": sha256_file(comparison_path),
        "taxonomy_sha256": sha256_file(taxonomy_path),
        "test_split_sha256": sha256_file(test_csv),
        "confidence_threshold": confidence_threshold,
        "preprocessing": preprocessing,
        "test_evaluation_budget": 1,
        "test_metrics_read_before_freeze": False,
    }


def evaluate_test_once(
    model: nn.Module,
    loader: DataLoader,
    dataset: DLCPD25Dataset,
    taxonomy: Taxonomy,
    device: torch.device,
    confidence_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[int]], list[dict[str, Any]]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    accumulator = ClassificationMetrics()
    loss_sum = 0.0
    sample_count = 0
    low_confidence_count = 0
    errors: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if logits.shape[1:] != (203,) or not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError("A3 test evaluation produced invalid logits or loss")
            probabilities = logits.softmax(dim=1)
            confidences, predictions = probabilities.max(dim=1)
            top_probabilities, top_ids = probabilities.topk(5, dim=1)
            accumulator.update(logits, targets)
            loss_sum += float(loss) * targets.numel()
            low_confidence_count += int((confidences < confidence_threshold).sum())
            cpu_targets = targets.cpu().tolist()
            cpu_predictions = predictions.cpu().tolist()
            cpu_confidences = confidences.cpu().tolist()
            cpu_top_probabilities = top_probabilities.cpu().tolist()
            cpu_top_ids = top_ids.cpu().tolist()
            for index, (target, prediction) in enumerate(zip(cpu_targets, cpu_predictions)):
                record = dataset.get_record(sample_count + index)
                if prediction == target:
                    continue
                true_class = taxonomy.resolve(target)
                predicted_class = taxonomy.resolve(prediction)
                errors.append(
                    {
                        "relative_path": record.relative_path,
                        "true_class_id": target,
                        "true_official_name": true_class.official_name,
                        "predicted_class_id": prediction,
                        "predicted_official_name": predicted_class.official_name,
                        "confidence": float(cpu_confidences[index]),
                        "top5": [
                            {
                                "class_id": int(class_id),
                                "official_name": taxonomy.resolve(int(class_id)).official_name,
                                "confidence": float(probability),
                            }
                            for class_id, probability in zip(
                                cpu_top_ids[index], cpu_top_probabilities[index]
                            )
                        ],
                    }
                )
            sample_count += targets.numel()
    summary, per_class = accumulator.compute()
    validate_metric_payload(summary)
    elapsed = time.perf_counter() - started
    metrics = {
        "loss": loss_sum / sample_count,
        **summary,
        "samples": sample_count,
        "low_confidence_count": low_confidence_count,
        "low_confidence_rate": low_confidence_count / sample_count,
        "duration_seconds": elapsed,
        "images_per_second": sample_count / elapsed,
    }
    errors.sort(key=lambda item: (-item["confidence"], item["relative_path"]))
    return metrics, per_class, accumulator.as_serializable_confusion(), errors[:ERROR_CASE_LIMIT]


def fixed_sample_check(
    source_checkpoint: Path,
    bundled_checkpoint: Path,
    data_root: Path,
    val_csv: Path,
    taxonomy_path: Path,
    image_size: int,
) -> list[dict[str, Any]]:
    dataset = DLCPD25Dataset(
        data_root,
        val_csv,
        taxonomy_path,
        transform=build_eval_transform(image_size),
    )
    indices = (0, len(dataset) // 2, len(dataset) - 1)
    batch = torch.stack([dataset[index][0] for index in indices])
    source_model, _ = build_classification_model("resnet50", 203, pretrained=False)
    bundled_model, _ = build_classification_model("resnet50", 203, pretrained=False)
    load_checkpoint(
        source_checkpoint,
        source_model,
        expected_architecture="resnet50",
        expected_num_classes=203,
    )
    load_checkpoint(
        bundled_checkpoint,
        bundled_model,
        expected_architecture="resnet50",
        expected_num_classes=203,
    )
    source_model.eval()
    bundled_model.eval()
    with torch.inference_mode():
        source_logits = source_model(batch)
        first_logits = bundled_model(batch)
        second_logits = bundled_model(batch)
    if not torch.equal(source_logits, first_logits) or not torch.equal(first_logits, second_logits):
        raise RuntimeError("fixed sample logits changed after bundling or repeated inference")
    probabilities = first_logits.softmax(dim=1)
    top_probabilities, top_ids = probabilities.topk(5, dim=1)
    return [
        {
            "relative_path": dataset.get_record(index).relative_path,
            "true_class_id": dataset.get_record(index).class_id,
            "top5_class_ids": [int(value) for value in top_ids[row].tolist()],
            "top5_confidences": [float(value) for value in top_probabilities[row].tolist()],
            "logits_sha256": hashlib.sha256(first_logits[row].numpy().tobytes()).hexdigest(),
            "repeat_logits_equal": True,
            "source_and_bundle_logits_equal": True,
        }
        for row, index in enumerate(indices)
    ]


def model_card(
    model_version: str,
    selected_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    confidence_threshold: float,
) -> str:
    val = selected_metrics["best_val"]
    return f"""# {model_version}

## Model

This is a 203-class image classifier based on ImageNet V2 pretrained ResNet-50. It does not perform object detection. Host and four-way category outputs are derived from the frozen taxonomy using the predicted class ID.

The selected checkpoint is weighted cross-entropy epoch {selected_metrics['best_epoch']}, chosen only by validation Macro-F1. Input preprocessing is RGB, resize 256, center crop 224, bicubic interpolation, and ImageNet normalization. The application low-confidence threshold is frozen at {confidence_threshold:.2f}.

## Validation

- Top-1: {val['accuracy']:.6f}
- Top-5: {val['top5_accuracy']:.6f}
- Macro-F1: {val['macro_f1']:.6f}
- Balanced Accuracy: {val['balanced_accuracy']:.6f}

## Test

- Samples: {test_metrics['samples']}
- Top-1: {test_metrics['accuracy']:.6f}
- Top-5: {test_metrics['top5_accuracy']:.6f}
- Macro-F1: {test_metrics['macro_f1']:.6f}
- Balanced Accuracy: {test_metrics['balanced_accuracy']:.6f}

The fixed test split was evaluated once after model, preprocessing, and threshold freeze. Test results were not used for model selection or tuning.

## Limitations

Predictions are not professional agricultural diagnosis. Long-tail classes, domain shift, image quality, occlusion, and visually similar conditions can reduce reliability. Confidence below the frozen threshold must be presented as uncertain.
"""


def evaluation_report(test_metrics: dict[str, Any], error_limit: int) -> str:
    return f"""# A3 Test Evaluation

- Test samples: {test_metrics['samples']}
- Top-1: {test_metrics['accuracy']:.6%}
- Top-5: {test_metrics['top5_accuracy']:.6%}
- Macro-F1: {test_metrics['macro_f1']:.6%}
- Balanced Accuracy: {test_metrics['balanced_accuracy']:.6%}
- Low-confidence rate: {test_metrics['low_confidence_rate']:.6%}
- Duration: {test_metrics['duration_seconds']:.2f} seconds
- Error cases retained: {error_limit}

The test split was evaluated exactly once after freeze. No test metric was used for tuning.
"""


def write_checksums(directory: Path) -> None:
    names = sorted(path.name for path in directory.iterdir() if path.is_file() and path.name != "checksums.sha256")
    (directory / "checksums.sha256").write_text(
        "".join(f"{sha256_file(directory / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def build_release(
    *,
    release_dir: Path,
    model_version: str,
    repo_commit: str,
    selected_run: Path,
    selected_metrics: dict[str, Any],
    resolved_config: dict[str, Any],
    taxonomy_path: Path,
    preprocessing: dict[str, Any],
    confidence_threshold: float,
    frozen: dict[str, Any],
    test_metrics: dict[str, Any],
    per_class: list[dict[str, Any]],
    confusion: list[list[int]],
    errors: list[dict[str, Any]],
    evaluation_dir: Path,
    data_root: Path,
    val_csv: Path,
) -> dict[str, Any]:
    if release_dir.exists():
        raise FileExistsError(f"refusing to overwrite release: {release_dir}")
    release_dir.mkdir(parents=True)
    shutil.copyfile(selected_run / "best.pt", release_dir / "best.pt")
    shutil.copyfile(taxonomy_path, release_dir / "taxonomy.json")
    write_json(release_dir / "preprocessing.json", preprocessing)
    release_config = {
        **resolved_config,
        "stage": "A3",
        "model_version": model_version,
        "frozen_checkpoint_sha256": frozen["selected_checkpoint_sha256"],
        "confidence_threshold": confidence_threshold,
        "test_evaluations": 1,
    }
    (release_dir / "resolved-config.yaml").write_text(
        yaml.safe_dump(release_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    env = environment_versions()
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "model_version": model_version,
        "data_version": "data-v1-d5-r1",
        "git_commit": repo_commit,
        "architecture": "resnet50",
        "num_classes": 203,
        "taxonomy_sha256": sha256_file(release_dir / "taxonomy.json"),
        "preprocessing_sha256": sha256_file(release_dir / "preprocessing.json"),
        "confidence_threshold": confidence_threshold,
        "image_size": preprocessing["image_size"],
        "color_mode": preprocessing["color_mode"],
        "resize": preprocessing["eval_resize"],
        "crop": preprocessing["eval_crop"],
        "interpolation": preprocessing["interpolation"],
        "mean": preprocessing["mean"],
        "std": preprocessing["std"],
        "torch_version": env["torch"],
        "torchvision_version": env["torchvision"],
    }
    write_json(release_dir / "manifest.json", manifest)
    release_metrics = {
        "schema_version": 1,
        "stage": "A3",
        "status": "completed_pending_project_lead_acceptance",
        "model_version": model_version,
        "selection": {
            "run_id": selected_metrics["run_id"],
            "loss_strategy": selected_metrics["loss_strategy"],
            "best_epoch": selected_metrics["best_epoch"],
            "metric": "val_macro_f1",
            "validation": selected_metrics["best_val"],
        },
        "test": test_metrics,
        "test_evaluations": 1,
        "test_used_for_tuning": False,
    }
    write_json(release_dir / "metrics.json", release_metrics)
    write_json(release_dir / "per-class-metrics.json", per_class)
    write_json(release_dir / "confusion-matrix.json", confusion)
    write_json(release_dir / "error-cases.json", errors)
    write_json(release_dir / "frozen-spec.json", frozen)
    (release_dir / "model-card.md").write_text(
        model_card(model_version, selected_metrics, test_metrics, confidence_threshold),
        encoding="utf-8",
    )
    (release_dir / "evaluation-report.md").write_text(
        evaluation_report(test_metrics, len(errors)), encoding="utf-8"
    )
    fixed_samples = fixed_sample_check(
        selected_run / "best.pt",
        release_dir / "best.pt",
        data_root,
        val_csv,
        taxonomy_path,
        int(preprocessing["image_size"]),
    )
    write_json(release_dir / "fixed-sample-predictions.json", fixed_samples)
    write_checksums(release_dir)
    load_bundle_manifest(release_dir)
    write_json(evaluation_dir / "release-summary.json", release_metrics)
    return release_metrics


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--selected-run", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--verify-inputs-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    selected_run = (repo_root / args.selected_run).resolve() if not args.selected_run.is_absolute() else args.selected_run.resolve()
    comparison_path = (repo_root / args.comparison).resolve() if not args.comparison.is_absolute() else args.comparison.resolve()
    selected_metrics, _comparison = validate_selection(selected_run, comparison_path)
    resolved_config = yaml.safe_load((selected_run / "resolved-config.yaml").read_text(encoding="utf-8"))
    project_root = repo_root / "project"
    data_root = resolve_project_path(project_root, str(resolved_config["data_root"]))
    split_dir = resolve_project_path(project_root, str(resolved_config["split_dir"]))
    taxonomy_path = resolve_project_path(project_root, str(resolved_config["taxonomy"]))
    preflight = run_preflight(
        repo_root,
        repo_root / "artifacts/data/v1/d5-r1/data-v1-release.json",
        data_root,
    )
    preprocessing = preprocessing_spec(int(resolved_config["model"]["image_size"]))
    repo_commit = git_commit(repo_root)
    frozen = frozen_spec(
        model_version=args.model_version,
        selected_run=selected_run,
        comparison_path=comparison_path,
        selected_metrics=selected_metrics,
        repo_commit=repo_commit,
        taxonomy_path=taxonomy_path,
        test_csv=split_dir / "test.csv",
        confidence_threshold=args.confidence_threshold,
        preprocessing=preprocessing,
    )
    if args.verify_inputs_only:
        print(json.dumps({"status": "inputs_valid", "freeze": frozen}, ensure_ascii=False))
        return 0

    evaluation_dir = repo_root / "artifacts/training" / args.evaluation_id
    release_dir = repo_root / "artifacts/releases" / args.model_version
    receipt_path = repo_root / TEST_EVALUATION_RECEIPT
    if receipt_path.exists():
        raise FileExistsError(
            f"refusing to read test again; repository receipt already exists: {receipt_path}"
        )
    if evaluation_dir.exists():
        raise FileExistsError(f"refusing to repeat A3 test evaluation: {evaluation_dir}")
    if release_dir.exists():
        raise FileExistsError(f"refusing to overwrite release: {release_dir}")
    evaluation_dir.mkdir(parents=True)
    write_json(evaluation_dir / "frozen-spec.json", frozen)
    write_json(evaluation_dir / "preflight.json", preflight)
    write_json(
        evaluation_dir / "test-read-started.json",
        {
            "started_at": utc_now(),
            "frozen_spec_sha256": sha256_file(evaluation_dir / "frozen-spec.json"),
            "test_evaluation_number": 1,
        },
    )
    receipt = claim_test_evaluation(
        receipt_path,
        evaluation_id=args.evaluation_id,
        frozen=frozen,
        frozen_spec_sha256=sha256_file(evaluation_dir / "frozen-spec.json"),
    )
    set_seed(int(resolved_config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A3 test evaluation requires CUDA on this machine")
    dataset = DLCPD25Dataset(
        data_root,
        split_dir / "test.csv",
        taxonomy_path,
        transform=build_eval_transform(int(preprocessing["image_size"])),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
    )
    model, _ = build_classification_model("resnet50", 203, pretrained=False)
    load_checkpoint(
        selected_run / "best.pt",
        model,
        expected_architecture="resnet50",
        expected_num_classes=203,
        map_location=device,
    )
    model.to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    taxonomy = Taxonomy(taxonomy_path)
    test_metrics, per_class, confusion, errors = evaluate_test_once(
        model,
        loader,
        dataset,
        taxonomy,
        device,
        args.confidence_threshold,
    )
    test_metrics["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
    write_json(evaluation_dir / "test-metrics.json", test_metrics)
    write_json(evaluation_dir / "per-class-metrics.json", per_class)
    write_json(evaluation_dir / "confusion-matrix.json", confusion)
    write_json(evaluation_dir / "error-cases.json", errors)
    release_metrics = build_release(
        release_dir=release_dir,
        model_version=args.model_version,
        repo_commit=repo_commit,
        selected_run=selected_run,
        selected_metrics=selected_metrics,
        resolved_config=resolved_config,
        taxonomy_path=taxonomy_path,
        preprocessing=preprocessing,
        confidence_threshold=args.confidence_threshold,
        frozen=frozen,
        test_metrics=test_metrics,
        per_class=per_class,
        confusion=confusion,
        errors=errors,
        evaluation_dir=evaluation_dir,
        data_root=data_root,
        val_csv=split_dir / "val.csv",
    )
    write_json(
        evaluation_dir / "test-evaluation-complete.json",
        {
            "completed_at": utc_now(),
            "test_evaluations": 1,
            "release": str(release_dir),
            "release_checksums_sha256": sha256_file(release_dir / "checksums.sha256"),
        },
    )
    complete_test_evaluation(
        receipt_path,
        receipt,
        release_checksums_sha256=sha256_file(release_dir / "checksums.sha256"),
        test_metrics_sha256=sha256_file(evaluation_dir / "test-metrics.json"),
    )
    write_checksums(evaluation_dir)
    print(json.dumps(release_metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
