from __future__ import annotations

import json
from pathlib import Path

import pytest
from dlcpd25_classifier.inference import BundleValidationError, load_bundle_manifest
from dlcpd25_classifier.training.a3 import (
    claim_test_evaluation,
    complete_test_evaluation,
    frozen_spec,
    sha256_file,
    validate_selection,
    write_checksums,
)

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY = ROOT / "metadata" / "class-taxonomy.json"


def _write_selected_run(root: Path) -> tuple[Path, Path]:
    run = root / "weighted"
    run.mkdir()
    (run / "best.pt").write_bytes(b"checkpoint")
    metrics = {
        "status": "completed_pending_project_lead_acceptance",
        "run_id": "weighted",
        "loss_strategy": "weighted_ce",
        "best_epoch": 4,
        "best_val": {"macro_f1": 0.7},
        "test_metrics_read": False,
    }
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run / "resolved-config.yaml").write_text("model: {image_size: 224}\n", encoding="utf-8")
    write_checksums(run)
    comparison = root / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "selected_run": "weighted",
                "selection_metric": "val_macro_f1",
                "test_metrics_read": False,
            }
        ),
        encoding="utf-8",
    )
    return run, comparison


def test_selection_and_freeze_are_fixed_before_test(tmp_path: Path) -> None:
    run, comparison = _write_selected_run(tmp_path)
    metrics, _ = validate_selection(run, comparison)
    test_csv = tmp_path / "test.csv"
    test_csv.write_text("frozen split\n", encoding="utf-8")
    frozen = frozen_spec(
        model_version="model-v1",
        selected_run=run,
        comparison_path=comparison,
        selected_metrics=metrics,
        repo_commit="a" * 40,
        taxonomy_path=TAXONOMY,
        test_csv=test_csv,
        confidence_threshold=0.55,
        preprocessing={"image_size": 224},
    )
    assert frozen["selected_loss_strategy"] == "weighted_ce"
    assert frozen["selected_checkpoint_sha256"] == sha256_file(run / "best.pt")
    assert frozen["test_evaluation_budget"] == 1
    assert frozen["test_metrics_read_before_freeze"] is False


def test_selection_rejects_test_contamination(tmp_path: Path) -> None:
    run, comparison = _write_selected_run(tmp_path)
    payload = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    payload["test_metrics_read"] = True
    (run / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    write_checksums(run)
    with pytest.raises(ValueError, match="test isolation"):
        validate_selection(run, comparison)


def test_repository_receipt_prevents_renamed_second_test_evaluation(tmp_path: Path) -> None:
    receipt_path = tmp_path / "metadata" / "a3-test-evaluation.json"
    frozen = {
        "model_version": "model-v1",
        "git_commit": "a" * 40,
        "test_split_sha256": "b" * 64,
        "selected_checkpoint_sha256": "c" * 64,
    }
    claim = claim_test_evaluation(
        receipt_path,
        evaluation_id="evaluation-v1",
        frozen=frozen,
        frozen_spec_sha256="d" * 64,
    )
    complete_test_evaluation(
        receipt_path,
        claim,
        release_checksums_sha256="e" * 64,
        test_metrics_sha256="f" * 64,
    )
    consumed = receipt_path.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to read test again"):
        claim_test_evaluation(
            receipt_path,
            evaluation_id="renamed-evaluation",
            frozen={**frozen, "model_version": "renamed-model"},
            frozen_spec_sha256="0" * 64,
        )

    assert receipt_path.read_bytes() == consumed


def test_bundle_loader_rejects_weight_hash_mismatch(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "best.pt").write_bytes(b"checkpoint")
    (bundle / "resolved-config.yaml").write_text("model: resnet50\n", encoding="utf-8")
    (bundle / "taxonomy.json").write_bytes(TAXONOMY.read_bytes())
    preprocessing = {
        "color_mode": "RGB",
        "image_size": 224,
        "eval_resize": 256,
        "eval_crop": "center",
        "interpolation": "bicubic",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    (bundle / "preprocessing.json").write_text(json.dumps(preprocessing), encoding="utf-8")
    (bundle / "metrics.json").write_text("{}\n", encoding="utf-8")
    (bundle / "model-card.md").write_text("# Model\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "model_version": "model-v1",
        "data_version": "data-v1-d5-r1",
        "git_commit": "a" * 40,
        "architecture": "resnet50",
        "num_classes": 203,
        "taxonomy_sha256": sha256_file(bundle / "taxonomy.json"),
        "preprocessing_sha256": sha256_file(bundle / "preprocessing.json"),
        "confidence_threshold": 0.55,
        "image_size": 224,
        "color_mode": "RGB",
        "resize": 256,
        "crop": "center",
        "interpolation": "bicubic",
        "mean": preprocessing["mean"],
        "std": preprocessing["std"],
        "torch_version": "2.11.0",
        "torchvision_version": "0.26.0",
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_checksums(bundle)
    load_bundle_manifest(bundle)
    (bundle / "best.pt").write_bytes(b"tampered")
    with pytest.raises(BundleValidationError) as error:
        load_bundle_manifest(bundle)
    assert error.value.code == "checksum_mismatch"
