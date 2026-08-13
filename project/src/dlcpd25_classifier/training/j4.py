"""Freeze J3, evaluate both test splits once, and build one joint model bundle."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from dlcpd25_classifier.data import DLCPD25Dataset
from dlcpd25_classifier.detection import (
    DetectionClassMapping,
    DirectResizeDetectionTransform,
    IP102DetectionDataset,
    build_shared_detection_model,
)
from dlcpd25_classifier.detection.evaluation import evaluate_detection
from dlcpd25_classifier.inference.joint_bundle import (
    JOINT_BUNDLE_SCHEMA_VERSION,
    load_joint_bundle_manifest,
    sha256_file,
)
from dlcpd25_classifier.training.j3 import _detection_counts
from dlcpd25_classifier.training.joint import collate_detection
from dlcpd25_classifier.training.metrics import (
    ClassificationMetrics,
    validate_metric_payload,
)
from dlcpd25_classifier.training.train import (
    environment_versions,
    git_commit,
    load_config,
    resolve_project_path,
    set_seed,
)
from dlcpd25_classifier.training.transforms import (
    build_direct_resize_eval_transform,
    direct_resize_preprocessing_spec,
)

STAGE = "J4"
REQUIRED_J3_ARTIFACTS = (
    "best-selection.json",
    "history.json",
    "initial-classification-validation.json",
    "joint-best.pt",
    "joint-last.pt",
    "metrics.json",
    "progress.csv",
    "resolved-config.yaml",
    "run-state.json",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def verify_checksum_manifest(directory: Path) -> int:
    manifest = directory / "checksums.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, raw_name = line.split(maxsplit=1)
        target = directory / raw_name.removeprefix("*")
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"J3 artifact checksum mismatch: {target.name}")
        count += 1
    return count


def verify_inputs(config: dict[str, Any], project_root: Path) -> dict[str, str]:
    keys = tuple(config["input_sha256"])
    required = {
        "selected_checkpoint",
        "classification_checkpoint",
        "classification_test_csv",
        "classification_taxonomy",
        "detection_train_split",
        "detection_test_split",
        "detection_annotations",
        "detection_mapping",
    }
    if set(keys) != required:
        raise ValueError("J4 input_sha256 keys do not match the frozen contract")
    if Path(str(config["classification_test_csv"])).name != "test.csv":
        raise ValueError("J4 classification input must be the frozen test.csv")
    if Path(str(config["detection_test_split"])).name != "test.txt":
        raise ValueError("J4 detection input must be the frozen test.txt")
    actual: dict[str, str] = {}
    for key in keys:
        path = resolve_project_path(project_root, str(config[key]))
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[key] = sha256_file(path)
        if actual[key] != str(config["input_sha256"][key]):
            raise ValueError(f"J4 frozen input checksum mismatch: {key}")
    return actual


def validate_config(config: dict[str, Any]) -> None:
    if config.get("model") != {
        "architecture": "joint-resnet50-fasterrcnn",
        "image_size": 224,
        "classification_classes": 203,
        "detection_classes": 96,
    }:
        raise ValueError("J4 model contract must remain joint/224/203/96")
    evaluation = config.get("evaluation", {})
    expected = {
        "classification_confidence_threshold": 0.55,
        "detection_score_threshold_for_ap": 0.05,
        "detection_score_threshold_for_precision_recall": 0.5,
        "detection_nms_iou_threshold": 0.5,
        "detection_max_detections_per_image": 100,
        "test_evaluation_budget": 1,
    }
    if any(evaluation.get(key) != value for key, value in expected.items()):
        raise ValueError("J4 frozen evaluation thresholds changed")


def validate_j3_run(run_dir: Path, selected_checkpoint: Path) -> dict[str, Any]:
    checksum_count = verify_checksum_manifest(run_dir)
    for name in REQUIRED_J3_ARTIFACTS:
        if not (run_dir / name).is_file():
            raise FileNotFoundError(run_dir / name)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    best = json.loads((run_dir / "best-selection.json").read_text(encoding="utf-8"))
    if metrics.get("status") != "completed_pending_project_lead_acceptance":
        raise ValueError("J4 selected J3 run is incomplete")
    if metrics.get("epochs_completed") != 10 or metrics.get("best_epoch") != 10:
        raise ValueError("J4 requires the accepted 10-epoch J3 selection")
    if metrics.get("test_metrics_read") is not False or not best.get("eligible"):
        raise ValueError("J3 selection does not prove test isolation and classification gate")
    if best.get("epoch") != metrics.get("best_epoch"):
        raise ValueError("J3 best selection and metrics disagree")
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    eligible_maps = [
        float(item["detection"]["map"]) for item in history if item["eligible"]
    ]
    if not eligible_maps or not math.isclose(
        float(best["detection"]["map"]), max(eligible_maps), abs_tol=1e-12
    ):
        raise ValueError("J3 selected checkpoint is not the highest eligible detection mAP")
    if selected_checkpoint.resolve() != (run_dir / "joint-best.pt").resolve():
        raise ValueError("J4 selected checkpoint is not J3 joint-best.pt")
    return {"checksum_entries": checksum_count, "metrics": metrics, "best": best}


def frozen_spec(
    config: dict[str, Any],
    *,
    input_sha256: dict[str, str],
    repo_commit: str,
    selected: dict[str, Any],
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    return {
        "schema_version": 1,
        "stage": "J4-freeze",
        "status": "frozen_before_test",
        "frozen_at": utc_now(),
        "model_version": config["model_version"],
        "git_commit": repo_commit,
        "selected_j3_run": selected["metrics"]["run_id"],
        "selected_epoch": selected["metrics"]["best_epoch"],
        "selection_rule": "classification_gate_then_highest_detection_map",
        "input_sha256": input_sha256,
        "preprocessing": direct_resize_preprocessing_spec(224),
        "postprocessing": {
            "classification_confidence_threshold": evaluation[
                "classification_confidence_threshold"
            ],
            "detection_score_threshold": evaluation[
                "detection_score_threshold_for_precision_recall"
            ],
            "detection_ap_score_threshold": evaluation[
                "detection_score_threshold_for_ap"
            ],
            "detection_nms_iou_threshold": evaluation[
                "detection_nms_iou_threshold"
            ],
            "detection_max_detections_per_image": evaluation[
                "detection_max_detections_per_image"
            ],
        },
        "test_evaluation_budget": 1,
        "test_metrics_read_before_freeze": False,
    }


def claim_test_once(
    receipt_path: Path, frozen: dict[str, Any], frozen_hash: str
) -> dict[str, Any]:
    receipt = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "test_read_started",
        "evaluation_number": 1,
        "model_version": frozen["model_version"],
        "started_at": utc_now(),
        "git_commit": frozen["git_commit"],
        "frozen_spec_sha256": frozen_hash,
        "classification_test_split_sha256": frozen["input_sha256"][
            "classification_test_csv"
        ],
        "detection_test_split_sha256": frozen["input_sha256"][
            "detection_test_split"
        ],
        "selected_checkpoint_sha256": frozen["input_sha256"][
            "selected_checkpoint"
        ],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to read J4 test again: {receipt_path}") from exc
    return receipt


def complete_receipt(
    receipt_path: Path,
    receipt: dict[str, Any],
    *,
    checksums_sha256: str,
    classification_metrics_sha256: str,
    detection_metrics_sha256: str,
) -> None:
    if json.loads(receipt_path.read_text(encoding="utf-8")) != receipt:
        raise RuntimeError("J4 test receipt changed after test access was claimed")
    write_json(
        receipt_path,
        {
            **receipt,
            "status": "consumed",
            "completed_at": utc_now(),
            "release_checksums_sha256": checksums_sha256,
            "classification_metrics_sha256": classification_metrics_sha256,
            "detection_metrics_sha256": detection_metrics_sha256,
        },
    )


def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    confidence_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[int]]]:
    """Evaluate the frozen classification test in exactly one inference pass."""
    accumulator = ClassificationMetrics()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    samples = 0
    low_confidence = 0
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model.forward_classification(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError("J4 classification test produced non-finite values")
            accumulator.update(logits, targets)
            loss_sum += float(loss) * targets.numel()
            samples += targets.numel()
            probabilities = logits.softmax(dim=1)
            low_confidence += int(
                (probabilities.max(dim=1).values < confidence_threshold).sum()
            )
    summary, per_class = accumulator.compute()
    validate_metric_payload(summary)
    elapsed = time.perf_counter() - started
    metrics: dict[str, Any] = {
        "loss": loss_sum / samples,
        **summary,
        "samples": samples,
        "duration_seconds": elapsed,
        "images_per_second": samples / elapsed,
        "inference_passes": 1,
        "confidence_threshold": confidence_threshold,
    }
    metrics["low_confidence_count"] = low_confidence
    metrics["low_confidence_rate"] = low_confidence / samples
    return metrics, per_class, accumulator.as_serializable_confusion()


def write_checksums(directory: Path) -> None:
    names = sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    (directory / "checksums.sha256").write_text(
        "".join(f"{sha256_file(directory / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def build_model_card(
    classification: dict[str, Any], detection: dict[str, Any]
) -> str:
    return f"""# DLCPD-25 + IP102 Joint Model v1

## Model

One `224 x 224` RGB tensor is processed by one shared ResNet-50 body. The classification head predicts all 203 DLCPD-25 image classes. The FPN/RPN/ROI detection branch localizes 96 IP102 pest classes. The bundle contains one joint checkpoint.

## Final Test

- Classification Top-1: {classification['accuracy']:.6%}
- Classification Top-5: {classification['top5_accuracy']:.6%}
- Classification Macro-F1: {classification['macro_f1']:.6%}
- Detection mAP@0.5:0.95: {detection['map']:.6%}
- Detection AP50: {detection['ap50']:.6%}
- Detection Precision: {detection['precision']:.6%}
- Detection Recall: {detection['recall']:.6%}

Both frozen test splits were evaluated once in J4 after checkpoint, preprocessing, mapping, score threshold, NMS and maximum detections were frozen. Test results were not used for training or tuning.

## Limitations

The detection branch only has box supervision for 96 mapped IP102 pest classes. Diseases, healthy classes and physiological defects can be classified but cannot be promised bounding boxes. IP102 source class 61 has no test support, corresponding to detector label 8. Small-object performance is limited by direct resize to 224.
"""


def run_j4(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    project_root = repo_root / "project"
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_config(config_path)
    validate_config(config)
    inputs = verify_inputs(config, project_root)
    selected_run = resolve_project_path(project_root, str(config["selected_run"]))
    selected_checkpoint = resolve_project_path(
        project_root, str(config["selected_checkpoint"])
    )
    selected = validate_j3_run(selected_run, selected_checkpoint)
    repo_commit = git_commit(repo_root)
    frozen = frozen_spec(
        config, input_sha256=inputs, repo_commit=repo_commit, selected=selected
    )
    if args.verify_inputs_only:
        return {"status": "inputs_valid", "freeze": frozen, "test_metrics_read": False}

    evaluation_dir = resolve_project_path(project_root, str(config["evaluation_dir"]))
    release_dir = resolve_project_path(project_root, str(config["release_dir"]))
    receipt_path = resolve_project_path(project_root, str(config["receipt"]))
    for path in (evaluation_dir, release_dir, receipt_path):
        if path.exists():
            raise FileExistsError(f"refusing to repeat J4 evaluation: {path}")
    evaluation_dir.mkdir(parents=True)
    write_json(evaluation_dir / "frozen-spec.json", frozen)
    frozen_hash = sha256_file(evaluation_dir / "frozen-spec.json")
    receipt = claim_test_once(receipt_path, frozen, frozen_hash)

    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("J4 final evaluation requires CUDA")
    mapping_path = resolve_project_path(project_root, str(config["detection_mapping"]))
    mapping = DetectionClassMapping(mapping_path)
    model, model_info = build_shared_detection_model(
        resolve_project_path(project_root, str(config["classification_checkpoint"])),
        mapping,
        trainable_backbone_layers=5,
    )
    checkpoint = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    evaluation = config["evaluation"]
    model.detector.roi_heads.score_thresh = float(
        evaluation["detection_score_threshold_for_ap"]
    )
    model.detector.roi_heads.nms_thresh = float(
        evaluation["detection_nms_iou_threshold"]
    )
    model.detector.roi_heads.detections_per_img = int(
        evaluation["detection_max_detections_per_image"]
    )

    classification_test = DLCPD25Dataset(
        resolve_project_path(project_root, str(config["classification_data_root"])),
        resolve_project_path(project_root, str(config["classification_test_csv"])),
        resolve_project_path(project_root, str(config["classification_taxonomy"])),
        transform=build_direct_resize_eval_transform(224),
    )
    detection_test = IP102DetectionDataset(
        resolve_project_path(project_root, str(config["detection_voc_root"])),
        resolve_project_path(project_root, str(config["detection_test_split"])),
        mapping_path,
        annotations_path=resolve_project_path(
            project_root, str(config["detection_annotations"])
        ),
        transforms=DirectResizeDetectionTransform(224),
    )
    detection_train = IP102DetectionDataset(
        resolve_project_path(project_root, str(config["detection_voc_root"])),
        resolve_project_path(project_root, str(config["detection_train_split"])),
        mapping_path,
        annotations_path=resolve_project_path(
            project_root, str(config["detection_annotations"])
        ),
        transforms=DirectResizeDetectionTransform(224),
    )
    if len(classification_test) != 22178 or len(detection_test) != 3798:
        raise ValueError("J4 frozen test cardinalities changed")
    workers = int(evaluation["workers"])
    classification_loader = DataLoader(
        classification_test,
        batch_size=int(evaluation["classification_batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
    )
    detection_loader = DataLoader(
        detection_test,
        batch_size=int(evaluation["detection_batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_detection,
    )
    class_names = {
        label: mapping.from_detector(label).dlcpd25_name
        for label in range(1, mapping.num_detector_classes + 1)
    }
    started = time.perf_counter()
    classification_metrics, classification_per_class, confusion = evaluate_classification(
        model,
        classification_loader,
        device,
        float(evaluation["classification_confidence_threshold"]),
    )
    detection_metrics, detection_per_class = evaluate_detection(
        model,
        detection_loader,
        device,
        score_threshold=float(
            evaluation["detection_score_threshold_for_precision_recall"]
        ),
        class_names=class_names,
        train_object_counts=_detection_counts(detection_train),
    )
    missing = [record for record in detection_per_class if record["val_objects"] == 0]
    if [record["detector_label"] for record in missing] != [8]:
        raise ValueError("J4 expected only detector label 8 to lack test support")
    detection_metrics["classes_with_test_support"] = 95
    detection_metrics["classes_without_test_support"] = [
        {
            "detector_label": record["detector_label"],
            "name": record["name"],
            "ip102_source_class_id": 61,
        }
        for record in missing
    ]

    release_dir.mkdir(parents=True)
    shutil.copyfile(selected_checkpoint, release_dir / "joint-best.pt")
    shutil.copyfile(
        resolve_project_path(project_root, str(config["classification_taxonomy"])),
        release_dir / "taxonomy.json",
    )
    shutil.copyfile(mapping_path, release_dir / "ip102-detection-class-map.json")
    preprocessing = frozen["preprocessing"]
    postprocessing = frozen["postprocessing"]
    write_json(release_dir / "preprocessing.json", preprocessing)
    write_json(release_dir / "postprocessing.json", postprocessing)
    write_json(release_dir / "metrics-classification.json", classification_metrics)
    write_json(release_dir / "metrics-detection.json", detection_metrics)
    write_json(
        release_dir / "metrics-classification-per-class.json", classification_per_class
    )
    write_json(release_dir / "classification-confusion.json", confusion)
    write_json(release_dir / "metrics-detection-per-class.json", detection_per_class)
    write_json(release_dir / "frozen-spec.json", frozen)
    release_config = {
        **config,
        "stage": STAGE,
        "input_git_commit": repo_commit,
        "test_evaluations": 1,
        "classification_test_inference_passes": 1,
        "detection_test_inference_passes": 1,
        "test_metrics_used_for_tuning": False,
    }
    (release_dir / "resolved-config.yaml").write_text(
        yaml.safe_dump(release_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (release_dir / "model-card.md").write_text(
        build_model_card(classification_metrics, detection_metrics), encoding="utf-8"
    )
    manifest = {
        "schema_version": JOINT_BUNDLE_SCHEMA_VERSION,
        "model_version": config["model_version"],
        "git_commit": repo_commit,
        "architecture": "joint-resnet50-fasterrcnn",
        "image_size": 224,
        "classification_classes": 203,
        "detection_classes": 96,
        "shared_body_forwards_per_joint_call": 1,
        "checkpoint_sha256": sha256_file(release_dir / "joint-best.pt"),
        "taxonomy_sha256": sha256_file(release_dir / "taxonomy.json"),
        "detection_mapping_sha256": sha256_file(
            release_dir / "ip102-detection-class-map.json"
        ),
        "preprocessing_sha256": sha256_file(release_dir / "preprocessing.json"),
        "postprocessing_sha256": sha256_file(release_dir / "postprocessing.json"),
        "torch_version": environment_versions()["torch"],
        "torchvision_version": environment_versions()["torchvision"],
        "pycocotools_version": importlib_metadata.version("pycocotools"),
    }
    write_json(release_dir / "manifest.json", manifest)
    write_checksums(release_dir)
    load_joint_bundle_manifest(release_dir)

    summary = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "completed_pending_project_lead_acceptance",
        "model_version": config["model_version"],
        "selected_epoch": selected["metrics"]["best_epoch"],
        "classification_test": classification_metrics,
        "detection_test": detection_metrics,
        "model": model_info.__dict__,
        "duration_seconds": time.perf_counter() - started,
        "test_evaluations": 1,
        "test_used_for_tuning": False,
    }
    write_json(evaluation_dir / "metrics.json", summary)
    write_json(evaluation_dir / "frozen-spec.json", frozen)
    write_json(evaluation_dir / "release-manifest.json", manifest)
    write_checksums(evaluation_dir)
    complete_receipt(
        receipt_path,
        receipt,
        checksums_sha256=sha256_file(release_dir / "checksums.sha256"),
        classification_metrics_sha256=sha256_file(
            release_dir / "metrics-classification.json"
        ),
        detection_metrics_sha256=sha256_file(release_dir / "metrics-detection.json"),
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[4]
    )
    parser.add_argument("--config", type=Path, default=Path("configs/j4.yaml"))
    parser.add_argument("--verify-inputs-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        result = run_j4(parse_args())
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"J4 failed: {exc}", file=os.sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "classification_top1": result.get("classification_test", {}).get(
                    "accuracy"
                ),
                "detection_map": result.get("detection_test", {}).get("map"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
