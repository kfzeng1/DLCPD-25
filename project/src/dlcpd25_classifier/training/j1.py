"""J1 classification adaptation to the joint model's direct-resize input contract."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dlcpd25_classifier.data import DLCPD25Dataset
from dlcpd25_classifier.models import build_classification_model
from dlcpd25_classifier.training.a2 import (
    class_counts,
    clipped_inverse_frequency_weights,
    evaluate_epoch,
    make_scheduler,
    train_epoch,
)
from dlcpd25_classifier.training.checkpoint import load_checkpoint, save_checkpoint
from dlcpd25_classifier.training.metrics import (
    epoch_duration_seconds,
    validate_metric_payload,
)
from dlcpd25_classifier.training.preflight import run_preflight
from dlcpd25_classifier.training.progress import append_progress
from dlcpd25_classifier.training.train import (
    environment_versions,
    git_commit,
    load_config,
    resolve_project_path,
    set_seed,
    sha256_file,
    write_checksums,
)
from dlcpd25_classifier.training.transforms import (
    build_direct_resize_eval_transform,
    build_direct_resize_train_transform,
    build_eval_transform,
    direct_resize_preprocessing_spec,
    preprocessing_spec,
)

STAGE = "J1"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _validate_config(config: dict[str, Any]) -> None:
    model = config.get("model", {})
    training = config.get("training", {})
    evaluation = config.get("evaluation", {})
    if model != {"name": "resnet50", "num_classes": 203, "image_size": 224}:
        raise ValueError("J1 requires ResNet-50, 203 classes, and 224 input")
    if training.get("class_weighting") != (
        "inverse_frequency_clipped_at_10_then_mean_normalized"
    ):
        raise ValueError("J1 must retain the selected weighted CE strategy")
    if evaluation.get("selection_metric") != "val_macro_f1":
        raise ValueError("J1 checkpoint selection must use val Macro-F1")
    if evaluation.get("test_metrics_read") is not False:
        raise ValueError("J1 must not read test metrics")


def _build_dataset(
    data_root: Path,
    split_csv: Path,
    taxonomy_path: Path,
    transform: Any,
) -> DLCPD25Dataset:
    if split_csv.name not in {"train.csv", "val.csv"}:
        raise ValueError("J1 only permits the frozen train and val splits")
    return DLCPD25Dataset(data_root, split_csv, taxonomy_path, transform=transform)


def _loader(
    dataset: DLCPD25Dataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
    )


def _load_initial_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    model, _ = build_classification_model("resnet50", 203, pretrained=False)
    payload = load_checkpoint(
        checkpoint_path,
        model,
        expected_architecture="resnet50",
        expected_num_classes=203,
    )
    metadata = payload.get("metadata", {})
    if metadata.get("loss_strategy") != "weighted_ce":
        raise ValueError("J1 initialization must be the selected weighted CE checkpoint")
    if metadata.get("test_metrics_read") is not False:
        raise ValueError("initial checkpoint has invalid test-isolation metadata")
    return model.to(device), payload


def _baseline_record(metrics: dict[str, float]) -> dict[str, float]:
    result = {
        key: float(metrics[key])
        for key in ("loss", "accuracy", "top5_accuracy", "macro_f1", "balanced_accuracy")
    }
    validate_metric_payload(result)
    return result


def _evaluate_baselines(
    checkpoint_path: Path,
    *,
    data_root: Path,
    val_csv: Path,
    taxonomy_path: Path,
    image_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> dict[str, Any]:
    model, payload = _load_initial_model(checkpoint_path, device)
    criterion = nn.CrossEntropyLoss()
    old_dataset = _build_dataset(
        data_root, val_csv, taxonomy_path, build_eval_transform(image_size)
    )
    direct_dataset = _build_dataset(
        data_root, val_csv, taxonomy_path, build_direct_resize_eval_transform(image_size)
    )
    old_metrics, _, _ = evaluate_epoch(
        model,
        _loader(old_dataset, batch_size=batch_size, workers=workers, shuffle=False),
        criterion,
        device,
    )
    direct_metrics, _, _ = evaluate_epoch(
        model,
        _loader(direct_dataset, batch_size=batch_size, workers=workers, shuffle=False),
        criterion,
        device,
    )
    return {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(payload["epoch"]),
        "samples": len(old_dataset),
        "old_preprocessing": {
            "spec": preprocessing_spec(image_size),
            "metrics": _baseline_record(old_metrics),
        },
        "direct_resize_pre_adaptation": {
            "spec": direct_resize_preprocessing_spec(image_size),
            "metrics": _baseline_record(direct_metrics),
        },
        "test_metrics_read": False,
    }


def _copy_best_checkpoint(source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def run_j1(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    project_root = repo_root / "project"
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_config(config_path)
    _validate_config(config)
    set_seed(int(config["seed"]))

    output_root = resolve_project_path(project_root, str(config["output_dir"]))
    output_dir = output_root / args.run_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite J1 run: {output_dir}")
    output_dir.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("J1 requires CUDA on the configured training machine")
    model_config = config["model"]
    training = config["training"]
    evaluation = config["evaluation"]
    image_size = int(model_config["image_size"])
    data_root = resolve_project_path(project_root, str(config["data_root"]))
    split_dir = resolve_project_path(project_root, str(config["split_dir"]))
    taxonomy_path = resolve_project_path(project_root, str(config["taxonomy"]))
    initial_checkpoint = resolve_project_path(project_root, str(config["initial_checkpoint"]))
    if not initial_checkpoint.is_file():
        raise FileNotFoundError(initial_checkpoint)

    release_path = repo_root / "artifacts/data/v1/d5-r1/data-v1-release.json"
    preflight = run_preflight(repo_root, release_path, data_root)
    baseline = _evaluate_baselines(
        initial_checkpoint,
        data_root=data_root,
        val_csv=split_dir / "val.csv",
        taxonomy_path=taxonomy_path,
        image_size=image_size,
        batch_size=int(training["batch_size"]),
        workers=int(training["workers"]),
        device=device,
    )
    _write_json(output_dir / "baseline-comparison.json", baseline)

    train_dataset = _build_dataset(
        data_root,
        split_dir / "train.csv",
        taxonomy_path,
        build_direct_resize_train_transform(image_size),
    )
    val_dataset = _build_dataset(
        data_root,
        split_dir / "val.csv",
        taxonomy_path,
        build_direct_resize_eval_transform(image_size),
    )
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = _loader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        workers=int(training["workers"]),
        shuffle=True,
        generator=generator,
    )
    val_loader = _loader(
        val_dataset,
        batch_size=int(training["batch_size"]),
        workers=int(training["workers"]),
        shuffle=False,
    )

    model, initial_payload = _load_initial_model(initial_checkpoint, device)
    weights = clipped_inverse_frequency_weights(class_counts(train_dataset))
    train_criterion = nn.CrossEntropyLoss(
        weight=weights.to(device), label_smoothing=float(training["label_smoothing"])
    )
    val_criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        fused=True,
    )
    scheduler = make_scheduler(
        optimizer, int(training["epochs"]), int(training["warmup_epochs"])
    )
    amp_enabled = bool(training["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    resolved = {
        **config,
        "stage": STAGE,
        "run_id": args.run_id,
        "input_git_commit": git_commit(repo_root),
        "input_checkpoint_sha256": baseline["checkpoint_sha256"],
        "actual": {
            "initial_checkpoint_epoch": int(initial_payload["epoch"]),
            "all_model_parameters_trainable": True,
            "selection_metric": "val_macro_f1",
            "preprocessing": direct_resize_preprocessing_spec(image_size),
            "test_metrics_read": False,
        },
    }
    (output_dir / "resolved-config.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    _write_json(output_dir / "preflight.json", preflight)
    _write_json(
        output_dir / "class-balance.json",
        {
            "counts": class_counts(train_dataset),
            "weights": weights.tolist(),
            "formula": training["class_weighting"],
        },
    )
    _write_json(
        output_dir / "run-state.json",
        {"status": "running", "last_epoch": 0, "test_metrics_read": False},
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, int(training["epochs"]) + 1):
        generator.manual_seed(int(config["seed"]) + epoch)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_epoch(
            model,
            train_loader,
            train_criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
        )
        val_metrics, per_class, confusion = evaluate_epoch(
            model, val_loader, val_criterion, device
        )
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        improved = val_metrics["macro_f1"] > best_macro_f1 + 1e-6
        if improved:
            best_epoch = epoch
            best_macro_f1 = float(val_metrics["macro_f1"])
            epochs_without_improvement = 0
            _write_json(output_dir / "best-per-class.json", per_class)
            _write_json(output_dir / "best-confusion-matrix.json", confusion)
        else:
            epochs_without_improvement += 1
        scheduler.step()
        checkpoint_metadata = {
            "stage": STAGE,
            "run_id": args.run_id,
            "data_version": preflight["data_version"],
            "taxonomy_sha256": preflight["taxonomy"]["sha256"],
            "git_commit": git_commit(repo_root),
            "initial_checkpoint_sha256": baseline["checkpoint_sha256"],
            "best_epoch": best_epoch,
            "best_macro_f1": best_macro_f1,
            "epochs_without_improvement": epochs_without_improvement,
            "preprocessing": direct_resize_preprocessing_spec(image_size),
            "test_metrics_read": False,
        }
        checkpoint_metrics = {
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_top5_accuracy": float(val_metrics["top5_accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
        }
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            architecture="resnet50",
            num_classes=203,
            epoch=epoch,
            metrics=checkpoint_metrics,
            metadata=checkpoint_metadata,
            scheduler=scheduler,
            scaler=scaler,
            overwrite=True,
        )
        if improved:
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                architecture="resnet50",
                num_classes=203,
                epoch=epoch,
                metrics=checkpoint_metrics,
                metadata=checkpoint_metadata,
                scheduler=scheduler,
                scaler=scaler,
                overwrite=True,
            )
        _write_json(output_dir / "history.json", history)
        append_progress(output_dir / "metrics.csv", record)
        _write_json(
            output_dir / "run-state.json",
            {
                "status": "running",
                "last_epoch": epoch,
                "best_epoch": best_epoch,
                "best_macro_f1": best_macro_f1,
                "epochs_without_improvement": epochs_without_improvement,
                "test_metrics_read": False,
            },
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if epochs_without_improvement >= int(training["early_stopping_patience"]):
            break

    duration = epoch_duration_seconds(history)
    fresh_model, _ = build_classification_model("resnet50", 203, pretrained=False)
    best_payload = load_checkpoint(
        output_dir / "best.pt",
        fresh_model,
        expected_architecture="resnet50",
        expected_num_classes=203,
    )
    best_val = {
        "accuracy": float(best_payload["metrics"]["val_accuracy"]),
        "top5_accuracy": float(best_payload["metrics"]["val_top5_accuracy"]),
        "macro_f1": float(best_payload["metrics"]["val_macro_f1"]),
        "balanced_accuracy": float(best_payload["metrics"]["val_balanced_accuracy"]),
    }
    validate_metric_payload(best_val)
    old_val = baseline["old_preprocessing"]["metrics"]
    direct_before = baseline["direct_resize_pre_adaptation"]["metrics"]
    top1_change_pp = 100.0 * (best_val["accuracy"] - old_val["accuracy"])
    drop_limit_pp = float(evaluation["top1_drop_limit_percentage_points"])
    gate_passed = top1_change_pp >= -drop_limit_pp
    if gate_passed:
        _copy_best_checkpoint(output_dir / "best.pt", output_dir / "classification-init.pt")
    final = {
        "schema_version": 1,
        "stage": STAGE,
        "status": (
            "completed_pending_project_lead_acceptance"
            if gate_passed
            else "blocked_top1_regression_requires_analysis"
        ),
        "run_id": args.run_id,
        "data_version": preflight["data_version"],
        "input_git_commit": git_commit(repo_root),
        "input_checkpoint_sha256": baseline["checkpoint_sha256"],
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "selection_metric": "val_macro_f1",
        "before": {
            "old_preprocessing": old_val,
            "direct_resize": direct_before,
        },
        "best_val": best_val,
        "changes_percentage_points": {
            "best_vs_old_top1": top1_change_pp,
            "best_vs_old_top5": 100.0
            * (best_val["top5_accuracy"] - old_val["top5_accuracy"]),
            "best_vs_old_macro_f1": 100.0
            * (best_val["macro_f1"] - old_val["macro_f1"]),
            "best_vs_old_balanced_accuracy": 100.0
            * (best_val["balanced_accuracy"] - old_val["balanced_accuracy"]),
            "best_vs_direct_before_top1": 100.0
            * (best_val["accuracy"] - direct_before["accuracy"]),
            "best_vs_direct_before_macro_f1": 100.0
            * (best_val["macro_f1"] - direct_before["macro_f1"]),
        },
        "top1_gate": {
            "reference": "initial_checkpoint_with_old_preprocessing_on_same_val",
            "maximum_drop_percentage_points": drop_limit_pp,
            "passed": gate_passed,
        },
        "duration_seconds": duration,
        "wall_duration_seconds": time.perf_counter() - started,
        "duration_scope": "sum_train_and_val_epoch_seconds",
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "batch_size": int(training["batch_size"]),
        "workers": int(training["workers"]),
        "amp_enabled": amp_enabled,
        "preprocessing": direct_resize_preprocessing_spec(image_size),
        "checkpoint_reload": {
            "status": "passed",
            "epoch": int(best_payload["epoch"]),
            "strict_model_state": True,
        },
        "test_metrics_read": False,
        "environment": environment_versions(),
    }
    _write_json(output_dir / "metrics.json", final)
    _write_json(
        output_dir / "run-state.json",
        {
            "status": final["status"],
            "last_epoch": len(history),
            "best_epoch": best_epoch,
            "test_metrics_read": False,
        },
    )
    artifact_names = [
        "baseline-comparison.json",
        "best-confusion-matrix.json",
        "best-per-class.json",
        "best.pt",
        "class-balance.json",
        "history.json",
        "last.pt",
        "metrics.csv",
        "metrics.json",
        "preflight.json",
        "resolved-config.yaml",
        "run-state.json",
    ]
    if gate_passed:
        artifact_names.append("classification-init.pt")
    write_checksums(output_dir, artifact_names)
    return final


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--config", type=Path, default=Path("configs/j1.yaml"))
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = run_j1(args)
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
        OSError,
        KeyError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"J1 failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": metrics["status"],
                "run_id": metrics["run_id"],
                "best_epoch": metrics["best_epoch"],
                "best_val": metrics["best_val"],
                "top1_gate": metrics["top1_gate"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if metrics["top1_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
