"""A2 full ResNet-50 train/validation engine."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dlcpd25_classifier.data import DLCPD25Dataset
from dlcpd25_classifier.models import build_classification_model
from dlcpd25_classifier.training.checkpoint import load_checkpoint, save_checkpoint
from dlcpd25_classifier.training.metrics import (
    ClassificationMetrics,
    epoch_duration_seconds,
    validate_metric_payload,
)
from dlcpd25_classifier.training.preflight import run_preflight
from dlcpd25_classifier.training.progress import append_progress
from dlcpd25_classifier.training.transforms import (
    build_eval_transform,
    build_train_transform,
)

LOSS_STRATEGIES = ("ce", "weighted_ce")


def class_counts(dataset: DLCPD25Dataset) -> list[int]:
    counts = Counter(record.class_id for record in dataset.records)
    if sorted(counts) != list(range(203)):
        raise ValueError("training split must cover all 203 classes")
    return [counts[class_id] for class_id in range(203)]


def clipped_inverse_frequency_weights(
    counts: list[int], max_raw_weight: float = 10.0
) -> torch.Tensor:
    if len(counts) != 203 or any(count <= 0 for count in counts):
        raise ValueError("class weights require 203 positive class counts")
    total = float(sum(counts))
    raw = torch.tensor(
        [total / (len(counts) * count) for count in counts], dtype=torch.float32
    )
    clipped = raw.clamp(max=max_raw_weight)
    return clipped / clipped.mean()


def make_scheduler(
    optimizer: AdamW, total_epochs: int, warmup_epochs: int
) -> LambdaLR:
    if total_epochs <= 0 or not 0 <= warmup_epochs < total_epochs:
        raise ValueError("warmup_epochs must be in [0, total_epochs)")

    def multiplier(epoch_index: int) -> float:
        if warmup_epochs and epoch_index < warmup_epochs:
            return (epoch_index + 1) / warmup_epochs
        decay_epochs = max(total_epochs - warmup_epochs - 1, 1)
        progress = min(max((epoch_index - warmup_epochs) / decay_epochs, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, multiplier)


def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    samples = 0
    started = time.perf_counter()
    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        if logits.shape != (targets.shape[0], 203) or not torch.isfinite(loss):
            raise RuntimeError("A2 training produced invalid logits or loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.detach()) * targets.numel()
        correct += int((logits.argmax(dim=1) == targets).sum())
        samples += targets.numel()
        if max_batches is not None and batch_index >= max_batches:
            break
    elapsed = time.perf_counter() - started
    return {
        "loss": loss_sum / samples,
        "accuracy": correct / samples,
        "samples": float(samples),
        "duration_seconds": elapsed,
        "images_per_second": samples / elapsed,
    }


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float | int]], list[list[int]]]:
    model.eval()
    accumulator = ClassificationMetrics()
    loss_sum = 0.0
    samples = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError("A2 validation produced non-finite values")
            accumulator.update(logits, targets)
            loss_sum += float(loss) * targets.numel()
            samples += targets.numel()
    summary, per_class = accumulator.compute()
    validate_metric_payload(summary)
    elapsed = time.perf_counter() - started
    return {
        "loss": loss_sum / samples,
        **summary,
        "samples": float(samples),
        "duration_seconds": elapsed,
        "images_per_second": samples / elapsed,
    }, per_class, accumulator.as_serializable_confusion()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_loaders(
    data_root: Path,
    split_dir: Path,
    taxonomy_path: Path,
    image_size: int,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[DLCPD25Dataset, DLCPD25Dataset, DataLoader, DataLoader, torch.Generator]:
    train_dataset = DLCPD25Dataset(
        data_root,
        split_dir / "train.csv",
        taxonomy_path,
        transform=build_train_transform(image_size),
    )
    val_dataset = DLCPD25Dataset(
        data_root,
        split_dir / "val.csv",
        taxonomy_path,
        transform=build_eval_transform(image_size),
    )
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, drop_last=False, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    return train_dataset, val_dataset, train_loader, val_loader, generator


def resource_probe(args: Any, common: dict[str, Any]) -> dict[str, Any]:
    config = common["config"]
    device = common["device"]
    train_dataset, _val_dataset, train_loader, _val_loader, generator = _build_loaders(
        common["data_root"],
        common["split_dir"],
        common["taxonomy_path"],
        int(config["model"]["image_size"]),
        args.batch_size,
        args.workers,
        int(config["seed"]),
    )
    generator.manual_seed(int(config["seed"]) + 1)
    model, _model_info = build_classification_model(
        str(config["model"]["name"]), 203, bool(config["model"]["pretrained"])
    )
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=float(config["training"]["weight_decay"]),
        fused=True,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["training"]["label_smoothing"]))
    amp_enabled = bool(config["training"]["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    metrics = train_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        scaler,
        device,
        amp_enabled,
        max_batches=args.probe_batches,
    )
    return {
        **metrics,
        "dataset_samples": len(train_dataset),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def prepare_common(args: Any) -> dict[str, Any]:
    from dlcpd25_classifier.training.train import (
        load_config,
        resolve_project_path,
        set_seed,
    )

    repo_root = args.repo_root.resolve()
    project_root = repo_root / "project"
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A2 requires CUDA on the configured training machine")
    set_seed(int(config["seed"]))
    return {
        "repo_root": repo_root,
        "project_root": project_root,
        "config_path": config_path,
        "config": config,
        "device": device,
        "data_root": resolve_project_path(project_root, str(config["data_root"])),
        "split_dir": resolve_project_path(project_root, str(config["split_dir"])),
        "taxonomy_path": resolve_project_path(project_root, str(config["taxonomy"])),
        "output_root": resolve_project_path(project_root, str(config["output_dir"])),
    }


def run_a2_training(args: Any) -> dict[str, Any]:
    from dlcpd25_classifier.training.train import (
        environment_versions,
        git_commit,
        preprocessing_spec,
        write_checksums,
    )

    if args.loss_strategy not in LOSS_STRATEGIES:
        raise ValueError(f"unsupported A2 loss strategy: {args.loss_strategy}")
    common = prepare_common(args)
    if args.a2_resource_probe:
        return resource_probe(args, common)

    config = common["config"]
    repo_root = common["repo_root"]
    output_dir = common["output_root"] / args.run_id
    if output_dir.exists() and args.resume is None:
        raise FileExistsError(f"refusing to overwrite A2 run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=args.resume is not None)
    release_path = repo_root / "artifacts/data/v1/d5-r1/data-v1-release.json"
    preflight = run_preflight(repo_root, release_path, common["data_root"])
    train_dataset, val_dataset, train_loader, val_loader, generator = _build_loaders(
        common["data_root"],
        common["split_dir"],
        common["taxonomy_path"],
        int(config["model"]["image_size"]),
        args.batch_size,
        args.workers,
        int(config["seed"]),
    )
    counts = class_counts(train_dataset)
    weights = clipped_inverse_frequency_weights(counts)
    class_weight = weights.to(common["device"]) if args.loss_strategy == "weighted_ce" else None
    train_criterion = nn.CrossEntropyLoss(
        weight=class_weight,
        label_smoothing=float(config["training"]["label_smoothing"]),
    )
    val_criterion = nn.CrossEntropyLoss()
    model, model_info = build_classification_model(
        str(config["model"]["name"]), 203, bool(config["model"]["pretrained"])
    )
    model.to(common["device"])
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=float(config["training"]["weight_decay"]),
        fused=True,
    )
    scheduler = make_scheduler(optimizer, args.epochs, int(config["training"]["warmup_epochs"]))
    amp_enabled = bool(config["training"]["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_epoch = 0
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    if args.resume is not None:
        resume_path = Path(args.resume).resolve()
        payload = load_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
            scaler,
            expected_architecture="resnet50",
            expected_num_classes=203,
            map_location=common["device"],
        )
        metadata = payload.get("metadata", {})
        if metadata.get("run_id") != args.run_id or metadata.get("loss_strategy") != args.loss_strategy:
            raise ValueError("resume checkpoint does not belong to this A2 run")
        start_epoch = int(payload["epoch"]) + 1
        best_epoch = int(metadata.get("best_epoch", 0))
        best_macro_f1 = float(metadata.get("best_macro_f1", -1.0))
        epochs_without_improvement = int(metadata.get("epochs_without_improvement", 0))
        history_path = output_dir / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []

    resolved = {
        **config,
        "stage": "A2",
        "run_id": args.run_id,
        "input_git_commit": git_commit(repo_root),
        "actual": {
            "loss_strategy": args.loss_strategy,
            "class_weight_formula": (
                "inverse_frequency_clipped_at_10_then_mean_normalized"
                if args.loss_strategy == "weighted_ce"
                else "none"
            ),
            "batch_size": args.batch_size,
            "workers": args.workers,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "scheduler": "linear_warmup_then_cosine",
            "selection_metric": "val_macro_f1",
            "early_stopping_patience": int(config["training"]["early_stopping_patience"]),
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
            "counts": counts,
            "weights": weights.tolist() if args.loss_strategy == "weighted_ce" else None,
            "formula": resolved["actual"]["class_weight_formula"],
        },
    )
    _write_json(
        output_dir / "run-state.json",
        {"status": "running", "start_epoch": start_epoch, "test_metrics_read": False},
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    patience = int(config["training"]["early_stopping_patience"])
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        generator.manual_seed(int(config["seed"]) + epoch)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_epoch(
            model,
            train_loader,
            train_criterion,
            optimizer,
            scaler,
            common["device"],
            amp_enabled,
        )
        val_metrics, per_class, confusion = evaluate_epoch(
            model, val_loader, val_criterion, common["device"]
        )
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        last_epoch = epoch
        improved = val_metrics["macro_f1"] > best_macro_f1 + 1e-6
        if improved:
            best_epoch = epoch
            best_macro_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            _write_json(output_dir / "best-per-class.json", per_class)
            _write_json(output_dir / "best-confusion-matrix.json", confusion)
        else:
            epochs_without_improvement += 1
        scheduler.step()
        checkpoint_metadata = {
            "stage": "A2",
            "run_id": args.run_id,
            "loss_strategy": args.loss_strategy,
            "data_version": preflight["data_version"],
            "taxonomy_sha256": preflight["taxonomy"]["sha256"],
            "git_commit": git_commit(repo_root),
            "best_epoch": best_epoch,
            "best_macro_f1": best_macro_f1,
            "epochs_without_improvement": epochs_without_improvement,
            "test_metrics_read": False,
        }
        checkpoint_metrics = {
            "val_accuracy": val_metrics["accuracy"],
            "val_top5_accuracy": val_metrics["top5_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
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
        if epochs_without_improvement >= patience:
            break

    duration = epoch_duration_seconds(history)
    fresh_model, _info = build_classification_model("resnet50", 203, pretrained=False)
    best_payload = load_checkpoint(
        output_dir / "best.pt",
        fresh_model,
        expected_architecture="resnet50",
        expected_num_classes=203,
    )
    best_metrics = {
        "accuracy": float(best_payload["metrics"]["val_accuracy"]),
        "top5_accuracy": float(best_payload["metrics"]["val_top5_accuracy"]),
        "macro_f1": float(best_payload["metrics"]["val_macro_f1"]),
        "balanced_accuracy": float(best_payload["metrics"]["val_balanced_accuracy"]),
    }
    validate_metric_payload(best_metrics)
    final = {
        "schema_version": 1,
        "stage": "A2",
        "status": "completed_pending_project_lead_acceptance",
        "run_id": args.run_id,
        "loss_strategy": args.loss_strategy,
        "data_version": preflight["data_version"],
        "input_git_commit": git_commit(repo_root),
        "model": model_info.__dict__,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs_completed": last_epoch,
        "best_epoch": best_epoch,
        "best_val": best_metrics,
        "duration_seconds": duration,
        "duration_scope": "sum_train_and_val_epoch_seconds",
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "amp_enabled": amp_enabled,
        "preprocessing": preprocessing_spec(int(config["model"]["image_size"])),
        "checkpoint_reload": {"status": "passed", "epoch": int(best_payload["epoch"])},
        "test_metrics_read": False,
        "environment": environment_versions(),
    }
    _write_json(output_dir / "metrics.json", final)
    _write_json(
        output_dir / "run-state.json",
        {
            "status": "completed_pending_project_lead_acceptance",
            "last_epoch": last_epoch,
            "best_epoch": best_epoch,
            "test_metrics_read": False,
        },
    )
    artifact_names = [
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
    write_checksums(output_dir, artifact_names)
    return final
