"""J3 full alternating training on frozen DLCPD-25 and IP102 train/val splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterator
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Sampler

from dlcpd25_classifier.data import DLCPD25Dataset
from dlcpd25_classifier.detection import (
    DetectionClassMapping,
    DirectResizeDetectionTransform,
    IP102DetectionDataset,
    build_shared_detection_model,
)
from dlcpd25_classifier.detection.evaluation import evaluate_detection
from dlcpd25_classifier.training.a2 import (
    class_counts,
    clipped_inverse_frequency_weights,
)
from dlcpd25_classifier.training.joint import (
    build_joint_optimizer,
    capture_rng_state,
    collate_detection,
    learning_rates_by_name,
    restore_rng_state,
    set_task_trainability,
)
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
    sha256_file,
    write_checksums,
)
from dlcpd25_classifier.training.transforms import (
    build_direct_resize_eval_transform,
    build_direct_resize_train_transform,
    direct_resize_preprocessing_spec,
)

STAGE = "J3"
CHECKPOINT_SCHEMA_VERSION = 1
PROGRESS_FIELDS = (
    "epoch",
    "pair",
    "pairs_in_epoch",
    "global_pair",
    "classification_loss_mean",
    "detection_loss_mean",
    "backbone_learning_rate",
    "classification_head_learning_rate",
    "detection_head_learning_rate",
    "pairs_per_second",
    "elapsed_seconds",
    "estimated_remaining_seconds",
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
        encoding="utf-8", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            default=_json_scalar,
        )
        stream.write("\n")
    os.replace(temporary, path)


def _json_scalar(value: Any) -> Any:
    """Convert NumPy/COCO scalar values without weakening structured output."""
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if isinstance(converted, (bool, float, int, str)) or converted is None:
            return converted
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_inputs(config: dict[str, Any], project_root: Path) -> dict[str, str]:
    keys = (
        "classification_checkpoint",
        "classification_train_csv", "classification_val_csv",
        "classification_taxonomy", "detection_train_split",
        "detection_val_split", "detection_annotations", "detection_mapping",
    )
    forbidden = ("test.csv", "test.txt")
    for key in ("classification_train_csv", "classification_val_csv", "detection_train_split", "detection_val_split"):
        if Path(str(config[key])).name in forbidden:
            raise ValueError("J3 must never select a test split")
    expected_names = {
        "classification_train_csv": "train.csv",
        "classification_val_csv": "val.csv",
        "detection_train_split": "train.txt",
        "detection_val_split": "val.txt",
    }
    if any(Path(str(config[key])).name != name for key, name in expected_names.items()):
        raise ValueError("J3 requires the frozen train/val split filenames")
    expected = config.get("input_sha256")
    if not isinstance(expected, dict) or set(expected) != set(keys):
        raise ValueError("J3 requires exact SHA-256 values for all frozen inputs")
    actual: dict[str, str] = {}
    for key in keys:
        path = resolve_project_path(project_root, str(config[key]))
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[key] = sha256_file(path)
        if actual[key] != expected[key]:
            raise ValueError(f"J3 frozen input checksum mismatch: {key}")
    return actual


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("model") != {
        "image_size": 224, "classification_classes": 203, "detection_classes": 96
    }:
        raise ValueError("J3 model contract must remain 224/203/96")
    training = config.get("training", {})
    evaluation = config.get("evaluation", {})
    if training.get("task_ratio") != [1, 1]:
        raise ValueError("J3 requires a fixed 1:1 task ratio")
    if training.get("classification_amp") is not True:
        raise ValueError("J3 classification training must retain AMP")
    if training.get("detection_amp") is not False:
        raise ValueError("J3 detection forward must use FP32 for numerical stability")
    backbone_lr = float(training.get("backbone_learning_rate", -1))
    classification_lr = float(training.get("classification_head_learning_rate", -1))
    detection_lr = float(training.get("detection_head_learning_rate", -1))
    if not (backbone_lr == classification_lr and detection_lr == 10.0 * backbone_lr):
        raise ValueError(
            "J3 requires equal low learning rates for the pretrained backbone and "
            "classification head, with a 10x detection-head learning rate"
        )
    reference = float(evaluation.get("classification_top1_reference", -1))
    drop = float(evaluation.get("maximum_top1_drop_percentage_points", -1)) / 100.0
    threshold = float(evaluation.get("classification_top1_eligibility_threshold", -1))
    abort_threshold = float(evaluation.get("classification_top1_abort_threshold", -1))
    if not math.isclose(reference - drop, threshold, abs_tol=1e-12):
        raise ValueError("J3 classification eligibility threshold is inconsistent")
    if not 0.0 < abort_threshold < threshold:
        raise ValueError("J3 classification abort threshold must be below the eligibility gate")
    if evaluation.get("checkpoint_selection") != "classification_gate_then_highest_detection_map":
        raise ValueError("J3 checkpoint selection rule is not frozen")
    if evaluation.get("test_metrics_read") is not False:
        raise ValueError("J3 must not read test metrics")


class FixedOrderSampler(Sampler[int]):
    def __init__(self, indices: list[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class DetectionBatchStream:
    """Infinite shuffled detector batches with serializable exact sample position."""

    def __init__(self, dataset: IP102DetectionDataset, *, batch_size: int, workers: int, seed: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.workers = workers
        self.generator = torch.Generator().manual_seed(seed)
        self.permutation: list[int] = []
        self.cursor = 0
        self.cycles_completed = 0
        self._iterator: Iterator[Any] | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "permutation": self.permutation,
            "cursor": self.cursor,
            "cycles_completed": self.cycles_completed,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["batch_size"]) != self.batch_size:
            raise ValueError("detection stream batch size changed across resume")
        permutation = [int(index) for index in state["permutation"]]
        cursor = int(state["cursor"])
        if permutation and sorted(permutation) != list(range(len(self.dataset))):
            raise ValueError("invalid saved detection permutation")
        if not 0 <= cursor <= len(permutation):
            raise ValueError("invalid saved detection cursor")
        self.permutation = permutation
        self.cursor = cursor
        self.cycles_completed = int(state["cycles_completed"])
        self.generator.set_state(state["generator_state"].cpu())
        self._iterator = None

    def _start_cycle(self) -> None:
        self.permutation = torch.randperm(len(self.dataset), generator=self.generator).tolist()
        self.cursor = 0
        self._iterator = None

    def _build_iterator(self) -> None:
        remaining = self.permutation[self.cursor :]
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=FixedOrderSampler(remaining),
            num_workers=self.workers,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=collate_detection,
            drop_last=False,
        )
        self._iterator = iter(loader)

    def next(self) -> tuple[Tensor, list[dict[str, Tensor]]]:
        if not self.permutation or self.cursor >= len(self.permutation):
            if self.permutation:
                self.cycles_completed += 1
            self._start_cycle()
        if self._iterator is None:
            self._build_iterator()
        try:
            batch = next(self._iterator)
        except StopIteration as exc:
            raise RuntimeError("detection stream ended before saved permutation") from exc
        self.cursor += int(batch[0].shape[0])
        return batch


def checkpoint_is_eligible(classification_accuracy: float, threshold: float) -> bool:
    return classification_accuracy >= threshold


def checkpoint_is_better(
    candidate: dict[str, Any], best: dict[str, Any] | None, threshold: float
) -> bool:
    if not checkpoint_is_eligible(float(candidate["classification"]["accuracy"]), threshold):
        return False
    return best is None or float(candidate["detection"]["map"]) > float(best["detection"]["map"])


def _optimizer(model: nn.Module, training: dict[str, Any]) -> torch.optim.AdamW:
    return build_joint_optimizer(
        model,
        backbone_learning_rate=float(training["backbone_learning_rate"]),
        classification_head_learning_rate=float(
            training["classification_head_learning_rate"]
        ),
        detection_head_learning_rate=float(training["detection_head_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        fused=True,
    )


def _classification_step(
    model: nn.Module, images: Tensor, targets: Tensor, criterion: nn.Module,
    optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    device: torch.device, amp: bool,
) -> float:
    set_task_trainability(model, "classification")
    model.train()
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
        logits = model.forward_classification(images)
        loss = criterion(logits, targets)
    if logits.shape != (targets.shape[0], 203) or not torch.isfinite(loss):
        raise RuntimeError("J3 classification step produced invalid values")
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach())


def _detection_step(
    model: nn.Module, images: Tensor, targets: list[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    device: torch.device, amp: bool,
) -> float:
    set_task_trainability(model, "detection")
    model.train()
    images = images.to(device, non_blocking=True)
    targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
        components = model.forward_detection(images, targets)
        loss = sum(components.values())
    non_finite = [
        name for name, value in components.items() if not torch.isfinite(value)
    ]
    if not components or non_finite or not torch.isfinite(loss):
        raise RuntimeError(
            f"J3 detection step produced non-finite loss components: {non_finite}"
        )
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach())


def _append_progress(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROGRESS_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row[key] for key in PROGRESS_FIELDS})
        stream.flush()


def _detection_counts(dataset: IP102DetectionDataset) -> dict[int, int]:
    if dataset._records is None:
        raise ValueError("J3 requires frozen T0 detection annotations")
    counts: Counter[int] = Counter()
    for image_id in dataset.image_ids:
        for obj in dataset._records[image_id]["objects"]:
            counts[dataset.mapping.from_ip102(int(obj["ip102_class_id"])).detector_label] += 1
    return dict(counts)


def evaluate_joint_classification(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float | int]], list[list[int]]]:
    """Evaluate the classification head without invoking a nonexistent default forward."""
    model.eval()
    accumulator = ClassificationMetrics()
    loss_sum = 0.0
    samples = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model.forward_classification(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError("J3 classification validation produced non-finite values")
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


def _save_j3_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR, scaler: torch.amp.GradScaler,
    detection_stream: DetectionBatchStream, *, epoch: int, global_pair: int,
    history: list[dict[str, Any]], metadata: dict[str, Any],
) -> None:
    _atomic_checkpoint(
        path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "architecture": "joint-resnet50-fasterrcnn",
            "classification_classes": 203,
            "detection_classes": 96,
            "image_size": 224,
            "completed_epoch": epoch,
            "global_pair": global_pair,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "rng_state": capture_rng_state(),
            "detection_stream_state": detection_stream.state_dict(),
            "history": history,
            "metadata": metadata,
        },
    )


def _load_j3_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR, scaler: torch.amp.GradScaler,
    detection_stream: DetectionBatchStream,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "architecture": "joint-resnet50-fasterrcnn",
        "classification_classes": 203,
        "detection_classes": 96,
        "image_size": 224,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("unsupported J3 checkpoint contract")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    scaler.load_state_dict(payload["scaler_state_dict"])
    restore_rng_state(payload["rng_state"])
    detection_stream.load_state_dict(payload["detection_stream_state"])
    return payload


def run_j3_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the J3 initialization without creating a run or taking a train step."""
    repo_root = args.repo_root.resolve()
    project_root = repo_root / "project"
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_config(config_path)
    _validate_config(config)
    inputs = _verify_inputs(config, project_root)
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("J3 preflight requires CUDA")

    mapping_path = resolve_project_path(project_root, str(config["detection_mapping"]))
    mapping = DetectionClassMapping(mapping_path)
    class_checkpoint = resolve_project_path(project_root, str(config["classification_checkpoint"]))
    model, model_info = build_shared_detection_model(
        class_checkpoint, mapping, trainable_backbone_layers=5
    )
    model.to(device)
    data_root = resolve_project_path(project_root, str(config["classification_data_root"]))
    taxonomy = resolve_project_path(project_root, str(config["classification_taxonomy"]))
    class_val = DLCPD25Dataset(
        data_root,
        resolve_project_path(project_root, str(config["classification_val_csv"])),
        taxonomy,
        transform=build_direct_resize_eval_transform(224),
    )
    if len(class_val) != 22178:
        raise ValueError("J3 frozen classification validation cardinality changed")
    evaluation = config["evaluation"]
    loader = DataLoader(
        class_val,
        batch_size=int(evaluation["classification_batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["workers"]),
        pin_memory=True,
        persistent_workers=False,
    )
    metrics, _, _ = evaluate_joint_classification(
        model, loader, nn.CrossEntropyLoss(), device
    )
    threshold = float(evaluation["classification_top1_eligibility_threshold"])
    return {
        "stage": STAGE,
        "status": "eligible_for_training" if checkpoint_is_eligible(
            float(metrics["accuracy"]), threshold
        ) else "blocked_initial_classification_below_eligibility_gate",
        "classification_val": metrics,
        "classification_top1_threshold": threshold,
        "input_sha256": inputs,
        "model": model_info.__dict__,
        "test_metrics_read": False,
        "artifacts_written": False,
        "training_steps_executed": 0,
    }


def run_j3(args: argparse.Namespace) -> dict[str, Any]:
    if args.preflight_only:
        return run_j3_preflight(args)
    repo_root = args.repo_root.resolve()
    project_root = repo_root / "project"
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_config(config_path)
    _validate_config(config)
    inputs = _verify_inputs(config, project_root)
    output_dir = resolve_project_path(project_root, str(config["output_dir"])) / args.run_id
    resume_path = output_dir / "joint-last.pt"
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite J3 run: {output_dir}")
    if args.resume and not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("J3 requires CUDA")
    training = config["training"]
    evaluation = config["evaluation"]
    mapping_path = resolve_project_path(project_root, str(config["detection_mapping"]))
    mapping = DetectionClassMapping(mapping_path)
    class_checkpoint = resolve_project_path(project_root, str(config["classification_checkpoint"]))
    model, model_info = build_shared_detection_model(class_checkpoint, mapping, trainable_backbone_layers=5)
    optimizer = _optimizer(model, training)
    scheduler = CosineAnnealingLR(optimizer, T_max=int(training["epochs"]), eta_min=0.0)
    classification_amp = bool(training["classification_amp"])
    detection_amp = bool(training["detection_amp"])
    scaler = torch.amp.GradScaler(
        "cuda", enabled=classification_amp, init_scale=float(training["amp_init_scale"]),
        growth_interval=int(training["amp_growth_interval"]),
    )
    model.to(device)

    data_root = resolve_project_path(project_root, str(config["classification_data_root"]))
    taxonomy = resolve_project_path(project_root, str(config["classification_taxonomy"]))
    class_train = DLCPD25Dataset(
        data_root, resolve_project_path(project_root, str(config["classification_train_csv"])), taxonomy,
        transform=build_direct_resize_train_transform(224),
    )
    class_val = DLCPD25Dataset(
        data_root, resolve_project_path(project_root, str(config["classification_val_csv"])), taxonomy,
        transform=build_direct_resize_eval_transform(224),
    )
    detection_root = resolve_project_path(project_root, str(config["detection_voc_root"]))
    annotations = resolve_project_path(project_root, str(config["detection_annotations"]))
    detection_train = IP102DetectionDataset(
        detection_root, resolve_project_path(project_root, str(config["detection_train_split"])), mapping_path,
        annotations_path=annotations, transforms=DirectResizeDetectionTransform(224),
    )
    detection_val = IP102DetectionDataset(
        detection_root, resolve_project_path(project_root, str(config["detection_val_split"])), mapping_path,
        annotations_path=annotations, transforms=DirectResizeDetectionTransform(224),
    )
    if (len(class_train), len(class_val), len(detection_train), len(detection_val)) != (177021, 22178, 12142, 3036):
        raise ValueError("J3 frozen dataset cardinalities changed")
    workers = int(training["workers"])
    detection_stream = DetectionBatchStream(
        detection_train, batch_size=int(training["detection_batch_size"]), workers=workers,
        seed=int(config["seed"]) + 1000,
    )
    class_val_loader = DataLoader(
        class_val, batch_size=int(evaluation["classification_batch_size"]), shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=False,
    )
    detection_val_loader = DataLoader(
        detection_val, batch_size=int(evaluation["detection_batch_size"]), shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=False,
        collate_fn=collate_detection,
    )
    class_names = {
        label: mapping.from_detector(label).dlcpd25_name
        for label in range(1, mapping.num_detector_classes + 1)
    }
    train_object_counts = _detection_counts(detection_train)
    class_weights = clipped_inverse_frequency_weights(class_counts(class_train)).to(device)
    class_train_criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=float(training["label_smoothing"])
    )
    class_val_criterion = nn.CrossEntropyLoss()

    history: list[dict[str, Any]] = []
    start_epoch = 1
    global_pair = 0
    if args.resume:
        resumed = _load_j3_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, detection_stream
        )
        history = list(resumed["history"])
        start_epoch = int(resumed["completed_epoch"]) + 1
        global_pair = int(resumed["global_pair"])
    threshold = float(evaluation["classification_top1_eligibility_threshold"])
    abort_threshold = float(evaluation["classification_top1_abort_threshold"])
    if not args.resume:
        baseline_metrics, _, _ = evaluate_joint_classification(
            model, class_val_loader, class_val_criterion, device
        )
        _atomic_json(
            output_dir / "initial-classification-validation.json", baseline_metrics
        )
        if float(baseline_metrics["accuracy"]) < threshold:
            _atomic_json(
                output_dir / "run-state.json",
                {
                    "status": "blocked_initial_classification_below_eligibility_gate",
                    "classification_val": baseline_metrics,
                    "classification_top1_threshold": threshold,
                    "test_metrics_read": False,
                },
            )
            raise RuntimeError(
                "J3 initial joint checkpoint is below the classification eligibility gate"
            )
    eligible_history = [record for record in history if record["eligible"]]
    best_record = max(eligible_history, key=lambda record: record["detection"]["map"], default=None)
    metadata = {
        "stage": STAGE, "run_id": args.run_id, "input_git_commit": git_commit(repo_root),
        "input_sha256": inputs, "model": model_info.__dict__, "task_ratio": [1, 1],
        "preprocessing": direct_resize_preprocessing_spec(224), "test_metrics_read": False,
    }
    resolved = {**config, "stage": STAGE, "run_id": args.run_id, "input_git_commit": metadata["input_git_commit"]}
    if not args.resume:
        (output_dir / "resolved-config.yaml").write_text(
            yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        _atomic_json(output_dir / "history.json", history)
    _atomic_json(
        output_dir / "run-state.json",
        {"status": "running", "epoch": start_epoch, "pair": 0, "test_metrics_read": False},
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    total_epochs = int(training["epochs"])
    pairs_per_epoch = math.ceil(len(class_train) / int(training["classification_batch_size"]))
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
        class_loader = DataLoader(
            class_train, batch_size=int(training["classification_batch_size"]), shuffle=True,
            generator=epoch_generator, num_workers=workers, pin_memory=True,
            persistent_workers=False, drop_last=False,
        )
        epoch_started = time.perf_counter()
        class_loss_sum = detection_loss_sum = 0.0
        class_samples = detection_samples = 0
        interval = int(training["progress_interval_pairs"])
        for pair, (class_images, class_targets) in enumerate(class_loader, 1):
            class_loss = _classification_step(
                model, class_images, class_targets, class_train_criterion,
                optimizer, scaler, device, classification_amp,
            )
            detection_images, detection_targets = detection_stream.next()
            detection_loss = _detection_step(
                model, detection_images, detection_targets, optimizer, scaler, device,
                detection_amp,
            )
            global_pair += 1
            class_loss_sum += class_loss * len(class_targets)
            class_samples += len(class_targets)
            detection_loss_sum += detection_loss * len(detection_targets)
            detection_samples += len(detection_targets)
            if pair % interval == 0 or pair == pairs_per_epoch:
                elapsed = time.perf_counter() - epoch_started
                rate = pair / elapsed
                remaining_pairs = (pairs_per_epoch - pair) + (total_epochs - epoch) * pairs_per_epoch
                learning_rates = learning_rates_by_name(optimizer)
                progress = {
                    "epoch": epoch, "pair": pair, "pairs_in_epoch": pairs_per_epoch,
                    "global_pair": global_pair,
                    "classification_loss_mean": class_loss_sum / class_samples,
                    "detection_loss_mean": detection_loss_sum / detection_samples,
                    "backbone_learning_rate": learning_rates["backbone"],
                    "classification_head_learning_rate": learning_rates[
                        "classification_head"
                    ],
                    "detection_head_learning_rate": learning_rates["detection_head"],
                    "pairs_per_second": rate, "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": remaining_pairs / max(rate, 1e-9),
                }
                _append_progress(output_dir / "progress.csv", progress)
                _atomic_json(
                    output_dir / "run-state.json",
                    {"status": "running", **progress, "validation": "pending", "test_metrics_read": False},
                )
        training_duration = time.perf_counter() - epoch_started
        _atomic_json(
            output_dir / "run-state.json",
            {"status": "running", "epoch": epoch, "pair": pairs_per_epoch, "validation": "classification", "test_metrics_read": False},
        )
        class_metrics, class_per_class, confusion = evaluate_joint_classification(
            model, class_val_loader, class_val_criterion, device
        )
        model.detector.roi_heads.score_thresh = float(evaluation["detection_score_threshold_for_ap"])
        model.detector.roi_heads.nms_thresh = float(evaluation["detection_nms_iou_threshold"])
        model.detector.roi_heads.detections_per_img = int(evaluation["detection_max_detections_per_image"])
        _atomic_json(
            output_dir / "run-state.json",
            {"status": "running", "epoch": epoch, "pair": pairs_per_epoch, "validation": "detection", "classification_val": class_metrics, "test_metrics_read": False},
        )
        detection_metrics, detection_per_class = evaluate_detection(
            model, detection_val_loader, device,
            score_threshold=float(evaluation["detection_score_threshold_for_precision_recall"]),
            class_names=class_names, train_object_counts=train_object_counts,
        )
        learning_rates = learning_rates_by_name(optimizer)
        record = {
            "epoch": epoch,
            "learning_rates": learning_rates,
            "train": {
                "classification_loss": class_loss_sum / class_samples,
                "detection_loss": detection_loss_sum / detection_samples,
                "classification_samples": class_samples,
                "detection_samples": detection_samples,
                "pairs": pairs_per_epoch,
                "duration_seconds": training_duration,
                "pairs_per_second": pairs_per_epoch / training_duration,
            },
            "classification": class_metrics,
            "detection": detection_metrics,
            "eligible": checkpoint_is_eligible(float(class_metrics["accuracy"]), threshold),
            "classification_top1_threshold": threshold,
        }
        history.append(record)
        scheduler.step()
        # Preserve the completed training and both validation summaries before
        # writing secondary reports so a reporting error cannot lose an epoch.
        _save_j3_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, detection_stream,
            epoch=epoch, global_pair=global_pair, history=history, metadata=metadata,
        )
        _atomic_json(output_dir / f"epoch-{epoch}-classification-per-class.json", class_per_class)
        _atomic_json(output_dir / f"epoch-{epoch}-classification-confusion.json", confusion)
        _atomic_json(output_dir / f"epoch-{epoch}-detection-per-class-ap.json", detection_per_class)
        _atomic_json(output_dir / "history.json", history)
        if checkpoint_is_better(record, best_record, threshold):
            best_record = record
            shutil.copyfile(resume_path, output_dir / "joint-best.pt")
            _atomic_json(output_dir / "best-selection.json", record)
        _atomic_json(
            output_dir / "run-state.json",
            {"status": "running", "epoch": epoch, "pair": pairs_per_epoch,
             "validation": "completed", "latest": record,
             "best_epoch": best_record["epoch"] if best_record else None,
             "test_metrics_read": False},
        )
        if float(class_metrics["accuracy"]) < abort_threshold:
            break
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            break
    paused = args.stop_after_epoch is not None and len(history) < total_epochs
    if history and float(history[-1]["classification"]["accuracy"]) < abort_threshold:
        status = "blocked_classification_below_safety_floor"
    elif paused:
        status = "paused_after_requested_epoch"
    elif best_record is None:
        status = "blocked_no_checkpoint_met_classification_gate"
    else:
        status = "completed_pending_project_lead_acceptance"
    final = {
        "schema_version": 1, "stage": STAGE, "status": status, "run_id": args.run_id,
        "input_git_commit": metadata["input_git_commit"], "input_sha256": inputs,
        "train_samples": {"classification": len(class_train), "detection": len(detection_train)},
        "val_samples": {"classification": len(class_val), "detection": len(detection_val)},
        "epochs_completed": len(history), "best_epoch": best_record["epoch"] if best_record else None,
        "best_validation": best_record, "classification_gate": {
            "reference": float(evaluation["classification_top1_reference"]),
            "maximum_drop_percentage_points": float(evaluation["maximum_top1_drop_percentage_points"]),
            "threshold": threshold,
            "abort_threshold": abort_threshold,
        },
        "task_ratio": [1, 1], "duration_seconds_this_process": time.perf_counter() - run_started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(), "amp_final_scale": scaler.get_scale(),
        "precision": {
            "classification": "AMP FP16",
            "detection": "FP32",
        },
        "test_metrics_read": False,
        "environment": {
            **environment_versions(),
            "pycocotools": importlib_metadata.version("pycocotools"),
        },
    }
    _atomic_json(output_dir / "metrics.json", final)
    _atomic_json(output_dir / "run-state.json", {"status": status, "latest": history[-1], "best_epoch": final["best_epoch"], "test_metrics_read": False})
    artifact_names = [
        "resolved-config.yaml", "history.json", "metrics.json", "run-state.json",
        "progress.csv", "joint-last.pt", "initial-classification-validation.json",
    ]
    artifact_names.extend(
        name for epoch in range(1, len(history) + 1)
        for name in (
            f"epoch-{epoch}-classification-per-class.json",
            f"epoch-{epoch}-classification-confusion.json",
            f"epoch-{epoch}-detection-per-class-ap.json",
        )
    )
    if best_record is not None:
        artifact_names.extend(["joint-best.pt", "best-selection.json"])
    write_checksums(output_dir, artifact_names)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--config", type=Path, default=Path("configs/j3.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate J2 initialization on classification val without creating a J3 run",
    )
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    if args.preflight_only and (args.resume or args.stop_after_epoch is not None):
        parser.error("--preflight-only cannot be combined with resume or epoch controls")
    if not args.preflight_only and not args.run_id:
        parser.error("--run-id is required for a training run")
    if args.stop_after_epoch is not None and args.stop_after_epoch <= 0:
        parser.error("--stop-after-epoch must be positive")
    return args


def main() -> int:
    try:
        result = run_j3(parse_args())
    except (FileExistsError, FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"J3 failed: {exc}", file=sys.stderr)
        return 1
    output = {"status": result["status"]}
    if "best_epoch" in result:
        output["best_epoch"] = result["best_epoch"]
    if "classification_val" in result:
        output["classification_val_top1"] = result["classification_val"]["accuracy"]
        output["classification_top1_threshold"] = result[
            "classification_top1_threshold"
        ]
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
