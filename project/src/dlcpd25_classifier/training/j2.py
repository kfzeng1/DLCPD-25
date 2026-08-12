"""J2 alternating classification/detection smoke trainer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from dlcpd25_classifier.data import DLCPD25Dataset
from dlcpd25_classifier.detection import (
    DetectionClassMapping,
    DirectResizeDetectionTransform,
    IP102DetectionDataset,
    build_shared_detection_model,
)
from dlcpd25_classifier.detection.checkpoint import (
    load_joint_checkpoint,
    save_joint_checkpoint,
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
from dlcpd25_classifier.training.transforms import build_direct_resize_train_transform


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _verify_frozen_inputs(config: dict[str, Any], project_root: Path) -> dict[str, str]:
    input_keys = (
        "classification_checkpoint",
        "classification_train_csv",
        "classification_taxonomy",
        "detection_train_split",
        "detection_annotations",
        "detection_mapping",
    )
    if Path(str(config["classification_train_csv"])).name != "train.csv":
        raise ValueError("J2 classification input must be the frozen train.csv")
    if Path(str(config["detection_train_split"])).name != "train.txt":
        raise ValueError("J2 detection input must be the frozen train.txt")
    expected = config.get("input_sha256")
    if not isinstance(expected, dict) or set(expected) != set(input_keys):
        raise ValueError("J2 requires exact SHA-256 values for every frozen input")
    actual: dict[str, str] = {}
    for key in input_keys:
        path = resolve_project_path(project_root, str(config[key]))
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[key] = sha256_file(path)
        if actual[key] != expected[key]:
            raise ValueError(f"J2 frozen input checksum mismatch: {key}")
    return actual


def _collate_detection(
    batch: list[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def _set_task_trainability(model: nn.Module, task: str) -> None:
    if task not in {"classification", "detection"}:
        raise ValueError(f"unknown J2 task: {task}")
    for parameter in model.shared_body.parameters():
        parameter.requires_grad = True
    for parameter in model.classification_head.parameters():
        parameter.requires_grad = task == "classification"
    for module in model.detection_head:
        for parameter in module.parameters():
            parameter.requires_grad = task == "detection"


def _finite_gradients(parameters: list[nn.Parameter]) -> bool:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    return bool(gradients) and all(torch.isfinite(gradient).all() for gradient in gradients)


def _parameter_snapshot(parameters: list[nn.Parameter]) -> list[Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _changed(parameters: list[nn.Parameter], before: list[Tensor]) -> bool:
    return any(not torch.equal(parameter.detach(), old) for parameter, old in zip(parameters, before))


def _predictions_equal(
    first: list[dict[str, Tensor]], second: list[dict[str, Tensor]]
) -> bool:
    if len(first) != len(second):
        return False
    return all(
        first_item.keys() == second_item.keys()
        and all(torch.equal(first_item[key], second_item[key]) for key in first_item)
        for first_item, second_item in zip(first, second)
    )


def _classification_step(
    model: nn.Module,
    images: Tensor,
    targets: Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, dict[str, bool]]:
    _set_task_trainability(model, "classification")
    model.train()
    images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
    shared = list(model.shared_body.parameters())
    classification = list(model.classification_head.parameters())
    detection = [parameter for module in model.detection_head for parameter in module.parameters()]
    before_shared = _parameter_snapshot(shared)
    before_detection = _parameter_snapshot(detection)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
        logits = model.forward_classification(images)
        loss = criterion(logits, targets)
    if logits.shape != (targets.shape[0], 203) or not torch.isfinite(loss):
        raise RuntimeError("classification step produced invalid logits or loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    finite = _finite_gradients(shared + classification)
    if not finite:
        raise RuntimeError("classification step produced non-finite gradients")
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach()), {
        "shared_body_updated": _changed(shared, before_shared),
        "classification_head_grad_finite": _finite_gradients(classification),
        "detection_head_unchanged": not _changed(detection, before_detection),
    }


def _detection_step(
    model: nn.Module,
    images: Tensor,
    targets: list[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, dict[str, bool]]:
    _set_task_trainability(model, "detection")
    model.train()
    images = images.to(device, non_blocking=True)
    targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]
    shared = list(model.shared_body.parameters())
    classification = list(model.classification_head.parameters())
    detection = [parameter for module in model.detection_head for parameter in module.parameters()]
    before_shared = _parameter_snapshot(shared)
    before_classification = _parameter_snapshot(classification)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
        losses = model.forward_detection(images, targets)
        if not isinstance(losses, dict) or not losses:
            raise RuntimeError("detection training did not return loss components")
        loss = sum(losses.values())
    if not torch.isfinite(loss):
        raise RuntimeError("detection step produced a non-finite loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    finite = _finite_gradients(shared + detection)
    if not finite:
        raise RuntimeError("detection step produced non-finite gradients")
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach()), {
        "shared_body_updated": _changed(shared, before_shared),
        "shared_grad_finite": _finite_gradients(shared),
        "detection_head_grad_finite": _finite_gradients(detection),
        "classification_head_unchanged": not _changed(classification, before_classification),
    }


def _build_classification_loader(config: dict[str, Any], repo_root: Path) -> tuple[DataLoader, list[int]]:
    train_dataset = DLCPD25Dataset(
        resolve_project_path(repo_root / "project", str(config["classification_data_root"])),
        resolve_project_path(repo_root / "project", str(config["classification_train_csv"])),
        resolve_project_path(repo_root / "project", str(config["classification_taxonomy"])),
        transform=build_direct_resize_train_transform(224),
    )
    sample_count = min(int(config["smoke"]["classification_samples"]), len(train_dataset))
    indices: list[int] = []
    covered_classes: set[int] = set()
    for index, record in enumerate(train_dataset.records):
        if record.class_id in covered_classes:
            continue
        indices.append(index)
        covered_classes.add(record.class_id)
        if len(indices) == sample_count:
            break
    if len(indices) != sample_count:
        raise ValueError("not enough distinct classification classes for J2 smoke")
    generator = torch.Generator().manual_seed(int(config["seed"]))
    return (
        DataLoader(
            Subset(train_dataset, indices),
            batch_size=int(config["smoke"]["classification_batch_size"]),
            shuffle=True,
            generator=generator,
            num_workers=int(config["smoke"]["workers"]),
            drop_last=False,
        ),
        indices,
    )


def _build_detection_loader(config: dict[str, Any], repo_root: Path) -> tuple[DataLoader, list[int]]:
    dataset = IP102DetectionDataset(
        resolve_project_path(repo_root / "project", str(config["detection_voc_root"])),
        resolve_project_path(repo_root / "project", str(config["detection_train_split"])),
        resolve_project_path(repo_root / "project", str(config["detection_mapping"])),
        annotations_path=resolve_project_path(repo_root / "project", str(config["detection_annotations"])),
        transforms=DirectResizeDetectionTransform(224),
    )
    sample_count = min(int(config["smoke"]["detection_samples"]), len(dataset))
    indices: list[int] = []
    covered_labels: set[int] = set()
    for index in range(len(dataset)):
        labels = set(_detection_labels(dataset, index))
        if not labels - covered_labels:
            continue
        indices.append(index)
        covered_labels.update(labels)
        if len(indices) == sample_count:
            break
    if len(indices) < sample_count:
        for index in range(len(dataset)):
            if index not in indices:
                indices.append(index)
            if len(indices) == sample_count:
                break
    generator = torch.Generator().manual_seed(int(config["seed"]) + 1)
    return (
        DataLoader(
            Subset(dataset, indices),
            batch_size=int(config["smoke"]["detection_batch_size"]),
            shuffle=True,
            generator=generator,
            num_workers=int(config["smoke"]["workers"]),
            collate_fn=_collate_detection,
            drop_last=False,
        ),
        indices,
    )


def _detection_labels(dataset: IP102DetectionDataset, index: int) -> tuple[int, ...]:
    image_id = dataset.image_ids[index]
    if dataset._records is None:
        # Formal J2 always uses T0 annotations; reject accidental raw-XML selection.
        raise ValueError("J2 detection selection requires frozen T0 annotations")
    ip102_ids = [
        int(obj["ip102_class_id"])
        for obj in dataset._records[image_id]["objects"]
    ]
    return tuple(
        sorted(
            {
                dataset.mapping.from_ip102(class_id).detector_label
                for class_id in ip102_ids
            }
        )
    )


def run_j2(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    project_root = repo_root / "project"
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_config(config_path)
    input_sha256 = _verify_frozen_inputs(config, project_root)
    output_dir = resolve_project_path(project_root, str(config["output_dir"])) / args.run_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite J2 run: {output_dir}")
    output_dir.mkdir(parents=True)
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("J2 requires CUDA on the configured training machine")

    classification_loader, classification_indices = _build_classification_loader(config, repo_root)
    detection_loader, detection_indices = _build_detection_loader(config, repo_root)
    mapping = DetectionClassMapping(
        resolve_project_path(project_root, str(config["detection_mapping"]))
    )
    checkpoint_path = resolve_project_path(project_root, str(config["classification_checkpoint"]))
    model, model_info = build_shared_detection_model(checkpoint_path, mapping, trainable_backbone_layers=5)
    model.to(device)
    head_lr = float(config["smoke"]["head_learning_rate"])
    backbone_lr = float(config["smoke"]["backbone_learning_rate"])
    backbone_parameters = list(model.shared_body.parameters())
    head_parameters = list(model.classification_head.parameters()) + [
        parameter for module in model.detection_head for parameter in module.parameters()
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_lr},
            {"params": head_parameters, "lr": head_lr},
        ],
        weight_decay=float(config["smoke"]["weight_decay"]),
    )
    amp_enabled = bool(config["smoke"]["amp"])
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=float(config["smoke"].get("amp_init_scale", 32)),
        growth_interval=int(config["smoke"].get("amp_growth_interval", 100000)),
    )
    classification_criterion = nn.CrossEntropyLoss()
    classification_generator = classification_loader.generator or torch.Generator().manual_seed(int(config["seed"]))
    detection_generator = detection_loader.generator or torch.Generator().manual_seed(int(config["seed"]) + 1)
    classification_iterator = iter(classification_loader)
    detection_iterator = iter(detection_loader)
    steps_per_task = int(config["smoke"]["steps_per_task"])
    if config["smoke"].get("task_ratio") != [1, 1]:
        raise ValueError("J2 requires a fixed 1:1 classification/detection task ratio")
    if (
        steps_per_task % len(classification_loader) != 0
        or steps_per_task % len(detection_loader) != 0
    ):
        raise ValueError("J2 must finish both loaders at cycle boundaries")
    history: list[dict[str, Any]] = []
    initial_classification_loss: float | None = None
    initial_detection_loss: float | None = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    global_step = 0
    for pair in range(steps_per_task):
        try:
            classification_batch = next(classification_iterator)
        except StopIteration:
            classification_iterator = iter(classification_loader)
            classification_batch = next(classification_iterator)
        class_loss, class_checks = _classification_step(
            model, classification_batch[0], classification_batch[1], classification_criterion,
            optimizer, scaler, device, amp_enabled,
        )
        global_step += 1
        if initial_classification_loss is None:
            initial_classification_loss = class_loss
        history.append({"global_step": global_step, "task": "classification", "loss": class_loss, "checks": class_checks})
        try:
            detection_batch = next(detection_iterator)
        except StopIteration:
            detection_iterator = iter(detection_loader)
            detection_batch = next(detection_iterator)
        detection_loss, detection_checks = _detection_step(
            model, detection_batch[0], detection_batch[1], optimizer, scaler, device, amp_enabled
        )
        global_step += 1
        if initial_detection_loss is None:
            initial_detection_loss = detection_loss
        history.append({"global_step": global_step, "task": "detection", "loss": detection_loss, "checks": detection_checks})
        print(json.dumps(history[-2:], ensure_ascii=False), flush=True)
    training_duration = time.perf_counter() - started

    metadata = {
        "stage": "J2",
        "run_id": args.run_id,
        "git_commit": git_commit(repo_root),
        "classification_checkpoint_sha256": sha256_file(checkpoint_path),
        "input_sha256": input_sha256,
        "task_ratio": [1, 1],
        "test_metrics_read": False,
        "model_info": model_info.__dict__,
    }
    checkpoint_path_out = output_dir / "joint-last.pt"
    save_joint_checkpoint(
        checkpoint_path_out, model, optimizer, scaler,
        global_step=global_step,
        next_task="classification",
        classification_generator=classification_generator,
        detection_generator=detection_generator,
        classification_batches_into_cycle=0,
        detection_batches_into_cycle=0,
        metadata=metadata,
    )

    # Reload on CUDA for exact fixed-prediction comparison, then smoke CPU separately.
    fixed_images = classification_batch[0][:1].detach().cpu()
    fixed_detection_images = detection_batch[0][:1].detach().cpu()
    reloaded_model, _ = build_shared_detection_model(
        checkpoint_path, mapping, trainable_backbone_layers=5
    )
    reloaded_model.to(device)
    reloaded_optimizer = torch.optim.AdamW(
        [
            {"params": list(reloaded_model.shared_body.parameters()), "lr": backbone_lr},
            {
                "params": list(reloaded_model.classification_head.parameters())
                + [
                    parameter
                    for module in reloaded_model.detection_head
                    for parameter in module.parameters()
                ],
                "lr": head_lr,
            },
        ],
        weight_decay=float(config["smoke"]["weight_decay"]),
    )
    reloaded_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    reloaded_class_generator = torch.Generator().manual_seed(99)
    reloaded_detection_generator = torch.Generator().manual_seed(98)
    payload = load_joint_checkpoint(
        checkpoint_path_out,
        reloaded_model,
        reloaded_optimizer,
        reloaded_scaler,
        reloaded_class_generator,
        reloaded_detection_generator,
        map_location=device,
        restore_rng=False,
    )
    model.eval()
    reloaded_model.eval()
    with torch.inference_mode():
        original_logits = model.forward_classification(fixed_images.to(device))
        reloaded_logits = reloaded_model.forward_classification(fixed_images.to(device))
        original_detections = model.forward_detection(fixed_detection_images.to(device))
        reloaded_detections = reloaded_model.forward_detection(
            fixed_detection_images.to(device)
        )
    if not torch.equal(original_logits, reloaded_logits):
        raise RuntimeError("fixed classification prediction changed after checkpoint reload")
    if not _predictions_equal(original_detections, reloaded_detections):
        raise RuntimeError("fixed detection prediction changed after checkpoint reload")
    cpu_model, _ = build_shared_detection_model(
        checkpoint_path, mapping, trainable_backbone_layers=5
    )
    load_joint_checkpoint(checkpoint_path_out, cpu_model, map_location="cpu", restore_rng=False)
    cpu_model.eval()
    with torch.inference_mode():
        cpu_logits, cpu_detections = cpu_model.forward_joint(fixed_detection_images)
    if cpu_logits.shape != (1, 203) or not torch.isfinite(cpu_logits).all():
        raise RuntimeError("CPU checkpoint smoke produced invalid classification logits")
    if len(cpu_detections) != 1 or not all(
        torch.isfinite(cpu_detections[0][key]).all()
        for key in ("boxes", "scores")
    ):
        raise RuntimeError("CPU checkpoint smoke produced invalid detections")

    # Restore the complete state into a fresh CUDA process state and continue 1:1.
    load_joint_checkpoint(
        checkpoint_path_out,
        reloaded_model,
        reloaded_optimizer,
        reloaded_scaler,
        reloaded_class_generator,
        reloaded_detection_generator,
        map_location=device,
        restore_rng=True,
    )
    resumed_classification_loader, _ = _build_classification_loader(config, repo_root)
    resumed_detection_loader, _ = _build_detection_loader(config, repo_root)
    resumed_classification_loader.generator = reloaded_class_generator
    resumed_detection_loader.generator = reloaded_detection_generator
    resumed_classification_loader.sampler.generator = reloaded_class_generator
    resumed_detection_loader.sampler.generator = reloaded_detection_generator
    resumed_classification_batch = next(iter(resumed_classification_loader))
    resumed_detection_batch = next(iter(resumed_detection_loader))
    resumed_classification_loss, resumed_classification_checks = _classification_step(
        reloaded_model,
        resumed_classification_batch[0],
        resumed_classification_batch[1],
        classification_criterion,
        reloaded_optimizer,
        reloaded_scaler,
        device,
        amp_enabled,
    )
    resumed_detection_loss, resumed_detection_checks = _detection_step(
        reloaded_model,
        resumed_detection_batch[0],
        resumed_detection_batch[1],
        reloaded_optimizer,
        reloaded_scaler,
        device,
        amp_enabled,
    )
    classification_losses = [record["loss"] for record in history if record["task"] == "classification"]
    detection_losses = [record["loss"] for record in history if record["task"] == "detection"]
    classification_cycle = len(classification_loader)
    detection_cycle = len(detection_loader)
    classification_initial_mean = sum(classification_losses[:classification_cycle]) / classification_cycle
    classification_final_mean = sum(classification_losses[-classification_cycle:]) / classification_cycle
    detection_initial_mean = sum(detection_losses[:detection_cycle]) / detection_cycle
    detection_final_mean = sum(detection_losses[-detection_cycle:]) / detection_cycle
    if classification_final_mean >= classification_initial_mean or detection_final_mean >= detection_initial_mean:
        raise RuntimeError("both J2 task losses must decrease on the fixed smoke samples")
    final = {
        "schema_version": 1,
        "stage": "J2",
        "status": "completed_pending_project_lead_acceptance",
        "run_id": args.run_id,
        "input_git_commit": git_commit(repo_root),
        "classification_checkpoint_sha256": metadata["classification_checkpoint_sha256"],
        "input_sha256": input_sha256,
        "model": model_info.__dict__,
        "task_ratio": [1, 1],
        "steps_per_task": steps_per_task,
        "global_steps": global_step,
        "classification_samples": len(classification_indices),
        "detection_samples": len(detection_indices),
        "classification_loss_initial_cycle_mean": classification_initial_mean,
        "classification_loss_final_cycle_mean": classification_final_mean,
        "detection_loss_initial_cycle_mean": detection_initial_mean,
        "detection_loss_final_cycle_mean": detection_final_mean,
        "losses_decreased": True,
        "amp_enabled": amp_enabled,
        "amp_initial_scale": float(config["smoke"].get("amp_init_scale", 32)),
        "amp_final_scale": scaler.get_scale(),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "training_duration_seconds": training_duration,
        "training_steps_per_second": global_step / training_duration,
        "training_images_per_second": (
            steps_per_task
            * (
                int(config["smoke"]["classification_batch_size"])
                + int(config["smoke"]["detection_batch_size"])
            )
            / training_duration
        ),
        "duration_seconds": time.perf_counter() - started,
        "gradient_boundary_checks_passed": all(
            all(record["checks"].values()) for record in history
        ),
        "checkpoint": {
            "path": checkpoint_path_out.name,
            "global_step": int(payload["global_step"]),
            "next_task": payload["next_task"],
            "reload": "passed",
            "fixed_classification_logits_equal": True,
            "fixed_detection_predictions_equal": True,
            "cpu_joint_smoke": "passed",
            "interruption_resume": {
                "status": "passed",
                "continued_tasks": ["classification", "detection"],
                "classification_loss": resumed_classification_loss,
                "detection_loss": resumed_detection_loss,
                "classification_checks": resumed_classification_checks,
                "detection_checks": resumed_detection_checks,
            },
        },
        "history": history,
        "test_metrics_read": False,
        "environment": environment_versions(),
    }
    _write_json(output_dir / "history.json", history)
    classification_dataset = classification_loader.dataset.dataset
    detection_dataset = detection_loader.dataset.dataset
    _write_json(
        output_dir / "sample-selection.json",
        {
            "classification": [
                {
                    "index": index,
                    "relative_path": classification_dataset.get_record(index).relative_path,
                    "class_id": classification_dataset.get_record(index).class_id,
                }
                for index in classification_indices
            ],
            "detection": [
                {
                    "index": index,
                    "image_id": detection_dataset.image_ids[index],
                    "detector_labels": list(_detection_labels(detection_dataset, index)),
                }
                for index in detection_indices
            ],
        },
    )
    _write_json(output_dir / "metrics.json", final)
    (output_dir / "resolved-config.yaml").write_text(
        yaml.safe_dump({**config, "stage": "J2", "run_id": args.run_id, "input_git_commit": git_commit(repo_root)}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    artifact_names = ["history.json", "joint-last.pt", "metrics.json", "resolved-config.yaml", "sample-selection.json"]
    write_checksums(output_dir, artifact_names)
    return final


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--config", type=Path, default=Path("configs/j2.yaml"))
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = run_j2(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, TypeError, ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"J2 failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": metrics["status"], "run_id": metrics["run_id"], "peak_memory_bytes": metrics["peak_memory_bytes"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
