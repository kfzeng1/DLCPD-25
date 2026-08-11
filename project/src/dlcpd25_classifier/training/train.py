"""Training entry point for bounded A1 smoke and A2 full training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from dlcpd25_classifier.data import DLCPD25Dataset
from dlcpd25_classifier.models import build_classification_model
from dlcpd25_classifier.training.checkpoint import load_checkpoint, save_checkpoint
from dlcpd25_classifier.training.preflight import run_preflight
from dlcpd25_classifier.training.transforms import (
    build_eval_transform,
    preprocessing_spec,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_fixed_balanced_indices(
    dataset: DLCPD25Dataset,
    sample_count: int = 32,
    class_count: int = 8,
) -> list[int]:
    if sample_count < 32 or sample_count > 64:
        raise ValueError("A1 sample_count must be between 32 and 64")
    if class_count <= 1 or sample_count % class_count:
        raise ValueError("class_count must divide sample_count and be greater than one")
    samples_per_class = sample_count // class_count
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        if len(by_class[record.class_id]) < samples_per_class:
            by_class[record.class_id].append(index)
    eligible_classes = [
        class_id for class_id in sorted(by_class) if len(by_class[class_id]) == samples_per_class
    ]
    if len(eligible_classes) < class_count:
        raise ValueError("not enough classes for the fixed balanced A1 subset")
    selected_classes = eligible_classes[:class_count]
    return [index for class_id in selected_classes for index in by_class[class_id]]


def evaluate_model(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    samples = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if logits.ndim != 2 or logits.shape[1] != 203 or not torch.isfinite(logits).all():
                raise RuntimeError("model produced invalid [N, 203] logits")
            probabilities = logits.softmax(dim=1)
            if not torch.isfinite(loss) or not torch.isfinite(probabilities).all():
                raise RuntimeError("loss or probabilities are not finite")
            loss_sum += float(loss) * targets.numel()
            correct += int((logits.argmax(dim=1) == targets).sum())
            samples += targets.numel()
    return {"loss": loss_sum / samples, "accuracy": correct / samples}


def overfit_subset(
    model: nn.Module,
    dataset: Dataset[tuple[torch.Tensor, int]],
    *,
    device: torch.device,
    seed: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    amp: bool,
) -> tuple[dict[str, Any], AdamW, list[dict[str, float]]]:
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    amp_enabled = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    model.to(device)
    initial = evaluate_model(model, eval_loader, criterion, device)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    completed_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            if logits.shape != (targets.shape[0], 203) or not torch.isfinite(loss):
                raise RuntimeError("invalid logits or loss during A1 optimization")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        metrics = evaluate_model(model, eval_loader, criterion, device)
        history.append({"epoch": float(epoch), **metrics})
        completed_epochs = epoch
        if metrics["accuracy"] >= 0.95 and metrics["loss"] <= initial["loss"] * 0.2:
            break
    final = history[-1]
    if final["accuracy"] < 0.95 or final["loss"] > initial["loss"] * 0.2:
        raise RuntimeError(
            "fixed A1 subset did not clearly overfit: "
            f"accuracy={final['accuracy']:.4f}, loss={final['loss']:.4f}"
        )
    return {
        "initial_loss": initial["loss"],
        "initial_accuracy": initial["accuracy"],
        "final_loss": final["loss"],
        "final_accuracy": final["accuracy"],
        "loss_ratio": final["loss"] / initial["loss"],
        "epochs_completed": completed_epochs,
        "duration_seconds": time.perf_counter() - started,
        "amp_enabled": amp_enabled,
    }, optimizer, history


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("training config must be a mapping")
    return payload


def resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def environment_versions() -> dict[str, str]:
    packages = ("Pillow", "PyYAML", "torch", "torchvision")
    return {
        "python": platform.python_version(),
        **{name.lower(): importlib.metadata.version(name) for name in packages},
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_checksums(output_dir: Path, names: Sequence[str]) -> None:
    lines = [f"{sha256_file(output_dir / name)}  {name}" for name in sorted(names)]
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_a1_smoke(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    project_root = repo_root / "project"
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_config(config_path)
    output_root = resolve_project_path(project_root, str(config["output_dir"]))
    output_dir = output_root / args.run_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite A1 run: {output_dir}")

    set_seed(int(config["seed"]))
    data_root = resolve_project_path(project_root, str(config["data_root"]))
    split_dir = resolve_project_path(project_root, str(config["split_dir"]))
    taxonomy_path = resolve_project_path(project_root, str(config["taxonomy"]))
    release_path = repo_root / "artifacts/data/v1/d5-r1/data-v1-release.json"
    preflight = run_preflight(repo_root, release_path, data_root)

    model_config = config["model"]
    image_size = int(model_config["image_size"])
    dataset = DLCPD25Dataset(
        data_root=data_root,
        split_csv=split_dir / "train.csv",
        taxonomy_path=taxonomy_path,
        transform=build_eval_transform(image_size),
    )
    indices = select_fixed_balanced_indices(dataset, args.samples, args.classes)
    subset = Subset(dataset, indices)
    selection = [
        {
            "index": index,
            "relative_path": dataset.get_record(index).relative_path,
            "class_id": dataset.get_record(index).class_id,
            "sha256": dataset.get_record(index).sha256,
        }
        for index in indices
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A1 requires the configured CUDA smoke on this machine")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, model_info = build_classification_model(
        str(model_config["name"]),
        int(model_config["num_classes"]),
        bool(model_config["pretrained"]),
    )
    overfit, optimizer, history = overfit_subset(
        model,
        subset,
        device=device,
        seed=int(config["seed"]),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=float(config["training"]["weight_decay"]),
        amp=bool(config["training"]["amp"]),
    )
    peak_memory = torch.cuda.max_memory_allocated()

    output_dir.mkdir(parents=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint_metrics = {
        "loss": float(overfit["final_loss"]),
        "accuracy": float(overfit["final_accuracy"]),
    }
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        architecture=model_info.architecture,
        num_classes=model_info.num_classes,
        epoch=int(overfit["epochs_completed"]),
        metrics=checkpoint_metrics,
        metadata={
            "stage": "A1",
            "run_id": args.run_id,
            "data_version": preflight["data_version"],
            "taxonomy_sha256": preflight["taxonomy"]["sha256"],
            "git_commit": git_commit(repo_root),
        },
    )

    original_cpu = model.to("cpu").eval()
    reloaded, _reloaded_info = build_classification_model(
        model_info.architecture, model_info.num_classes, pretrained=False
    )
    checkpoint_payload = load_checkpoint(
        checkpoint_path,
        reloaded,
        expected_architecture=model_info.architecture,
        expected_num_classes=model_info.num_classes,
    )
    reloaded.eval()
    cpu_images, _cpu_targets = next(iter(DataLoader(subset, batch_size=2, shuffle=False)))
    with torch.inference_mode():
        original_logits = original_cpu(cpu_images)
        reloaded_logits = reloaded(cpu_images)
        probabilities = reloaded_logits.softmax(dim=1)
    if not torch.equal(original_logits, reloaded_logits):
        raise RuntimeError("checkpoint reload changed CPU logits")
    if reloaded_logits.shape != (2, 203) or not torch.isfinite(probabilities).all():
        raise RuntimeError("CPU checkpoint smoke produced invalid output")

    resolved_config = {
        **config,
        "stage": "A1",
        "run_id": args.run_id,
        "input_git_commit": git_commit(repo_root),
        "a1_smoke": {
            "samples": args.samples,
            "classes": args.classes,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "transform": "deterministic-eval-for-overfit-diagnosis",
        },
    }
    (output_dir / "resolved-config.yaml").write_text(
        yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    write_json(output_dir / "preflight.json", preflight)
    write_json(output_dir / "sample-selection.json", selection)
    write_json(output_dir / "training-log.json", history)
    metrics = {
        "schema_version": 1,
        "stage": "A1",
        "status": "completed_pending_project_lead_acceptance",
        "run_id": args.run_id,
        "data_version": preflight["data_version"],
        "input_git_commit": git_commit(repo_root),
        "model": model_info.__dict__,
        "preprocessing": preprocessing_spec(image_size),
        "subset": {
            "samples": len(selection),
            "classes": sorted({item["class_id"] for item in selection}),
            "samples_per_class": len(selection) // len({item["class_id"] for item in selection}),
        },
        "overfit": overfit,
        "cuda_smoke": {
            "device": torch.cuda.get_device_name(0),
            "logits_shape": [args.batch_size, 203],
            "finite_loss_and_probabilities": True,
            "amp_enabled": overfit["amp_enabled"],
            "peak_memory_bytes": peak_memory,
        },
        "cpu_checkpoint_smoke": {
            "logits_shape": list(reloaded_logits.shape),
            "finite_probabilities": True,
            "logits_equal_before_after_reload": True,
            "checkpoint_epoch": checkpoint_payload["epoch"],
        },
        "environment": environment_versions(),
    }
    write_json(output_dir / "metrics.json", metrics)
    artifact_names = [
        "checkpoint.pt",
        "metrics.json",
        "preflight.json",
        "resolved-config.yaml",
        "sample-selection.json",
        "training-log.json",
    ]
    write_checksums(output_dir, artifact_names)
    return metrics


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--a1-smoke", action="store_true")
    parser.add_argument("--a2-train", action="store_true")
    parser.add_argument("--a2-resource-probe", action="store_true")
    parser.add_argument("--loss-strategy", choices=("ce", "weighted_ce"), default="ce")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--probe-batches", type=int, default=100)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modes = sum((args.a1_smoke, args.a2_train, args.a2_resource_probe))
    if modes != 1:
        print("choose exactly one training mode", file=sys.stderr)
        return 2
    try:
        if args.a1_smoke:
            metrics = run_a1_smoke(args)
        else:
            from dlcpd25_classifier.training.a2 import run_a2_training

            metrics = run_a2_training(args)
    except (
        FileExistsError,
        RuntimeError,
        TypeError,
        ValueError,
        OSError,
        KeyError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        return 1
    if args.a1_smoke:
        summary = {
            "status": metrics["status"],
            "run_id": metrics["run_id"],
            "final_accuracy": metrics["overfit"]["final_accuracy"],
            "final_loss": metrics["overfit"]["final_loss"],
            "epochs": metrics["overfit"]["epochs_completed"],
            "peak_memory_bytes": metrics["cuda_smoke"]["peak_memory_bytes"],
        }
    else:
        summary = metrics
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
