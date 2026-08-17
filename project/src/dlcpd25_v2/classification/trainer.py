"""Training loop for the DLCPD-25 Plan-A classification expert."""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from dlcpd25_v2.classification.metrics import (
    accuracy_from_logits,
    balanced_accuracy_from_confusion,
    confusion_matrix_from_pairs,
    macro_f1_from_confusion,
)
from dlcpd25_v2.classification.model import ConvNextClassifier, build_model
from dlcpd25_v2.classification.transforms import build_transforms
from dlcpd25_v2.data.classification_dataset import ManifestClassificationDataset


class ModelEMA:
    """Exponential moving average of a model, used for validation and saving."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_parameter, parameter in zip(self.ema_model.parameters(), model.parameters()):
            ema_parameter.mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)
        for ema_buffer, buffer in zip(self.ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "ema_model_state_dict": self.ema_model.state_dict()}

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.decay = float(payload["decay"])
        self.ema_model.load_state_dict(payload["ema_model_state_dict"])


class ProgressWriter:
    """Atomically writes training progress for the web dashboard."""

    def __init__(self, run_dir: Path, run_id: str, total_epochs: int, total_steps: int) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = run_dir / "state.json"
        self.history_path = run_dir / "history.json"
        self.started_at = time.time()
        self.run_id = run_id
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self.history: list[dict[str, Any]] = []
        self._write_json(self.state_path, self._base_state(status="starting"))

    def _base_state(self, status: str) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": status,
            "started_at": self.started_at,
            "updated_at": time.time(),
            "total_epochs": self.total_epochs,
            "total_steps": self.total_steps,
        }

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def update(
        self,
        *,
        status: str,
        epoch: int,
        global_step: int,
        steps_per_epoch: int,
        batch_in_epoch: int,
        batch_size: int,
        loss: float,
        avg_epoch_loss: float,
        lr: float,
        best_metric: float | None,
        elapsed_seconds: float,
        eta_total_seconds: float | None,
        eta_epoch_seconds: float | None,
        device: str,
        amp_dtype: str,
        gpu_memory_mb: float | None,
    ) -> None:
        state = self._base_state(status)
        state.update(
            {
                "epoch": epoch,
                "global_step": global_step,
                "steps_per_epoch": steps_per_epoch,
                "batch_in_epoch": batch_in_epoch,
                "batch_size": batch_size,
                "loss_recent": round(float(loss), 6),
                "avg_epoch_loss": round(float(avg_epoch_loss), 6),
                "lr": float(lr),
                "best_metric": best_metric,
                "selection_metric": "val_macro_f1",
                "elapsed_seconds": round(float(elapsed_seconds), 1),
                "eta_total_seconds": None if eta_total_seconds is None else round(float(eta_total_seconds), 1),
                "eta_epoch_seconds": None if eta_epoch_seconds is None else round(float(eta_epoch_seconds), 1),
                "device": device,
                "amp_dtype": amp_dtype,
                "gpu_memory_mb": gpu_memory_mb,
            }
        )
        self._write_json(self.state_path, state)

    def set_history(self, history: list[dict[str, Any]]) -> None:
        self.history = history
        self._write_json(self.history_path, history)


def _decay_groups(parameters: Any, weight_decay: float) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in parameters:
        if parameter.ndim <= 1:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return decay, no_decay


@dataclass
class TrainConfig:
    config: dict[str, Any]
    run_dir: Path
    device: str
    amp_dtype: str
    total_epochs: int
    resume_path: Path | None
    limit_train_batches: int | None
    limit_val_batches: int | None


def _metric_value(metrics: dict[str, float], name: str) -> float:
    return float(metrics[name])


def train(cfg: TrainConfig) -> dict[str, Any]:
    config = cfg.config
    data_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]

    seed = int(train_cfg.get("seed", 20260817))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    image_size = int(model_cfg["input_size"])
    num_classes = int(model_cfg.get("num_classes", data_cfg.get("num_classes", 203)))
    num_hosts = int(model_cfg["auxiliary_heads"]["host"])
    num_categories = int(model_cfg["auxiliary_heads"]["category"])
    batch_size = int(train_cfg["batch_size"])
    workers = int(train_cfg["workers"])
    epochs = cfg.total_epochs

    # Frozen data contracts.
    manifest_path = Path(data_cfg["manifest"])
    if not manifest_path.is_absolute():
        from dlcpd25_v2.common import repo_root

        manifest_path = repo_root() / manifest_path
    taxonomy_path = Path(data_cfg["taxonomy"])
    if not taxonomy_path.is_absolute():
        from dlcpd25_v2.common import repo_root

        taxonomy_path = repo_root() / taxonomy_path

    train_dataset = ManifestClassificationDataset(
        manifest_path,
        taxonomy_path,
        split="train",
        transform=build_transforms(image_size, train=True),
    )
    val_dataset = ManifestClassificationDataset(
        manifest_path,
        taxonomy_path,
        split="val",
        transform=build_transforms(image_size, train=False),
    )
    train_sampler = train_dataset.balanced_sampler()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    if cfg.limit_train_batches:
        train_loader = _limit_loader(train_loader, cfg.limit_train_batches)
    if cfg.limit_val_batches:
        val_loader = _limit_loader(val_loader, cfg.limit_val_batches)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    schedule_epochs = int(train_cfg.get("schedule_epochs", epochs))
    schedule_steps = steps_per_epoch * schedule_epochs
    print(f"[data] train={len(train_dataset)} val={len(val_dataset)} batches_per_epoch={steps_per_epoch}")

    device = torch.device(cfg.device)
    model, model_info = build_model(
        architecture=model_cfg["architecture"],
        num_classes=num_classes,
        num_hosts=num_hosts,
        num_categories=num_categories,
        pretrained=bool(model_cfg.get("pretrained", True)),
    )
    channels_last = bool(train_cfg.get("channels_last", True)) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    print(f"[model] {model_info.architecture} params={model_info.parameter_count/1e6:.2f}M channels_last={channels_last}")

    from dlcpd25_v2.classification.losses import ClassificationLoss

    criterion = ClassificationLoss(
        num_classes=num_classes,
        class_counts=train_dataset.class_counts,
        focal_gamma=float(train_cfg["loss"].get("focal_gamma", 2.0)),
        label_smoothing=float(train_cfg["loss"].get("label_smoothing", 0.1)),
        host_weight=float(train_cfg["loss"].get("aux_weight_host", 0.2)),
        category_weight=float(train_cfg["loss"].get("aux_weight_category", 0.05)),
    ).to(device)

    backbone_lr = float(train_cfg["backbone_learning_rate"])
    head_lr = float(train_cfg["head_learning_rate"])
    weight_decay = float(train_cfg["weight_decay"])
    backbone_decay, backbone_no_decay = _decay_groups(model.features.parameters(), weight_decay)
    head_parameters = list(model.classifier.parameters()) + list(model.host_classifier.parameters()) + list(model.category_classifier.parameters())
    head_decay, head_no_decay = _decay_groups(iter(head_parameters), weight_decay)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_decay, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
            {"params": head_decay, "lr": head_lr, "weight_decay": weight_decay},
            {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0},
        ],
        lr=head_lr,
    )

    warmup_epochs = int(train_cfg.get("warmup_epochs", 3))
    warmup_steps = min(warmup_epochs * steps_per_epoch, max(1, schedule_steps - 1))
    cosine_steps = max(1, schedule_steps - warmup_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps),
        ],
        milestones=[warmup_steps],
    )

    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    use_amp = bool(train_cfg.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ModelEMA(model, decay=float(train_cfg.get("ema_decay", 0.999)))
    ema.ema_model = ema.ema_model.to(device)

    progress = ProgressWriter(cfg.run_dir, cfg.run_dir.name, epochs, total_steps)
    history: list[dict[str, Any]] = []
    best_metric = -1.0
    best_epoch = 0
    start_epoch = 1
    global_step = 0
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 6))
    epochs_without_improvement = 0

    if cfg.resume_path is not None:
        checkpoint = torch.load(cfg.resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as exc:  # noqa: BLE001
            print(f"[resume] scheduler state mismatch ({type(exc).__name__}); rebuilding from step {global_step}")
            for _ in range(global_step):
                scheduler.step()
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        ema.load_state_dict(checkpoint["ema"])
        ema.ema_model = ema.ema_model.to(device)
        best_metric = float(checkpoint.get("best_metric", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        history = checkpoint.get("history", [])
        progress.set_history(history)
        print(f"[resume] checkpoint={cfg.resume_path} start_epoch={start_epoch}")

    def validate(step: int) -> dict[str, Any]:
        model_for_eval = ema.ema_model
        model_for_eval.eval()
        all_logits: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        running_loss = 0.0
        seen = 0
        with torch.inference_mode():
            for images, targets, host_targets, category_targets in val_loader:
                if channels_last:
                    images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
                else:
                    images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                host_targets = host_targets.to(device, non_blocking=True)
                category_targets = category_targets.to(device, non_blocking=True)
                logits, host_logits, category_logits = model_for_eval(images)
                loss, _ = criterion(logits, host_logits, category_logits, targets, host_targets, category_targets)
                batch_size_actual = targets.shape[0]
                running_loss += float(loss.item()) * batch_size_actual
                seen += batch_size_actual
                all_logits.append(logits.detach().cpu())
                all_targets.append(targets.detach().cpu())
        logits_cat = torch.cat(all_logits, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        confusion = confusion_matrix_from_pairs(logits_cat.argmax(dim=1), targets_cat, num_classes)
        top1, top5 = accuracy_from_logits(logits_cat, targets_cat, (1, 5))
        metrics = {
            "top1": round(top1, 4),
            "top5": round(top5, 4),
            "macro_f1": round(macro_f1_from_confusion(confusion), 4),
            "balanced_accuracy": round(balanced_accuracy_from_confusion(confusion), 4),
            "loss": round(running_loss / max(seen, 1), 6),
            "support": seen,
        }
        print(
            f"[eval] epoch={step:03d} top1={metrics['top1']:.2f} top5={metrics['top5']:.2f} "
            f"macro_f1={metrics['macro_f1']:.2f} loss={metrics['loss']:.4f}"
        )
        return metrics

    def save_checkpoint(tag: str) -> Path:
        path = cfg.run_dir / "checkpoints" / f"{tag}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "architecture": model_cfg["architecture"],
            "config": config,
            "epoch": current_epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "steps_per_epoch": steps_per_epoch,
            "schedule_steps": schedule_steps,
            "model_state_dict": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return path

    current_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        current_epoch = epoch
        model.train()
        running_loss = 0.0
        running_top1 = 0.0
        seen = 0
        epoch_started = time.time()
        steps_done_before = global_step
        batch_in_epoch = 0
        for images, targets, host_targets, category_targets in train_loader:
            batch_in_epoch += 1
            if channels_last:
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            else:
                images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            host_targets = host_targets.to(device, non_blocking=True)
            category_targets = category_targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits, host_logits, category_logits = model(images)
                    loss, loss_parts = criterion(
                        logits, host_logits, category_logits, targets, host_targets, category_targets
                    )
            else:
                logits, host_logits, category_logits = model(images)
                loss, loss_parts = criterion(
                    logits, host_logits, category_logits, targets, host_targets, category_targets
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            global_step += 1

            batch_size_actual = targets.shape[0]
            seen += batch_size_actual
            running_loss += float(loss.item()) * batch_size_actual
            batch_top1 = accuracy_from_logits(logits.detach().float(), targets, (1,))[0]
            running_top1 += batch_top1 * batch_size_actual

            if batch_in_epoch % int(train_cfg.get("log_every_steps", 10)) == 0 or batch_in_epoch == steps_per_epoch:
                elapsed = time.time() - progress.started_at
                avg_epoch_loss = running_loss / max(seen, 1)
                steps_done = global_step
                steps_remaining_total = max(total_steps - steps_done, 0)
                eta_total = (elapsed / max(steps_done, 1)) * steps_remaining_total
                eta_epoch = (time.time() - epoch_started) / max(batch_in_epoch, 1) * max(steps_per_epoch - batch_in_epoch, 0)
                gpu_mem = (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                    if device.type == "cuda"
                    else None
                )
                progress.update(
                    status="running",
                    epoch=epoch,
                    global_step=global_step,
                    steps_per_epoch=steps_per_epoch,
                    batch_in_epoch=batch_in_epoch,
                    batch_size=batch_size,
                    loss=float(loss.item()),
                    avg_epoch_loss=avg_epoch_loss,
                    lr=float(scheduler.get_last_lr()[0]),
                    best_metric=None if best_metric < 0 else best_metric,
                    elapsed_seconds=elapsed,
                    eta_total_seconds=eta_total,
                    eta_epoch_seconds=eta_epoch,
                    device=str(device),
                    amp_dtype=cfg.amp_dtype,
                    gpu_memory_mb=gpu_mem,
                )

        train_metrics = {
            "epoch": epoch,
            "train_loss": round(running_loss / max(seen, 1), 6),
            "train_top1": round(running_top1 / max(seen, 1), 4),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        val_metrics = validate(epoch)
        selected = _metric_value(val_metrics, eval_cfg.get("selection_metric", "val_macro_f1").replace("val_", ""))
        improved = selected > best_metric
        if improved:
            best_metric = selected
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint("best")
        else:
            epochs_without_improvement += 1
        history.append(
            {
                **train_metrics,
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "timestamp": time.time(),
            }
        )
        progress.set_history(history)
        progress.update(
            status="validating" if epoch < epochs else "completed",
            epoch=epoch,
            global_step=global_step,
            steps_per_epoch=steps_per_epoch,
            batch_in_epoch=batch_in_epoch,
            batch_size=batch_size,
            loss=float(loss.item()),
            avg_epoch_loss=running_loss / max(seen, 1),
            lr=float(scheduler.get_last_lr()[0]),
            best_metric=best_metric,
            elapsed_seconds=time.time() - progress.started_at,
            eta_total_seconds=None if epoch >= epochs else (time.time() - progress.started_at) / max(epoch, 1) * (epochs - epoch),
            eta_epoch_seconds=None,
            device=str(device),
            amp_dtype=cfg.amp_dtype,
            gpu_memory_mb=(torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None),
        )
        save_checkpoint("last")
        if improved:
            print(f"[checkpoint] best.pt epoch={epoch} val_macro_f1={best_metric:.4f}")
        if epochs_without_improvement >= early_stopping_patience:
            print(f"[early-stop] no improvement for {early_stopping_patience} epochs")
            break

    try:
        final_state = json.loads(progress.state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        final_state = progress._base_state(status="completed")
    final_state.update(
        {
            "status": "completed",
            "epoch": current_epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "steps_per_epoch": steps_per_epoch,
            "schedule_steps": schedule_steps,
            "updated_at": time.time(),
        }
    )
    progress._write_json(progress.state_path, final_state)
    return {
        "run_dir": str(cfg.run_dir),
        "epochs_completed": current_epoch,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "history": history,
    }


def _limit_loader(loader: DataLoader, limit: int) -> _LoaderSlice:
    return _LoaderSlice(loader, limit)


class _LoaderSlice:
    def __init__(self, loader: DataLoader, limit: int) -> None:
        self.loader = loader
        self.limit = limit

    def __iter__(self):
        iterator = iter(self.loader)
        for _ in range(self.limit):
            yield next(iterator)

    def __len__(self):
        return self.limit
