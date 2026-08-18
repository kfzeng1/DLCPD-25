"""Training loop for the IP102 Plan-A detection expert."""

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

from dlcpd25_v2.detection.dataset import IP102DetectionDataset, collate_detection
from dlcpd25_v2.detection.metrics import evaluate_coco
from dlcpd25_v2.detection.model import (
    build_detection_model,
    load_classification_backbone,
)


class DetectionEMA:
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


class DetectionProgressWriter:
    def __init__(self, run_dir: Path, total_epochs: int, total_steps: int, selection_metric: str) -> None:
        self.run_dir = run_dir
        self.state_path = run_dir / "state.json"
        self.history_path = run_dir / "history.json"
        self.started_at = time.time()
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self.selection_metric = selection_metric
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_state(self._state(status="starting"))

    def _state(self, status: str) -> dict[str, Any]:
        return {
            "run_id": self.run_dir.name,
            "task": "detection",
            "status": status,
            "started_at": self.started_at,
            "updated_at": time.time(),
            "total_epochs": self.total_epochs,
            "total_steps": self.total_steps,
        }

    def _write_state(self, state: dict[str, Any]) -> None:
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.state_path)

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
        state = self._state(status)
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
                "selection_metric": self.selection_metric,
                "elapsed_seconds": round(float(elapsed_seconds), 1),
                "eta_total_seconds": None if eta_total_seconds is None else round(float(eta_total_seconds), 1),
                "eta_epoch_seconds": None if eta_epoch_seconds is None else round(float(eta_epoch_seconds), 1),
                "device": device,
                "amp_dtype": amp_dtype,
                "gpu_memory_mb": gpu_memory_mb,
            }
        )
        self._write_state(state)

    def set_history(self, history: list[dict[str, Any]]) -> None:
        self.history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _clean_target(target: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in target.items()
        if key in {"boxes", "labels", "image_id", "area", "iscrowd"}
    }


def _decay_groups(parameters, weight_decay: float):
    decay, no_decay = [], []
    for parameter in parameters:
        if parameter.ndim <= 1:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return decay, no_decay


@dataclass
class DetectionTrainConfig:
    config: dict[str, Any]
    run_dir: Path
    device: str
    amp_dtype: str
    total_epochs: int
    resume_path: Path | None
    limit_train_batches: int | None
    limit_val_batches: int | None


def train_detection(cfg: DetectionTrainConfig) -> dict[str, Any]:
    config = cfg.config
    data_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]

    seed = int(train_cfg.get("seed", 20260817))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    from dlcpd25_v2.common import repo_root

    root = repo_root()
    voc_root = root / data_cfg["voc_root"]
    contract_dir = root / data_cfg["contract_dir"]
    batch_size = int(train_cfg["batch_size"])
    workers = int(train_cfg["workers"])
    epochs = cfg.total_epochs
    selection_metric = eval_cfg.get("selection_metric", "val_mAP_50_95")

    train_dataset = IP102DetectionDataset(voc_root, contract_dir, split="train", train=True)
    val_dataset = IP102DetectionDataset(voc_root, contract_dir, split="val", train=False)
    train_sampler = train_dataset.repeat_factor_sampler()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_detection,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_detection,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    if cfg.limit_train_batches:
        train_loader = _LoaderSlice(train_loader, cfg.limit_train_batches)
    if cfg.limit_val_batches:
        val_loader = _LoaderSlice(val_loader, cfg.limit_val_batches)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    schedule_epochs = int(train_cfg.get("schedule_epochs", epochs))
    schedule_steps = steps_per_epoch * schedule_epochs
    print(f"[data] train={len(train_dataset)} val={len(val_dataset)} batches_per_epoch={steps_per_epoch}")

    device = torch.device(cfg.device)
    model, info = build_detection_model(
        num_classes=int(data_cfg["num_classes"]) + 1,
        image_size=int(model_cfg["input_size"]["min_side"]),
        pretrained_backbone=bool(model_cfg.get("pretrained", True)),
        box_score_thresh=float(model_cfg.get("box_score_thresh", 0.05)),
        box_nms_thresh=float(model_cfg.get("box_nms_thresh", 0.5)),
        box_detections_per_img=int(model_cfg.get("max_detections_per_image", 30)),
        box_batch_size_per_img=int(train_cfg.get("roi_batch_size_per_image", 128)),
        rpn_batch_size_per_img=int(train_cfg.get("rpn_batch_size_per_image", 256)),
    )
    init_backbone = train_cfg.get("init_backbone")
    if init_backbone:
        load_classification_backbone(root / init_backbone, model)
    model = model.to(device)
    print(f"[model] {info.backbone} params={info.parameter_count/1e6:.2f}M")

    backbone_lr = float(train_cfg["backbone_learning_rate"])
    head_lr = float(train_cfg["head_learning_rate"])
    weight_decay = float(train_cfg["weight_decay"])
    backbone_parameters = list(model.backbone.body.parameters())
    head_parameters = [p for name, p in model.named_parameters() if not name.startswith("backbone.body.")]
    backbone_decay, backbone_no_decay = _decay_groups(backbone_parameters, weight_decay)
    head_decay, head_no_decay = _decay_groups(head_parameters, weight_decay)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_decay, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
            {"params": head_decay, "lr": head_lr, "weight_decay": weight_decay},
            {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0},
        ],
        lr=head_lr,
    )

    warmup_epochs = int(train_cfg.get("warmup_epochs", 1))
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
    ema = DetectionEMA(model, decay=float(train_cfg.get("ema_decay", 0.999)))
    ema.ema_model = ema.ema_model.to(device)

    progress = DetectionProgressWriter(cfg.run_dir, epochs, total_steps, selection_metric)
    history: list[dict[str, Any]] = []
    best_metric = -1.0
    best_epoch = 0
    start_epoch = 1
    global_step = 0
    patience = int(train_cfg.get("early_stopping_patience", 6))
    epochs_without_improvement = 0

    if cfg.resume_path is not None:
        checkpoint = torch.load(cfg.resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        scheduled_initial_lrs = scheduler.get_last_lr()
        for group, initial_lr in zip(optimizer.param_groups, scheduled_initial_lrs):
            group["lr"] = initial_lr
        for _ in range(global_step):
            optimizer.step()
            scheduler.step()
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        ema.load_state_dict(checkpoint["ema"])
        ema.ema_model = ema.ema_model.to(device)
        best_metric = float(checkpoint.get("best_metric", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        history = checkpoint.get("history", [])
        progress.set_history(history)
        print(f"[resume] start_epoch={start_epoch} global_step={global_step}")

    def validate(epoch: int) -> dict[str, Any]:
        model_for_eval = ema.ema_model
        model_for_eval.eval()
        records = [val_dataset.records[image_id] for image_id in val_dataset.ids]
        image_ids: list[str] = []
        predictions: list[list[dict[str, Any]]] = []
        with torch.inference_mode():
            for images, targets in val_loader:
                images = [image.to(device, non_blocking=True) for image in images]
                batch_ids = [str(target["image_id"]) for target in targets]
                model_targets = [_clean_target(target) for target in targets]
                model_targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in model_targets]
                outputs = model_for_eval(images, None)
                outputs = [{k: v.cpu() for k, v in output.items()} for output in outputs]
                image_ids.extend(batch_ids)
                predictions.extend(outputs)
        metrics = evaluate_coco(records, image_ids, predictions)
        metrics = {f"val_{key}": value for key, value in metrics.items()}
        print(
            f"[eval] epoch={epoch:03d} "
            f"mAP={metrics['val_mAP_50_95']:.2f} AP50={metrics['val_AP50']:.2f} "
            f"AP75={metrics['val_AP75']:.2f}"
        )
        return metrics

    def save_checkpoint(tag: str) -> Path:
        path = cfg.run_dir / "checkpoints" / f"{tag}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "architecture": info.backbone,
            "config": config,
            "epoch": current_epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
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
        seen = 0
        epoch_started = time.time()
        batch_in_epoch = 0
        for images, targets in train_loader:
            batch_in_epoch += 1
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in _clean_target(target).items()
                }
                for target in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast("cuda", dtype=amp_dtype):
                    loss_dict = model(images, targets)
                    loss = sum(value for value in loss_dict.values())
            else:
                loss_dict = model(images, targets)
                loss = sum(value for value in loss_dict.values())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            global_step += 1

            batch_actual = len(images)
            seen += batch_actual
            running_loss += float(loss.item()) * batch_actual
            if batch_in_epoch % int(train_cfg.get("log_every_steps", 10)) == 0 or batch_in_epoch == steps_per_epoch:
                elapsed = time.time() - progress.started_at
                avg_epoch_loss = running_loss / max(seen, 1)
                eta_total = (elapsed / max(global_step, 1)) * max(total_steps - global_step, 0)
                eta_epoch = (time.time() - epoch_started) / max(batch_in_epoch, 1) * max(steps_per_epoch - batch_in_epoch, 0)
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
                    gpu_memory_mb=(torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None),
                )

        train_metrics = {
            "epoch": epoch,
            "train_loss": round(running_loss / max(seen, 1), 6),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        val_metrics = validate(epoch)
        selected_key = selection_metric
        selected = float(val_metrics[selected_key])
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
                **val_metrics,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "timestamp": time.time(),
            }
        )
        progress.set_history(history)
        save_checkpoint("last")
        progress.update(
            status="completed" if epoch >= epochs else "running",
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
        if improved:
            print(f"[checkpoint] best.pt epoch={epoch} {selection_metric}={best_metric:.4f}")
        if epochs_without_improvement >= patience:
            print(f"[early-stop] no improvement for {patience} epochs")
            break

    # preserve latest fields when completed
    try:
        final_state = json.loads(progress.state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        final_state = progress._state(status="completed")
    final_state.update(
        {
            "status": "completed",
            "epoch": current_epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "updated_at": time.time(),
        }
    )
    progress._write_state(final_state)
    return {
        "run_dir": str(cfg.run_dir),
        "epochs_completed": current_epoch,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "history": history,
    }


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
