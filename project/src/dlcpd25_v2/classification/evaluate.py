"""Final frozen evaluation for the DLCPD-25 Plan-A classification model.

This script must only be run after training/configuration are frozen.
It reads the test split from the data contract and never modifies it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dlcpd25_v2.classification.losses import ClassificationLoss
from dlcpd25_v2.classification.metrics import (
    accuracy_from_logits,
    balanced_accuracy_from_confusion,
    confusion_matrix_from_pairs,
    macro_f1_from_confusion,
)
from dlcpd25_v2.classification.model import build_model
from dlcpd25_v2.classification.transforms import build_transforms
from dlcpd25_v2.common import repo_path, repo_root
from dlcpd25_v2.data.classification_dataset import ManifestClassificationDataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/training/classification/convnext-tiny-384-plan-a-v1/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to the checkpoint run directory",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _per_class_rows(confusion: torch.Tensor, taxonomy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    class_map = {
        int(item["class_id"]): (str(item["official_name"]), str(item["host_zh"]), str(item["category"]))
        for item in taxonomy["classes"]
    }
    tp = confusion.diagonal().to(torch.float64)
    true_sum = confusion.sum(dim=1).to(torch.float64)
    pred_sum = confusion.sum(dim=0).to(torch.float64)
    for class_id in range(confusion.shape[0]):
        precision = float(tp[class_id] / pred_sum[class_id]) if pred_sum[class_id] > 0 else 0.0
        recall = float(tp[class_id] / true_sum[class_id]) if true_sum[class_id] > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        name, host_zh, category = class_map.get(class_id, (str(class_id), "", ""))
        rows.append(
            {
                "class_id": class_id,
                "name": name,
                "host_zh": host_zh,
                "category": category,
                "support": int(true_sum[class_id]),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
            }
        )
    return rows


def _host_breakdown(
    confusion: torch.Tensor, taxonomy: dict[str, Any]
) -> list[dict[str, Any]]:
    class_map = {
        int(item["class_id"]): (str(item["host_id"]), str(item["host_zh"]))
        for item in taxonomy["classes"]
    }
    host_index = {
        str(host["id"]): (idx, str(host["name_zh"]))
        for idx, host in enumerate(taxonomy["hosts"])
    }
    host_count = len(host_index)
    host_conf = torch.zeros((host_count, host_count), dtype=torch.int64)
    for true_class in range(confusion.shape[0]):
        true_host = class_map[true_class][0]
        for pred_class in range(confusion.shape[0]):
            host_conf[host_index[true_host][0], host_index[class_map[pred_class][0]][0]] += int(
                confusion[true_class, pred_class]
            )
    rows = []
    for host_id, (idx, name_zh) in host_index.items():
        support = int(host_conf[idx].sum())
        correct = int(host_conf[idx, idx])
        accuracy = correct / support if support else 0.0
        rows.append(
            {
                "host_id": host_id,
                "host_zh": name_zh,
                "support": support,
                "correct": correct,
                "accuracy": round(accuracy, 6),
            }
        )
    rows.sort(key=lambda row: row["accuracy"])
    return rows


def main() -> int:
    args = parse_args()
    root = repo_root()
    checkpoint_path = repo_path(args.checkpoint)
    print(f"[evaluate] checkpoint={checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    data_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    device = torch.device(args.device)
    manifest_path = repo_path(data_cfg["manifest"])
    taxonomy_path = repo_path(data_cfg["taxonomy"])
    image_size = int(model_cfg["input_size"])

    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    test_dataset = ManifestClassificationDataset(
        manifest_path,
        taxonomy_path,
        split="test",
        transform=build_transforms(image_size, train=False),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    print(f"[evaluate] test samples={len(test_dataset)} batches={len(test_loader)}")

    model, info = build_model(
        architecture=model_cfg["architecture"],
        num_classes=test_dataset.num_classes,
        num_hosts=test_dataset.num_hosts,
        num_categories=test_dataset.num_categories,
        pretrained=False,
    )
    ema_payload = payload.get("ema") or {}
    ema_state = ema_payload.get("ema_model_state_dict") if isinstance(ema_payload, dict) else None
    state_dict = ema_state if ema_state is not None else payload["model_state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print(f"[evaluate] loaded {'EMA' if ema_state is not None else 'raw'} weights; params={info.parameter_count/1e6:.2f}M")

    # Loss alpha must match training: derive from the frozen train split.
    train_dataset_for_weights = ManifestClassificationDataset(
        manifest_path,
        taxonomy_path,
        split="train",
        transform=build_transforms(image_size, train=False),
    )
    criterion = ClassificationLoss(
        num_classes=test_dataset.num_classes,
        class_counts=train_dataset_for_weights.class_counts,
        focal_gamma=float(train_cfg["loss"].get("focal_gamma", 2.0)),
        label_smoothing=float(train_cfg["loss"].get("label_smoothing", 0.1)),
        host_weight=float(train_cfg["loss"].get("aux_weight_host", 0.2)),
        category_weight=float(train_cfg["loss"].get("aux_weight_category", 0.05)),
    ).to(device)

    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    running_loss = 0.0
    seen = 0
    started = time.time()
    with torch.inference_mode():
        for images, targets, host_targets, category_targets in test_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            host_targets = host_targets.to(device, non_blocking=True)
            category_targets = category_targets.to(device, non_blocking=True)
            logits, host_logits, category_logits = model(images)
            loss, _ = criterion(
                logits, host_logits, category_logits, targets, host_targets, category_targets
            )
            n = targets.shape[0]
            running_loss += float(loss.item()) * n
            seen += n
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    confusion = confusion_matrix_from_pairs(
        logits_cat.argmax(dim=1), targets_cat, test_dataset.num_classes
    )
    top1, top5 = accuracy_from_logits(logits_cat, targets_cat, (1, 5))
    metrics = {
        "test_loss": round(running_loss / max(seen, 1), 6),
        "test_top1": round(top1, 4),
        "test_top5": round(top5, 4),
        "test_macro_f1": round(macro_f1_from_confusion(confusion), 4),
        "test_balanced_accuracy": round(balanced_accuracy_from_confusion(confusion), 4),
        "test_support": seen,
        "evaluation_seconds": round(time.time() - started, 1),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    output_dir = args.output_dir or checkpoint_path.parent.parent
    output_dir = repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_class = _per_class_rows(confusion, taxonomy)
    host_breakdown = _host_breakdown(confusion, taxonomy)
    report_payload = {
        "schema_version": 1,
        "model": {
            "architecture": info.architecture,
            "input_size": image_size,
            "parameters_m": round(info.parameter_count / 1e6, 2),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "weights": "EMA" if ema_state is not None else "raw",
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "best_epoch": int(payload.get("best_epoch", -1)),
        },
        "test_metrics": metrics,
        "per_class": per_class,
        "host_breakdown": host_breakdown,
        "generated_at": time.time(),
    }
    (output_dir / "test-evaluation.json").write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "test-metrics-per-class.json").write_text(
        json.dumps(per_class, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(confusion, output_dir / "test-confusion-matrix.pt")
    print(f"[evaluate] saved artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
