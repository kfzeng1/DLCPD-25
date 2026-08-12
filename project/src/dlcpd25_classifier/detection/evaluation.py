"""COCO detection validation and frozen-threshold operating metrics."""

from __future__ import annotations

import contextlib
import io
import time
from collections import Counter, defaultdict
from typing import Any

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.ops import box_iou


def _xywh(box: list[float]) -> list[float]:
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]


def _threshold_counts(
    prediction: dict[str, Tensor], target: dict[str, Tensor], score_threshold: float
) -> tuple[int, int, int]:
    keep = prediction["scores"] >= score_threshold
    boxes = prediction["boxes"][keep].cpu()
    labels = prediction["labels"][keep].cpu()
    target_boxes = target["boxes"].cpu()
    target_labels = target["labels"].cpu()
    matched: set[int] = set()
    true_positive = 0
    order = prediction["scores"][keep].cpu().argsort(descending=True)
    overlaps = box_iou(boxes, target_boxes) if len(boxes) and len(target_boxes) else None
    for prediction_index in order.tolist():
        candidates = [
            index
            for index in range(len(target_boxes))
            if index not in matched and target_labels[index] == labels[prediction_index]
        ]
        if not candidates or overlaps is None:
            continue
        best = max(candidates, key=lambda index: float(overlaps[prediction_index, index]))
        if float(overlaps[prediction_index, best]) >= 0.5:
            matched.add(best)
            true_positive += 1
    return true_positive, len(boxes) - true_positive, len(target_boxes) - true_positive


def _mean_valid(values: Tensor) -> float | None:
    valid = values[values > -1]
    return float(valid.mean()) if valid.numel() else None


def evaluate_detection(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    score_threshold: float,
    class_names: dict[int, str],
    train_object_counts: dict[int, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one fixed validation split with standard COCO bbox metrics."""
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score threshold must be within [0, 1]")
    model.eval()
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    val_support: Counter[int] = Counter()
    size_support: Counter[str] = Counter()
    threshold_tp = threshold_fp = threshold_fn = 0
    annotation_id = 1
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_images, targets in loader:
            batch_images = batch_images.to(device, non_blocking=True)
            predictions = model.forward_detection(batch_images)
            for prediction, target in zip(predictions, targets):
                image_id = int(target["image_id"])
                images.append({"id": image_id, "width": 224, "height": 224})
                boxes = target["boxes"].tolist()
                labels = target["labels"].tolist()
                areas = target["area"].tolist()
                for box, label, area in zip(boxes, labels, areas):
                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": _xywh(box),
                            "area": float(area),
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1
                    val_support[int(label)] += 1
                    size_support["small" if area < 32**2 else "medium" if area < 96**2 else "large"] += 1
                cpu_prediction = {key: value.detach().cpu() for key, value in prediction.items()}
                for box, label, score in zip(
                    cpu_prediction["boxes"].tolist(),
                    cpu_prediction["labels"].tolist(),
                    cpu_prediction["scores"].tolist(),
                ):
                    detections.append(
                        {
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": _xywh(box),
                            "score": float(score),
                        }
                    )
                tp, fp, fn = _threshold_counts(cpu_prediction, target, score_threshold)
                threshold_tp += tp
                threshold_fp += fp
                threshold_fn += fn
    categories = [{"id": label, "name": class_names[label]} for label in sorted(class_names)]
    ground_truth = COCO()
    ground_truth.dataset = {
        "info": {"description": "IP102 frozen validation split resized to 224"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth.createIndex()
        if detections:
            results = ground_truth.loadRes(detections)
        else:
            results = COCO()
            results.dataset = {
                "info": ground_truth.dataset["info"],
                "images": images,
                "annotations": [],
                "categories": categories,
            }
            results.createIndex()
        evaluator = COCOeval(ground_truth, results, "bbox")
        evaluator.params.imgIds = [item["id"] for item in images]
        evaluator.params.catIds = sorted(class_names)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = torch.as_tensor(evaluator.eval["precision"])
    per_class: list[dict[str, Any]] = []
    for class_index, label in enumerate(evaluator.params.catIds):
        label = int(label)
        ap = _mean_valid(precision[:, :, class_index, 0, 2])
        ap50 = _mean_valid(precision[0, :, class_index, 0, 2])
        per_class.append(
            {
                "detector_label": label,
                "name": class_names[label],
                "train_objects": int(train_object_counts.get(label, 0)),
                "val_objects": int(val_support[label]),
                "ap": ap,
                "ap50": ap50,
            }
        )
    ranked = sorted(
        (record for record in per_class if record["train_objects"] > 0),
        key=lambda record: (record["train_objects"], record["detector_label"]),
    )
    tail_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(ranked):
        group = ("tail", "middle", "head")[min(2, index * 3 // max(len(ranked), 1))]
        tail_groups[group].append(record)
    long_tail = {
        group: {
            "classes": len(records),
            "train_objects_min": min((record["train_objects"] for record in records), default=0),
            "train_objects_max": max((record["train_objects"] for record in records), default=0),
            "mean_ap": (
                sum(record["ap"] for record in records if record["ap"] is not None)
                / sum(record["ap"] is not None for record in records)
                if any(record["ap"] is not None for record in records)
                else None
            ),
        }
        for group, records in tail_groups.items()
    }
    elapsed = time.perf_counter() - started
    summary = {
        "map": float(evaluator.stats[0]),
        "ap50": float(evaluator.stats[1]),
        "ap75": float(evaluator.stats[2]),
        "ap_small": float(evaluator.stats[3]),
        "ap_medium": float(evaluator.stats[4]),
        "ap_large": float(evaluator.stats[5]),
        "precision": threshold_tp / max(threshold_tp + threshold_fp, 1),
        "recall": threshold_tp / max(threshold_tp + threshold_fn, 1),
        "precision_recall_score_threshold": score_threshold,
        "precision_recall_iou_threshold": 0.5,
        "true_positive": threshold_tp,
        "false_positive": threshold_fp,
        "false_negative": threshold_fn,
        "images": len(images),
        "objects": len(annotations),
        "predictions": len(detections),
        "object_size_support": dict(size_support),
        "long_tail": long_tail,
        "duration_seconds": elapsed,
        "images_per_second": len(images) / elapsed,
        "implementation": "pycocotools COCOeval bbox",
    }
    return summary, per_class
