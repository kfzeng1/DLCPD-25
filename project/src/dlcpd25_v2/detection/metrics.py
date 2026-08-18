"""COCO-style mAP evaluation for IP102 detection."""

from __future__ import annotations

import contextlib
import io
from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def build_coco_groundtruth_object(records: list[dict[str, Any]]) -> COCO:
    images = []
    annotations = []
    annotation_id = 1
    for record in records:
        image_id = record["image_id"]
        images.append(
            {
                "id": image_id,
                "width": int(record["width"]),
                "height": int(record["height"]),
            }
        )
        for obj in record["objects"]:
            xmin, ymin, xmax, ymax = (float(v) for v in obj["bbox"])
            width = max(0.0, xmax - xmin)
            height = max(0.0, ymax - ymin)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(obj["detector_label"]),
                    "bbox": [xmin, ymin, width, height],
                    "area": max(0.0, width * height),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    coco = COCO()
    coco.dataset = {"images": images, "annotations": annotations, "categories": []}
    coco.createIndex()
    return coco


def predictions_to_coco(
    image_ids: list[str],
    predictions: list[dict[str, Any]],
    score_threshold: float = 0.05,
) -> list[dict[str, Any]]:
    rows = []
    for image_id, prediction in zip(image_ids, predictions):
        boxes = prediction["boxes"].cpu()
        labels = prediction["labels"].cpu()
        scores = prediction["scores"].cpu()
        for box, label, box_score in zip(boxes, labels, scores):
            if float(box_score.item()) < score_threshold:
                continue
            box = box.tolist()
            xmin, ymin, xmax, ymax = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            width = max(0.0, xmax - xmin)
            height = max(0.0, ymax - ymin)
            rows.append(
                {
                    "image_id": image_id,
                    "category_id": int(label.item()),
                    "bbox": [xmin, ymin, width, height],
                    "score": float(box_score.item()),
                }
            )
    return rows


def evaluate_coco(records: list[dict[str, Any]], image_ids: list[str], predictions: list[dict[str, Any]]) -> dict[str, float]:
    rows = predictions_to_coco(image_ids, predictions)
    if not rows:
        return {
            "mAP_50_95": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AR_100": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
        }
    coco_gt = build_coco_groundtruth_object(records)
    coco_dt = coco_gt.loadRes(rows)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = sorted(coco_gt.getImgIds())
    coco_eval.params.maxDets = [1, 10, 100]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    stats = coco_eval.stats
    return {
        "mAP_50_95": round(float(stats[0]) * 100.0, 4),
        "AP50": round(float(stats[1]) * 100.0, 4),
        "AP75": round(float(stats[2]) * 100.0, 4),
        "AR_100": round(float(stats[8]) * 100.0, 4),
        "AP_small": round(float(stats[3]) * 100.0, 4),
        "AP_medium": round(float(stats[4]) * 100.0, 4),
        "AP_large": round(float(stats[5]) * 100.0, 4),
    }
