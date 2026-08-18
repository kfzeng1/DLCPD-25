"""Final frozen evaluation for the IP102 Plan-A detection expert."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import torch
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from dlcpd25_v2.common import repo_path
from dlcpd25_v2.detection.dataset import IP102DetectionDataset, collate_detection
from dlcpd25_v2.detection.metrics import (
    build_coco_groundtruth_object,
    predictions_to_coco,
)
from dlcpd25_v2.detection.model import build_detection_model


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
            "artifacts/training/detection/fasterrcnn-convnext-tiny-640-v1/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to the checkpoint run directory",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _run_coco_eval(coco_gt, coco_dt, cat_ids=None):
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = sorted(coco_gt.getImgIds())
    coco_eval.params.maxDets = [1, 10, 100]
    if cat_ids is not None:
        coco_eval.params.catIds = cat_ids
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return coco_eval.stats


def precision_recall_at_iou50(
    predictions: list[dict[str, Any]],
    image_ids: list[str],
    records_by_id: dict[str, dict[str, Any]],
    score_threshold: float,
) -> dict[str, float]:
    detections_by_image: dict[str, list[dict[str, Any]]] = {image_id: [] for image_id in image_ids}
    for image_id, prediction in zip(image_ids, predictions):
        for box, label, score in zip(
            prediction["boxes"].tolist(), prediction["labels"].tolist(), prediction["scores"].tolist()
        ):
            if score >= score_threshold:
                detections_by_image[image_id].append(
                    {"box": box, "label": int(label), "score": float(score)}
                )

    total_gt = 0
    matched_gt = 0
    tp = fp = 0
    for image_id, detections in detections_by_image.items():
        record = records_by_id[image_id]
        gt_by_label: dict[int, list[tuple[float, float, float, float]]] = {}
        for obj in record["objects"]:
            label = int(obj["detector_label"])
            gt_by_label.setdefault(label, []).append(tuple(float(v) for v in obj["bbox"]))
            total_gt += 1
        used_gt: set[tuple[int, int]] = set()
        for label, boxes in gt_by_label.items():
            candidates = [d for d in detections if d["label"] == label]
            candidates.sort(key=lambda d: d["score"], reverse=True)
            for candidate in candidates:
                best_iou = 0.0
                best_index = -1
                for index, gt_box in enumerate(boxes):
                    if (label, index) in used_gt:
                        continue
                    iou = _box_iou(candidate["box"], gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_index = index
                if best_iou >= 0.5:
                    used_gt.add((label, best_index))
                    tp += 1
                    matched_gt += 1
                else:
                    fp += 1
    fn = total_gt - matched_gt
    return {
        "precision_at_0.5": round(tp / (tp + fp), 6) if tp + fp > 0 else 0.0,
        "recall_at_0.5": round(tp / (tp + fn), 6) if tp + fn > 0 else 0.0,
        "tp_at_0.5": tp,
        "fp_at_0.5": fp,
        "fn_at_0.5": fn,
    }


def _box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b - inter > 0 else 0.0


def main() -> int:
    args = parse_args()
    checkpoint_path = repo_path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    data_cfg = config["dataset"]
    model_cfg = config["model"]

    from dlcpd25_v2.common import repo_root

    root = repo_root()
    voc_root = root / data_cfg["voc_root"]
    contract_dir = root / data_cfg["contract_dir"]
    test_dataset = IP102DetectionDataset(voc_root, contract_dir, split="test", train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_detection,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    print(f"[evaluate] test samples={len(test_dataset)} batches={len(test_loader)}")

    device = torch.device(args.device)
    model, info = build_detection_model(
        num_classes=int(data_cfg["num_classes"]) + 1,
        image_size=int(model_cfg["input_size"]["min_side"]),
        pretrained_backbone=False,
        box_score_thresh=float(model_cfg.get("box_score_thresh", 0.05)),
        box_nms_thresh=float(model_cfg.get("box_nms_thresh", 0.5)),
        box_detections_per_img=int(model_cfg.get("max_detections_per_image", 30)),
    )
    ema_payload = payload.get("ema") or {}
    ema_state = ema_payload.get("ema_model_state_dict") if isinstance(ema_payload, dict) else None
    model.load_state_dict(ema_state if ema_state is not None else payload["model_state_dict"])
    model = model.to(device).eval()
    print(f"[evaluate] loaded {'EMA' if ema_state is not None else 'raw'} weights; params={info.parameter_count/1e6:.2f}M")

    records_by_id = {image_id: test_dataset.records[image_id] for image_id in test_dataset.ids}
    records = [records_by_id[image_id] for image_id in test_dataset.ids]
    image_ids: list[str] = []
    predictions: list[dict[str, Any]] = []
    started = time.time()
    with torch.inference_mode():
        for images, targets in test_loader:
            batch_ids = [str(target["image_id"]) for target in targets]
            images = [image.to(device, non_blocking=True) for image in images]
            outputs = model(images, None)
            outputs = [{k: v.cpu() for k, v in output.items()} for output in outputs]
            image_ids.extend(batch_ids)
            predictions.extend(outputs)
    inference_seconds = round(time.time() - started, 1)
    print(f"[evaluate] inference {len(image_ids)} images in {inference_seconds}s")

    rows = predictions_to_coco(image_ids, predictions, score_threshold=args.score_threshold)
    coco_gt = build_coco_groundtruth_object(records)
    coco_dt = coco_gt.loadRes(rows)
    stats = _run_coco_eval(coco_gt, coco_dt)
    overall = {
        "test_mAP_50_95": round(float(stats[0]) * 100.0, 4),
        "test_AP50": round(float(stats[1]) * 100.0, 4),
        "test_AP75": round(float(stats[2]) * 100.0, 4),
        "test_AP_small": round(float(stats[3]) * 100.0, 4),
        "test_AP_medium": round(float(stats[4]) * 100.0, 4),
        "test_AP_large": round(float(stats[5]) * 100.0, 4),
        "test_AR_100": round(float(stats[8]) * 100.0, 4),
    }
    overall.update(
        precision_recall_at_iou50(predictions, image_ids, records_by_id, score_threshold=0.5)
    )
    overall["test_support_images"] = len(image_ids)
    overall["evaluation_seconds"] = inference_seconds
    print(json.dumps(overall, ensure_ascii=False, indent=2))

    # Per-class AP.
    class_map = json.loads((root / data_cfg["class_map"]).read_text(encoding="utf-8"))
    name_by_label = {int(item["detector_label"]): str(item["ip102_name"]) for item in class_map["classes"]}
    per_class = []
    for label in sorted({int(item["detector_label"]) for item in class_map["classes"]}):
        cat_stats = _run_coco_eval(coco_gt, coco_dt, cat_ids=[label])
        per_class.append(
            {
                "detector_label": label,
                "name": name_by_label.get(label, f"class_{label}"),
                "ap": round(float(cat_stats[0]) * 100.0, 4),
                "ap50": round(float(cat_stats[1]) * 100.0, 4),
                "ap75": round(float(cat_stats[2]) * 100.0, 4),
            }
        )

    output_dir = repo_path(args.output_dir) if args.output_dir else checkpoint_path.parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "schema_version": 1,
        "model": {
            "architecture": info.backbone,
            "input_size": info.image_size,
            "parameters_m": round(info.parameter_count / 1e6, 2),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "weights": "EMA" if ema_state is not None else "raw",
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "best_epoch": int(payload.get("best_epoch", -1)),
        },
        "test_metrics": overall,
        "per_class": per_class,
        "generated_at": time.time(),
    }
    (output_dir / "test-evaluation.json").write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "test-metrics-per-class.json").write_text(
        json.dumps(per_class, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[evaluate] saved artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
