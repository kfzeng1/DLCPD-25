"""Joint predictor: ConvNeXt-Tiny classifier + ConvNeXt-Tiny-FPN detector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from dlcpd25_v2.classification.model import build_model
from dlcpd25_v2.classification.transforms import build_transforms
from dlcpd25_v2.common import repo_root
from dlcpd25_v2.detection.model import build_detection_model

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


def _load_checkpoint_state(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    ema_payload = payload.get("ema") or {}
    ema_state = ema_payload.get("ema_model_state_dict") if isinstance(ema_payload, dict) else None
    return payload, ema_state


class JointPredictor:
    """Loads both frozen experts and analyzes RGB images."""

    def __init__(
        self,
        classification_checkpoint: Path | str,
        detection_checkpoint: Path | str,
        classification_taxonomy: Path | str,
        detection_class_map: Path | str,
        detection_score_threshold: float = 0.30,
        detection_max_detections: int = 30,
        annotated_max_side: int = 1200,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.classification_checkpoint = Path(classification_checkpoint)
        self.detection_checkpoint = Path(detection_checkpoint)
        self.detection_score_threshold = float(detection_score_threshold)
        self.detection_max_detections = int(detection_max_detections)
        self.annotated_max_side = int(annotated_max_side)

        # Frozen metadata.
        taxonomy = json.loads(Path(classification_taxonomy).read_text(encoding="utf-8"))
        self.class_map = {
            int(item["class_id"]): {
                "name": str(item["official_name"]),
                "host_id": str(item["host_id"]),
                "host_zh": str(item["host_zh"]),
                "category": str(item["category"]),
                "category_zh": str(item["category_zh"]),
            }
            for item in taxonomy["classes"]
        }
        detection_map = json.loads(Path(detection_class_map).read_text(encoding="utf-8"))
        self.detection_label_map = {
            int(item["detector_label"]): {
                "class_id": int(item["dlcpd25_class_id"]),
                "ip102_name": str(item["ip102_name"]),
            }
            for item in detection_map["classes"]
        }

        # Classification model.
        cls_payload, cls_ema = _load_checkpoint_state(self.classification_checkpoint)
        cls_config = cls_payload["config"]
        self.classification_image_size = int(cls_config["model"]["input_size"])
        self.classifier, cls_info = build_model(
            architecture=cls_config["model"]["architecture"],
            num_classes=int(cls_config["model"]["num_classes"]),
            num_hosts=int(cls_config["model"]["auxiliary_heads"]["host"]),
            num_categories=int(cls_config["model"]["auxiliary_heads"]["category"]),
            pretrained=False,
        )
        self.classifier.load_state_dict(cls_ema if cls_ema is not None else cls_payload["model_state_dict"])
        self.classifier = self.classifier.to(self.device).eval()
        self.classification_transform = build_transforms(self.classification_image_size, train=False)

        # Detection model.
        det_payload, det_ema = _load_checkpoint_state(self.detection_checkpoint)
        det_config = det_payload["config"]
        self.detector, det_info = build_detection_model(
            num_classes=int(det_config["dataset"]["num_classes"]) + 1,
            image_size=int(det_config["model"]["input_size"]["min_side"]),
            pretrained_backbone=False,
            box_score_thresh=float(det_config["model"].get("box_score_thresh", 0.05)),
            box_nms_thresh=float(det_config["model"].get("box_nms_thresh", 0.5)),
            box_detections_per_img=int(det_config["model"].get("max_detections_per_image", 30)),
        )
        self.detector.load_state_dict(det_ema if det_ema is not None else det_payload["model_state_dict"])
        self.detector = self.detector.to(self.device).eval()

        self.model_info = {
            "device": str(self.device),
            "classification_parameters_m": round(cls_info.parameter_count / 1e6, 2),
            "detection_parameters_m": round(det_info.parameter_count / 1e6, 2),
        }

    @staticmethod
    def load_image(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")

    def classify(self, image: Image.Image, top_k: int = 5) -> dict[str, Any]:
        tensor = self.classification_transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits, host_logits, category_logits = self.classifier(tensor)
        probabilities = torch.softmax(logits.float(), dim=1)[0]
        host_probs = torch.softmax(host_logits.float(), dim=1)[0]
        category_probs = torch.softmax(category_logits.float(), dim=1)[0]
        top_k = min(top_k, probabilities.shape[0])
        values, indices = probabilities.topk(top_k)
        top5 = []
        for value, index in zip(values.tolist(), indices.tolist()):
            info = self.class_map[int(index)]
            top5.append(
                {
                    "class_id": int(index),
                    "name": info["name"],
                    "host_zh": info["host_zh"],
                    "category": info["category"],
                    "category_zh": info["category_zh"],
                    "probability": round(float(value), 6),
                }
            )
        return {
            "top5": top5,
            "host": {
                "host_zh": self.class_map[int(indices[0])]["host_zh"],
                "probability": round(float(host_probs.max()), 6),
            },
            "category": {
                "category_zh": self.class_map[int(indices[0])]["category_zh"],
                "probability": round(float(category_probs.max()), 6),
            },
        }

    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        from torchvision.transforms.functional import to_tensor

        tensor = to_tensor(image).to(self.device)
        with torch.inference_mode():
            outputs = self.detector([tensor])[0]
        detections: list[dict[str, Any]] = []
        for box, label, score in zip(
            outputs["boxes"].tolist(), outputs["labels"].tolist(), outputs["scores"].tolist()
        ):
            if score < self.detection_score_threshold:
                continue
            mapped = self.detection_label_map.get(int(label))
            class_id = mapped["class_id"] if mapped else -1
            class_info = self.class_map.get(class_id, {})
            detections.append(
                {
                    "detector_label": int(label),
                    "ip102_name": mapped["ip102_name"] if mapped else f"class_{label}",
                    "class_id": class_id,
                    "name": class_info.get("name", ""),
                    "host_zh": class_info.get("host_zh", ""),
                    "score": round(float(score), 6),
                    "box": [round(float(v), 2) for v in box],
                }
            )
        detections.sort(key=lambda item: item["score"], reverse=True)
        return detections[: self.detection_max_detections]

    def annotate(self, image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        width, height = annotated.size
        line_width = max(2, round(min(width, height) / 350))
        font_size = max(16, round(min(width, height) / 32))
        font = None
        for candidate in FONT_CANDIDATES:
            path = Path(candidate)
            if path.is_file():
                try:
                    font = ImageFont.truetype(str(path), size=font_size)
                    break
                except OSError:
                    continue
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            x1 = max(0, min(width, x1))
            y1 = max(0, min(height, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))
            draw.rectangle((x1, y1, x2, y2), outline=(56, 189, 248), width=line_width)
            label = f"{detection['ip102_name']} {detection['score']:.2f}"
            text_box = draw.textbbox((x1, y1), label, font=font)
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]
            top = max(0, y1 - text_h - 4)
            draw.rectangle((x1, top, x1 + text_w + 8, top + text_h + 4), fill=(14, 165, 233))
            draw.text((x1 + 4, top + 2), label, fill=(255, 255, 255), font=font)
        if max(annotated.size) > self.annotated_max_side:
            ratio = self.annotated_max_side / max(annotated.size)
            annotated = annotated.resize(
                (round(annotated.width * ratio), round(annotated.height * ratio)),
                Image.Resampling.LANCZOS,
            )
        return annotated
