"""Gradio presentation layer for the single-image classification workflow."""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any

from dlcpd25_classifier.inference import (
    AppSettings,
    ImageLimits,
    Predictor,
    create_fake_predictor,
)
from dlcpd25_classifier.inference.errors import InferenceError

LOGGER = logging.getLogger(__name__)

APP_CSS = """
.gradio-container {
    width: min(1180px, 100%) !important;
    max-width: 100% !important;
    margin: 0 auto !important;
    box-sizing: border-box !important;
}
.app-title { margin: 4px 0 18px 0; }
.app-title h1 { font-size: 24px !important; line-height: 1.25 !important; letter-spacing: 0 !important; }
.result-status { min-height: 42px; }
.result-status p { margin: 0; }
.primary-button { border-radius: 6px !important; }
@media (max-width: 640px) {
    .gradio-container { padding-left: 12px !important; padding-right: 12px !important; }
    .app-title h1 { font-size: 20px !important; }
}
"""


def predictor_from_settings(settings: AppSettings) -> Predictor:
    if settings.mode == "bundle":
        return Predictor.from_bundle(settings.model_bundle, device=settings.device)
    return create_fake_predictor(
        settings.taxonomy_path,
        class_id=settings.fake_class_id,
        model_version=settings.model_version,
        data_version=settings.data_version,
        config_sha256=settings.config_sha256,
        git_commit=settings.git_commit,
        image_size=settings.image_size,
        confidence_threshold=settings.confidence_threshold,
        top_k=settings.top_k,
        image_limits=ImageLimits(
            max_upload_bytes=settings.max_upload_bytes,
            max_image_pixels=settings.max_image_pixels,
        ),
    )


def _empty_result(status: str) -> tuple[Any, ...]:
    return status, "", "", "", 0.0, [], "", "", 0.0


def classify_image(image: Any, predictor: Predictor) -> tuple[Any, ...]:
    """Translate Predictor results and public errors into Gradio component values."""
    if image is None:
        return _empty_result("请先选择一张图片。")
    try:
        result = predictor.predict(image)
    except InferenceError as exc:
        return _empty_result(exc.user_message)
    except Exception:
        LOGGER.exception("unexpected application inference failure")
        return _empty_result("图片处理失败，请稍后重试。")

    if result.low_confidence:
        status = "结果不确定，图片可能不属于系统已知类别。"
    elif result.model_version.startswith("p1-fixed-logits"):
        status = "P1 固定假模型输出，仅用于应用联调。"
    else:
        status = "分类完成。"
    rows = [
        [
            item.rank,
            item.class_id,
            item.host_zh,
            item.category_zh,
            item.official_name,
            round(item.confidence * 100.0, 2),
        ]
        for item in result.top_k
    ]
    version = (
        f"模型 {result.model_version} | 数据 {result.data_version} | "
        f"设备 {result.device} | Schema v{result.schema_version}"
    )
    trace = f"配置 SHA-256 {result.config_sha256} | Git {result.git_commit}"
    return (
        status,
        result.host_zh,
        result.category_zh,
        result.detail_name,
        round(result.confidence * 100.0, 2),
        rows,
        version,
        trace,
        result.inference_ms,
    )


def build_app(predictor: Predictor):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio is not installed; install the project app dependencies") from exc

    with gr.Blocks(title="DLCPD-25 农产品图像分类", css=APP_CSS) as demo:
        gr.Markdown("# DLCPD-25 农产品病虫害与缺陷分类", elem_classes="app-title")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=320):
                image = gr.Image(
                    label="待分类图片",
                    type="filepath",
                    image_mode=None,
                    height=410,
                    sources=["upload", "clipboard"],
                )
                classify = gr.Button("开始分类", variant="primary", elem_classes="primary-button")
            with gr.Column(scale=6, min_width=360):
                status = gr.Markdown("等待图片。", elem_classes="result-status")
                with gr.Row():
                    host = gr.Textbox(label="宿主作物", interactive=False)
                    category = gr.Textbox(label="四大类属性", interactive=False)
                detail = gr.Textbox(label="具体类别", interactive=False)
                with gr.Row():
                    confidence = gr.Number(label="置信度 (%)", interactive=False, precision=2)
                    inference_ms = gr.Number(label="推理耗时 (ms)", interactive=False, precision=3)
        top_k = gr.Dataframe(
            headers=["排名", "Class ID", "宿主", "属性", "具体类别", "置信度 (%)"],
            datatype=["number", "number", "str", "str", "str", "number"],
            label="Top-5",
            interactive=False,
            wrap=True,
        )
        version = gr.Markdown()
        trace = gr.Markdown()
        outputs = [
            status,
            host,
            category,
            detail,
            confidence,
            top_k,
            version,
            trace,
            inference_ms,
        ]
        handler = partial(classify_image, predictor=predictor)
        classify.click(handler, inputs=[image], outputs=outputs, api_name="classify")
        image.upload(handler, inputs=[image], outputs=outputs, api_name=False)
    return demo


def load_app(config_path: str | Path):
    settings = AppSettings.from_yaml(config_path)
    return build_app(predictor_from_settings(settings))
