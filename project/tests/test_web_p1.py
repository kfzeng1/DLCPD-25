import os
from pathlib import Path

from dlcpd25_classifier.inference import create_fake_predictor
from dlcpd25_classifier.web import build_app, classify_image
from dlcpd25_classifier.web.__main__ import ensure_local_no_proxy
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = ROOT / "metadata" / "class-taxonomy.json"


def test_web_adapter_exposes_all_classification_fields() -> None:
    predictor = create_fake_predictor(TAXONOMY_PATH, class_id=178)
    output = classify_image(Image.new("RGB", (64, 48), (40, 100, 60)), predictor)
    assert len(output) == 9
    status, host, category, detail, confidence, rows, version, trace, elapsed = output
    assert "固定假模型" in status
    assert host == "番茄"
    assert category == "植物病害"
    assert detail == "tomato bacterial spot"
    assert confidence > 50.0
    assert len(rows) == 5
    assert "p1-fixed-logits-v1" in version
    assert "配置 SHA-256" in trace
    assert elapsed >= 0.0


def test_web_adapter_returns_public_error_without_traceback() -> None:
    predictor = create_fake_predictor(TAXONOMY_PATH)
    output = classify_image(b"broken", predictor)
    assert output[0] == "图片已损坏或无法识别，请重新选择有效图片。"
    assert all(value in ("", 0.0, []) for value in output[1:])


def test_gradio_app_builds_with_bound_predictor() -> None:
    predictor = create_fake_predictor(TAXONOMY_PATH)
    app = build_app(predictor)
    config = app.get_config_file()
    labels = {component.get("props", {}).get("label") for component in config["components"]}
    assert {"待分类图片", "宿主作物", "四大类属性", "具体类别", "Top-5"} <= labels
    image_component = next(
        component
        for component in config["components"]
        if component.get("props", {}).get("label") == "待分类图片"
    )
    assert image_component["props"]["type"] == "filepath"


def test_local_launcher_adds_exact_proxy_bypass(monkeypatch) -> None:
    monkeypatch.setenv("NO_PROXY", "127.0.0.0/8")
    monkeypatch.setenv("no_proxy", "localhost")
    ensure_local_no_proxy("127.0.0.1")
    assert "127.0.0.1" in os.environ["NO_PROXY"].split(",")
    assert "127.0.0.1" in os.environ["no_proxy"].split(",")
