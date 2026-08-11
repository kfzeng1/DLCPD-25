import json
from pathlib import Path

import pytest
import torch
import yaml
from dlcpd25_classifier.training.a2 import (
    clipped_inverse_frequency_weights,
    make_scheduler,
)
from dlcpd25_classifier.training.compare import compare_runs
from dlcpd25_classifier.training.dashboard import PAGE, build_payload
from dlcpd25_classifier.training.metrics import (
    ClassificationMetrics,
    epoch_duration_seconds,
)
from dlcpd25_classifier.training.progress import append_progress
from torch import nn
from torch.optim import AdamW


def test_streaming_metrics_match_known_predictions() -> None:
    metrics = ClassificationMetrics(num_classes=3)
    logits = torch.tensor(
        [[9.0, 1.0, 0.0], [0.0, 8.0, 1.0], [0.0, 7.0, 6.0], [0.0, 1.0, 9.0]]
    )
    targets = torch.tensor([0, 1, 2, 2])
    metrics.update(logits, targets)
    summary, per_class = metrics.compute()
    assert summary["accuracy"] == pytest.approx(0.75)
    assert summary["top5_accuracy"] == 1.0
    assert summary["balanced_accuracy"] == pytest.approx((1.0 + 1.0 + 0.5) / 3)
    assert len(per_class) == 3
    assert metrics.as_serializable_confusion() == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]


def test_clipped_inverse_frequency_weights_favor_rare_classes() -> None:
    weights = clipped_inverse_frequency_weights([3] + [100] * 202)
    assert weights.shape == (203,)
    assert weights.mean() == pytest.approx(1.0)
    assert weights[0] > weights[1]
    assert torch.isfinite(weights).all()


def test_warmup_cosine_scheduler_reaches_zero() -> None:
    model = nn.Linear(2, 2)
    optimizer = AdamW(model.parameters(), lr=1.0)
    scheduler = make_scheduler(optimizer, total_epochs=5, warmup_epochs=2)
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] == pytest.approx(0.5)
    assert rates[1] == pytest.approx(1.0)
    assert rates[-1] == pytest.approx(0.0)


def write_fake_run(root: Path, run_id: str, strategy: str, macro_f1: float) -> Path:
    run = root / run_id
    run.mkdir()
    metrics = {
        "stage": "A2",
        "status": "completed_pending_project_lead_acceptance",
        "run_id": run_id,
        "loss_strategy": strategy,
        "best_epoch": 2,
        "best_val": {
            "accuracy": 0.8,
            "top5_accuracy": 0.95,
            "macro_f1": macro_f1,
            "balanced_accuracy": 0.75,
        },
        "duration_seconds": 10.0,
        "peak_memory_bytes": 100,
        "test_metrics_read": False,
    }
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run / "best.pt").write_bytes(run_id.encode())
    config = {
        "seed": 1,
        "data_root": "data",
        "taxonomy": "taxonomy",
        "split_dir": "splits",
        "model": {"name": "resnet50", "num_classes": 203},
        "training": {
            "amp": True,
            "weight_decay": 0.0001,
            "label_smoothing": 0.1,
            "warmup_epochs": 2,
        },
        "actual": {
            "batch_size": 16,
            "workers": 6,
            "epochs": 5,
            "learning_rate": 0.001,
        },
    }
    (run / "resolved-config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return run


def test_comparison_selects_only_by_validation_macro_f1(tmp_path: Path) -> None:
    ce = write_fake_run(tmp_path, "ce", "ce", 0.6)
    weighted = write_fake_run(tmp_path, "weighted", "weighted_ce", 0.7)
    (ce / "history.json").write_text(
        json.dumps(
            [
                {
                    "train": {"duration_seconds": 3.0},
                    "val": {"duration_seconds": 2.0},
                }
            ]
        ),
        encoding="utf-8",
    )
    result = compare_runs([ce, weighted])
    assert result["selected_run"] == "weighted"
    assert result["selection_metric"] == "val_macro_f1"
    assert result["test_metrics_read"] is False
    assert result["runs"][0]["duration_seconds"] == 5.0
    assert result["runs"][1]["duration_seconds"] == 10.0


def test_comparison_rejects_non_loss_configuration_difference(tmp_path: Path) -> None:
    ce = write_fake_run(tmp_path, "ce", "ce", 0.6)
    weighted = write_fake_run(tmp_path, "weighted", "weighted_ce", 0.7)
    config_path = weighted / "resolved-config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["actual"]["learning_rate"] = 0.002
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="outside loss strategy: actual"):
        compare_runs([ce, weighted])


def test_epoch_duration_includes_history_before_resume() -> None:
    history = [
        {"train": {"duration_seconds": 10.0}, "val": {"duration_seconds": 2.0}},
        {"train": {"duration_seconds": 9.0}, "val": {"duration_seconds": 3.0}},
    ]
    assert epoch_duration_seconds(history) == 24.0


def test_progress_csv_appends_each_epoch_once(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    record = {
        "epoch": 1,
        "learning_rate": 0.0001,
        "train": {
            "loss": 1.0,
            "accuracy": 0.5,
            "images_per_second": 100.0,
            "duration_seconds": 10.0,
        },
        "val": {
            "loss": 0.8,
            "accuracy": 0.6,
            "top5_accuracy": 0.9,
            "macro_f1": 0.55,
            "balanced_accuracy": 0.58,
            "images_per_second": 120.0,
            "duration_seconds": 2.0,
        },
    }
    assert append_progress(path, record) is True
    assert append_progress(path, record) is False
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("epoch,learning_rate,train_loss")
    assert lines[1].startswith("1,0.0001,1.0,0.5")


def test_dashboard_payload_and_page_are_run_scoped(tmp_path: Path) -> None:
    run = tmp_path / "a2-run"
    run.mkdir()
    (run / "history.json").write_text('[{"epoch": 1}]', encoding="utf-8")
    (run / "run-state.json").write_text('{"status": "running"}', encoding="utf-8")
    payload = build_payload(run)
    assert payload["run_id"] == "a2-run"
    assert payload["history"] == [{"epoch": 1}]
    assert payload["state"]["status"] == "running"
    assert "Validation Scores" in PAGE
    assert "setInterval(refresh,5000)" in PAGE
