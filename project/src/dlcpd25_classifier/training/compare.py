"""Compare completed A2 runs using validation metrics only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from dlcpd25_classifier.training.metrics import epoch_duration_seconds


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((path / "resolved-config.yaml").read_text(encoding="utf-8"))
    if metrics.get("stage") != "A2" or metrics.get("status") != "completed_pending_project_lead_acceptance":
        raise ValueError(f"incomplete A2 run: {path}")
    if metrics.get("test_metrics_read") is not False:
        raise ValueError(f"A2 run does not prove test isolation: {path}")
    return metrics, config


def recorded_duration(path: Path, metrics: dict[str, Any]) -> float:
    history_path = path / "history.json"
    if not history_path.is_file():
        return float(metrics["duration_seconds"])
    history = json.loads(history_path.read_text(encoding="utf-8"))
    return epoch_duration_seconds(history)


def compare_runs(run_paths: list[Path]) -> dict[str, Any]:
    if len(run_paths) != 2:
        raise ValueError("A2 comparison requires exactly two runs")
    loaded = [(path, *load_run(path)) for path in run_paths]
    strategies = {metrics["loss_strategy"] for _path, metrics, _config in loaded}
    if strategies != {"ce", "weighted_ce"}:
        raise ValueError("A2 comparison requires CE and weighted CE")
    first_config = loaded[0][2]
    comparable_fields = (
        "seed",
        "data_root",
        "taxonomy",
        "split_dir",
        "model",
        "training",
    )
    for _path, _metrics, config in loaded[1:]:
        for field in comparable_fields:
            if config[field] != first_config[field]:
                raise ValueError(f"A2 runs differ outside loss strategy: {field}")
        allowed_actual_differences = {"loss_strategy", "class_weight_formula"}
        comparable_actual = {
            key: value
            for key, value in first_config["actual"].items()
            if key not in allowed_actual_differences
        }
        candidate_actual = {
            key: value
            for key, value in config["actual"].items()
            if key not in allowed_actual_differences
        }
        if candidate_actual != comparable_actual:
            raise ValueError("A2 runs differ outside loss strategy: actual")
    selected = max(loaded, key=lambda item: item[1]["best_val"]["macro_f1"])
    return {
        "schema_version": 1,
        "stage": "A2-comparison",
        "status": "completed_pending_project_lead_acceptance",
        "selection_metric": "val_macro_f1",
        "selected_run": selected[1]["run_id"],
        "selected_loss_strategy": selected[1]["loss_strategy"],
        "test_metrics_read": False,
        "runs": [
            {
                "run_id": metrics["run_id"],
                "loss_strategy": metrics["loss_strategy"],
                "best_epoch": metrics["best_epoch"],
                "best_val": metrics["best_val"],
                "duration_seconds": recorded_duration(path, metrics),
                "duration_scope": "sum_train_and_val_epoch_seconds",
                "peak_memory_bytes": metrics["peak_memory_bytes"],
                "metrics_sha256": sha256_file(path / "metrics.json"),
                "best_checkpoint_sha256": sha256_file(path / "best.pt"),
            }
            for path, metrics, _config in loaded
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        print(f"refusing to overwrite comparison: {args.output_dir}", file=sys.stderr)
        return 1
    try:
        result = compare_runs([path.resolve() for path in args.run])
        args.output_dir.mkdir(parents=True)
        result_path = args.output_dir / "comparison.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "checksums.sha256").write_text(
            f"{sha256_file(result_path)}  comparison.json\n", encoding="utf-8"
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"A2 comparison failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
