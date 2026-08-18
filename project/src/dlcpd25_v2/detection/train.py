"""CLI entry point for IP102 Plan-A detection training.

Example:
    python -m dlcpd25_v2.detection.train \
        --config configs/plan-a/detection.yaml \
        --epochs 1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from dlcpd25_v2.common import repo_path, repo_root
from dlcpd25_v2.detection.trainer import DetectionTrainConfig, train_detection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/plan-a/detection.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", type=str, choices=["bfloat16", "float16"], default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    config = load_yaml(repo_path(args.config))
    data_cfg = config["dataset"]
    train_cfg = config["training"]
    run_cfg = config["run"]

    schedule_epochs = int(train_cfg.get("schedule_epochs", int(train_cfg["epochs"])))
    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    train_cfg["schedule_epochs"] = schedule_epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.workers is not None:
        train_cfg["workers"] = int(args.workers)
    if args.amp_dtype is not None:
        train_cfg["amp_dtype"] = args.amp_dtype

    run_id = args.run_id or run_cfg.get("run_id") or datetime.now(timezone.utc).strftime("fasterrcnn-convnext-tiny-640-%Y%m%d-%H%M%S")
    output_dir = repo_path(run_cfg.get("output_dir", "artifacts/training/detection"))
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resume_path = repo_path(args.resume) if args.resume else None
    if resume_path is None and run_cfg.get("resume") == "last":
        last = run_dir / "checkpoints" / "last.pt"
        if last.is_file():
            resume_path = last

    cfg = DetectionTrainConfig(
        config=config,
        run_dir=run_dir,
        device=args.device,
        amp_dtype=train_cfg.get("amp_dtype", "bfloat16"),
        total_epochs=int(train_cfg["epochs"]),
        resume_path=resume_path,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )
    result = train_detection(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def load_yaml(path: Path):
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


if __name__ == "__main__":
    raise SystemExit(main())
