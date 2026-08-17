"""CLI entry point for DLCPD-25 Plan-A classification training.

Example:
    python -m dlcpd25_v2.classification.train \
        --config configs/plan-a/classification.yaml \
        --epochs 1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from dlcpd25_v2.classification.trainer import TrainConfig, train
from dlcpd25_v2.common import repo_path, repo_root


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/plan-a/classification.yaml"),
        help="Plan-A classification YAML config, relative to repo root",
    )
    parser.add_argument("--epochs", type=int, default=None, help="override total epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", type=str, choices=["bfloat16", "float16"], default=None)
    parser.add_argument("--resume", type=Path, default=None, help="resume from a checkpoint path")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="initialize model/EMA from a checkpoint and start a new run")
    parser.add_argument("--limit-train-batches", type=int, default=None, help="smoke-test override")
    parser.add_argument("--limit-val-batches", type=int, default=None, help="smoke-test override")
    parser.add_argument("--validate-only", action="store_true", help="run validation using best checkpoint")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    config_path = repo_path(args.config)
    config = load_yaml(config_path)
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

    run_id = args.run_id or run_cfg.get("run_id") or datetime.now(timezone.utc).strftime("convnext-tiny-384-%Y%m%d-%H%M%S")
    output_dir = repo_path(run_cfg.get("output_dir", "artifacts/training/classification"))
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resume_path = args.resume
    if resume_path is None and run_cfg.get("resume") == "last":
        last = run_dir / "checkpoints" / "last.pt"
        if last.is_file():
            resume_path = last
    if resume_path is not None:
        resume_path = repo_path(resume_path)

    if args.validate_only:
        raise SystemExit("--validate-only is not implemented yet; train one epoch first to get a checkpoint.")

    init_checkpoint = repo_path(args.init_checkpoint) if args.init_checkpoint else None
    train_config = TrainConfig(
        config=config,
        run_dir=run_dir,
        device=args.device,
        amp_dtype=train_cfg.get("amp_dtype", "bfloat16"),
        total_epochs=int(train_cfg["epochs"]),
        resume_path=resume_path,
        init_checkpoint=init_checkpoint,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )
    result = train(train_config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
