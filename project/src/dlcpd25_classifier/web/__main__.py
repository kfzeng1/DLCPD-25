"""Launch the P1 Gradio application."""

from __future__ import annotations

import argparse
from pathlib import Path

from .app import load_app

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = ROOT / "project" / "configs" / "app.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the DLCPD-25 classification application")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = load_app(args.config)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
