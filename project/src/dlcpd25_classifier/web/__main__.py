"""Launch the DLCPD-25 joint classification/detection application."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dlcpd25_classifier.inference.errors import InferenceError

from .app import load_app

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = ROOT / "project" / "configs" / "app.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the DLCPD-25 joint application")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def ensure_local_no_proxy(host: str) -> None:
    if host not in {"127.0.0.1", "localhost", "0.0.0.0"}:
        return
    for name in ("NO_PROXY", "no_proxy"):
        entries = [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]
        for local_host in ("127.0.0.1", "localhost"):
            if local_host not in entries:
                entries.append(local_host)
        os.environ[name] = ",".join(entries)


def main() -> None:
    args = parse_args()
    ensure_local_no_proxy(args.host)
    try:
        app = load_app(args.config)
    except InferenceError as exc:
        raise SystemExit(f"应用启动失败 [{exc.code}]：{exc.user_message}") from None
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"应用配置无效：{exc}") from None
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
