"""Shared repo-root and path helpers for DLCPD-25 Plan-A training code."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    override = os.environ.get("DLCPD25_ROOT")
    if override:
        path = Path(override).expanduser().resolve()
        if path.is_dir():
            return path
    path = Path.cwd().resolve()
    for candidate in (path, *path.parents):
        if (candidate / "configs" / "plan-a").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(
        "DLCPD-25 repo root not found. Run from the repo root or set DLCPD25_ROOT."
    )


def repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve()
