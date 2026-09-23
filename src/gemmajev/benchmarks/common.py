"""Shared pinned-source I/O. Dataset bodies are kept outside version control."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..assets import download_file, workspace_root

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "configs/benchmarks"
if not CONFIG_ROOT.is_dir():
    CONFIG_ROOT = PACKAGE / "configs"


def _verified_bytes(path: Path, record: dict) -> bytes:
    data = path.read_bytes()
    if len(data) != record["size_bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"Pinned source hash or size mismatch: {path}")
    return data


def _download_source(record: dict, destination: Path) -> None:
    download_file(record["url"], destination, sha256=record["sha256"], size=record["size_bytes"])


def cache_root(workspace=None) -> Path:
    return workspace_root(workspace) / ".datasets"
