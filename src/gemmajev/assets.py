"""Workspace paths and verified downloads for the supported local environment."""

from __future__ import annotations

import fcntl
import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

MODELS = ("e4b", "12b")


def workspace_root(workspace: str | Path | None = None) -> Path:
    if workspace is not None:
        return Path(workspace).expanduser().resolve()
    if os.environ.get("GEMMAJEV_WORKSPACE"):
        return Path(os.environ["GEMMAJEV_WORKSPACE"]).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "native").is_dir():
            return parent
    return Path.cwd().resolve()


def check_platform() -> dict:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("GemmaJev currently supports macOS on Apple Silicon only.")
    memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    if memory < 16 * 2**30:
        raise RuntimeError("The supported minimum is 16 GiB of unified memory.")
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    return {"system": "macOS", "chip": chip, "memory_bytes": memory}


def model_config(name: str, workspace: str | Path | None = None) -> dict:
    if name not in MODELS:
        raise ValueError("model must be e4b or 12b")
    root = workspace_root(workspace)
    manifest = Path(__file__).with_name("models.toml")
    with manifest.open("rb") as stream:
        config = tomllib.load(stream)["models"][name].copy()
    config["model_path"] = str(root / config["model_path"])
    return config


def runtime_path(workspace: str | Path | None = None) -> Path:
    return workspace_root(workspace) / ".runtime/build/bin/gemmajev-worker"


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _matches(path: Path, size: int | None, digest: str | None) -> bool:
    return (
        path.is_file()
        and (size is None or path.stat().st_size == size)
        and (digest is None or file_sha256(path) == digest)
    )


def download_file(
    url: str,
    destination: str | Path,
    *,
    sha256: str | None = None,
    size: int | None = None,
) -> Path:
    """Download atomically with system TLS; preserve an old file on failure."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_name(destination.name + ".download.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _matches(destination, size, sha256):
            return destination
        if size is not None and shutil.disk_usage(destination.parent).free < size + 2**30:
            raise RuntimeError(f"Not enough disk space to download {destination.name}.")
        if shutil.which("curl") is None:
            raise RuntimeError("The macOS curl command is required for downloads.")
        fd, filename = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".part", dir=destination.parent
        )
        os.close(fd)
        partial = Path(filename)
        try:
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "3",
                    "--retry-delay",
                    "1",
                    "--connect-timeout",
                    "30",
                    "--output",
                    str(partial),
                    "--",
                    url,
                ],
                check=True,
            )
            if not _matches(partial, size, sha256):
                raise RuntimeError(f"Downloaded bytes failed verification: {destination.name}")
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
    return destination


def prepare_model(name: str, workspace: str | Path | None = None) -> Path:
    config = model_config(name, workspace)
    url = f"https://huggingface.co/{config['repo_id']}/resolve/{config['revision']}/{config['model_file']}"
    return download_file(
        url, config["model_path"], size=config["model_size"], sha256=config["model_sha256"]
    )


def build_runtime(workspace: str | Path | None = None) -> Path:
    check_platform()
    root = workspace_root(workspace)
    script = Path(__file__).resolve().parents[2] / "scripts/build_runtime.sh"
    if not script.is_file():
        script = Path.cwd() / "scripts/build_runtime.sh"
    if not script.is_file():
        raise RuntimeError(
            "Run setup from a GemmaJev source checkout containing scripts/build_runtime.sh."
        )
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["bash", str(script)],
        cwd=script.parent.parent,
        env={**os.environ, "GEMMAJEV_WORKSPACE": str(root)},
        check=True,
    )
    return runtime_path(root)
