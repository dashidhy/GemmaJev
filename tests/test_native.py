"""CPU-only checks of the compiled artifact and its reproducible provenance."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / ".runtime/build/bin/gemmajev-worker"


@pytest.fixture
def binary():
    if not BINARY.is_file():
        pytest.skip("Run scripts/build_runtime.sh to check the compiled worker")
    return BINARY


def invoke(binary, *arguments):
    return subprocess.run(
        [str(binary), *arguments], capture_output=True, text=True, timeout=10, check=False
    )


def test_worker_provenance_and_non_generating_template(binary):
    response = invoke(binary, "--self-test")
    assert response.returncode == 0 and not response.stderr
    result = json.loads(response.stdout)
    assert result["ok"]
    assert result["answer_prefixes"] == {
        "e4b": "<|turn>model\n",
        "12b": "<|turn>model\n<|channel>thought\n<channel|>",
    }
    assert (
        result["worker_source_sha256"]
        == hashlib.sha256((ROOT / "native/gemmajev.cpp").read_bytes()).hexdigest()
    )
    assert result["runtime_patches"] == [
        {
            "name": "gemma4-raw-logits.patch",
            "sha256": hashlib.sha256(
                (ROOT / "native/patches/gemma4-raw-logits.patch").read_bytes()
            ).hexdigest(),
        }
    ]
    assert result["context_size"] == 4096
    assert result["batch_size"] == 512 and result["microbatch_size"] == 128


@pytest.mark.parametrize(
    "arguments",
    [
        ["--model-size", "unsupported"],
        ["--ctx-size", "4097"],
        ["--ctx-size", "513"],
        ["--ctx-size", "4000oops"],
        ["--threads", "0"],
        ["--unknown", "1"],
    ],
)
def test_invalid_configuration_rejected_before_model_load(binary, arguments):
    response = invoke(binary, "--model", "/does/not/exist.gguf", *arguments)
    assert response.returncode == 1 and not response.stderr
    result = json.loads(response.stdout)
    assert result["ready"] is False
    assert "load" not in result["error"]
