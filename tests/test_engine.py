"""Exercise real child-process lifecycle and protocol failures without model weights."""

from __future__ import annotations

import hashlib
import math
import sys

import pytest

import gemmajev.engine as module
from gemmajev.engine import GemmaJev

REQUEST = {
    "task": "Classify the message.",
    "state": "Please refund my order.",
    "query": "Which department?",
    "options": {"refund": "Refund requests", "delivery": "Delivery inquiries"},
}


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    weights = tmp_path / "weights.gguf"
    weights.write_bytes(b"fake")
    binary = tmp_path / "worker"
    monkeypatch.setattr(module, "check_platform", dict)
    monkeypatch.setattr(module, "runtime_path", lambda *_: binary)
    monkeypatch.setattr(
        module,
        "model_config",
        lambda *_: {
            "model_path": str(weights),
            "model_size": 4,
            "model_sha256": hashlib.sha256(b"fake").hexdigest(),
        },
    )

    def install(mode="ok"):
        binary.write_text(
            f"#!{sys.executable}\n"
            + """
import json, math, sys, time
MODE = """
            + repr(mode)
            + """
def emit(value):
    print(json.dumps(value), flush=True)
size = sys.argv[sys.argv.index('--model-size') + 1]
if MODE == 'startup_error':
    emit({'ready': False, 'error': 'load failed'})
    sys.exit(1)
emit({'ready': True, 'model_size': size, 'context_size': 4096, 'load_ms': 2,
      'runtime_revision': 'pinned', 'worker_source_sha256': 'a' * 64,
      'runtime_patches': [], 'batch_size': 512, 'microbatch_size': 128})
for line in sys.stdin:
    request = json.loads(line)
    if MODE == 'timeout':
        time.sleep(20)
    if MODE == 'eof':
        sys.exit(0)
    if MODE == 'malformed':
        print('not json', flush=True)
        continue
    if MODE == 'nonobject':
        emit([])
        continue
    if MODE == 'native_error':
        emit({'ok': False, 'error': 'context overflow'})
        continue
    labels = request['labels']
    logits = [-i for i in range(len(labels))]
    denom = sum(math.exp(x) for x in logits)
    candidates = [{'label': label, 'logit': logit, 'probability': math.exp(logit) / denom}
                  for label, logit in zip(labels, logits)]
    if MODE == 'reordered':
        candidates.reverse()
    if MODE == 'not_softmax':
        for c in candidates:
            c['probability'] = 1 / len(candidates)
    if MODE == 'nonfinite':
        candidates[0]['logit'] = float('nan')
    tokens = 10 if MODE != 'bad_counts' else 11
    emit({'ok': True, 'candidates': candidates, 'prompt_tokens': tokens,
          'kv_cache': {'reused_tokens': 2, 'evaluated_tokens': 8},
          'timings_ms': {'total': 1.5}})
"""
        )
        binary.chmod(0o755)
        return GemmaJev(workspace=tmp_path, timeout=3)

    return install


def test_lazy_load_context_manager_distribution_and_usage(fake_worker):
    engine = fake_worker()
    assert not engine.loaded and engine.last_usage == {}
    with engine as active:
        assert active is engine and engine.loaded
        assert engine.load() is engine
        result = engine.decide(REQUEST)
        assert list(result) == ["refund", "delivery"]
        assert result["refund"] == pytest.approx(1 / (1 + math.exp(-1)))
        assert engine.last_usage["prompt_tokens"] == 10
        assert engine.last_usage["cached_tokens"] == 2
        assert engine.last_usage["native_ms"] == 1.5
        assert "prompt" not in engine.last_usage
    assert not engine.loaded
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.decide(REQUEST)


def test_bad_caller_input_does_not_load_or_close_worker(fake_worker):
    engine = fake_worker()
    with pytest.raises(ValueError):
        engine.decide({})
    assert not engine.loaded
    engine.decide(REQUEST)
    with pytest.raises(ValueError):
        engine.decide({})
    assert engine.loaded
    engine.close()


def test_lock_blocks_second_model_and_releases_after_close(fake_worker, tmp_path):
    first = fake_worker()
    second = GemmaJev(model="e4b", workspace=tmp_path, timeout=3)
    try:
        first.load()
        with pytest.raises(RuntimeError, match="Another model"):
            second.load()
        assert not second.loaded
        first.close()
        second.load()
        assert second.loaded
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize(
    "mode",
    [
        "eof",
        "malformed",
        "nonobject",
        "native_error",
        "reordered",
        "not_softmax",
        "nonfinite",
        "bad_counts",
    ],
)
def test_failed_response_destroys_worker_and_cannot_desynchronize(fake_worker, mode):
    engine = fake_worker(mode)
    engine.load()
    process = engine._process
    with pytest.raises(RuntimeError):
        engine.decide(REQUEST)
    assert not engine.loaded and process.poll() is not None and engine.last_usage == {}
    with pytest.raises(RuntimeError, match="closed"):
        engine.decide(REQUEST)


def test_timeout_terminates_owned_child(fake_worker):
    engine = fake_worker("timeout")
    engine.load()
    engine.timeout = 0.05
    process = engine._process
    with pytest.raises(TimeoutError):
        engine.decide(REQUEST)
    assert process.poll() is not None and not engine.loaded


def test_failed_load_releases_lock(fake_worker, tmp_path):
    engine = fake_worker("startup_error")
    with pytest.raises(RuntimeError, match="load failed"):
        engine.load()
    assert not engine.loaded and engine._lock_file is None
    replacement = fake_worker()
    with replacement:
        assert replacement.loaded


def test_missing_assets_dont_start_process(fake_worker, monkeypatch, tmp_path):
    engine = fake_worker()
    monkeypatch.setattr(module, "runtime_path", lambda *_: tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="gemmajev setup"):
        engine.load()
    assert engine._process is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "unknown"},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"timeout": True},
        {"context_size": 256},
        {"context_size": 513},
        {"context_size": True},
    ],
)
def test_invalid_constructor(kwargs):
    with pytest.raises(ValueError):
        GemmaJev(**kwargs)


def test_same_size_corrupt_weights_fail_before_spawning_and_release_lock(
    fake_worker, tmp_path, monkeypatch
):
    engine = fake_worker()
    (tmp_path / "weights.gguf").write_bytes(b"bad!")
    spawned = []
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: spawned.append(args))
    with pytest.raises(ValueError, match="SHA-256"):
        engine.load()
    assert not spawned and not engine.loaded
    assert engine._process is None and engine._lock_file is None
    assert engine.last_usage == {}


def test_unpaired_surrogate_rejected_before_model_load(fake_worker):
    engine = fake_worker()
    request = {**REQUEST, "state": "invalid \ud800"}
    with pytest.raises(ValueError, match="UTF-8"):
        engine.decide(request)
    assert not engine.loaded and engine._process is None
    engine.close()
