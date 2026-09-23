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
    projector = tmp_path / "projector.gguf"
    projector.write_bytes(b"fake projector")
    binary = tmp_path / "worker"
    monkeypatch.setattr(module, "check_platform", dict)
    monkeypatch.setattr(module, "runtime_path", lambda *_, **kwargs: binary)
    monkeypatch.setattr(
        module,
        "model_config",
        lambda *_: {
            "model_path": str(weights),
            "model_size": 4,
            "model_sha256": hashlib.sha256(b"fake").hexdigest(),
            "mmproj_path": str(projector),
            "mmproj_size": len(b"fake projector"),
            "mmproj_sha256": hashlib.sha256(b"fake projector").hexdigest(),
        },
    )

    def install(mode="ok", **engine_kwargs):
        binary.write_text(
            f"#!{sys.executable}\n"
            + """
import json, math, sys, time
from pathlib import Path
MODE = """
            + repr(mode)
            + """
def emit(value):
    print(json.dumps(value), flush=True)
size = sys.argv[sys.argv.index('--model-size') + 1]
vision = '--mmproj' in sys.argv
budget = int(sys.argv[sys.argv.index('--image-tokens') + 1]) if vision else None
if MODE == 'startup_error':
    emit({'ready': False, 'error': 'load failed'})
    sys.exit(1)
emit({'ready': True, 'model_size': size, 'context_size': 4096, 'load_ms': 2,
      'runtime_revision': 'pinned', 'worker_source_sha256': 'a' * 64,
      'runtime_patches': [], 'batch_size': 512, 'microbatch_size': 128,
      'input_modality': 'image' if vision and MODE != 'wrong_vision_ready' else 'text',
      'image_token_budget': budget if MODE != 'wrong_image_budget' else 140})
for line in sys.stdin:
    request = json.loads(line)
    Path(__file__).with_suffix('.request.json').write_text(json.dumps(request))
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
    has_image = 'image_path' in request
    tokens = (74 if has_image else 10) if MODE != 'bad_counts' else 11
    reused = 0 if has_image else 2
    if MODE == 'image_prefix_cache' and has_image: reused = 8
    if MODE == 'image_cache_beyond_prefix' and has_image: reused = 9
    if request.get('reset_cache') and MODE != 'ignored_reset': reused = 0
    image_tokens = 64 if has_image and MODE != 'ignored_image' else 0
    if MODE == 'bad_image_tokens': image_tokens = -1
    result = {'ok': True, 'candidates': candidates, 'prompt_tokens': tokens,
          'input_modality': 'image' if has_image and MODE != 'wrong_modality' else 'text',
          'image_tokens': image_tokens,
          'image_position': 'after_text' if has_image else None,
          'cacheable_prefix_tokens': 8 if has_image else tokens,
          'kv_cache': {'enabled': True, 'reused_tokens': reused,
                       'evaluated_tokens': tokens - reused if MODE != 'bad_counts' else 8},
          'timings_ms': {'total': 1.5, 'prefill': 1.0,
                         'vision_encode': float('nan') if MODE == 'bad_vision_timing' else 0.2}}
    if MODE == 'missing_image_position': result.pop('image_position')
    if MODE == 'missing_cacheable_prefix': result.pop('cacheable_prefix_tokens')
    if MODE == 'negative_cacheable_prefix': result['cacheable_prefix_tokens'] = -1
    if MODE == 'bool_cacheable_prefix': result['cacheable_prefix_tokens'] = True
    if MODE == 'oversized_cacheable_prefix': result['cacheable_prefix_tokens'] = 10
    if MODE == 'image_before_text': result['image_position'] = 'before_text'
    if MODE == 'legacy_text':
        result.pop('image_position')
        result.pop('cacheable_prefix_tokens')
    emit(result)
"""
        )
        binary.chmod(0o755)
        return GemmaJev(workspace=tmp_path, timeout=3, **engine_kwargs)

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
        {"vision": "yes"},
        {"image_tokens": 0},
        {"image_tokens": 100},
        {"image_tokens": True},
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


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "image.ppm"
    path.write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
    return path


def test_vision_loads_separate_worker_and_scores_images_and_text(fake_worker, image_file, monkeypatch):
    selected = []
    original_path = module.runtime_path

    def runtime(workspace, **kwargs):
        selected.append(kwargs)
        return original_path(workspace, **kwargs)

    monkeypatch.setattr(module, "runtime_path", runtime)
    with fake_worker(vision=True, image_tokens=70) as engine:
        assert selected == [{"vision": True}]
        process = engine._process
        assert process.args[-4:] == [
            "--mmproj", str(image_file.parent / "projector.gguf"), "--image-tokens", "70"
        ]
        probabilities = engine.decide(REQUEST, image_path=image_file)
        assert list(probabilities) == list(REQUEST["options"])
        usage = engine.last_usage
        assert usage["input_modality"] == "image"
        assert usage["image_tokens"] == 64 and usage["cached_tokens"] == 0
        assert usage["kv_cache_enabled"] is True
        assert usage["image_token_budget"] == 70
        assert usage["image_position"] == "after_text"
        assert usage["cacheable_prefix_tokens"] == 8
        assert usage["image_sha256"] == hashlib.sha256(image_file.read_bytes()).hexdigest()
        assert usage["mmproj_sha256"] == hashlib.sha256(b"fake projector").hexdigest()
        assert usage["vision_encode_ms"] == 0.2 and usage["prefill_ms"] == 1.0
        payload = module.json.loads((image_file.parent / "worker.request.json").read_text())
        assert payload["image_path"] == str(image_file.resolve())

        engine.decide(REQUEST)
        assert engine._process is process
        assert engine.last_usage["input_modality"] == "text"
        assert engine.last_usage["image_tokens"] == 0
        assert engine.last_usage["image_sha256"] is None
        assert engine.last_usage["image_position"] is None
        assert engine.last_usage["cacheable_prefix_tokens"] == engine.last_usage["prompt_tokens"]
        payload = module.json.loads((image_file.parent / "worker.request.json").read_text())
        assert "image_path" not in payload


def test_image_on_text_engine_rejected_before_load(fake_worker, image_file):
    engine = fake_worker()
    with pytest.raises(ValueError, match="vision=True"):
        engine.decide(REQUEST, image_path=image_file)
    assert not engine.loaded and engine._process is None
    engine.close()


@pytest.mark.parametrize("kind", ["missing", "directory", "empty", "text", "nul", "surrogate"])
def test_invalid_image_rejected_without_loading(fake_worker, tmp_path, kind):
    engine = fake_worker(vision=True)
    path = tmp_path / "input"
    if kind == "directory":
        path.mkdir()
    elif kind == "empty":
        path.write_bytes(b"")
    elif kind == "text":
        path.write_text("not an image")
    elif kind == "nul":
        path = "invalid\0path.png"
    elif kind == "surrogate":
        path = "invalid\ud800.png"
    with pytest.raises(ValueError, match="image"):
        engine.decide(REQUEST, image_path=path)
    assert not engine.loaded and engine._process is None
    engine.close()


def test_oversized_image_rejected_without_loading(fake_worker, image_file, monkeypatch):
    engine = fake_worker(vision=True)
    monkeypatch.setattr(module, "MAX_IMAGE_BYTES", 4)
    with pytest.raises(ValueError, match="20 MiB"):
        engine.decide(REQUEST, image_path=image_file)
    assert not engine.loaded and engine._process is None
    engine.close()


def test_missing_or_corrupt_projector_does_not_spawn(fake_worker, tmp_path, monkeypatch):
    engine = fake_worker(vision=True)
    projector = tmp_path / "projector.gguf"
    projector.unlink()
    spawned = []
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: spawned.append(args))
    with pytest.raises(FileNotFoundError, match="setup.*--vision"):
        engine.load()
    assert not spawned and engine._lock_file is None
    projector.write_bytes(b"wrong size")
    with pytest.raises(ValueError, match="unexpected size"):
        engine.load()
    assert not spawned and engine._lock_file is None
    projector.write_bytes(b"bad projector!")
    with pytest.raises(ValueError, match="Projector SHA-256"):
        engine.load()
    assert not spawned and engine._lock_file is None


@pytest.mark.parametrize("mode", ["wrong_vision_ready", "wrong_image_budget"])
def test_vision_startup_mismatch_releases_worker(fake_worker, mode):
    engine = fake_worker(mode, vision=True)
    with pytest.raises(RuntimeError, match="vision configuration"):
        engine.load()
    assert not engine.loaded and engine._process is None and engine._lock_file is None


@pytest.mark.parametrize(
    "mode", [
        "wrong_modality", "image_cache_beyond_prefix", "bad_image_tokens", "ignored_image",
        "bad_vision_timing", "missing_image_position", "missing_cacheable_prefix",
        "negative_cacheable_prefix", "bool_cacheable_prefix", "oversized_cacheable_prefix",
        "image_before_text",
    ]
)
def test_vision_response_mismatch_destroys_worker(fake_worker, image_file, mode):
    engine = fake_worker(mode, vision=True)
    with engine:
        process = engine._process
        with pytest.raises(RuntimeError, match="Native worker returned"):
            engine.decide(REQUEST, image_path=image_file)
        assert process.poll() is not None and engine.last_usage == {}
        assert not engine.loaded and engine._lock_file is None


def test_image_text_prefix_cache_and_explicit_reset_preserve_distribution(fake_worker, image_file):
    with fake_worker("image_prefix_cache", vision=True) as engine:
        reused = engine.decide(REQUEST, image_path=image_file)
        assert engine.last_usage["cached_tokens"] == 8
        assert engine.last_usage["cached_tokens"] == engine.last_usage["cacheable_prefix_tokens"]
        payload_path = image_file.parent / "worker.request.json"
        assert "reset_cache" not in module.json.loads(payload_path.read_text())

        reset = engine.decide(REQUEST, image_path=image_file, reset_cache=True)
        assert reused == reset
        assert engine.last_usage["cached_tokens"] == 0
        assert module.json.loads(payload_path.read_text())["reset_cache"] is True

        engine.decide(REQUEST)
        assert engine.last_usage["input_modality"] == "text"
        assert engine.last_usage["image_tokens"] == 0
        assert engine.last_usage["image_position"] is None
        assert engine.last_usage["cacheable_prefix_tokens"] == 10
        payload = module.json.loads(payload_path.read_text())
        assert "image_path" not in payload and "reset_cache" not in payload


def test_text_cache_reset_and_legacy_response_metadata_remain_supported(fake_worker):
    with fake_worker("legacy_text") as engine:
        reused = engine.decide(REQUEST)
        assert engine.last_usage["cached_tokens"] == 2
        assert engine.last_usage["image_position"] is None
        assert engine.last_usage["cacheable_prefix_tokens"] == engine.last_usage["prompt_tokens"]
        assert engine.decide(REQUEST, reset_cache=True) == reused
        assert engine.last_usage["cached_tokens"] == 0


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_invalid_reset_flag_is_rejected_before_loading(fake_worker, value):
    engine = fake_worker()
    with pytest.raises(ValueError, match="reset_cache must be a boolean"):
        engine.decide(REQUEST, reset_cache=value)
    assert not engine.loaded and engine._process is None
    engine.close()


def test_native_ignored_reset_destroys_worker(fake_worker):
    with fake_worker("ignored_reset") as engine:
        process = engine._process
        with pytest.raises(RuntimeError, match="despite reset_cache"):
            engine.decide(REQUEST, reset_cache=True)
        assert process.poll() is not None and not engine.loaded
        assert engine.last_usage == {} and engine._lock_file is None
