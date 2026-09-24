"""Playground lifecycle and local HTTP behavior, without weights or GPU calls."""

import hashlib
import http.client
import io
import json
import socket
import struct
import subprocess
import threading
import wave
import webbrowser
import zlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from gemmajev.demo import (
    EXAMPLES,
    Playground,
    _engine_factory,
    create_server,
    load_example,
)


class FakeEngine:
    instances: ClassVar[list] = []

    def __init__(self, model, workspace):
        self.model = model
        self.loaded = False
        self.closed = False
        self.calls = 0
        self.load_calls = 0
        self.decisions = []
        self.audio_decisions = []
        self.instances.append(self)

    def load(self):
        self.load_calls += 1
        assert not any(item.loaded for item in self.instances if item is not self)
        assert not self.closed
        self.loaded = True

    def decide(self, request, *, image_path=None, audio_path=None):
        assert self.loaded and not self.closed
        assert image_path is None or audio_path is None
        self.calls += 1
        self.decisions.append((request, image_path))
        self.audio_decisions.append((request, audio_path))
        ids = list(request["options"])
        return {key: (0.8 if index == 0 else 0.2 / (len(ids) - 1)) for index, key in enumerate(ids)}

    def close(self):
        self.closed = True
        self.loaded = False


@pytest.fixture
def local_audio_samples(tmp_path):
    records = []
    for index in (1, 2):
        data = wav_audio(value=index * 100)
        path = tmp_path / f"private-source-language-{index}.wav"
        path.write_bytes(data)
        records.append({
            "id": f"sample_{index:02d}", "title": f"Local voice {index}", "path": path,
            "sha256": hashlib.sha256(data).hexdigest(), "duration_seconds": 0.05,
        })
    return records


@pytest.fixture(autouse=True)
def no_real_audio_synthesis(monkeypatch):
    from gemmajev import audio_samples, demo

    def unexpected(*args, **kwargs):
        raise AssertionError("Demo tests must never synthesize system voices")

    monkeypatch.setattr(audio_samples, "generate", unexpected)
    monkeypatch.setattr(demo, "prepare_samples", lambda workspace: [], raising=False)


@pytest.fixture
def manager(tmp_path):
    FakeEngine.instances = []
    value = Playground(workspace=tmp_path, engine_factory=FakeEngine)
    yield value
    value.close()


@pytest.fixture
def server_factory(manager):
    running = []

    def start(*, audio_samples=()):
        value = create_server(
            port=0, manager=manager, workspace=manager.workspace, audio_samples=audio_samples,
        )
        value.server_activate()
        thread = threading.Thread(target=value.serve_forever, kwargs={"poll_interval": 0.02})
        thread.start()
        running.append((value, thread))
        return value

    try:
        yield start
    finally:
        for value, thread in reversed(running):
            value.shutdown()
            value.server_close()
            thread.join(timeout=3)
            assert not thread.is_alive()


@pytest.fixture
def server(server_factory):
    return server_factory()


def exchange(server, method="GET", path="/api/config", *, payload=None, body=None, headers=None):
    """Use a real local socket, not a handler mock or third-party HTTP client."""
    request_headers = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    connection = http.client.HTTPConnection(*server.server_address, timeout=3)
    try:
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        content = response.read()
        response_headers = dict(response.getheaders())
        if method != "HEAD" and response.getheader("Content-Type", "").startswith("application/json"):
            content = json.loads(content)
        return response.status, response_headers, content
    finally:
        connection.close()


def decision_payload(model="12b"):
    return {"model": model, "request": load_example("customer_support")}


def png_image(width=2, height=2, *, color_type=6, fill=96, scanlines=None):
    """A complete PNG with valid compressed pixels and chunk CRCs; no image library."""
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]

    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))

    if scanlines is None:
        scanlines = (b"\0" + bytes([fill]) * width * channels) * height
    return b"\x89PNG\r\n\x1a\n" + b"".join([
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)),
        chunk(b"IDAT", zlib.compress(scanlines)),
        chunk(b"IEND", b""),
    ])


def upload(server, image, *, headers=None):
    return exchange(
        server, "POST", "/api/image", body=image,
        headers={"Content-Type": "image/png", **(headers or {})},
    )


def wav_audio(frames=800, *, value=1000, channels=1, rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(value.to_bytes(2, "little", signed=True) * frames * channels)
    return buffer.getvalue()


def upload_audio(server, data, *, headers=None):
    return exchange(
        server, "POST", "/api/audio", body=data,
        headers={"Content-Type": "audio/wav", **(headers or {})},
    )


def test_packaged_examples_have_semantic_ids_and_self_consistent_terms():
    for name in EXAMPLES:
        request = load_example(name)
        assert set(request) == {"task", "state", "query", "options"}
        assert len(request["options"]) == {"image_style": 4, "audio_language": 6}.get(name, 3)
        assert all(len(option_id) > 1 for option_id in request["options"])
    evidence = load_example("evidence_check")
    assert evidence["state"].startswith("Evidence:")
    assert evidence["query"].startswith("Claim:")
    assert "evidence" in evidence["task"] and "claim" in evidence["task"]
    image = load_example("image_style")
    assert "image" in image["task"].lower()
    assert "image" in image["state"].lower()
    assert "style" in image["query"].lower()
    audio = load_example("audio_language")
    assert "audio" in audio["task"].lower() and "audio" in audio["state"].lower()
    assert "language" in audio["query"].lower()
    assert list(audio["options"]) == ["english", "japanese", "spanish", "french", "other", "unclear"]


def test_demo_uses_smaller_context_without_changing_injected_factory(monkeypatch):
    import gemmajev.engine

    monkeypatch.setattr(gemmajev.engine, "GemmaJev", lambda **kwargs: kwargs)
    assert _engine_factory(model="12b", workspace=None) == {
        "model": "12b",
        "workspace": None,
        "context_size": 2048,
        "vision": True,
        "audio": True,
    }


def test_model_reuse_switch_and_failure_release(manager):
    request = load_example("customer_support")
    first = manager.run("12b", request)
    second = manager.run("12b", request)
    assert len(FakeEngine.instances) == 1
    assert FakeEngine.instances[0].calls == 2
    assert first["load_ms"] >= 0 and second["load_ms"] == 0
    manager.switch("e4b")
    assert FakeEngine.instances[0].closed
    assert len(FakeEngine.instances) == 2
    active = FakeEngine.instances[-1]
    assert active.loaded and active.load_calls == 1 and active.calls == 0
    manager.switch("e4b")
    result = manager.run("e4b", request)
    assert result["load_ms"] == 0 and active.load_calls == 1 and active.calls == 1
    assert len(FakeEngine.instances) == 2
    active.decide = lambda _: {"unrelated": 1.0}
    with pytest.raises(RuntimeError, match="invalid probability"):
        manager.run("e4b", request)
    assert active.closed
    manager.run("e4b", request)
    assert len(FakeEngine.instances) == 3


def test_preloading_is_idempotent_and_first_decision_reuses_the_engine(manager):
    assert manager.load("12b") is None
    manager.load("12b")
    assert len(FakeEngine.instances) == 1
    engine = FakeEngine.instances[0]
    assert engine.loaded and engine.load_calls == 1 and engine.calls == 0

    request = load_example("customer_support")
    result = manager.run("12b", request)
    assert result["load_ms"] == 0
    assert engine.load_calls == 1 and engine.calls == 1
    assert len(FakeEngine.instances) == 1
    request["state"] = "Changed input"
    request["options"]["delivery"] = "Changed meaning"
    assert result["request"] == load_example("customer_support")


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_failed_preload_releases_partial_engine_and_can_retry(manager, monkeypatch, error_type):
    original_load = FakeEngine.load

    def fail_load(engine):
        engine.loaded = True  # Simulate native resources allocated before failure.
        raise error_type("Loading interrupted")

    monkeypatch.setattr(FakeEngine, "load", fail_load)
    with pytest.raises(error_type, match="Loading interrupted"):
        manager.load("12b")
    assert len(FakeEngine.instances) == 1
    assert FakeEngine.instances[0].closed and not FakeEngine.instances[0].loaded

    monkeypatch.setattr(FakeEngine, "load", original_load)
    manager.load("12b")
    assert len(FakeEngine.instances) == 2
    assert FakeEngine.instances[1].loaded and FakeEngine.instances[1].calls == 0


def test_switch_waits_for_current_request_before_releasing(manager):
    request = load_example("customer_support")
    manager.run("12b", request)
    engine = FakeEngine.instances[0]
    entered = threading.Event()
    release = threading.Event()
    finished_switch = threading.Event()
    original = engine.decide

    def decide(value):
        entered.set()
        assert release.wait(timeout=3)
        assert not engine.closed
        return original(value)

    def switch():
        manager.switch("e4b")
        finished_switch.set()

    engine.decide = decide
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(manager.run, "12b", request)
        assert entered.wait(timeout=3)
        switching = pool.submit(switch)
        try:
            assert not finished_switch.wait(timeout=0.05)
        finally:
            release.set()
        running.result(timeout=3)
        switching.result(timeout=3)
    assert engine.closed and finished_switch.is_set()


def test_server_binds_before_loading_but_does_not_listen(manager):
    server = create_server(model="e4b", port=0, manager=manager)
    try:
        assert server.manager is manager and server.model == "e4b"
        assert server.server_address[0] == "127.0.0.1"
        assert server.server_address[1] > 0
        assert FakeEngine.instances == []
        with pytest.raises(OSError):
            socket.create_connection(server.server_address, timeout=0.2)
    finally:
        server.server_close()


def fake_launch_server(events, *, load_error=None):
    def load(model):
        events.append(("load", model))
        if load_error is not None:
            raise load_error("Loading interrupted")

    return SimpleNamespace(
        manager=SimpleNamespace(load=load, close=lambda: events.append("engine-close")),
        server_address=("127.0.0.1", 7862),
        server_port=7862,
        model="e4b",
        server_activate=lambda: events.append("listen"),
        serve_forever=lambda **kwargs: events.append("serve"),
        server_close=lambda: events.append("server-close"),
    )


@pytest.mark.parametrize("open_browser", [False, True])
def test_launch_loads_before_listening_and_opening_browser(monkeypatch, open_browser):
    from gemmajev import demo

    events = []
    server = fake_launch_server(events)

    def build_server(model, workspace, port, *, audio_samples=()):
        assert model == "e4b" and workspace == "/test/workspace" and port == 7862
        assert list(audio_samples) == []
        events.append("bind")
        return server

    monkeypatch.setattr(demo, "create_server", build_server)
    monkeypatch.setattr(webbrowser, "open", lambda url: events.append(("browser", url)))
    demo.launch(
        model="e4b", workspace="/test/workspace", port=7862, open_browser=open_browser,
    )
    expected = ["bind", ("load", "e4b"), "listen"]
    if open_browser:
        expected.append(("browser", "http://127.0.0.1:7862"))
    expected.extend(["serve", "server-close", "engine-close"])
    assert events == expected


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_launch_load_failure_never_listens_or_opens_browser(monkeypatch, error_type):
    from gemmajev import demo

    events = []
    server = fake_launch_server(events, load_error=error_type)
    monkeypatch.setattr(demo, "create_server", lambda *args, **kwargs: server)
    monkeypatch.setattr(webbrowser, "open", lambda url: events.append("browser"))
    expected_error = (
        nullcontext() if error_type is KeyboardInterrupt
        else pytest.raises(error_type, match="Loading interrupted")
    )
    with expected_error:
        demo.launch(model="12b")
    assert events == [("load", "12b"), "server-close", "engine-close"]


def test_config_exposes_examples_without_inference(server):
    status, _, config = exchange(server)
    assert status == 200 and config["model"] == "12b"
    assert config["examples"] == [
        {"id": name, "title": title, "request": load_example(name),
         "modality": {"image_style": "image", "audio_language": "audio"}.get(name, "text"),
         **({"image_id": "sample_1"} if name == "image_style" else {}),
         **({"audio_id": None} if name == "audio_language" else {})}
        for name, title in EXAMPLES.items()
    ]
    assert len(config["image_samples"]) == 4
    assert config["audio_samples"] == []
    for index, sample in enumerate(config["image_samples"], start=1):
        assert sample["id"] == f"sample_{index}" and sample["title"]
        assert sample["url"] == f"/api/images/sample_{index}"
    assert FakeEngine.instances == []


def test_static_files_are_packaged_and_arbitrary_paths_are_not_served(server):
    for path, content_type in [
        ("/", "text/html"), ("/app.js", "javascript"), ("/style.css", "text/css"),
    ]:
        status, headers, body = exchange(server, path=path)
        assert status == 200 and body
        assert content_type in headers["Content-Type"]
    for path in [
        "/README.md", "/../pyproject.toml", "/%2e%2e/pyproject.toml",
        "/.models/anything", "/api/not-a-route",
    ]:
        status, _, _ = exchange(server, path=path)
        assert status == 404
    assert FakeEngine.instances == []


def test_decision_uses_preloaded_model_preserves_option_order_and_switches(server, manager):
    manager.load("12b")
    payload = decision_payload()
    payload["request"]["options"] = dict(reversed(list(payload["request"]["options"].items())))
    status, _, result = exchange(server, "POST", "/api/decide", payload=payload)
    assert status == 200 and result["request"] == payload["request"]
    assert list(result["probabilities"]) == list(payload["request"]["options"])
    assert result["model"] == "12b" and result["load_ms"] == 0 and result["request_ms"] >= 0
    assert result["image_id"] is None
    assert result["audio_id"] is None
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    first = FakeEngine.instances[0]
    assert first.load_calls == 1 and first.calls == 1

    status, _, result = exchange(server, "POST", "/api/model", payload={"model": "e4b"})
    assert status == 200 and result == {"model": "e4b"}
    assert first.closed and len(FakeEngine.instances) == 2
    selected = FakeEngine.instances[-1]
    assert selected.loaded and selected.load_calls == 1 and selected.calls == 0
    assert exchange(server)[2]["model"] == "e4b"
    status, _, result = exchange(server, "POST", "/api/decide", payload=decision_payload("e4b"))
    assert status == 200 and result["model"] == "e4b" and result["load_ms"] == 0
    assert len(FakeEngine.instances) == 2 and selected.calls == 1 and selected.load_calls == 1


def test_invalid_requests_never_load_model_and_later_valid_request_succeeds(server):
    bad_request = {**load_example("customer_support"), "task": ""}
    cases = [
        ("/api/decide", {}),
        ("/api/decide", []),
        ("/api/decide", {"model": "unknown", "request": load_example("customer_support")}),
        ("/api/decide", {"model": "12b", "request": bad_request}),
        ("/api/decide", {"model": "12b", "request": {
            **load_example("customer_support"), "options": {"single": "Only option"},
        }}),
        ("/api/model", {"model": "unknown"}),
    ]
    for path, payload in cases:
        status, _, content = exchange(server, "POST", path, payload=payload)
        assert status == 400 and isinstance(content["error"], str) and content["error"]
    malformed_bodies = [
        b"not JSON",
        b'{"model":"e4b","model":"12b"}',
        b'{"model":"12b","request":{"options":{"same":"First","same":"Second"}}}',
        b"\xff",
    ]
    for body in malformed_bodies:
        status, _, content = exchange(
            server, "POST", "/api/decide", body=body,
            headers={"Content-Type": "application/json"},
        )
        assert status == 400 and content["error"]
    assert FakeEngine.instances == []
    assert exchange(server, "POST", "/api/decide", payload=decision_payload())[0] == 200


def test_non_json_and_oversized_requests_are_rejected_without_loading(server):
    body = json.dumps(decision_payload()).encode()
    for content_type in ["text/plain", "application/x-www-form-urlencoded"]:
        status, _, content = exchange(
            server, "POST", "/api/decide", body=body,
            headers={"Content-Type": content_type},
        )
        assert status == 415 and content["error"]
    status, _, content = exchange(
        server, "POST", "/api/decide", body=b" " * (256 * 1024 + 1),
        headers={"Content-Type": "application/json"},
    )
    assert status == 413 and content["error"]
    assert FakeEngine.instances == []


def test_foreign_hosts_and_cross_origin_requests_cannot_run_model(server):
    port = server.server_address[1]
    for headers in [
        {"Host": "attacker.invalid"},
        {"Origin": "https://attacker.invalid"},
        {"Host": f"127.0.0.1:{port}", "Origin": f"http://localhost:{port}"},
        {"Origin": "null"},
    ]:
        status, _, content = exchange(
            server, "POST", "/api/decide", payload=decision_payload(), headers=headers,
        )
        assert status == 403 and content["error"]
    assert exchange(server, headers={"Host": "attacker.invalid"})[0] == 403
    assert FakeEngine.instances == []
    for hostname in ["127.0.0.1", "localhost"]:
        headers = {"Host": f"{hostname}:{port}", "Origin": f"http://{hostname}:{port}"}
        assert exchange(server, "POST", "/api/decide", payload=decision_payload(),
                        headers=headers)[0] == 200


def test_runtime_failure_releases_engine_and_next_request_recovers(server, manager):
    manager.load("12b")
    failed = FakeEngine.instances[0]
    payload = decision_payload()
    payload["request"]["state"] = "long input " * 1000
    received = []

    def fail(request):
        received.append(request)
        raise RuntimeError("Native inference failed: input exceeds the context window")

    failed.decide = fail
    status, _, content = exchange(server, "POST", "/api/decide", payload=payload)
    assert status == 503 and "2,048" in content["error"]
    assert "nothing was truncated" in content["error"].lower()
    assert received == [payload["request"]]  # Never silently truncate input to make it fit.
    assert failed.closed
    assert exchange(server, "POST", "/api/decide", payload=decision_payload())[0] == 200
    assert len(FakeEngine.instances) == 2


def test_missing_model_is_reported_as_unavailable(server, monkeypatch):
    def missing(engine):
        raise FileNotFoundError("Model file is missing. Run gemmajev setup --model 12b.")

    monkeypatch.setattr(FakeEngine, "load", missing)
    status, _, content = exchange(server, "POST", "/api/decide", payload=decision_payload())
    assert status == 503 and "setup" in content["error"]
    assert FakeEngine.instances[0].closed


def test_concurrent_decision_and_switch_are_rejected_instead_of_queued(server, manager):
    manager.load("12b")
    engine = FakeEngine.instances[0]
    entered = threading.Event()
    release = threading.Event()
    original = engine.decide

    def blocking(request):
        entered.set()
        assert release.wait(timeout=5)
        assert not engine.closed
        return original(request)

    engine.decide = blocking
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(exchange, server, "POST", "/api/decide", payload=decision_payload())
        assert entered.wait(timeout=3)
        try:
            for path, payload in [
                ("/api/decide", decision_payload()), ("/api/model", {"model": "e4b"}),
            ]:
                status, _, content = exchange(server, "POST", path, payload=payload)
                assert status == 409 and content["error"]
            assert exchange(server)[0] == 200  # Static/config reads remain responsive.
            assert not engine.closed
        finally:
            release.set()
        assert running.result(timeout=3)[0] == 200
    assert len(FakeEngine.instances) == 1 and engine.calls == 1
    assert exchange(server, "POST", "/api/model", payload={"model": "e4b"})[0] == 200
    assert engine.closed


def test_missing_packaged_asset_returns_error_without_loading(server, monkeypatch, tmp_path):
    from gemmajev import demo

    monkeypatch.setattr(demo, "files", lambda _: tmp_path)
    status, _, content = exchange(server, path="/app.js")
    assert status == 503 and content["error"]
    assert FakeEngine.instances == []


def test_switch_load_failure_returns_error_releases_resources_and_can_retry(
    server, manager, monkeypatch,
):
    manager.load("12b")
    previous = FakeEngine.instances[0]
    original_load = FakeEngine.load

    def fail_load(engine):
        assert previous.closed
        engine.loaded = True  # Loading may allocate native resources before failing.
        raise RuntimeError("Model loading failed.")

    monkeypatch.setattr(FakeEngine, "load", fail_load)
    status, _, content = exchange(server, "POST", "/api/model", payload={"model": "e4b"})
    assert status == 503 and "loading failed" in content["error"]
    failed = FakeEngine.instances[-1]
    assert previous.closed and failed.closed and not failed.loaded
    assert all(engine.calls == 0 for engine in FakeEngine.instances)
    assert exchange(server)[2]["model"] == "12b"  # Only successful switches change config.

    monkeypatch.setattr(FakeEngine, "load", original_load)
    status, _, content = exchange(server, "POST", "/api/model", payload={"model": "e4b"})
    assert status == 200 and content == {"model": "e4b"}
    selected = FakeEngine.instances[-1]
    assert len(FakeEngine.instances) == 3 and selected.loaded
    assert selected.load_calls == 1 and selected.calls == 0
    status, _, result = exchange(server, "POST", "/api/decide", payload=decision_payload("e4b"))
    assert status == 200 and result["load_ms"] == 0
    assert selected.calls == 1 and selected.load_calls == 1


def test_switch_responds_only_after_load_and_rejects_other_model_operations(
    server, manager, monkeypatch,
):
    manager.load("12b")
    previous = FakeEngine.instances[0]
    entered = threading.Event()
    release = threading.Event()
    original_load = FakeEngine.load

    def blocking_load(engine):
        assert previous.closed  # Release the old model before allocating the replacement.
        entered.set()
        assert release.wait(timeout=5)
        original_load(engine)

    monkeypatch.setattr(FakeEngine, "load", blocking_load)
    with ThreadPoolExecutor(max_workers=1) as pool:
        switching = pool.submit(exchange, server, "POST", "/api/model", payload={"model": "e4b"})
        assert entered.wait(timeout=3)
        try:
            assert not switching.done()
            assert not FakeEngine.instances[-1].loaded
            for path, payload in [
                ("/api/decide", decision_payload()), ("/api/model", {"model": "12b"}),
            ]:
                status, _, content = exchange(server, "POST", path, payload=payload)
                assert status == 409 and content["error"]
            status, _, config = exchange(server)
            assert status == 200 and config["model"] == "12b"
            assert not switching.done()
        finally:
            release.set()
        status, _, content = switching.result(timeout=3)
        assert status == 200 and content == {"model": "e4b"}
    selected = FakeEngine.instances[-1]
    assert selected.loaded and selected.load_calls == 1 and selected.calls == 0
    assert exchange(server)[2]["model"] == "e4b"
    status, _, result = exchange(server, "POST", "/api/decide", payload=decision_payload("e4b"))
    assert status == 200 and result["load_ms"] == 0
    assert selected.calls == 1 and selected.load_calls == 1


def test_image_decision_reuses_loaded_engine_and_keeps_attachment_out_of_request(manager, tmp_path):
    request = load_example("image_style")
    attachment = tmp_path / "private-cat-photo.png"
    attachment.write_bytes(png_image())
    manager.load("12b")
    result = manager.run("12b", request, image_path=attachment)
    engine = FakeEngine.instances[0]
    assert engine.load_calls == 1 and engine.calls == 1
    assert engine.decisions == [(request, attachment)]
    assert result["request"] == request and result["load_ms"] == 0
    assert str(attachment) not in json.dumps(result)
    assert set(result) == {"request", "model", "probabilities", "request_ms", "load_ms"}
    manager.run("12b", request)
    assert engine.decisions[-1] == (request, None)


def test_packaged_images_are_served_without_running_inference(server):
    config = exchange(server)[2]
    for sample in config["image_samples"]:
        status, headers, body = exchange(server, path=sample["url"])
        assert status == 200 and headers["Content-Type"] == "image/png"
        assert body.startswith(b"\x89PNG\r\n\x1a\n")
    assert FakeEngine.instances == []


def test_sample_decision_uses_registered_path_without_leaking_image_identity(server):
    request = load_example("image_style")
    status, _, result = exchange(
        server, "POST", "/api/decide",
        payload={"model": "12b", "request": request, "image_id": "sample_1"},
    )
    assert status == 200 and result["image_id"] == "sample_1"
    passed_request, passed_image = FakeEngine.instances[0].decisions[-1]
    assert passed_request == request and result["request"] == request
    assert isinstance(passed_image, Path) and passed_image.is_file()
    assert "sample_1" not in json.dumps(passed_request)


@pytest.mark.parametrize("color_type", [0, 2, 4, 6])
def test_upload_roundtrip_is_deduplicated_and_does_not_run_model(server, color_type):
    image = png_image(color_type=color_type)
    digest = hashlib.sha256(image).hexdigest()
    status, _, uploaded = upload(server, image)
    assert status == 200
    assert uploaded == {
        "image_id": digest, "url": f"/api/images/{digest}", "width": 2, "height": 2,
    }
    assert upload(server, image)[2] == uploaded
    assert FakeEngine.instances == []
    status, headers, data = exchange(server, path=uploaded["url"])
    assert status == 200 and headers["Content-Type"] == "image/png" and data == image


def test_uploaded_image_can_be_scored_without_filename_or_path_in_prompt(server, manager):
    image = png_image()
    status, _, uploaded = upload(
        server, image,
        headers={"Content-Disposition": 'attachment; filename="secret-answer-is-cat.png"'},
    )
    assert status == 200 and FakeEngine.instances == []
    request = load_example("image_style")
    status, _, result = exchange(
        server, "POST", "/api/decide",
        payload={"model": "12b", "request": request, "image_id": uploaded["image_id"]},
    )
    assert status == 200 and result["image_id"] == uploaded["image_id"]
    passed_request, passed_image = FakeEngine.instances[0].decisions[-1]
    assert passed_request == request
    assert isinstance(passed_image, Path) and passed_image.read_bytes() == image
    assert passed_image.is_relative_to(manager.workspace / ".runtime")
    assert "secret-answer-is-cat" not in str(passed_image)
    assert str(passed_image) not in json.dumps(result)
    assert uploaded["image_id"] not in json.dumps(passed_request)


@pytest.mark.parametrize("image_id", [
    "not-registered", "a" * 64, "../README.md", "/etc/passwd", "sample_1/../../README.md",
    "%2e%2e%2fREADME.md", "https://example.test/image.png", 1, [], {},
])
def test_unknown_or_pathlike_image_ids_cannot_reach_model(server, image_id):
    status, _, content = exchange(
        server, "POST", "/api/decide",
        payload={**decision_payload(), "image_id": image_id},
    )
    assert status == 400 and content["error"]
    assert FakeEngine.instances == []


@pytest.mark.parametrize("path", [
    "/api/images/unknown", "/api/images/../README.md", "/api/images/%2e%2e%2fREADME.md",
    "/api/images/sample_1/anything", "/api/images/" + "a" * 64,
])
def test_unknown_image_routes_return_not_found(server, path):
    status, _, content = exchange(server, path=path)
    assert status == 404 and content["error"]
    assert FakeEngine.instances == []


@pytest.mark.parametrize("image", [
    b"not a PNG", b"\x89PNG\r\n\x1a\n", png_image()[:-12],
    png_image()[:32] + bytes([png_image()[32] ^ 1]) + png_image()[33:],
    png_image(scanlines=b"\0"), png_image(scanlines=b"\0" * 1000),
    png_image(width=0), png_image(width=1025), png_image(height=1025),
])
def test_malformed_and_oversized_dimensions_are_rejected_before_inference(server, image):
    status, _, content = upload(server, image)
    assert status in {400, 413} and content["error"]
    assert FakeEngine.instances == []


def test_upload_rejects_wrong_mime_and_oversized_body_without_loading(server, monkeypatch):
    from gemmajev import demo

    for mime in ["image/jpeg", "application/octet-stream", "application/json"]:
        status, _, content = upload(server, png_image(), headers={"Content-Type": mime})
        assert status == 415 and content["error"]
    monkeypatch.setattr(demo, "MAX_IMAGE_BYTES", 128)
    status, _, content = upload(server, b"x" * 129)
    assert status == 413 and content["error"]
    assert FakeEngine.instances == []


def test_upload_quota_counts_unique_images_and_preserves_existing_uploads(
    server_factory, monkeypatch,
):
    from gemmajev import demo

    first, second = png_image(fill=90), png_image(fill=91)
    monkeypatch.setattr(demo, "MAX_UPLOAD_BYTES", len(first) + len(second) - 1)
    server = server_factory()
    status, _, uploaded = upload(server, first)
    assert status == 200
    assert upload(server, first)[0] == 200  # Deduplication does not consume quota twice.
    status, _, content = upload(server, second)
    assert status == 413 and content["error"]
    assert exchange(server, path=uploaded["url"])[2] == first
    assert FakeEngine.instances == []


def test_images_preserve_local_host_and_origin_restrictions(server):
    for headers in [{"Host": "attacker.invalid"}, {"Origin": "https://attacker.invalid"}]:
        status, _, content = upload(server, png_image(), headers=headers)
        assert status == 403 and content["error"]
        assert exchange(server, path="/api/images/sample_1", headers=headers)[0] == 403
    assert FakeEngine.instances == []


@pytest.mark.parametrize("upload_first", [False, True])
def test_sample_and_upload_alias_never_rewrite_an_existing_image(tmp_path, monkeypatch, upload_first):
    from gemmajev import demo_images

    data = png_image()
    digest = hashlib.sha256(data).hexdigest()
    package = tmp_path / "package"
    sample = package / "examples/images/sample.png"
    sample.parent.mkdir(parents=True)
    sample.write_bytes(data)
    monkeypatch.setattr(demo_images, "files", lambda _: package)
    store = demo_images.ImageStore(tmp_path, samples={"sample_1": ("Photo", "sample.png")})
    try:
        if upload_first:
            store.add(data)
            original_path = store.resolve(digest)
        else:
            original_path = store.resolve("sample_1")
        write_bytes = Path.write_bytes

        def guarded_write(path, content):
            # A live model/preview can be reading this file while another alias
            # is registered. Any rewrite would briefly truncate that input.
            assert path != original_path, "Registered image bytes must remain immutable"
            return write_bytes(path, content)

        monkeypatch.setattr(Path, "write_bytes", guarded_write)
        if upload_first:
            assert store.resolve("sample_1") == original_path
        else:
            assert store.add(data)["image_id"] == digest
        assert store.resolve(digest) == store.resolve("sample_1") == original_path
        assert original_path.read_bytes() == data
    finally:
        store.close()


def test_uploaded_image_files_are_removed_when_server_closes(manager, tmp_path):
    value = create_server(port=0, manager=manager, workspace=tmp_path)
    value.server_activate()
    thread = threading.Thread(target=value.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    try:
        status, _, uploaded = upload(value, png_image())
        assert status == 200
        status, _, _ = exchange(
            value, "POST", "/api/decide",
            payload={**decision_payload(), "image_id": uploaded["image_id"]},
        )
        assert status == 200
        path = FakeEngine.instances[0].decisions[-1][1]
        assert path.is_file()
    finally:
        value.shutdown()
        value.server_close()
        thread.join(timeout=3)
    assert not thread.is_alive() and not path.exists()


def test_launch_prepares_audio_before_binding_and_loading(monkeypatch, local_audio_samples):
    from gemmajev import demo

    events = []
    server = fake_launch_server(events)

    def prepare(workspace):
        events.append("prepare-audio")
        return local_audio_samples

    def build(*args, audio_samples):
        events.append("bind")
        assert audio_samples == local_audio_samples
        return server

    monkeypatch.setattr(demo, "prepare_samples", prepare)
    monkeypatch.setattr(demo, "create_server", build)
    demo.launch(open_browser=False)
    assert events == ["prepare-audio", "bind", ("load", "12b"), "listen", "serve",
                      "server-close", "engine-close"]


@pytest.mark.parametrize("error", [
    RuntimeError("A system voice is unavailable"), subprocess.CalledProcessError(1, ["say"]),
    subprocess.TimeoutExpired(["say"], 60),
])
def test_launch_audio_generation_failure_preserves_upload_only_playground(monkeypatch, capsys, error):
    from gemmajev import demo

    events = []
    server = fake_launch_server(events)

    def unavailable(workspace):
        raise error

    def build(*args, audio_samples):
        assert audio_samples == []
        return server

    monkeypatch.setattr(demo, "prepare_samples", unavailable)
    monkeypatch.setattr(demo, "create_server", build)
    demo.launch(open_browser=False)
    assert ("load", "12b") in events and "listen" in events
    output = capsys.readouterr().out
    assert str(error) in output and "upload" in output.lower()


def test_audio_samples_config_and_playback_reveal_no_filesystem_paths(
    server_factory, local_audio_samples,
):
    server = server_factory(audio_samples=local_audio_samples)
    status, _, config = exchange(server)
    assert status == 200
    assert config["audio_samples"] == [
        {"id": row["id"], "title": row["title"], "url": f"/api/audio/{row['id']}",
         "duration_seconds": row["duration_seconds"]}
        for row in local_audio_samples
    ]
    audio_example = next(row for row in config["examples"] if row["id"] == "audio_language")
    assert audio_example["modality"] == "audio"
    assert audio_example["audio_id"] == local_audio_samples[0]["id"]
    for sample in local_audio_samples:
        status, headers, content = exchange(server, path=f"/api/audio/{sample['id']}")
        assert status == 200 and headers["Content-Type"].startswith("audio/wav")
        assert content == sample["path"].read_bytes()
        assert str(sample["path"]) not in json.dumps(config)
    assert FakeEngine.instances == []


def test_audio_manager_reuses_engine_without_adding_filename_to_request(manager, tmp_path):
    request = load_example("audio_language")
    path = tmp_path / "secret-spanish.wav"
    path.write_bytes(wav_audio())
    manager.load("12b")
    result = manager.run("12b", request, audio_path=path)
    engine = FakeEngine.instances[0]
    assert engine.audio_decisions == [(request, path)]
    assert engine.calls == 1 and engine.load_calls == 1 and result["load_ms"] == 0
    assert result["request"] == request and str(path) not in json.dumps(result)
    manager.run("12b", request)
    assert engine.audio_decisions[-1] == (request, None) and engine.load_calls == 1


def test_manager_rejects_combined_attachments_before_loading(manager, tmp_path):
    with pytest.raises(ValueError):
        manager.run("12b", load_example("audio_language"),
                    audio_path=tmp_path / "audio.wav", image_path=tmp_path / "image.png")
    assert FakeEngine.instances == []


def test_audio_upload_roundtrip_deduplication_and_metadata_without_inference(server):
    data = wav_audio()
    digest = hashlib.sha256(data).hexdigest()
    status, _, uploaded = upload_audio(server, data)
    assert status == 200 and uploaded == {
        "audio_id": digest, "url": f"/api/audio/{digest}",
        "duration_seconds": 0.05, "sample_rate": 16000, "channels": 1,
    }
    assert upload_audio(server, data)[2] == uploaded
    status, headers, content = exchange(server, path=uploaded["url"])
    assert status == 200 and content == data
    assert headers["Content-Type"].startswith("audio/wav")
    assert FakeEngine.instances == []


def test_audio_decision_uses_registered_path_and_preserves_only_public_request(
    server_factory, local_audio_samples,
):
    server = server_factory(audio_samples=local_audio_samples)
    request = load_example("audio_language")
    sample = local_audio_samples[0]
    status, _, result = exchange(server, "POST", "/api/decide", payload={
        "model": "12b", "request": request, "audio_id": sample["id"],
    })
    assert status == 200 and result["audio_id"] == sample["id"] and result["image_id"] is None
    received, path = FakeEngine.instances[0].audio_decisions[-1]
    assert received == request and result["request"] == request
    assert isinstance(path, Path) and path.read_bytes() == sample["path"].read_bytes()
    assert path != sample["path"]  # Session copy freezes the bytes seen by the engine.
    assert sample["id"] not in json.dumps(received)
    assert str(sample["path"]) not in json.dumps(result) and str(path) not in json.dumps(result)


def test_uploaded_audio_has_no_original_filename_and_no_image_carryover(server):
    data = wav_audio()
    status, _, uploaded = upload_audio(server, data, headers={
        "Content-Disposition": 'attachment; filename="answer-is-mandarin.wav"',
    })
    assert status == 200
    request = load_example("audio_language")
    status, _, result = exchange(server, "POST", "/api/decide", payload={
        "model": "12b", "request": request, "audio_id": uploaded["audio_id"],
    })
    assert status == 200 and result["image_id"] is None
    engine = FakeEngine.instances[0]
    received, path = engine.audio_decisions[-1]
    assert received == request and isinstance(path, Path) and path.read_bytes() == data
    assert "answer-is-mandarin" not in str(path) and path.stem == uploaded["audio_id"]
    assert engine.decisions[-1] == (request, None)
    status, _, result = exchange(server, "POST", "/api/decide", payload=decision_payload())
    assert status == 200 and result["audio_id"] is None and result["image_id"] is None
    assert engine.audio_decisions[-1][1] is None and engine.calls == 2


@pytest.mark.parametrize("audio_id", ["unknown", "a" * 64, "../secret.wav", "/etc/passwd", 1, [], {}])
def test_invalid_audio_ids_never_reach_the_engine(server, audio_id):
    status, _, content = exchange(server, "POST", "/api/decide", payload={
        **decision_payload(), "audio_id": audio_id,
    })
    assert status == 400 and content["error"]
    assert FakeEngine.instances == []


def test_http_rejects_simultaneous_audio_and_image_attachments(server):
    audio_id = upload_audio(server, wav_audio())[2]["audio_id"]
    status, _, content = exchange(server, "POST", "/api/decide", payload={
        **decision_payload(), "audio_id": audio_id, "image_id": "sample_1",
    })
    assert status == 400 and content["error"] and FakeEngine.instances == []


@pytest.mark.parametrize("path", ["/api/audio/unknown", "/api/audio/../secret.wav",
                                  "/api/audio/%2e%2e%2fsecret.wav", "/api/audio/" + "a" * 64])
def test_unknown_audio_routes_are_not_served(server, path):
    assert exchange(server, path=path)[0] == 404
    assert FakeEngine.instances == []


@pytest.mark.parametrize("data", [
    b"not WAV", wav_audio()[:-1], wav_audio(frames=1), wav_audio(frames=639),
    wav_audio(channels=2), wav_audio(rate=44100), wav_audio(frames=30 * 16000 + 1),
])
def test_malformed_noncanonical_or_out_of_range_audio_rejected_before_inference(server, data):
    status, _, content = upload_audio(server, data)
    assert status in {400, 413} and content["error"] and FakeEngine.instances == []


def test_audio_wrong_mime_or_oversized_body_rejected_without_loading(server, monkeypatch):
    from gemmajev import demo

    for mime in ("audio/mpeg", "application/json", "text/plain"):
        status, _, content = upload_audio(server, wav_audio(), headers={"Content-Type": mime})
        assert status == 415 and content["error"]
    monkeypatch.setattr(demo, "MAX_AUDIO_BYTES", 128)
    status, _, content = upload_audio(server, b"x" * 129)
    assert status == 413 and content["error"] and FakeEngine.instances == []


def test_audio_quota_deduplication_preserves_existing_samples(server_factory, monkeypatch):
    from gemmajev import demo

    first, second = wav_audio(value=100), wav_audio(value=200)
    original = demo.AudioStore

    def limited(*args, **kwargs):
        kwargs["max_bytes"] = len(first) + len(second) - 1
        return original(*args, **kwargs)

    monkeypatch.setattr(demo, "AudioStore", limited)
    server = server_factory()
    status, _, uploaded = upload_audio(server, first)
    assert status == 200 and upload_audio(server, first)[0] == 200
    status, _, content = upload_audio(server, second)
    assert status == 413 and content["error"]
    assert exchange(server, path=uploaded["url"])[2] == first and FakeEngine.instances == []


def test_audio_obeys_local_host_origin_and_fetch_site_restrictions(server):
    for headers in [
        {"Host": "attacker.invalid"}, {"Origin": "https://attacker.invalid"},
        {"Sec-Fetch-Site": "cross-site"},
    ]:
        assert upload_audio(server, wav_audio(), headers=headers)[0] == 403
        assert exchange(server, path="/api/audio/sample_01", headers=headers)[0] == 403
    assert FakeEngine.instances == []


@pytest.mark.parametrize("upload_first", [False, True])
def test_audio_sample_and_upload_aliases_never_rewrite_registered_bytes(
    tmp_path, local_audio_samples, monkeypatch, upload_first,
):
    from gemmajev.demo_audio import AudioStore

    sample = local_audio_samples[0]
    data = sample["path"].read_bytes()
    store = AudioStore(tmp_path, samples=local_audio_samples)
    try:
        if upload_first:
            store.add(data)
            path = store.resolve(sample["sha256"])
        else:
            path = store.resolve(sample["id"])
        write_bytes = Path.write_bytes

        def protected(target, content):
            assert target != path, "Registered audio bytes must remain immutable"
            return write_bytes(target, content)

        monkeypatch.setattr(Path, "write_bytes", protected)
        assert store.add(data)["audio_id"] == sample["sha256"]
        assert store.resolve(sample["sha256"]) == store.resolve(sample["id"]) == path
        assert path.read_bytes() == data
    finally:
        store.close()


def test_audio_files_cleaned_after_server_close_but_source_samples_preserved(
    manager, tmp_path, local_audio_samples,
):
    value = create_server(
        port=0, manager=manager, workspace=tmp_path, audio_samples=local_audio_samples,
    )
    value.server_activate()
    thread = threading.Thread(target=value.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    try:
        status, _, uploaded = upload_audio(value, wav_audio())
        assert status == 200
        status, _, _ = exchange(value, "POST", "/api/decide", payload={
            **decision_payload(), "audio_id": uploaded["audio_id"],
        })
        assert status == 200
        cached_audio = FakeEngine.instances[0].audio_decisions[-1][1]
        assert cached_audio.is_file()
    finally:
        value.shutdown()
        value.server_close()
        thread.join(timeout=3)
    assert not thread.is_alive() and not cached_audio.exists()
    assert all(sample["path"].is_file() for sample in local_audio_samples)


def test_server_waits_for_audio_inference_before_deleting_attachment(manager, tmp_path):
    value = create_server(port=0, manager=manager, workspace=tmp_path)
    value.server_activate()
    thread = threading.Thread(target=value.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    manager.load("12b")
    engine = FakeEngine.instances[0]
    original = engine.decide
    attachment = []

    def blocking(request, *, audio_path):
        attachment.append(audio_path)
        assert audio_path.is_file()
        entered.set()
        assert release.wait(timeout=5)
        assert audio_path.is_file(), "Server cleanup must wait until inference finishes"
        return original(request, audio_path=audio_path)

    def close_server():
        value.server_close()
        closed.set()

    engine.decide = blocking
    try:
        audio_id = upload_audio(value, wav_audio())[2]["audio_id"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            running = pool.submit(exchange, value, "POST", "/api/decide", payload={
                **decision_payload(), "audio_id": audio_id,
            })
            try:
                assert entered.wait(timeout=3)
                value.shutdown()
                closing = pool.submit(close_server)
                assert not closed.wait(timeout=0.05)
                assert attachment[0].is_file()
            finally:
                release.set()
            assert running.result(timeout=3)[0] == 200
            closing.result(timeout=3)
        assert closed.is_set() and not attachment[0].exists()
    finally:
        release.set()
        value.shutdown()
        value.server_close()
        thread.join(timeout=3)
    assert not thread.is_alive()


@pytest.fixture(params=["sample", "upload"])
def audio_resource(request, server_factory, local_audio_samples):
    server = server_factory(audio_samples=local_audio_samples)
    if request.param == "sample":
        sample = local_audio_samples[0]
        return server, f"/api/audio/{sample['id']}", sample["path"].read_bytes()
    data = wav_audio(value=777)
    status, _, uploaded = upload_audio(server, data)
    assert status == 200
    return server, uploaded["url"], data


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_full_audio_response_advertises_byte_ranges(audio_resource, method):
    server, url, data = audio_resource
    status, headers, content = exchange(server, method, url)
    assert status == 200 and headers["Accept-Ranges"] == "bytes"
    assert headers["Content-Length"] == str(len(data))
    assert headers["Content-Type"].startswith("audio/wav")
    assert "Content-Range" not in headers
    assert content == (data if method == "GET" else b"")
    assert FakeEngine.instances == []


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_audio_byte_ranges_are_inclusive_and_clamped(audio_resource, method):
    server, url, data = audio_resource
    total = len(data)
    for range_header, start, end in [
        ("bytes=0-0", 0, 0), ("bytes=4-15", 4, 15),
        ("bytes=44-", 44, total - 1), ("bytes=-19", total - 19, total - 1),
        ("bytes=7-999999", 7, total - 1), ("bytes=-999999", 0, total - 1),
        ("bytes=0-", 0, total - 1),
    ]:
        status, headers, content = exchange(server, method, url, headers={"Range": range_header})
        assert status == 206, range_header
        assert headers["Accept-Ranges"] == "bytes"
        assert headers["Content-Range"] == f"bytes {start}-{end}/{total}"
        assert headers["Content-Length"] == str(end - start + 1)
        assert headers["Content-Type"].startswith("audio/wav")
        assert content == (data[start:end + 1] if method == "GET" else b"")
    assert FakeEngine.instances == []


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_invalid_or_multiple_audio_ranges_report_unsatisfied_total(audio_resource, method):
    server, url, data = audio_resource
    for range_header in [
        f"bytes={len(data)}-", "bytes=999999-", "bytes=15-4", "bytes=-0", "bytes=-",
        "items=0-10", "bytes=abc-def", "bytes=0-1,4-5", "bytes=--1", "bytes=0",
    ]:
        status, headers, content = exchange(server, method, url, headers={"Range": range_header})
        assert status == 416, range_header
        assert headers["Content-Range"] == f"bytes */{len(data)}"
        if method == "HEAD":
            assert content == b""
    assert FakeEngine.instances == []


def test_audio_range_requests_preserve_origin_restrictions(audio_resource):
    server, url, _ = audio_resource
    status, _, _ = exchange(server, path=url, headers={
        "Range": "bytes=0-15", "Origin": "https://attacker.invalid",
    })
    assert status == 403 and FakeEngine.instances == []
