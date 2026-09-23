"""A local Playground served with Python's standard library."""

from __future__ import annotations

import json
import math
import threading
import time
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import urlsplit

from .assets import workspace_root
from .demo_images import IMAGE_SAMPLES, MAX_IMAGE_BYTES, MAX_UPLOAD_BYTES, ImageStore
from .request import validate_request

EXAMPLES = {
    "customer_support": "Customer support",
    "evidence_check": "Evidence check",
    "answer_usefulness": "Answer usefulness",
    "image_style": "Image style",
}


def load_example(name: str) -> dict:
    if name not in EXAMPLES:
        raise ValueError("Unknown example")
    source = files("gemmajev").joinpath("examples", f"{name}.json")
    return validate_request(json.loads(source.read_text(encoding="utf-8")))


def _engine_factory(**kwargs):
    from .engine import GemmaJev

    return GemmaJev(context_size=2048, vision=True, **kwargs)


class Playground:
    """Serialize model ownership; never keep two loaded models in memory."""

    def __init__(self, workspace=None, engine_factory: Callable = _engine_factory):
        self.workspace = workspace
        self.engine_factory = engine_factory
        self._engine = None
        self._lock = threading.Lock()

    def _select(self, model: str):
        if not isinstance(model, str) or model not in {"e4b", "12b"}:
            raise ValueError("Choose E4B or 12B.")
        if self._engine is not None and self._engine.model != model:
            self._engine.close()
            self._engine = None

    def switch(self, model: str):
        """Release the old model and fully load the selected model before returning."""
        with self._lock:
            self._load(model)

    def _load(self, model: str) -> float:
        """Load under the caller's lock, returning time spent loading in milliseconds."""
        self._select(model)
        if self._engine is None:
            self._engine = self.engine_factory(model=model, workspace=self.workspace)
        if self._engine.loaded:
            return 0.0
        start = time.perf_counter()
        try:
            self._engine.load()
        except BaseException:
            self._engine.close()
            self._engine = None
            raise
        return (time.perf_counter() - start) * 1000

    def load(self, model: str):
        """Prepare the engine without evaluating an example or populating a result."""
        with self._lock:
            self._load(model)

    def run(self, model: str, request: dict, *, image_path=None) -> dict:
        request = validate_request(request)
        with self._lock:
            load_ms = self._load(model)
            try:
                start = time.perf_counter()
                probabilities = (
                    self._engine.decide(request, image_path=image_path)
                    if image_path is not None else self._engine.decide(request)
                )
                request_ms = (time.perf_counter() - start) * 1000
                if (
                    set(probabilities) != set(request["options"])
                    or any(
                        not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
                        for p in probabilities.values()
                    )
                    or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5)
                ):
                    raise RuntimeError("The model returned an invalid probability distribution.")
            except Exception:
                self._engine.close()
                self._engine = None
                raise
        return {
            "request": request,
            "model": model,
            "probabilities": dict(probabilities),
            "request_ms": request_ms,
            "load_ms": load_ms,
        }

    def close(self):
        with self._lock:
            if self._engine is not None:
                self._engine.close()
                self._engine = None



MAX_BODY_BYTES = 256 * 1024
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


class PlaygroundServer(ThreadingHTTPServer):
    """One resident engine shared by local browser tabs; never queue inference."""

    daemon_threads = False
    block_on_close = True

    def __init__(self, model, workspace, port, manager):
        self.model = model
        self.manager = manager if manager is not None else Playground(workspace)
        self.operation = threading.Lock()
        self.images = ImageStore(workspace_root(workspace), samples=IMAGE_SAMPLES, max_bytes=MAX_UPLOAD_BYTES)
        super().__init__(("127.0.0.1", port), PlaygroundHandler, bind_and_activate=False)
        try:
            self.server_bind()
        except BaseException:
            self.server_close()
            raise

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(10)
        return connection, address

    def server_close(self):
        try:
            super().server_close()
        finally:
            self.images.close()


class PlaygroundHandler(BaseHTTPRequestHandler):
    server_version = "GemmaJev"
    sys_version = ""

    def log_message(self, format, *args):
        # Keep the terminal focused on engine startup and errors, not every asset.
        pass

    def _send(self, status, body, content_type="application/json; charset=utf-8"):
        if isinstance(body, dict):
            body = json.dumps(body, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'none'",
        )
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Closing a tab does not kill or unload the resident engine.

    def _error(self, status, message):
        self._send(status, {"error": message})

    def _local_request(self):
        host = self.headers.get("Host", "").lower()
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if host not in allowed:
            self._error(403, "Use the local Playground address printed in the terminal.")
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin != f"http://{host}":
            self._error(403, "Cross-origin requests are not allowed.")
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self._error(403, "Cross-site requests are not allowed.")
            return False
        return True

    def do_GET(self):
        if not self._local_request():
            return
        path = urlsplit(self.path).path
        if path.startswith("/api/images/"):
            try:
                content = self.server.images.resolve(path.removeprefix("/api/images/")).read_bytes()
            except ValueError as error:
                self._error(404, str(error))
                return
            except OSError:
                self._error(503, "Sample image is unavailable. Reinstall GemmaJev.")
                return
            self._send(200, content, "image/png")
        elif path in STATIC_FILES:
            name, content_type = STATIC_FILES[path]
            try:
                content = files("gemmajev").joinpath("playground", name).read_bytes()
            except OSError:
                self._error(503, "Playground assets are missing or unreadable. Reinstall GemmaJev.")
                return
            self._send(200, content, content_type)
        elif path == "/api/config":
            self._send(200, {
                "model": self.server.model,
                "examples": [
                    {"id": name, "title": title, "request": load_example(name),
                     **({"image_id": "sample_1"} if name == "image_style" else {})}
                    for name, title in EXAMPLES.items()
                ],
                "image_samples": [
                    {"id": name, "title": value[0], "url": f"/api/images/{name}"}
                    for name, value in IMAGE_SAMPLES.items()
                ],
            })
        else:
            self._error(404, "Not found.")

    def do_HEAD(self):
        self.do_GET()

    def _upload_image(self):
        if self.headers.get_content_type() != "image/png":
            self._error(415, "Upload PNG image bytes using the image picker.")
            return
        if "Transfer-Encoding" in self.headers:
            self._error(400, "Chunked request bodies are not supported.")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._error(411, "A valid Content-Length is required.")
            return
        if not 0 < length <= MAX_IMAGE_BYTES:
            self._error(413, "Image exceeds the 20 MiB upload limit.")
            return
        try:
            data = self.rfile.read(length)
            if len(data) != length:
                raise ValueError("Incomplete image upload.")
            self._send(200, self.server.images.add(data))
        except ValueError as error:
            self._error(400, str(error))
        except OverflowError as error:
            self._error(413, str(error))
        except (TimeoutError, ConnectionError):
            self._error(408, "Image upload timed out or was interrupted.")
        except OSError:
            self._error(503, "Could not save the uploaded image.")

    def do_POST(self):
        if not self._local_request():
            return
        path = urlsplit(self.path).path
        if path == "/api/image":
            self._upload_image()
            return
        if path not in {"/api/decide", "/api/model"}:
            self._error(404, "Not found.")
            return
        if self.headers.get_content_type() != "application/json":
            self._error(415, "Send an application/json request.")
            return
        if "Transfer-Encoding" in self.headers:
            self._error(400, "Chunked request bodies are not supported.")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._error(411, "A valid Content-Length is required.")
            return
        if length < 0 or length > MAX_BODY_BYTES:
            self._error(413, "Request exceeds the 256 KiB limit.")
            return
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete request body.")
            body = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
            if not isinstance(body, dict):
                raise ValueError("Request must be a JSON object.")  # noqa: TRY004
            model = body.get("model")
            if not isinstance(model, str) or model not in {"e4b", "12b"}:
                raise ValueError("Choose E4B or 12B.")
            request = validate_request(body.get("request")) if path == "/api/decide" else None
            image_id = body.get("image_id") if path == "/api/decide" else None
            image_path = self.server.images.resolve(image_id) if image_id is not None else None
        except (ValueError, UnicodeError) as error:
            self._error(400, str(error))
            return
        except (TimeoutError, ConnectionError):
            self._error(408, "Request body timed out or was interrupted.")
            return
        except OSError:
            self._error(503, "The selected image is unavailable. Choose or upload it again.")
            return
        if not self.server.operation.acquire(blocking=False):
            self._error(409, "The engine is busy. Wait for the current operation to finish.")
            return
        try:
            if path == "/api/model":
                self.server.manager.switch(model)
                self.server.model = model
                self._send(200, {"model": model})
            else:
                result = (
                    self.server.manager.run(model, request, image_path=image_path)
                    if image_path is not None else self.server.manager.run(model, request)
                )
                result["image_id"] = image_id
                self.server.model = model
                self._send(200, result)
        except (ValueError, RuntimeError, OSError, TimeoutError) as error:
            message = str(error)
            if "input exceeds the context window" in message:
                message = "Input exceeds 2,048 tokens. Shorten the text; nothing was truncated."
            self._error(503, message)
        finally:
            self.server.operation.release()


def create_server(model="12b", workspace=None, port=7860, *, manager=None):
    """Bind a loopback port without loading a model or accepting requests yet."""
    if not isinstance(model, str) or model not in {"e4b", "12b"}:
        raise ValueError("Choose E4B or 12B.")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("Port must be an integer from 0 to 65535.")
    return PlaygroundServer(model, workspace, port, manager)


def launch(model="12b", workspace=None, port=7860, open_browser=True):
    server = create_server(model, workspace, port)
    try:
        print(f"Loading Gemma 4 {model.upper()} inference engine...", flush=True)
        started = time.perf_counter()
        server.manager.load(model)
        server.server_activate()
        address = f"http://127.0.0.1:{server.server_port}"
        print(
            f"Engine ready in {time.perf_counter() - started:.1f} s. Playground: {address}",
            flush=True,
        )
        print("Press Ctrl-C to stop and release the model.", flush=True)
        if open_browser:
            webbrowser.open(address)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping Playground...", flush=True)
    finally:
        server.server_close()
        server.manager.close()
