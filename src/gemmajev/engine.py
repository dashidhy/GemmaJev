"""A resident, serial local scorer with a small semantic decision API."""

from __future__ import annotations

import fcntl
import json
import math
import queue
import subprocess
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Self

from .assets import check_platform, file_sha256, model_config, runtime_path, workspace_root
from .request import render_prompt


class GemmaJev:
    """Score options with E4B or 12B. Construction does not load the model.

    Use a context manager to release model memory promptly. Requests are serial
    and share a KV prefix cache; a workspace lock permits only one loaded worker.
    A closed instance cannot be reopened. Create another instance after closing
    or after a native failure, including a timeout.
    """

    def __init__(
        self,
        model: str = "12b",
        workspace: Path | None = None,
        timeout: float = 60,
        context_size: int = 4096,
    ) -> None:
        if model not in ("e4b", "12b"):
            raise ValueError("model must be e4b or 12b")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be a positive finite number")  # noqa: TRY004
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        if isinstance(context_size, bool) or not isinstance(context_size, int):
            raise ValueError("context_size must be a multiple of 256 from 512 to 4096")  # noqa: TRY004
        if not 512 <= context_size <= 4096 or context_size % 256:
            raise ValueError("context_size must be a multiple of 256 from 512 to 4096")
        self._model = model
        self.workspace = workspace_root(workspace)
        self.timeout = float(timeout)
        self.context_size = context_size
        self.last_usage: dict = {}
        self._mutex = threading.RLock()
        self._closed = False
        self._process: subprocess.Popen | None = None
        self._messages: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None
        self._lock_file = None
        self._stderr = None
        self._ready: dict = {}
        self._config: dict = {}
        self.log_path: Path | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def loaded(self) -> bool:
        return self._process is not None and self._process.poll() is None and not self._closed

    def load(self) -> GemmaJev:
        """Load once; check assets before acquiring the exclusive worker lock."""
        with self._mutex:
            self._require_open()
            if self.loaded:
                return self
            if self._process is not None:
                self.close()
                raise RuntimeError("Native worker exited; create a new GemmaJev instance")
            check_platform()
            config = model_config(self.model, self.workspace)
            binary = runtime_path(self.workspace)
            weights = Path(config["model_path"])
            if not binary.is_file() or not weights.is_file():
                raise FileNotFoundError(
                    f"Model or runtime missing. Run `uv run gemmajev setup --model {self.model}` "
                    f"in workspace {self.workspace}."
                )
            if weights.stat().st_size != config["model_size"]:
                raise ValueError(f"Model file has an unexpected size; run setup again: {weights}")
            runtime_dir = self.workspace / ".runtime"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            self._lock_file = (runtime_dir / "worker.lock").open("a+")
            try:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                self._lock_file.close()
                self._lock_file = None
                raise RuntimeError(
                    "Another model is loaded in this workspace; close it first"
                ) from error
            try:
                if file_sha256(weights) != config["model_sha256"]:
                    raise ValueError(
                        f"Model SHA-256 differs from the pinned checkpoint; run setup again: {weights}"
                    )
                logs = runtime_dir / "logs"
                logs.mkdir(parents=True, exist_ok=True)
                self.log_path = logs / f"{self.model}-{uuid.uuid4().hex}.log"
                self._stderr = self.log_path.open("w", encoding="utf-8")
                self._process = subprocess.Popen(
                    [
                        str(binary),
                        "--model",
                        str(weights),
                        "--model-size",
                        self.model,
                        "--ctx-size",
                        str(self.context_size),
                        "--threads",
                        "4",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=self._stderr,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                    # Retain the lock until this worker exits even if its Python
                    # parent dies while inference is still running.
                    pass_fds=(self._lock_file.fileno(),),
                )
                self._reader = threading.Thread(
                    target=self._read_output,
                    args=(self._process.stdout, self._messages),
                    daemon=True,
                )
                self._reader.start()
                ready = self._receive()
                if ready.get("ready") is not True or ready.get("model_size") != self.model:
                    raise RuntimeError("Native worker did not initialize the requested model")
                if ready.get("context_size") != self.context_size:
                    raise RuntimeError("Native worker context differs from the requested size")
                self._ready, self._config = ready, config
            except BaseException:
                self.close()
                raise
            return self

    def decide(self, request: Mapping[str, object]) -> dict[str, float]:
        """Return candidate-only softmax probabilities, in the caller's ID order."""
        with self._mutex:
            self._require_open()
            # Invalid caller input never loads a model or mutates its cache.
            prompt, labels = render_prompt(request)
            self.last_usage = {}
            self.load()
            started = time.perf_counter()
            try:
                assert self._process is not None and self._process.stdin is not None
                self._process.stdin.write(
                    json.dumps({"prompt": prompt, "labels": list(labels)}) + "\n"
                )
                self._process.stdin.flush()
                response = self._receive()
                probabilities = self._distribution(response, labels)
                cache, timings = response["kv_cache"], response["timings_ms"]
                self.last_usage = {
                    "model": self.model,
                    "model_sha256": self._config["model_sha256"],
                    "context_size": self.context_size,
                    "prompt_tokens": response["prompt_tokens"],
                    "cached_tokens": cache["reused_tokens"],
                    "evaluated_tokens": cache["evaluated_tokens"],
                    "native_ms": timings["total"],
                    "request_ms": (time.perf_counter() - started) * 1000,
                    "load_ms": self._ready["load_ms"],
                    "runtime_revision": self._ready["runtime_revision"],
                    "worker_source_sha256": self._ready["worker_source_sha256"],
                    "runtime_patches": self._ready["runtime_patches"],
                    "batch_size": self._ready["batch_size"],
                    "microbatch_size": self._ready["microbatch_size"],
                }
                return probabilities
            except BaseException:
                # A response arriving after a timeout must never be mistaken
                # for the next request's scores. Discard this worker entirely.
                self.last_usage = {}
                self.close()
                raise

    @staticmethod
    def _distribution(response: dict, labels: dict[str, str]) -> dict[str, float]:
        try:
            candidates = response["candidates"]
            if response.get("ok") is not True or [c["label"] for c in candidates] != list(labels):
                raise ValueError("incomplete or reordered candidates")
            logits = [c["logit"] for c in candidates]
            probabilities = [c["probability"] for c in candidates]
            if any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                for x in logits + probabilities
            ):
                raise ValueError("non-finite candidate scores")
            if any(not 0 <= p <= 1 for p in probabilities) or abs(sum(probabilities) - 1) > 1e-5:
                raise ValueError("candidate distribution is not normalized")
            maximum = max(logits)
            denominator = sum(math.exp(x - maximum) for x in logits)
            if any(
                abs(p - math.exp(x - maximum) / denominator) > 1e-5
                for p, x in zip(probabilities, logits)
            ):
                raise ValueError("probabilities disagree with candidate softmax")
            cache = response["kv_cache"]
            counts = [response["prompt_tokens"], cache["reused_tokens"], cache["evaluated_tokens"]]
            if any(type(n) is not int or n < 0 for n in counts) or counts[0] != sum(counts[1:]):
                raise ValueError("invalid token accounting")
            timing = response["timings_ms"]["total"]
            if not isinstance(timing, (float, int)) or not math.isfinite(timing) or timing < 0:
                raise ValueError("invalid runtime timing")
            return {option_id: float(p) for option_id, p in zip(labels.values(), probabilities)}
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Invalid native response: {error}") from error

    @staticmethod
    def _read_output(stream, messages: queue.Queue) -> None:
        try:
            for line in stream:
                messages.put(line)
        except (OSError, UnicodeError, ValueError) as error:
            messages.put(error)
        finally:
            messages.put(None)

    def _receive(self) -> dict:
        try:
            message = self._messages.get(timeout=self.timeout)
        except queue.Empty as error:
            raise TimeoutError(
                f"Native worker exceeded {self.timeout:g}s; see {self.log_path}"
            ) from error
        if message is None or isinstance(message, Exception):
            raise RuntimeError(
                f"Native worker exited or returned invalid text; see {self.log_path}"
            )
        try:
            result = json.loads(message)
        except (ValueError, TypeError) as error:
            raise RuntimeError("Native worker returned malformed JSON") from error
        if not isinstance(result, dict):
            raise RuntimeError("Native worker returned a non-object response")  # noqa: TRY004
        if result.get("ok") is False or result.get("ready") is False:
            raise RuntimeError(result.get("error", "Native worker failed"))
        return result

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("GemmaJev is closed; create a new instance")

    def close(self) -> None:
        """Release the model and lock. Safe to call repeatedly."""
        with self._mutex:
            self._closed = True
            process, self._process = self._process, None
            if process is not None:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                if self._reader is not None:
                    self._reader.join(timeout=1)
                if process.stdout is not None:
                    process.stdout.close()
            if self._stderr is not None:
                self._stderr.close()
                self._stderr = None
            if self._lock_file is not None:
                self._lock_file.close()
                self._lock_file = None

    def __enter__(self) -> Self:
        return self.load()

    def __exit__(self, *_args) -> None:
        self.close()
