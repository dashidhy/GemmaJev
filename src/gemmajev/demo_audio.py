"""Local WAV attachment storage and generated demo sample registration."""

from __future__ import annotations

import hashlib
import tempfile
import threading
from pathlib import Path

from .audio import MAX_AUDIO_BYTES, wav_info
from .audio_samples import ensure_samples


def prepare_samples(workspace) -> list[dict]:
    directory, manifest = ensure_samples(workspace)
    names = {"english": "English", "mandarin": "Mandarin", "japanese": "Japanese",
             "spanish": "Spanish", "french": "French", "german": "German"}
    result = []
    for row in manifest["samples"]:
        if row["kind"] == "speech":
            title = f"{names[row['language']]} · {row['voice_gender']} voice"
        else:
            title = "Silence" if row["kind"] == "silence" else "Non-speech tones"
        result.append({"id": row["id"], "title": title, "path": directory / row["file"],
                       "sha256": row["sha256"], "duration_seconds": row["duration_seconds"]})
    # Keep a recognizable English clip first; all variants remain available.
    return sorted(result, key=lambda row: (not row["title"].startswith("English"), row["id"]))


class AudioStore:
    def __init__(self, workspace: Path, *, samples=(), max_bytes=64 * 1024 * 1024):
        runtime = workspace / ".runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        self._directory = tempfile.TemporaryDirectory(prefix="playground-audio-", dir=runtime)
        self._root = Path(self._directory.name)
        self._samples = {row["id"]: row for row in samples}
        self._paths = {}
        self._bytes = 0
        self._max_bytes = max_bytes
        self._lock = threading.Lock()

    def _save(self, data: bytes, digest: str) -> Path:
        # Called under the store lock; registered content is never rewritten.
        if digest not in self._paths:
            if self._bytes + len(data) > self._max_bytes:
                raise OverflowError("Audio storage is full. Restart the Playground to clear it.")
            path = self._root / f"{digest}.wav"
            path.write_bytes(data)
            self._paths[digest] = path
            self._bytes += len(data)
        return self._paths[digest]

    def add(self, data: bytes) -> dict:
        info = wav_info(data)
        digest = hashlib.sha256(data).hexdigest()
        with self._lock:
            self._save(data, digest)
        return {"audio_id": digest, "url": f"/api/audio/{digest}",
                **{key: info[key] for key in ("duration_seconds", "sample_rate", "channels")}}

    def resolve(self, audio_id: str) -> Path:
        if not isinstance(audio_id, str):
            raise ValueError("Choose a sample or upload a WAV file first.")  # noqa: TRY004
        with self._lock:
            if audio_id in self._paths:
                return self._paths[audio_id]
            if audio_id not in self._samples:
                raise ValueError("Unknown audio. Choose a sample or upload it again.")
            sample = self._samples[audio_id]
            with Path(sample["path"]).open("rb") as source:
                data = source.read(MAX_AUDIO_BYTES + 1)
            wav_info(data)
            digest = hashlib.sha256(data).hexdigest()
            if digest != sample["sha256"]:
                raise ValueError("Audio sample changed. Generate the local samples again.")
            path = self._save(data, digest)
            self._paths[audio_id] = path
            return path

    def close(self):
        self._directory.cleanup()
