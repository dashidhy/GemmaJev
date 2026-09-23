"""Session-local PNG storage for the Playground; no image-library dependency."""

from __future__ import annotations

import hashlib
import struct
import tempfile
import threading
import zlib
from importlib.resources import files
from pathlib import Path

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_IMAGE_DIMENSION = 1024
IMAGE_SAMPLES = {
    "sample_1": ("Photo", "sample_1.png"),
    "sample_2": ("Artwork", "sample_2.png"),
    "sample_3": ("Anime", "sample_3.png"),
    "sample_4": ("Graphic", "sample_4.png"),
}


def png_dimensions(data: bytes, *, max_dimension=MAX_IMAGE_DIMENSION) -> tuple[int, int]:
    """Validate normalized PNG framing and bound its decompressed allocation."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) < 45:
        raise ValueError("Upload a valid PNG image.")
    position, width, height, channels = 8, 0, 0, 0
    compressed = bytearray()
    ended = False
    while position < len(data):
        if position + 12 > len(data):
            raise ValueError("Truncated PNG image.")
        length = struct.unpack_from(">I", data, position)[0]
        end = position + 12 + length
        if end > len(data):
            raise ValueError("Truncated PNG image.")
        kind = data[position + 4 : position + 8]
        payload = data[position + 8 : end - 4]
        crc = struct.unpack_from(">I", data, end - 4)[0]
        if zlib.crc32(kind + payload) != crc:
            raise ValueError("PNG image checksum is invalid.")
        if position == 8:
            if kind != b"IHDR" or length != 13:
                raise ValueError("PNG image header is invalid.")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if not 1 <= width <= max_dimension or not 1 <= height <= max_dimension:
                raise ValueError(f"Image dimensions must be between 1 and {max_dimension} pixels.")
            channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color, 0)
            if depth != 8 or not channels or compression or filtering or interlace:
                raise ValueError("Use the image picker to convert this image to a supported PNG.")
        elif kind == b"IHDR":
            raise ValueError("PNG image has multiple headers.")
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            if length or end != len(data):
                raise ValueError("PNG image ending is invalid.")
            ended = True
            break
        position = end
    if not ended or not compressed:
        raise ValueError("PNG image data is incomplete.")
    expected = (1 + width * channels) * height
    try:
        decoder = zlib.decompressobj()
        pixels = decoder.decompress(compressed, expected + 1)
        if len(pixels) != expected or not decoder.eof or decoder.unused_data:
            raise ValueError("PNG image data has an invalid size.")
    except zlib.error as error:
        raise ValueError("PNG image data is invalid.") from error
    if any(pixels[y * (1 + width * channels)] > 4 for y in range(height)):
        raise ValueError("PNG image filter is invalid.")
    return width, height


class ImageStore:
    def __init__(self, workspace: Path, *, samples=IMAGE_SAMPLES, max_bytes=MAX_UPLOAD_BYTES):
        runtime = workspace / ".runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        self._directory = tempfile.TemporaryDirectory(prefix="playground-images-", dir=runtime)
        self._root = Path(self._directory.name)
        self._samples = samples
        self._paths = {}
        self._upload_bytes = 0
        self._max_bytes = max_bytes
        self._lock = threading.Lock()

    def add(self, data: bytes) -> dict:
        width, height = png_dimensions(data)
        image_id = hashlib.sha256(data).hexdigest()
        with self._lock:
            if image_id not in self._paths:
                if self._upload_bytes + len(data) > self._max_bytes:
                    raise OverflowError("Image storage is full. Restart the Playground to clear it.")
                path = self._root / f"{image_id}.png"
                path.write_bytes(data)
                self._paths[image_id] = path
                self._upload_bytes += len(data)
        return {"image_id": image_id, "url": f"/api/images/{image_id}", "width": width, "height": height}

    def resolve(self, image_id: str) -> Path:
        if not isinstance(image_id, str):
            raise ValueError("Choose a sample image or upload an image first.")  # noqa: TRY004
        with self._lock:
            if image_id in self._paths:
                return self._paths[image_id]
            if image_id not in self._samples:
                raise ValueError("Unknown image. Choose a sample or upload it again.")
            name = self._samples[image_id][1]
            data = files("gemmajev").joinpath("examples", "images", name).read_bytes()
            # Bundled samples are trusted project assets, not arbitrary upload paths.
            png_dimensions(data, max_dimension=4096)
            digest = hashlib.sha256(data).hexdigest()
            path = self._paths.get(digest)
            if path is None:
                path = self._root / f"{digest}.png"
                path.write_bytes(data)
                self._paths[digest] = path
            # Share one immutable content file across sample and upload IDs.
            # Readers use it outside this lock, so an alias must not rewrite it.
            self._paths[image_id] = path
            return path

    def close(self):
        self._directory.cleanup()
