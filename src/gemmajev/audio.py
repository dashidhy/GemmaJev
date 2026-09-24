"""Bounded, dependency-free validation of normalized audio attachments."""

from __future__ import annotations

import struct

MAX_AUDIO_BYTES = 20 * 1024 * 1024
MAX_AUDIO_SECONDS = 30
AUDIO_SAMPLE_RATE = 16000
MIN_AUDIO_FRAMES = 640  # One 40 ms audio token; shorter clips cannot be encoded safely.


def wav_info(data: bytes) -> dict:
    """Accept complete PCM16, mono, 16 kHz RIFF/WAVE files of at most 30 seconds."""
    if not 44 <= len(data) <= MAX_AUDIO_BYTES:
        raise ValueError("Provide a nonempty WAV file of at most 20 MiB.")
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Audio must be a PCM16 WAV file.")
    if struct.unpack_from("<I", data, 4)[0] + 8 != len(data):
        raise ValueError("WAV file size does not match its header.")
    position, format_found, frames = 12, False, None
    while position < len(data):
        if position + 8 > len(data):
            raise ValueError("Truncated WAV chunk header.")
        kind = data[position:position + 4]
        size = struct.unpack_from("<I", data, position + 4)[0]
        start, end = position + 8, position + 8 + size
        padded_end = end + size % 2
        if padded_end > len(data):
            raise ValueError("Truncated WAV chunk.")
        if kind == b"fmt ":
            if format_found or size < 16:
                raise ValueError("WAV format header is invalid.")
            encoding, channels, rate, byte_rate, alignment, bits = struct.unpack_from(
                "<HHIIHH", data, start
            )
            if (encoding, channels, rate, byte_rate, alignment, bits) != (
                1, 1, AUDIO_SAMPLE_RATE, AUDIO_SAMPLE_RATE * 2, 2, 16
            ):
                raise ValueError("Audio must be PCM16, mono, 16 kHz WAV.")
            format_found = True
        elif kind == b"data":
            if frames is not None or not size or size % 2:
                raise ValueError("WAV audio data is empty or invalid.")
            frames = size // 2
            if frames < MIN_AUDIO_FRAMES:
                raise ValueError("Audio must be at least 40 milliseconds long.")
            if frames > MAX_AUDIO_SECONDS * AUDIO_SAMPLE_RATE:
                raise ValueError("Audio must be no longer than 30 seconds.")
        position = padded_end
    if not format_found or frames is None:
        raise ValueError("WAV format or audio data is missing.")
    return {
        "duration_seconds": frames / AUDIO_SAMPLE_RATE,
        "sample_rate": AUDIO_SAMPLE_RATE,
        "channels": 1,
        "frames": frames,
    }
