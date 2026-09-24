"""Normalized WAV validation without audio libraries, models or GPU calls."""

import io
import struct
import wave

import pytest

import gemmajev.audio as module
from gemmajev.audio import wav_info


def chunk(kind, data):
    return kind + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b"")


def riff(*chunks):
    payload = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(payload)) + payload


def fmt(encoding=1, channels=1, rate=16000, byte_rate=32000, alignment=2, bits=16):
    return chunk(b"fmt ", struct.pack("<HHIIHH", encoding, channels, rate, byte_rate, alignment, bits))


def wav(frames=640):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x00\x10" * frames)
    return buffer.getvalue()


@pytest.mark.parametrize("frames", [640, 16000, 30 * 16000])
def test_pcm16_wav_returns_exact_duration_including_thirty_second_boundary(frames):
    assert wav_info(wav(frames)) == {
        "duration_seconds": frames / 16000, "sample_rate": 16000, "channels": 1, "frames": frames,
    }


def test_metadata_chunks_and_odd_padding_do_not_count_as_audio_frames():
    source = riff(
        chunk(b"JUNK", b"x"), fmt(), chunk(b"LIST", b"INFO"),
        chunk(b"data", b"\x01\x00" * 800), chunk(b"JUNK", b"extra"),
    )
    assert wav_info(source)["duration_seconds"] == 0.05
    assert wav_info(source)["frames"] == 800


@pytest.mark.parametrize("source", [
    b"", b"not audio" * 10, b"RIFX" + wav()[4:], b"RF64" + wav()[4:],
    wav()[:8] + b"AVI " + wav()[12:],
    wav()[:-1], wav() + b"trailing",  # RIFF size must match the actual file.
    riff(fmt(), chunk(b"JUNK", b"metadata"), b"data"),  # Incomplete chunk header.
    riff(fmt(), b"data" + struct.pack("<I", 100) + b"\x00\x00"),
    riff(fmt(), b"JUNK" + struct.pack("<I", 1) + b"x"),  # Missing odd-byte padding.
    riff(fmt(), chunk(b"data", b"")), riff(fmt(), chunk(b"data", b"\x01")),
    riff(fmt(), fmt(), chunk(b"data", b"\0\0")),
    riff(fmt(), chunk(b"data", b"\0\0" * 640), chunk(b"data", b"\0\0" * 640)),
    riff(chunk(b"fmt ", b"x" * 15), chunk(b"data", b"\0\0")),
    riff(fmt(), chunk(b"JUNK", b"no audio data")),
    riff(chunk(b"data", b"\0\0" * 640)),
])
def test_malformed_truncated_or_ambiguous_wav_is_rejected(source):
    with pytest.raises(ValueError):
        wav_info(source)


@pytest.mark.parametrize("format_chunk", [
    fmt(encoding=3), fmt(encoding=65534), fmt(channels=2), fmt(rate=44100),
    fmt(byte_rate=16000), fmt(alignment=4), fmt(bits=8), fmt(bits=24), fmt(bits=32),
])
def test_only_pcm16_mono_sixteen_khz_encoding_is_accepted(format_chunk):
    with pytest.raises(ValueError, match="PCM16, mono, 16 kHz"):
        wav_info(riff(format_chunk, chunk(b"data", b"\0\0" * 20)))


def test_audio_over_thirty_seconds_is_rejected_without_truncation():
    with pytest.raises(ValueError, match="no longer than 30 seconds"):
        wav_info(wav(30 * 16000 + 1))


@pytest.mark.parametrize("frames", [1, 639])
def test_subtoken_audio_is_rejected_before_native_preprocessing(frames):
    with pytest.raises(ValueError, match="at least 40 milliseconds"):
        wav_info(wav(frames))


def test_total_bytes_limit_includes_metadata_chunks(monkeypatch):
    source = riff(fmt(), chunk(b"JUNK", b"x" * 128), chunk(b"data", b"\0\0" * 640))
    monkeypatch.setattr(module, "MAX_AUDIO_BYTES", len(source) - 1)
    with pytest.raises(ValueError, match="20 MiB"):
        wav_info(source)
