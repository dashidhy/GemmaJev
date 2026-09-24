"""Create local, varied language-identification probes with macOS system voices.

No model weights, online service, or third-party Python package is used. Speech
files are local test material; this script does not grant redistribution rights
to Apple's generated voice output. Outputs default to the ignored .runtime/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from importlib.resources import files
from pathlib import Path

from .assets import workspace_root
from .audio import MAX_AUDIO_BYTES, wav_info

ROOT = workspace_root()
SAMPLE_RATE = 16_000
PROMPT = files("gemmajev").joinpath("examples", "audio_language.json")

# The genders below describe the chosen synthetic voice presets, not people
# identified from audio. Each language has unrelated, independently authored text.
SPEECH = {
    "english": {
        "male": ["Eddy (English (US))", "Reed (English (UK))"],
        "female": ["Samantha", "Karen"],
        "utterances": [
            "The library closes early today, so I will return these books after lunch.",
            "Please leave the spare key under the small flowerpot beside the front door.",
            "A blue bicycle was parked near the bakery when we arrived this morning.",
            "I found a recipe for vegetable soup and bought fresh carrots at the market.",
        ],
    },
    "mandarin": {
        "male": ["Eddy (Chinese (China mainland))", "Reed (Chinese (China mainland))"],
        "female": ["Tingting", "Meijia"],
        "utterances": [
            "阳台上的番茄已经熟了，周末我们可以摘下来做一盘凉菜。",
            "下午的会议推迟了半小时，你可以先去楼下买一杯热茶。",
            "这把雨伞好像不是我的，我记得自己的伞柄是木头做的。",
            "沿着河边走到第二座桥，左手边就能看见新开的花店。",
        ],
    },
    "japanese": {
        "male": ["Eddy (Japanese (Japan))", "Reed (Japanese (Japan))"],
        "female": ["Kyoko", "Flo (Japanese (Japan))"],
        "utterances": [
            "昨日、公園で小さな白い犬を見かけました。とても元気に走っていました。",
            "新しい靴は少し大きいので、厚い靴下を履いて出かけます。",
            "週末は山に登る予定ですが、天気が悪ければ家で映画を見ます。",
            "窓を開けると涼しい風が入ってきました。もう秋になったようです。",
        ],
    },
    "spanish": {
        "male": ["Eddy (Spanish (Spain))", "Reed (Spanish (Mexico))"],
        "female": ["Mónica", "Paulina"],
        "utterances": [
            "Mi hermana prepara una tarta de manzana para la cena del viernes.",
            "La estación está detrás del museo, pero hoy la entrada principal está cerrada.",
            "Necesito cambiar las cortinas porque entra demasiada luz por la mañana.",
            "El vecino dejó una caja de herramientas junto a la puerta del garaje.",
        ],
    },
    "french": {
        "male": ["Thomas", "Jacques"],
        "female": ["Amélie", "Flo (French (France))"],
        "utterances": [
            "Le jardin est couvert de feuilles et nous allons les ramasser demain matin.",
            "J'ai oublié mon carnet dans le train, mais le contrôleur l'a retrouvé.",
            "Cette petite lampe éclaire très bien le coin du salon près du fauteuil.",
        ],
    },
    "german": {
        "male": ["Eddy (German (Germany))", "Reed (German (Germany))"],
        "female": ["Anna", "Flo (German (Germany))"],
        "utterances": [
            "Der kleine Laden an der Ecke verkauft frisches Brot und selbstgemachte Marmelade.",
            "Ich habe die Fahrkarten bereits gekauft und warte jetzt am Bahnsteig.",
            "Auf dem Küchentisch liegt ein Brief, den ich heute noch beantworten möchte.",
        ],
    },
}


def plan_samples(seed: int) -> list[dict]:
    """Select balanced voices and distinct utterances before shuffling neutral IDs."""
    rng = random.Random(seed)
    records = []
    for language in ("english", "mandarin", "japanese", "spanish"):
        texts = rng.sample(SPEECH[language]["utterances"], 2)
        for gender, text in zip(("male", "female"), texts):
            records.append({
                "kind": "speech", "language": language,
                "expected_option": "other" if language == "mandarin" else language,
                "voice_gender": gender, "voice": rng.choice(SPEECH[language][gender]),
                "transcript": text, "rate_wpm": rng.choice((155, 170, 185)),
            })
    other_genders = ["male", "female"]
    rng.shuffle(other_genders)
    for language, gender in zip(("french", "german"), other_genders):
        records.append({
            "kind": "speech", "language": language,
            "expected_option": "french" if language == "french" else "other",
            "voice_gender": gender, "voice": rng.choice(SPEECH[language][gender]),
            "transcript": rng.choice(SPEECH[language]["utterances"]),
            "rate_wpm": rng.choice((155, 170, 185)),
        })
    # Retain the original clips and add a distinct complementary voice/text.
    for language, first_gender in zip(("french", "german"), other_genders):
        gender = "female" if first_gender == "male" else "male"
        used = {row["transcript"] for row in records if row["language"] == language}
        texts = [text for text in SPEECH[language]["utterances"] if text not in used]
        records.append({
            "kind": "speech", "language": language,
            "expected_option": "french" if language == "french" else "other",
            "voice_gender": gender, "voice": rng.choice(SPEECH[language][gender]),
            "transcript": rng.choice(texts), "rate_wpm": rng.choice((155, 170, 185)),
        })
    records.extend([
        {"kind": "silence", "language": None, "expected_option": "unclear",
         "voice_gender": None, "voice": None, "transcript": None, "rate_wpm": None,
         "control_seconds": 2.0},
        {"kind": "non_speech_tones", "language": None, "expected_option": "unclear",
         "voice_gender": None, "voice": None, "transcript": None, "rate_wpm": None,
         "control_seconds": 2.4, "frequencies_hz": rng.sample((330, 440, 550, 660, 880), 2)},
    ])
    rng.shuffle(records)
    return [{"id": f"sample_{index:02d}", **record} for index, record in enumerate(records, 1)]


def installed_voices() -> set[str]:
    output = subprocess.check_output(["say", "-v", "?"], text=True)
    names = set()
    for line in output.splitlines():
        match = re.match(r"^(.*?)\s+[a-z]{2,3}_[A-Z0-9]{2,3}\s+#", line)
        if match:
            names.add(match.group(1).strip())
    return names


def write_control(path: Path, record: dict) -> None:
    frames = bytearray()
    count = round(record["control_seconds"] * SAMPLE_RATE)
    for index in range(count):
        sample = 0
        if record["kind"] == "non_speech_tones":
            seconds = index / SAMPLE_RATE
            envelope = min(1, seconds / 0.03, (count - index - 1) / SAMPLE_RATE / 0.03)
            signal = sum(math.sin(2 * math.pi * f * seconds) for f in record["frequencies_hz"])
            sample = round(0.12 * 32767 * envelope * signal)
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        output.writeframes(frames)


def inspect_wav(path: Path) -> dict:
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, SAMPLE_RATE):
            raise ValueError(f"Expected mono 16 kHz PCM16 audio: {path}")
        duration = source.getnframes() / SAMPLE_RATE
        if not 0 < duration <= 30:
            raise ValueError(f"Expected an audio clip of at most 30 seconds: {path}")
        samples = struct.iter_unpack("<h", source.readframes(source.getnframes()))
        peak = max(abs(value[0]) for value in samples)
    return {
        "duration_seconds": duration, "sample_rate": SAMPLE_RATE, "channels": 1,
        "sample_width_bytes": 2, "peak_pcm16": peak, "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def generate(output: Path, seed: int) -> dict:
    if platform.system() != "Darwin" or shutil.which("say") is None:
        raise RuntimeError("This local sample generator requires macOS and its say command.")
    records = plan_samples(seed)
    missing = sorted({r["voice"] for r in records if r["voice"]} - installed_voices())
    if missing:
        raise RuntimeError("These macOS voices are not installed: " + ", ".join(missing))
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty; choose a new --output path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    prompt_bytes = PROMPT.read_bytes()
    # This separate JSON is the only textual model input; manifest metadata is not.
    (output / "request.json").write_bytes(prompt_bytes)
    manifest = {
        "version": 1, "seed": seed, "purpose": "Offline language-identification feasibility check",
        "generator": "macOS say for speech; Python standard library for controls",
        "macos_version": platform.mac_ver()[0], "python_version": platform.python_version(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "request_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "request_file": "request.json", "randomization": "seeded voices, utterances, rates and order",
        "redistribution": "Local test material; Apple voice-output redistribution is not asserted.",
        "complete": False, "samples": [],
    }
    manifest_path = output / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    save()
    for record in records:
        path = output / f"{record['id']}.wav"
        if record["kind"] == "speech":
            subprocess.run([
                "say", "-v", record["voice"], "-r", str(record["rate_wpm"]),
                "--file-format=WAVE", "--data-format=LEI16@16000", "--channels=1",
                "-o", str(path), "-f", "-",
            ], input=record["transcript"], text=True, check=True, timeout=60)
        else:
            write_control(path, record)
        metadata = inspect_wav(path)
        if record["kind"] == "speech" and metadata["peak_pcm16"] == 0:
            raise RuntimeError(f"The selected voice produced silent audio: {record['voice']}")
        manifest["samples"].append({**record, "file": path.name, **metadata})
        save()
        print(f"{record['id']}: {metadata['duration_seconds']:.2f} s", flush=True)
    manifest["complete"] = True
    save()
    return manifest


def ensure_samples(workspace=None, *, seed=0) -> tuple[Path, dict]:
    """Generate once locally, then verify immutable cached samples before reuse."""
    runtime = workspace_root(workspace) / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(
        Path(__file__).read_bytes() + PROMPT.read_bytes() + platform.mac_ver()[0].encode()
    ).hexdigest()[:16]
    output = runtime / f"audio-samples-{fingerprint}-{seed}"
    if not output.exists():
        staging = Path(tempfile.mkdtemp(prefix="audio-samples-pending-", dir=runtime))
        try:
            generate(staging, seed)
            try:
                staging.rename(output)
            except OSError:
                if not output.exists():
                    raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    manifest = json.loads((output / "manifest.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("complete") is not True or manifest.get("seed") != seed:
        raise ValueError("Local audio samples are incomplete. Remove their cache and generate again.")
    prompt_bytes = PROMPT.read_bytes()
    request_path = (output / "request.json").resolve()
    if (
        not request_path.is_relative_to(output.resolve())
        or request_path.stat().st_size != len(prompt_bytes)
        or request_path.read_bytes() != prompt_bytes
        or manifest.get("request_sha256") != hashlib.sha256(prompt_bytes).hexdigest()
    ):
        raise ValueError("Local audio sample request differs. Remove its cache and generate again.")
    records = manifest.get("samples", [])
    required = {"id", "file", "sha256", "kind", "language", "voice_gender", "duration_seconds"}
    if (
        not isinstance(records, list) or len(records) != len(plan_samples(seed))
        or any(not isinstance(row, dict) or not required <= row.keys() for row in records)
        or any(not isinstance(row["id"], str) or not isinstance(row["file"], str) for row in records)
        or len({row["id"] for row in records}) != len(records)
    ):
        raise ValueError("Local audio sample manifest is invalid.")
    for row in records:
        name = row["file"]
        if not re.fullmatch(r"sample_\d{2}\.wav", name) or row["id"] != Path(name).stem:
            raise ValueError("Local audio sample name is invalid.")
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or not 0 < path.stat().st_size <= MAX_AUDIO_BYTES:
            raise ValueError("Local audio sample file is invalid.")
        data = path.read_bytes()
        wav_info(data)
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Local audio sample hash differs. Remove its cache and generate again.")
    return output, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/audio-language-check")
    args = parser.parse_args()
    try:
        output = args.output.expanduser().resolve()
        manifest = generate(output, args.seed)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Audio sample generation failed: {error}", file=sys.stderr)
        return 1
    total = sum(row["duration_seconds"] for row in manifest["samples"])
    print(f"Saved {len(manifest['samples'])} clips ({total:.2f} s) and manifest to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
