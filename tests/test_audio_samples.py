"""Deterministic corpus planning and immutable cache checks, without system TTS."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import wave
from collections import Counter
from pathlib import Path

import pytest

from gemmajev import audio_samples as module
from gemmajev.audio import wav_info


def write_pcm(path, value=1000):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(value.to_bytes(2, "little", signed=True) * 800)


@pytest.fixture
def fake_generation(monkeypatch):
    calls = []

    def generate(output, seed):
        calls.append((output, seed))
        output.mkdir(parents=True, exist_ok=True)
        request = module.PROMPT.read_bytes()
        (output / "request.json").write_bytes(request)
        records = []
        for index, record in enumerate(module.plan_samples(seed)):
            path = output / f"{record['id']}.wav"
            write_pcm(path, index + 1)
            records.append({**record, "file": path.name, **module.inspect_wav(path)})
        manifest = {
            "complete": True, "seed": seed, "request_file": "request.json",
            "request_sha256": hashlib.sha256(request).hexdigest(), "samples": records,
        }
        (output / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    monkeypatch.setattr(module, "generate", generate)
    monkeypatch.setattr(module.platform, "mac_ver", lambda: ("test-version", ("", "", ""), ""))
    return calls


def rewrite_manifest(folder, update):
    path = folder / "manifest.json"
    value = json.loads(path.read_text())
    update(value)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("seed", [0, 1, 37])
def test_plans_have_expected_labels_balanced_genders_distinct_content_and_neutral_ids(seed):
    plan = module.plan_samples(seed)
    assert len(plan) == 14
    assert Counter(row["expected_option"] for row in plan) == {
        "english": 2, "japanese": 2, "spanish": 2, "french": 2, "other": 4, "unclear": 2,
    }
    assert [row["id"] for row in plan] == [f"sample_{index:02d}" for index in range(1, 15)]
    speech = [row for row in plan if row["kind"] == "speech"]
    assert len({row["transcript"] for row in speech}) == 12
    assert Counter(row["voice_gender"] for row in speech) == {"male": 6, "female": 6}
    for language in ("english", "mandarin", "japanese", "spanish", "french", "german"):
        group = [row for row in speech if row["language"] == language]
        assert {row["voice_gender"] for row in group} == {"male", "female"}
        assert len({row["voice"] for row in group}) == 2
        assert len({row["transcript"] for row in group}) == 2
        assert all(row["transcript"] in module.SPEECH[language]["utterances"] for row in group)
        assert all(
            row["expected_option"] == ("other" if language in {"mandarin", "german"} else language)
            for row in group
        )
    assert {row["language"] for row in speech if row["expected_option"] == "other"} == {
        "mandarin", "german",
    }


def test_seed_is_reproducible_but_changes_voices_utterances_rates_and_order():
    first = module.plan_samples(0)
    assert first == module.plan_samples(0)
    second = module.plan_samples(1)
    assert [row["language"] for row in first] != [row["language"] for row in second]
    first_speech = {(row["language"], row["voice_gender"]): row for row in first if row["voice"]}
    second_speech = {(row["language"], row["voice_gender"]): row for row in second if row["voice"]}
    common = first_speech.keys() & second_speech.keys()
    for attribute in ("voice", "transcript", "rate_wpm"):
        assert any(first_speech[key][attribute] != second_speech[key][attribute] for key in common)


def test_real_generator_uses_selected_voices_and_keeps_answers_only_in_manifest(
    tmp_path, monkeypatch,
):
    calls = []
    available = {
        voice for choices in module.SPEECH.values()
        for gender in ("male", "female") for voice in choices[gender]
    }
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(module.shutil, "which", lambda command: "/usr/bin/say")
    monkeypatch.setattr(module, "installed_voices", lambda: available)

    def synthesize(arguments, *, input, **kwargs):
        calls.append((arguments, input, kwargs))
        write_pcm(Path(arguments[arguments.index("-o") + 1]))

    monkeypatch.setattr(module.subprocess, "run", synthesize)
    output = tmp_path / "generated"
    manifest = module.generate(output, 0)
    assert len(calls) == 12 and manifest["complete"] is True
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert (output / "request.json").read_bytes() == module.PROMPT.read_bytes()
    speech = [row for row in manifest["samples"] if row["kind"] == "speech"]
    for (arguments, text, kwargs), record in zip(calls, speech):
        assert arguments[:2] == ["say", "-v"]
        assert arguments[2] == record["voice"] and text == record["transcript"]
        assert "--data-format=LEI16@16000" in arguments and "--channels=1" in arguments
        assert arguments[-2:] == ["-f", "-"] and kwargs["check"] is True
    prompt = json.loads((output / "request.json").read_text())
    assert set(prompt) == {"task", "state", "query", "options"}
    assert all(row["transcript"] not in json.dumps(prompt) for row in speech)
    for row in manifest["samples"]:
        raw = (output / row["file"]).read_bytes()
        assert wav_info(raw)["duration_seconds"] == row["duration_seconds"]
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]


def test_cache_is_verified_and_reused_without_regenerating(tmp_path, fake_generation):
    first_folder, first = module.ensure_samples(tmp_path, seed=0)
    first_bytes = (first_folder / "manifest.json").read_bytes()
    second_folder, second = module.ensure_samples(tmp_path, seed=0)
    assert first_folder == second_folder and first == second
    assert (first_folder / "manifest.json").read_bytes() == first_bytes
    assert len(fake_generation) == 1
    assert first_folder.parent == tmp_path / ".runtime"
    assert not list((tmp_path / ".runtime").glob("audio-samples-pending-*"))
    other_folder, other = module.ensure_samples(tmp_path, seed=1)
    assert other_folder != first_folder and other["seed"] == 1
    assert len(fake_generation) == 2


def test_changed_prompt_or_macos_version_gets_a_separate_cache(
    tmp_path, fake_generation, monkeypatch,
):
    original, _ = module.ensure_samples(tmp_path)
    changed_prompt = tmp_path / "changed-request.json"
    request = json.loads(module.PROMPT.read_text())
    request["query"] += " Use the supplied options."
    changed_prompt.write_text(json.dumps(request))
    monkeypatch.setattr(module, "PROMPT", changed_prompt)
    changed, _ = module.ensure_samples(tmp_path)
    assert changed != original and original.is_dir()
    monkeypatch.setattr(module.platform, "mac_ver", lambda: ("new-version", ("", "", ""), ""))
    upgraded, _ = module.ensure_samples(tmp_path)
    assert upgraded not in {original, changed} and len(fake_generation) == 3


@pytest.mark.parametrize("tamper_file", [False, True])
def test_tampered_audio_or_manifest_hash_is_rejected_without_regeneration(
    tmp_path, fake_generation, tamper_file,
):
    folder, manifest = module.ensure_samples(tmp_path)
    audio = folder / manifest["samples"][0]["file"]
    if tamper_file:
        write_pcm(audio, 9999)  # Still valid WAV, so the hash guard must reject it.
    else:
        rewrite_manifest(folder, lambda value: value["samples"][0].update(sha256="0" * 64))
    contents = audio.read_bytes()
    with pytest.raises(ValueError, match="hash differs"):
        module.ensure_samples(tmp_path)
    assert audio.read_bytes() == contents and len(fake_generation) == 1


@pytest.mark.parametrize("invalid_name", ["../outside.wav", "/tmp/outside.wav", "sample_01.wav/child"])
def test_manifest_paths_cannot_escape_the_cache(tmp_path, fake_generation, invalid_name):
    folder, _ = module.ensure_samples(tmp_path)
    rewrite_manifest(folder, lambda value: value["samples"][0].update(file=invalid_name))
    with pytest.raises(ValueError, match="sample name"):
        module.ensure_samples(tmp_path)
    assert len(fake_generation) == 1


def test_sample_symlink_to_an_external_file_is_rejected(tmp_path, fake_generation):
    folder, manifest = module.ensure_samples(tmp_path)
    audio = folder / manifest["samples"][0]["file"]
    external = tmp_path / "outside.wav"
    external.write_bytes(audio.read_bytes())
    audio.unlink()
    audio.symlink_to(external)
    with pytest.raises(ValueError, match="sample file"):
        module.ensure_samples(tmp_path)
    assert external.is_file() and len(fake_generation) == 1


@pytest.mark.parametrize("mode", ["incomplete", "seed", "missing_record", "duplicate_id", "mismatch_id"])
def test_incomplete_or_inconsistent_manifest_is_rejected(tmp_path, fake_generation, mode):
    folder, manifest = module.ensure_samples(tmp_path)
    changed = copy.deepcopy(manifest)
    if mode == "incomplete":
        changed["complete"] = False
    elif mode == "seed":
        changed["seed"] = 99
    elif mode == "missing_record":
        changed["samples"].pop()
    elif mode == "duplicate_id":
        changed["samples"][1]["id"] = changed["samples"][0]["id"]
    else:
        changed["samples"][0]["id"] = "unrelated"
    (folder / "manifest.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        module.ensure_samples(tmp_path)
    assert len(fake_generation) == 1


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_failed_generation_removes_staging_and_never_publishes_cache(
    tmp_path, monkeypatch, error_type,
):
    def fail(output, seed):
        (output / "partial.wav").write_bytes(b"partial")
        raise error_type("Synthesis interrupted")

    monkeypatch.setattr(module, "generate", fail)
    with pytest.raises(error_type, match="Synthesis interrupted"):
        module.ensure_samples(tmp_path)
    assert list((tmp_path / ".runtime").iterdir()) == []


def test_concurrent_publication_reuses_winning_cache_and_removes_own_staging(
    tmp_path, fake_generation, monkeypatch,
):
    original_rename = Path.rename

    def publish_other_first(staging, target):
        shutil.copytree(staging, target)  # A second builder publishes a complete cache first.
        return original_rename(staging, target)

    monkeypatch.setattr(Path, "rename", publish_other_first)
    folder, manifest = module.ensure_samples(tmp_path)
    assert manifest["complete"] and len(manifest["samples"]) == 14
    assert list((tmp_path / ".runtime").iterdir()) == [folder]
    assert len(fake_generation) == 1
