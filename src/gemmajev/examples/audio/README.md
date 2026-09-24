# Audio-language examples

This directory documents the Playground's locally generated audio samples and
offline checks. It contains no generated speech recordings. The request in
[audio_language.json](../audio_language.json)
classifies one clip into `english`, `japanese`, `spanish`, `french`, `other`, or
`unclear`, using the same request structure as the text and image examples.

Generate a varied local set on macOS from the repository root:

```bash
uv run python scripts/generate_audio_samples.py --seed 0
```

The Playground also generates these recipes automatically on first launch,
before loading the inference engine. It verifies and reuses the local cache
under `.runtime/audio-samples-*/` on later launches. If a required system voice
is unavailable, the other examples still work and you can upload your own WAV.
All model inputs use the same task, state, query and option descriptions;
voice names, transcripts and expected answers are not passed to the model.

The script uses installed macOS `say` voices for speech and Python's standard
library for controls. It requires no additional Python package, network service,
model weights or GPU. It reports missing voices instead of downloading them.
Generated WAV files and their manifest stay in the Git-ignored
`.runtime/audio-language-check/` directory. An existing nonempty directory is
never overwritten; use `--output .runtime/audio-language-check-seed-1 --seed 1`
for another selection.

The 14 clips contain:

| Content | Clips |
| --- | ---: |
| English, Japanese, Spanish and French | 2 per language |
| Mandarin Chinese and German, mapped to `other` | 2 per language |
| Silence and a non-speech tone mixture, both mapped to `unclear` | 2 |

Each of the six spoken languages has one male-presenting and one
female-presenting synthetic voice, so the twelve speech clips have six of each.
These labels describe voice presets,
not people identified from recordings. The seed controls voice selection,
utterances, speaking rates and shuffled sample order. The utterances are original,
neutral sentences about different situations; they are not translations of a
shared sentence and do not announce their language.

Every file is mono 16 kHz signed 16-bit PCM WAV. The API accepts 40 ms–30 s;
the supplied examples are approximately 2–7 seconds long.
`manifest.json` records the seed, voice preset, intended language, expected option,
transcript, duration, file SHA-256 and generator/request hashes. It also records
the macOS version: a seed fixes sample selection, but synthesized bytes can vary
between runs or operating-system/voice versions. `complete: true` is written only after
all generated files pass format and duration validation.

Use only `request.json` and a selected audio file as model input. Filenames are
neutral sample IDs; the manifest's labels, transcripts, voices and hashes are
evaluation metadata and must not enter the prompt. No model is run by this script.
This small synthetic set does not establish accuracy on real speakers, accents,
background noise, code-switching or arbitrary languages. Silence and tone probes
test abstention; they do not measure environmental-sound recognition.

Generated system-voice output is local test material and is not bundled with
this repository. Apple's [macOS license, section 2F](https://www.apple.com/legal/sla/docs/macOSTahoe.pdf)
restricts system-voice output to personal, non-commercial use and prohibits
public redistribution. The authored utterances and generator source follow
the repository's MIT license; that does not grant rights to Apple's voices or
generated recordings. Use appropriately licensed audio for other uses.

## Development check

The default model is 12B. Mandarin recordings are retained, but Mandarin is
intentionally not a named output category: they belong to `other`, alongside
German. Correctly selecting `other` does not demonstrate identifying
Mandarin by name. This demo distinguishes four named languages from other
speech and unclear audio.

The interface displays the model's actual probability distribution; it does
not substitute expected labels or select different prompts by model. Users
can edit the input fields and compare. A local 12B check with the seed-0 recipes
selected the expected category for all 14 clips, including both French voices
and both non-speech controls. This is a small synthetic smoke check, not a
language-identification benchmark.
