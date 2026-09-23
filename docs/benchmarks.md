# Benchmarks

GemmaJev scores fixed options using the same public input contract and prompt template on every task. The release contains measurements for **Gemma 4 E4B and 12B**, using Google's instruction-tuned QAT Q4_0 checkpoints, on a Mac with an M2 Pro and 16 GiB of unified memory. No task-specific fine-tuning or calibration is applied.

The tables below retain the completed measurements. Repository restructuring did not rerun the models. The [summary](results/summary.json), [per-decision archive](results/predictions.jsonl.gz), and [external source records](results/references.json) contain the values and provenance used here.

## Compared with TypeSafe Jev

Each task has 200 examples. Accuracy is higher-is-better; HelpSteer2 uses Spearman rank correlation between expected helpfulness scores and human ratings.

| Task | Metric | TypeSafe Jev | GemmaJev E4B | GemmaJev 12B |
| --- | --- | ---: | ---: | ---: |
| AG News | Accuracy | 87.00% | 84.50% | 82.50% |
| BoolQ | Accuracy | 91.50% | 85.00% | 89.50% |
| MMLU | Accuracy | 93.50% | 66.00% | 73.00% |
| LogiQA | Accuracy | 76.50% | 34.50% | 49.00% |
| ANLI R3 | Accuracy | 66.00% | 45.00% | 52.50% |
| Emotion | Accuracy | 50.50% | 49.50% | 47.50% |
| HelpSteer2 | Spearman | 0.4902 | 0.4686 | 0.5676 |

**The Jev column is a community measurement of the original TypeSafe API**, not a TypeSafe-published benchmark or a local Jev run. It comes from [OmarMujahid/jev-decision-bench, pinned report](https://github.com/OmarMujahid/jev-decision-bench/blob/0883fa781729ab571005253f6b288ec67f49dd29/report.json), requesting `jev-1.13.0` on 2026-09-18.

We reconstruct its published seed-7 sampling rules, balancing and answer shuffling against pinned dataset files. The original evaluator did not publish immutable dataset revisions or historical input snapshots. The published Jev predictions reproduce the reported primary scores when scored against our reconstructed gold labels; that does **not** establish identical historical input text. Treat these as task-level comparisons, without paired significance claims across systems.

12B approaches Jev on reading comprehension and exceeds the reported helpfulness-ranking correlation, while remaining substantially behind on knowledge and complex reasoning. Higher rank correlation does not imply better calibrated absolute scores: Jev's HelpSteer2 MAE was not published. Local HelpSteer2 MAE is 1.0815 for E4B and 1.0228 for 12B, on the original 0–4 scale.

## Compared with community implementations

The AnyJev comparison follows its published seed-0 selection: 300 test items each for Banking20, 20 Newsgroups and prompt injection, plus all 400 Typed-decisions test cases with five questions each. The first three columns below use **macro-F1**, not accuracy. Typed accuracy covers 2,000 questions; Typed MAE covers only its 800 ordinal-score questions.

**Task-specific training** indicates additional model-weight training for decision tasks, excluding the base checkpoint's original pretraining and instruction tuning. Configuration links point to the recorded benchmark sources.

| System and published configuration | Task-specific training | Banking20 F1 | Newsgroups F1 | Injection F1 | Typed accuracy | Typed MAE ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GemmaJev E4B | No | 83.72% | 64.75% | 82.75% | 67.40% | 0.4236 |
| GemmaJev 12B | No | 85.84% | 68.31% | 92.88% | 72.50% | 0.4530 |
| [AnyJev](https://github.com/nokia-applied-research/AnyJev) · Qwen2.5-7B-Instruct · [raw](https://github.com/nokia-applied-research/AnyJev/blob/78ca550268eefa4b4ed7b34f377b475597e4f51b/bench/results_batchprior_v0/2026-09-20/Qwen__Qwen2.5-7B-Instruct.json) ([Typed](https://github.com/nokia-applied-research/AnyJev/blob/78ca550268eefa4b4ed7b34f377b475597e4f51b/bench/results_typed/2026-09-21/Qwen__Qwen2.5-7B-Instruct.json)) | No | 73.77% | 66.99% | 67.46% | 62.05% | 0.4375 |
| [AnyJev](https://github.com/nokia-applied-research/AnyJev) · Qwen2.5-7B-Instruct · [L1](https://github.com/nokia-applied-research/AnyJev/blob/78ca550268eefa4b4ed7b34f377b475597e4f51b/bench/results_batchprior_v0/2026-09-20/Qwen__Qwen2.5-7B-Instruct.json) ([Typed](https://github.com/nokia-applied-research/AnyJev/blob/78ca550268eefa4b4ed7b34f377b475597e4f51b/bench/results_typed/2026-09-21/Qwen__Qwen2.5-7B-Instruct.json)) | No | 78.12% | 71.35% | 77.79% | 63.20% | 0.4427 |
| [AnyJev](https://github.com/nokia-applied-research/AnyJev) · Qwen3-32B · [L1](https://github.com/nokia-applied-research/AnyJev/blob/78ca550268eefa4b4ed7b34f377b475597e4f51b/bench/results_typed/2026-09-21/Qwen__Qwen3-32B.json) | No | — | — | — | 70.05% | 0.4122 |
| [Laya](https://huggingface.co/convaiinnovations/laya-typed-decisions) · laya-typed-decisions · [fine-tuned](https://github.com/nokia-applied-research/AnyJev/blob/78ca550268eefa4b4ed7b34f377b475597e4f51b/bench/results_typed/2026-09-21/laya__convaiinnovations__laya-typed-decisions.json) | [Yes](https://huggingface.co/convaiinnovations/laya-typed-decisions#training) | — | — | — | 76.80% | 0.2425 |
| [TypeSafe Jev 1.13.0](https://docs.typesafe.ai/introduction) · [reported API](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/blob/ea9306458d6e9563628369a3d1e72e362fb381d2/README.md) | [Yes](https://docs.typesafe.ai/introduction/machine-learning-primer#rlcd-and-calibrated-decisions) | — | — | — | 72.70% | 0.3910 |

[AnyJev's pinned results](https://github.com/nokia-applied-research/AnyJev/tree/78ca550268eefa4b4ed7b34f377b475597e4f51b/bench) supply the AnyJev and Laya columns. `raw` is its direct checkpoint readout; `L1` is its calibrated pipeline, including the preceding corrections. These are fixed named configurations, not a per-task selection of the best result. Missing cells were not evaluated in those published configurations. Exact readout conditions and source hashes are in [references.json](results/references.json).

The Typed-decisions Jev figure comes from the [dataset author's reported API measurement](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/blob/ea9306458d6e9563628369a3d1e72e362fb381d2/README.md). It is not an independent Jev run by AnyJev or Laya. Laya is fine-tuned on this task's training split; GemmaJev and the selected AnyJev models are frozen. Typed-decisions inputs and targets are synthetic, with teacher-derived labels and scores. These results measure agreement with that reference, not human decision quality.

| System | Task-specific training | Authored144 accuracy | WANLI accuracy |
| --- | --- | ---: | ---: |
| GemmaJev E4B | No | 81.94% | 70.14% |
| GemmaJev 12B | No | 95.83% | 75.69% |
| [SemIf](https://github.com/TheoLeeCJ/SemIf) · Qwen3.5-4B · [direct](https://github.com/TheoLeeCJ/SemIf/blob/1f2dea3e25379f9dfc98cb83c324f00ab5deda37/results/raw/predictions/direct-authored144.jsonl) ([WANLI](https://github.com/TheoLeeCJ/SemIf/blob/1f2dea3e25379f9dfc98cb83c324f00ab5deda37/results/raw/predictions/direct-wanli256.jsonl)) | No | 80.56% | 64.58% |
| [SemIf](https://github.com/TheoLeeCJ/SemIf) · Qwen3.8-27B-exl3 · [direct](https://github.com/TheoLeeCJ/SemIf/blob/1f2dea3e25379f9dfc98cb83c324f00ab5deda37/exl3-bridge/results/authored144-27b-exl3.jsonl) | No | 95.83% | — |

[SemIf's pinned per-item results](https://github.com/TheoLeeCJ/SemIf/tree/1f2dea3e25379f9dfc98cb83c324f00ab5deda37/results/raw/predictions) supply these frozen-model baselines. Authored144 retains all 144 authored examples, in three families and 36 groups of four related variants. WANLI uses 144 held-out examples from SemIf's published 256-item pool, balanced across entailment, contradiction and neutral; the same selected item IDs are used for the SemIf comparison. The split is seed-20260923 and disjoint by connected premise/pair-ID groups. These two panels are small; Authored144 is an authored diagnostic rather than a representative production distribution.

## Reproduce

Use the supported macOS / Apple Silicon environment described in the [README](../README.md). All downloaded data, model weights and compiled/runtime output remain in ignored `.datasets/`, `.models/` and `.runtime/` directories.

First, verify every published local score **offline**, without downloading models or datasets:

```bash
uv run gemmajev benchmark verify
uv run gemmajev benchmark report
```

Verification checks the prediction archive and external-reference hashes, every source-manifest hash, complete two-model coverage, probability validity, and all recomputed local summary values. External published aggregates are verified as archived source records; GemmaJev does not claim to have rerun their models.

To prepare data and run fresh inference:

```bash
uv run --extra data gemmajev benchmark prepare --dataset all
uv run gemmajev setup --model 12b
uv run --extra data gemmajev benchmark run --model 12b --dataset boolq --limit 3
uv run --extra data gemmajev benchmark run --model 12b --dataset all
uv run gemmajev setup --model e4b
uv run --extra data gemmajev benchmark run --model e4b --dataset all
```

`--limit N` selects the first N items **per dataset** for a smoke check and writes to a separate output directory. Omit it for the frozen full panel. Sampling seeds are fixed by the published recipes; a different seed is a different benchmark. Downloads use immutable revisions, byte sizes and SHA-256 checks from [the manifests](../configs/benchmarks/). Dataset code is reconstructed locally; no remote dataset code is executed.

The runner is serial and releases the model after 16 requests or 45 seconds of completed work, then cools for 30 seconds. Repeating the command resumes the same ordered protocol without repeating saved predictions. Errors stop the run and preserve progress; partial panels are not silently reported as complete. Fresh measurements and model/runtime fingerprints are saved under `.runtime/benchmarks/`. Use `--output` for a separate run. Measurements may vary slightly across builds and cache boundaries; the archived values are the original recorded measurements, not a guarantee of bit-identical future inference.

The recorded runs used the pinned llama.cpp revision `8086439a4cea94c71a5dfb8fe4ad1546aebd640f`. Their batching differed by panel: the seven-task comparison used a 64-request / 45-second threshold, the five-task community panel used 64 requests / 60 seconds, and WANLI used fixed batches of 64, 64 and 16 requests per model. The time-based panels finished the current source group before releasing the worker; all panels cooled for 30 seconds. These recorded settings and the original binary hash are retained in the summary, separately from the release runner's current defaults.

## Input and metric details

Only `task`, `state`, `query` and the ordered semantic `options` map enter the model. Task instructions name their own evidence/context and question fields consistently; gold labels, source IDs and annotation metadata are kept outside the API input. Every task uses the final generic prompt from [request.py](../src/gemmajev/request.py).

The reconstruction preserves all selected examples and their original gold answers. Before the recorded measurements, the adapter made three input corrections: BoolQ's native boolean proposition became a decision question; MMLU's shuffled “all/none of the above” and referenced-answer options were expanded without changing their meaning; LogiQA's source-parser truncations were repaired against its pinned original `Test.txt`. The manifest records affected IDs and hashes. These are explicit input adaptations and another reason to avoid claiming identical historical Jev inputs.

- **Accuracy:** argmax semantic option ID equals the source gold ID. Ordinal level accuracy also uses argmax, not a rounded expected score.
- **Macro-F1:** arithmetic mean across observed gold/predicted class IDs on fixed-class tasks. It is not pooled across unrelated answer choices in knowledge questions.
- **MAE:** mean absolute error of `sum(probability × score level)` against the source target, in the original rubric units. Smaller is better. Typed-decisions mixes 0–3 and 0–4 rubrics; its MAE is not normalized.
- **Spearman:** correlation of average-tie ranks of expected and human scores; larger is better. Undefined constant panels return null.
- **Latency:** serial local request wall time, including both cold and prefix-cache-reused requests, excluding model loading and cooling. It is not directly comparable with external batched throughput or hosted server time.

The archive contains 9,176 decision records across 4,588 examples per model. It stores IDs, probabilities, gold labels, score mappings and request hashes, without source passages or model weights. Dataset and model licenses remain their upstream licenses. SemIf attribution is preserved in [its license notice](licenses/semif.txt); dataset license metadata and source links are retained in the manifests.
