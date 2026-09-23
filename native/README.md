# Text scoring runtime

`bash scripts/build_runtime.sh` builds `.runtime/build/bin/gemmajev-worker`
with Metal on macOS Apple Silicon. It downloads the pinned llama.cpp revision
`8086439a4cea94c71a5dfb8fe4ad1546aebd640f` to a private `.runtime/` source tree,
applies the one raw-logits patch, and publishes the executable atomically.
The default build uses two jobs; `GEMMAJEV_BUILD_JOBS` can override that.
`GEMMAJEV_WORKSPACE` selects the asset root and `GEMMAJEV_LLAMA_SOURCE` can point
to a local Git checkout of that exact revision. No other source tree is modified.

The worker accepts one JSON object per physical line containing `prompt` and
`labels`. It emits a startup object, then one response per request. Diagnostics
go to stderr. Closing stdin releases the model. The Python API serializes calls,
holds an exclusive workspace lock and destroys the worker on a timeout or
protocol failure so a late response cannot be assigned to another request.

The worker tokenizes caller content as literal text, separately from Gemma's
chat delimiters. It uses the official non-generating response position:
E4B ends at `<|turn>model\n`; 12B additionally appends an empty
`<|channel>thought\n<channel|>` block to disable reasoning. This implements the
[Gemma response format](https://ai.google.dev/gemma/docs/capabilities/thinking).
Each label must be exactly one unchanged token at that position. Candidate
probabilities are a stable softmax of those raw logits, without sampling,
generation, grammars or a second normalization in Python.

The patch disables upstream's generation-time vocabulary suppression mask.
It leaves model computation and softcapping intact. The worker records the
patch hash, its source hash and the llama.cpp revision, alongside runtime
parameters and token counts. Full-vocabulary diagnostics are internal only;
the public API returns semantic ID probabilities.

The model remains resident and reuses the exact common token prefix between
successive requests. Full sliding-window cache storage is enabled. Every
request removes the previous suffix and reevaluates at least its final token,
including for identical and shortened prompts. Missing history or failed
suffix removal triggers a complete recomputation. Decode failures synchronize
and clear both physical KV and its token provenance. `reset_cache: true` is an
internal diagnostic option for comparing cached and full-prefill scores.

`--self-test` reports build provenance and template prefixes without loading
weights or initializing a GPU. The supported context is 512–4096 tokens in multiples of 256, with
512-token batches, 128-token microbatches and four CPU threads. Cache reuse can
change floating-point roundoff through different Metal kernels; quality
comparisons should inspect decisions as well as probability differences.
