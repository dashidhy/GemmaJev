#include "ggml-backend.h"
#include "llama.h"
#ifdef GEMMAJEV_VISION
#include "mtmd.h"
#include "mtmd-helper.h"
#endif
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

namespace {
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double peak_rss_mb() {
#if defined(__unix__) || defined(__APPLE__)
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
#else
    return 0;
#endif
}

template<class T, void (*Free)(T *)>
using Handle = std::unique_ptr<T, decltype(Free)>;

#ifdef GEMMAJEV_VISION
// Validate a bounded canonical WAV before allocating/decoding its waveform.
// The public audio API accepts only PCM16, mono, 16 kHz and at most 30 seconds.
std::vector<float> read_audio(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("audio_path must name a readable WAV file");
    const auto length = file.tellg();
    if (length < 44 || length > 20 * 1024 * 1024) {
        throw std::runtime_error("audio WAV must be nonempty and at most 20 MiB");
    }
    std::vector<uint8_t> data(static_cast<size_t>(length));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char *>(data.data()), length)) {
        throw std::runtime_error("truncated WAV file");
    }
    auto tag = [&](size_t offset) { return std::string(reinterpret_cast<const char *>(data.data() + offset), 4); };
    auto u16 = [&](size_t offset) { return uint32_t(data[offset]) | (uint32_t(data[offset + 1]) << 8); };
    auto u32 = [&](size_t offset) {
        return uint32_t(data[offset]) | (uint32_t(data[offset + 1]) << 8)
            | (uint32_t(data[offset + 2]) << 16) | (uint32_t(data[offset + 3]) << 24);
    };
    if (tag(0) != "RIFF" || tag(8) != "WAVE" || uint64_t(u32(4)) + 8 != data.size()) {
        throw std::runtime_error("invalid RIFF/WAVE header or file size");
    }
    bool format_found = false;
    size_t pcm_start = 0, pcm_bytes = 0;
    for (size_t position = 12; position < data.size();) {
        if (position + 8 > data.size()) throw std::runtime_error("truncated WAV chunk header");
        const auto kind = tag(position);
        const size_t size = u32(position + 4), start = position + 8;
        const size_t end = start + size, padded_end = end + size % 2;
        if (padded_end > data.size()) throw std::runtime_error("truncated WAV chunk");
        if (kind == "fmt ") {
            if (format_found || size < 16) throw std::runtime_error("invalid WAV format header");
            if (u16(start) != 1 || u16(start + 2) != 1 || u32(start + 4) != 16000
                    || u32(start + 8) != 32000 || u16(start + 12) != 2 || u16(start + 14) != 16) {
                throw std::runtime_error("audio must be PCM16, mono, 16 kHz WAV");
            }
            format_found = true;
        } else if (kind == "data") {
            if (pcm_start || size == 0 || size % 2) throw std::runtime_error("invalid WAV audio data");
            if (size / 2 < 640) throw std::runtime_error("audio must be at least 40 milliseconds long");
            if (size / 2 > 30 * 16000) throw std::runtime_error("audio must be no longer than 30 seconds");
            pcm_start = start;
            pcm_bytes = size;
        }
        position = padded_end;
    }
    if (!format_found || !pcm_start) throw std::runtime_error("WAV format or audio data is missing");
    std::vector<float> pcm(pcm_bytes / 2);
    for (size_t i = 0; i < pcm.size(); ++i) {
        int32_t sample = static_cast<int32_t>(u16(pcm_start + 2 * i));
        if (sample >= 32768) sample -= 65536;
        pcm[i] = static_cast<float>(sample) / 32768.0f;
    }
    return pcm;
}
#endif

struct Options {
    std::string model;
    std::string model_size = "12b";
    std::string mmproj;
    int image_tokens = 70;
    int context = 4096;
    int threads = 4;
    int gpu_layers = 99;
    bool self_test = false;
};

Options options(int argc, char ** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            std::cout << "gemmajev-worker --model FILE --model-size e4b|12b "
                         "[--ctx-size 4096] [--threads 4] [--gpu-layers 99]\n"
                         "Multimodal builds also accept --mmproj FILE --image-tokens 70|140|280|560|1120\n"
                         "gemmajev-worker --self-test\n";
            std::exit(0);
        }
        if (key == "--self-test") { result.self_test = true; continue; }
        if (++i == argc) throw std::runtime_error("missing value for " + key);
        const std::string value = argv[i];
        if (key == "--model") result.model = value;
        else if (key == "--model-size") result.model_size = value;
        else if (key == "--mmproj") result.mmproj = value;
        else if (key == "--ctx-size" || key == "--threads" || key == "--gpu-layers" || key == "--image-tokens") {
            size_t consumed = 0;
            const int number = std::stoi(value, &consumed);
            if (consumed != value.size()) throw std::runtime_error("invalid integer for " + key);
            if (key == "--ctx-size") result.context = number;
            else if (key == "--threads") result.threads = number;
            else if (key == "--gpu-layers") result.gpu_layers = number;
            else result.image_tokens = number;
        } else throw std::runtime_error("unknown argument: " + key);
    }
    if (result.model_size != "e4b" && result.model_size != "12b") {
        throw std::runtime_error("model size must be e4b or 12b");
    }
    if (result.context < 512 || result.context > 4096 || result.context % 256 != 0 || result.threads < 1 || result.gpu_layers < 0) {
        throw std::runtime_error("ctx-size must be a multiple of 256 from 512–4096, threads >= 1, gpu-layers >= 0");
    }
    const std::set<int> budgets{70, 140, 280, 560, 1120};
    if (!budgets.count(result.image_tokens)) throw std::runtime_error("unsupported image token budget");
#ifndef GEMMAJEV_VISION
    if (!result.mmproj.empty()) throw std::runtime_error("build the vision worker to use --mmproj");
#endif
    if (!result.self_test && result.model.empty()) throw std::runtime_error("--model is required");
    return result;
}

void log_stderr(enum ggml_log_level level, const char * text, void *) {
    if (level != GGML_LOG_LEVEL_DEBUG) std::fputs(text, stderr);
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text,
                                  bool add_special, bool parse_special) {
    if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("text is too long");
    }
    int count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                               nullptr, 0, add_special, parse_special);
    if (count == 0) return {};
    if (count > 0 || count == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("failed to size tokenization");
    }
    std::vector<llama_token> result(static_cast<size_t>(-count));
    count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                           result.data(), static_cast<int32_t>(result.size()), add_special, parse_special);
    if (count < 0) throw std::runtime_error("tokenization failed");
    result.resize(static_cast<size_t>(count));
    return result;
}

std::string piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> output(128);
    int count = llama_token_to_piece(vocab, token, output.data(), static_cast<int>(output.size()), 0, true);
    if (count < 0) {
        output.resize(static_cast<size_t>(-count));
        count = llama_token_to_piece(vocab, token, output.data(), static_cast<int>(output.size()), 0, true);
    }
    if (count < 0) throw std::runtime_error("token decoding failed");
    return std::string(output.data(), static_cast<size_t>(count));
}

// Explicitly render one user turn and the model-specific response prefix.
// No response tokens are generated: only next-token candidate logits are read.
// Official reference: https://ai.google.dev/gemma/docs/capabilities/thinking
std::string answer_prefix(const std::string & size) {
    return "<|turn>model\n" + (size == "12b" ? std::string("<|channel>thought\n<channel|>") : "");
}

// Tokenize caller text as literal text, keeping model delimiters separate.
// Keeping "user\n" with the body preserves normal BPE boundaries from the
// complete template, while embedded control-looking strings stay plain text.
std::vector<llama_token> chat_tokens(const llama_vocab * vocab, const std::string & prompt,
                                     const std::string & size, const std::string & label = "") {
    auto tokens = tokenize(vocab, "<|turn>", true, true);
    const auto body = tokenize(vocab, "user\n" + prompt, false, false);
    tokens.insert(tokens.end(), body.begin(), body.end());
    const auto suffix = tokenize(vocab, "<turn|>\n" + answer_prefix(size) + label, false, true);
    tokens.insert(tokens.end(), suffix.begin(), suffix.end());
    return tokens;
}

json runtime_patches() {
    auto patches = json::array({
        {{"name", "gemma4-raw-logits.patch"}, {"sha256", GEMMAJEV_PATCH_SHA256}},
    });
#ifdef GEMMAJEV_VISION
    patches.push_back({{"name", "gemma4-image-budget.patch"}, {"sha256", GEMMAJEV_IMAGE_PATCH_SHA256}});
#endif
    return patches;
}

constexpr bool vision_build() {
#ifdef GEMMAJEV_VISION
    return true;
#else
    return false;
#endif
}

class Worker {
public:
    explicit Worker(const Options & opts) : opts_(opts) {
        const auto started = Clock::now();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = opts.gpu_layers;
        model_.reset(llama_model_load_from_file(opts.model.c_str(), mp));
        if (!model_) throw std::runtime_error("failed to load language model");
        char architecture[128];
        if (llama_model_meta_val_str(model_.get(), "general.architecture", architecture, sizeof(architecture)) < 0
                || std::string(architecture) != "gemma4") {
            throw std::runtime_error("worker requires a Gemma 4 language model");
        }
        vocab_ = llama_model_get_vocab(model_.get());
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = static_cast<uint32_t>(opts.context);
        cp.n_batch = std::min<uint32_t>(cp.n_ctx, opts.mmproj.empty() ? 512 : 2048);
        // Gemma's non-causal image must fit in one microbatch.
        cp.n_ubatch = std::min<uint32_t>(cp.n_batch, opts.mmproj.empty() ? 128 : std::max(512, opts.image_tokens));
        cp.n_threads = opts.threads;
        cp.n_threads_batch = opts.threads;
        // The pinned runtime already defaults to this; make the prefix-retention
        // requirement explicit for Gemma's alternating full/SWA attention.
        cp.swa_full = true;
        context_.reset(llama_init_from_model(model_.get(), cp));
        if (!context_) throw std::runtime_error("failed to create language context");
#ifdef GEMMAJEV_VISION
        if (!opts.mmproj.empty()) {
            auto params = mtmd_context_params_default();
            params.use_gpu = opts.gpu_layers > 0;
            params.n_threads = opts.threads;
            params.image_max_tokens = opts.image_tokens;
            params.warmup = false;
            params.print_timings = false;
            vision_.reset(mtmd_init_from_file(opts.mmproj.c_str(), model_.get(), params));
            if (!vision_ || !mtmd_support_vision(vision_.get())) {
                throw std::runtime_error("failed to load a compatible vision projector");
            }
        }
#endif
        load_ms_ = milliseconds(started);
    }

    json ready() const {
        return {{"ready", true}, {"model_size", opts_.model_size},
                {"runtime_revision", GEMMAJEV_LLAMA_REVISION}, {"load_ms", load_ms_},
                {"worker_source_sha256", GEMMAJEV_WORKER_SHA256},
                {"runtime_patches", runtime_patches()},
                {"kv_cache", true},
                {"input_modality", opts_.mmproj.empty() ? "text" : "image"},
                {"audio_supported", audio_supported()},
                {"image_token_budget", opts_.mmproj.empty() ? json(nullptr) : json(opts_.image_tokens)},
                {"batch_size", llama_n_batch(context_.get())},
                {"microbatch_size", llama_n_ubatch(context_.get())},
                {"context_size", llama_n_ctx(context_.get())}, {"threads", opts_.threads},
                {"gpu_layers_requested", opts_.gpu_layers}, {"vocab_size", llama_vocab_n_tokens(vocab_)},
                {"architecture", "gemma4"}, {"process_peak_rss_mb", peak_rss_mb()}};
    }

    json score(const json & request) {
        bool memory_mutated = false;
        try {
            return score_impl(request, memory_mutated);
        } catch (...) {
            // Malformed requests fail before mutation and preserve a valid
            // prefix. Decode/scoring failures may leave partially updated KV.
            if (memory_mutated) {
                // A failed/aborted decode can leave asynchronous GPU work from
                // earlier microbatches; finish it before clearing shared buffers.
                llama_synchronize(context_.get());
                llama_memory_clear(llama_get_memory(context_.get()), true);
                cached_tokens_.clear();
                cached_positions_ = 0;
                previous_request_failed_ = true;
            }
            throw;
        }
    }

private:
    bool audio_supported() const {
#ifdef GEMMAJEV_VISION
        return vision_ && mtmd_support_audio(vision_.get());
#else
        return false;
#endif
    }

    json score_impl(const json & request, bool & memory_mutated) {
        const auto started = Clock::now();
        if (!request.is_object()) throw std::runtime_error("request must be a JSON object");
        const bool reset_cache = request.value("reset_cache", false);
        const std::string prompt = request.at("prompt").get<std::string>();
        if (prompt.find('\0') != std::string::npos) {
            throw std::runtime_error("prompt must not contain NUL bytes");
        }
        const auto labels = request.at("labels").get<std::vector<std::string>>();
        const std::string size = request.value("model_size", opts_.model_size);
        if (size != opts_.model_size) throw std::runtime_error("request model_size differs from loaded model_size");
        if (labels.size() < 2 || labels.size() > 26) throw std::runtime_error("provide between 2 and 26 labels");
        const std::string prefix = answer_prefix(size);
        auto prefix_tokens = chat_tokens(vocab_, prompt, size);
        std::vector<llama_token> token_ids;
        std::set<llama_token> unique_tokens;
        for (const auto & label : labels) {
            if (label.size() != 1 || label[0] < 'A' || label[0] > 'Z') {
                throw std::runtime_error("labels must be single uppercase letters A–Z");
            }
            const auto bare = tokenize(vocab_, label, false, false);
            const auto appended = chat_tokens(vocab_, prompt, size, label);
            if (bare.size() != 1 || appended.size() != prefix_tokens.size() + 1
                    || !std::equal(prefix_tokens.begin(), prefix_tokens.end(), appended.begin())
                    || appended.back() != bare.front() || piece(vocab_, bare.front()) != label) {
                throw std::runtime_error("label must be exactly one unchanged token at the answer position: " + label);
            }
            if (!unique_tokens.insert(bare.front()).second) throw std::runtime_error("labels must be distinct");
            token_ids.push_back(bare.front());
        }

        const bool has_image = request.contains("image_path");
        const bool has_audio = request.contains("audio_path");
        if (has_image && has_audio) throw std::runtime_error("provide only one image or audio attachment");
        const bool has_media = has_image || has_audio;
        const std::string modality = has_audio ? "audio" : has_image ? "image" : "text";
        size_t n_tokens = prefix_tokens.size();
        llama_pos n_positions = static_cast<llama_pos>(n_tokens);
        size_t image_tokens = 0;
        size_t audio_tokens = 0;
        double audio_seconds = 0;
        json image_size = nullptr;
#ifdef GEMMAJEV_VISION
        Handle<mtmd_bitmap, mtmd_bitmap_free> bitmap{nullptr, mtmd_bitmap_free};
        Handle<mtmd_input_chunks, mtmd_input_chunks_free> chunks{nullptr, mtmd_input_chunks_free};
        std::vector<float> audio_pcm;
        std::vector<llama_token> media_prefix, media_suffix;
        if (has_media) {
            if (!vision_) throw std::runtime_error("media input requires a loaded multimodal projector");
            const auto path = request.at(has_audio ? "audio_path" : "image_path").get<std::string>();
            if (path.empty() || path.find('\0') != std::string::npos) {
                throw std::runtime_error("invalid attachment path");
            }
            if (has_audio) {
                if (!audio_supported()) throw std::runtime_error("the loaded projector does not support audio");
                audio_pcm = read_audio(path);
                audio_seconds = audio_pcm.size() / 16000.0;
                bitmap.reset(mtmd_bitmap_init_from_audio(audio_pcm.size(), audio_pcm.data()));
                if (!bitmap || !mtmd_bitmap_is_audio(bitmap.get())) throw std::runtime_error("audio initialization failed");
            } else {
                auto wrapped = mtmd_helper_bitmap_init_from_file(vision_.get(), path.c_str(), false);
                bitmap.reset(wrapped.bitmap);
                Handle<mtmd_helper_video, mtmd_helper_video_free> video{wrapped.video_ctx, mtmd_helper_video_free};
                if (!bitmap || video || mtmd_bitmap_is_audio(bitmap.get())) {
                    throw std::runtime_error("image_path must name a readable image file");
                }
            }
            // Only the trusted media marker reaches mtmd. Caller text is literal.
            chunks.reset(mtmd_input_chunks_init());
            const mtmd_bitmap * input_bitmap = bitmap.get();
            const mtmd_input_text input{mtmd_get_marker(vision_.get()), false, false};
            if (mtmd_tokenize(vision_.get(), chunks.get(), &input, &input_bitmap, 1) != 0) {
                throw std::runtime_error("multimodal tokenization failed");
            }
            // Everything before the attachment is causal text and is reusable.
            // Media and the response suffix are always recomputed for this request.
            media_prefix = tokenize(vocab_, "<|turn>", true, true);
            const auto body = tokenize(vocab_, "user\n" + prompt + "\n", false, false);
            media_prefix.insert(media_prefix.end(), body.begin(), body.end());
            media_suffix = tokenize(vocab_, "\n<turn|>\n" + prefix, false, true);
            prefix_tokens = media_prefix;
            n_tokens = media_prefix.size() + mtmd_helper_get_n_tokens(chunks.get()) + media_suffix.size();
            n_positions = media_prefix.size() + mtmd_helper_get_n_pos(chunks.get()) + media_suffix.size();
            if (has_image) image_size = {mtmd_bitmap_get_nx(bitmap.get()), mtmd_bitmap_get_ny(bitmap.get())};
            for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
                const auto * chunk = mtmd_input_chunks_get(chunks.get(), i);
                if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                    if (!has_image) throw std::runtime_error("unexpected image in audio input");
                    const auto count = mtmd_input_chunk_get_n_tokens(chunk);
                    if (count > llama_n_ubatch(context_.get())) {
                        throw std::runtime_error("image embeddings exceed the non-causal microbatch");
                    }
                    image_tokens += count;
                } else if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
                    if (!has_audio) throw std::runtime_error("unexpected audio in image input");
                    audio_tokens += mtmd_input_chunk_get_n_tokens(chunk);
                }
            }
            if (has_image && image_tokens == 0) throw std::runtime_error("image produced no visual tokens");
            if (has_audio && audio_tokens == 0) throw std::runtime_error("audio produced no audio tokens");
        }
#else
        if (has_media) throw std::runtime_error("media input requires the multimodal worker");
#endif
        if (n_positions > static_cast<llama_pos>(llama_n_ctx(context_.get()))) {
            throw std::runtime_error("input exceeds the context window");
        }
        const double preprocess_ms = milliseconds(started);
        const auto reset_start = Clock::now();
        memory_mutated = true;
        const auto cache = prepare_cache(prefix_tokens, reset_cache, modality);
        double prefill_ms = milliseconds(reset_start);
        double vision_ms = 0;
        double audio_ms = 0;
        if (!has_media) {
            const auto decode_start = Clock::now();
            decode_text(prefix_tokens, cache.reused_tokens, 0, true);
            prefill_ms += milliseconds(decode_start);
        }
#ifdef GEMMAJEV_VISION
        else {
            const auto prefix_start = Clock::now();
            llama_pos n_past = decode_text(media_prefix, cache.reused_tokens, 0, false);
            prefill_ms += milliseconds(prefix_start);
            for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
                const auto * chunk = mtmd_input_chunks_get(chunks.get(), i);
                const auto type = mtmd_input_chunk_get_type(chunk);
                if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
                    const auto encode_start = Clock::now();
                    if (mtmd_encode_chunk(vision_.get(), chunk) != 0) {
                        throw std::runtime_error("media encoding failed");
                    }
                    if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) vision_ms += milliseconds(encode_start);
                    else audio_ms += milliseconds(encode_start);
                    const auto decode_start = Clock::now();
                    if (mtmd_helper_decode_image_chunk(vision_.get(), context_.get(), chunk,
                            mtmd_get_output_embd(vision_.get()), n_past, 0,
                            static_cast<int32_t>(llama_n_batch(context_.get())), &n_past, nullptr, nullptr) != 0) {
                        throw std::runtime_error("media embedding prefill failed");
                    }
                    prefill_ms += milliseconds(decode_start);
                } else if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    const auto decode_start = Clock::now();
                    if (mtmd_helper_eval_chunk_single(vision_.get(), context_.get(), chunk, n_past, 0,
                            static_cast<int32_t>(llama_n_batch(context_.get())), false, &n_past) != 0) {
                        throw std::runtime_error("media boundary prefill failed");
                    }
                    prefill_ms += milliseconds(decode_start);
                } else {
                    throw std::runtime_error("unexpected media chunk");
                }
            }
            const auto suffix_start = Clock::now();
            n_past = decode_text(media_suffix, 0, n_past, true);
            if (n_past != n_positions) throw std::runtime_error("multimodal position accounting mismatch");
            prefill_ms += milliseconds(suffix_start);
        }
#endif
        const auto score_start = Clock::now();
        // This synchronizes pending GPU work before reading the unmodified logits.
        const float * logits = llama_get_logits_ith(context_.get(), -1);
        prefill_ms += milliseconds(score_start);
        const auto math_start = Clock::now();
        if (!logits) throw std::runtime_error("no answer-position logits produced");
        const auto vocab_size = llama_vocab_n_tokens(vocab_);
        llama_token greedy = 0;
        for (llama_token t = 0; t < vocab_size; ++t) {
            if (!std::isfinite(logits[t])) throw std::runtime_error("non-finite raw vocabulary logits");
            if (logits[t] > logits[greedy]) greedy = t;
        }
        const double full_max = logits[greedy];
        double full_sum = 0;
        for (llama_token t = 0; t < vocab_size; ++t) full_sum += std::exp(logits[t] - full_max);
        const double full_log_z = full_max + std::log(full_sum);
        size_t selected = 0;
        for (size_t i = 1; i < labels.size(); ++i) {
            if (logits[token_ids[i]] > logits[token_ids[selected]]) selected = i;
        }
        const double candidate_max = logits[token_ids[selected]];
        double candidate_sum = 0;
        for (const auto token : token_ids) candidate_sum += std::exp(logits[token] - candidate_max);
        const double candidate_log_z = candidate_max + std::log(candidate_sum);
        json candidates = json::array();
        std::vector<double> raw_logits;
        std::vector<double> probabilities;
        double entropy = 0;
        for (size_t i = 0; i < labels.size(); ++i) {
            const auto token = token_ids[i];
            const double probability = std::exp(logits[token] - candidate_log_z);
            if (probability > 0) entropy -= probability * std::log(probability);
            raw_logits.push_back(logits[token]);
            probabilities.push_back(probability);
            candidates.push_back({{"label", labels[i]}, {"token_id", token}, {"logit", logits[token]},
                                  {"probability", probability},
                                  {"vocab_probability", std::exp(logits[token] - full_log_z)}});
        }
        const double score_ms = milliseconds(math_start);
        json result = {{"ok", true}, {"candidates", candidates}, {"labels", labels}, {"token_ids", token_ids},
                {"logits", raw_logits}, {"probabilities", probabilities},
                {"selected_index", selected}, {"selected_label", labels[selected]},
                {"candidate_mass", std::exp(candidate_log_z - full_log_z)}, {"entropy", entropy},
                {"greedy_token", {{"token_id", greedy}, {"piece", piece(vocab_, greedy)},
                                  {"logit", logits[greedy]}, {"probability", std::exp(logits[greedy] - full_log_z)}}},
                {"prompt_tokens", n_tokens}, {"image_tokens", image_tokens}, {"image_size", image_size},
                {"input_modality", modality},
                {"image_position", has_image ? json("after_text") : json(nullptr)},
                {"audio_tokens", audio_tokens},
                {"audio_position", has_audio ? json("after_text") : json(nullptr)},
                {"audio_duration_seconds", has_audio ? json(audio_seconds) : json(nullptr)},
                {"audio_sample_rate", has_audio ? json(16000) : json(nullptr)},
                {"cacheable_prefix_tokens", prefix_tokens.size()},
                {"image_token_budget", has_image ? json(opts_.image_tokens) : json(nullptr)},
                {"image_resize_mode", has_image ? json("gemma4_max_budget_floor") : json(nullptr)},
                {"image_resize_interpolation", has_image ? json("bilinear") : json(nullptr)},
                {"answer_prefix", prefix}, {"model_size", opts_.model_size},
                {"vocab_size", vocab_size}, {"process_peak_rss_mb", peak_rss_mb()},
                {"runtime_revision", GEMMAJEV_LLAMA_REVISION},
                {"runtime_patches", runtime_patches()},
                {"kv_cache", {{"enabled", true}, {"reused_tokens", cache.reused_tokens},
                              {"evaluated_tokens", n_tokens - cache.reused_tokens}, {"prompt_tokens", n_tokens},
                              {"reset_reason", cache.reset_reason.empty() ? json(nullptr) : json(cache.reset_reason)}}},
                {"timings_ms", {{"preprocess", preprocess_ms}, {"vision_encode", vision_ms}, {"audio_encode", audio_ms},
                                {"prefill", prefill_ms}, {"score", score_ms}, {"total", milliseconds(started)}}}};
        // Publish token provenance only after decoding and score construction
        // both succeed. The selected candidate itself is never appended to KV.
        // For media requests, provenance stops before the first media boundary.
        // Physical KV still contains the media/suffix until the next trim.
        cached_tokens_ = std::move(prefix_tokens);
        cached_positions_ = n_positions;
        cached_modality_ = modality;
        return result;
    }

    llama_pos decode_text(const std::vector<llama_token> & tokens, size_t offset,
                          llama_pos base_position, bool logits_last) {
        const size_t capacity = llama_n_batch(context_.get());
        while (offset < tokens.size()) {
            const size_t count = std::min(capacity, tokens.size() - offset);
            llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data() + offset), static_cast<int32_t>(count));
            std::vector<llama_pos> positions(count);
            std::vector<int32_t> sequence_counts(count, 1);
            llama_seq_id sequence_id = 0;
            std::vector<llama_seq_id *> sequences(count, &sequence_id);
            for (size_t i = 0; i < count; ++i) positions[i] = base_position + static_cast<llama_pos>(offset + i);
            std::vector<int8_t> output_flags(count, 0);
            output_flags.back() = logits_last && offset + count == tokens.size();
            batch.pos = positions.data();
            batch.n_seq_id = sequence_counts.data();
            batch.seq_id = sequences.data();
            batch.logits = output_flags.data();
            if (llama_decode(context_.get(), batch) != 0) throw std::runtime_error("text prefill failed");
            offset += count;
        }
        return base_position + static_cast<llama_pos>(tokens.size());
    }

    struct CacheDecision {
        size_t reused_tokens = 0;
        std::string reset_reason;
    };

    CacheDecision prepare_cache(const std::vector<llama_token> & tokens, bool requested_reset,
                                const std::string & modality) {
        const bool full_prefix_allowed = modality != "text";
        const auto memory = llama_get_memory(context_.get());
        const bool previous_failed = previous_request_failed_;
        previous_request_failed_ = false;
        auto clear = [&](const char * reason) {
            llama_memory_clear(memory, true);
            cached_tokens_.clear();
            cached_positions_ = 0;
            cached_modality_ = "text";
            return CacheDecision{0, reason};
        };
        if (requested_reset) return clear("requested");
        if (cached_tokens_.empty()) return clear(previous_failed ? "previous_request_failed" : "empty");
        // Keep media prefix reuse within one identical textual task. Different
        // prefill shapes can produce different Metal rounding, so transitions
        // between text/image/audio modes or media-task prompts start afresh.
        if (cached_modality_ != modality) return clear("modality_changed");
        if (full_prefix_allowed && cached_tokens_ != tokens) {
            return clear(modality == "image" ? "image_text_changed" : "audio_text_changed");
        }

        // The pinned API guarantees all positions between min and max exist.
        // Gemma 4's ISWA implementation reports the SWA cache; its full-attention
        // cache is a superset. A nonzero min means old prefix KV was evicted.
        if (llama_memory_seq_pos_min(memory, 0) != 0
                || llama_memory_seq_pos_max(memory, 0) != cached_positions_ - 1) {
            return clear("incomplete_prefix");
        }
        size_t common = 0;
        const size_t available = tokens.empty() ? 0 : tokens.size() - (full_prefix_allowed ? 0 : 1);
        const size_t limit = std::min(cached_tokens_.size(), available);
        while (common < limit && cached_tokens_[common] == tokens[common]) ++common;
        if (common == 0) return clear("no_common_prefix");

        // Text-only requests re-evaluate at least their final token for logits.
        // Image requests can reuse all causal prefix text: fresh image and
        // response-suffix decodes will provide new answer-position logits.
        // In either case, discard every previous position after this prefix.
        if (!llama_memory_seq_rm(memory, 0, static_cast<llama_pos>(common), -1)) {
            return clear("suffix_remove_failed");
        }
        if (llama_memory_seq_pos_min(memory, 0) != 0
                || llama_memory_seq_pos_max(memory, 0) != static_cast<llama_pos>(common - 1)) {
            return clear("incomplete_prefix_after_trim");
        }
        return {common, ""};
    }

    Options opts_;
    Handle<llama_model, llama_model_free> model_{nullptr, llama_model_free};
    Handle<llama_context, llama_free> context_{nullptr, llama_free};
#ifdef GEMMAJEV_VISION
    Handle<mtmd_context, mtmd_free> vision_{nullptr, mtmd_free};
#endif
    const llama_vocab * vocab_ = nullptr;
    double load_ms_ = 0;
    std::vector<llama_token> cached_tokens_;
    llama_pos cached_positions_ = 0;
    std::string cached_modality_ = "text";
    bool previous_request_failed_ = false;
};

void emit(const json & data) {
    // Some vocabulary pieces are incomplete UTF-8 bytes. Replace invalid bytes
    // in diagnostics while preserving every numerical result.
    std::cout << data.dump(-1, ' ', false, json::error_handler_t::replace) << '\n' << std::flush;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        const auto opts = options(argc, argv);
        if (opts.self_test) {
            emit({{"ok", true}, {"runtime_revision", GEMMAJEV_LLAMA_REVISION},
                  {"worker_source_sha256", GEMMAJEV_WORKER_SHA256},
                  {"runtime_patches", runtime_patches()},
                  {"vision_build", vision_build()},
                  {"answer_prefixes", {{"e4b", answer_prefix("e4b")}, {"12b", answer_prefix("12b")}}},
                  {"context_size", opts.context}, {"batch_size", 512}, {"microbatch_size", 128}});
            return 0;
        }
        llama_log_set(log_stderr, nullptr);
#ifdef GEMMAJEV_VISION
        mtmd_helper_log_set(log_stderr, nullptr);
#endif
        ggml_backend_load_all();
        llama_backend_init();
        {
            Worker worker(opts);
            emit(worker.ready());
            std::string line;
            while (std::getline(std::cin, line)) {
                try {
                    emit(worker.score(json::parse(line)));
                } catch (const std::exception & error) {
                    emit({{"ok", false}, {"error", error.what()}});
                }
            }
        }
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        emit({{"ready", false}, {"ok", false}, {"error", error.what()}});
        return 1;
    }
}
