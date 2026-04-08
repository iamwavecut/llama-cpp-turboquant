#include "common.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama-cpp.h"
#include "llama.h"

#include "../src/llama-arch.h"
#include "../src/llama-model.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-spectral.h"

#include <cinttypes>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    std::hash<std::string> hasher;
    std::mt19937 gen(hasher(tensor->name) + *(const size_t *) userdata);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; ++i) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; ++i) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static bool silent_model_load_progress(float, void *) {
    return true;
}

static gguf_context_ptr make_test_gguf_ctx() {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_LLAMA, ret.get());

    const uint32_t n_ctx = 128;
    const uint32_t n_vocab = 128;
    const uint32_t n_embd = 256;
    const uint32_t n_head = 2;
    const uint32_t n_ff = 384;
    const uint32_t n_layer = 2;
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(LLM_ARCH_LLAMA));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,       n_ff);
    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,     false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,               1.0f);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,      n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,   n_head);
    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS,  8.0f);
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,       1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,      n_embd_head);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,   std::vector<uint32_t>({ n_embd_head / 4, n_embd_head / 4, n_embd_head / 4, n_embd_head / 4 }));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,           "no_vocab");

    for (uint32_t il = 0; il < n_layer; ++il) {
        ggml_tensor t;
        std::memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }

    return ret;
}

static llama_model_ptr make_test_model(size_t seed) {
    gguf_context_ptr gguf_ctx = make_test_gguf_ctx();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.progress_callback = silent_model_load_progress;

    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &seed, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }

    return model;
}

static std::vector<float> make_calibration_samples(uint32_t dim, uint32_t n_samples, uint32_t layer, uint32_t head, bool is_key) {
    std::vector<float> flat;
    flat.reserve(size_t(dim) * n_samples);

    const uint32_t semantic_span = is_key ? 20u : 24u;
    const float layer_scale = 1.0f + 0.08f * float(layer);
    const float head_scale = 1.0f + 0.05f * float(head);

    for (uint32_t row = 0; row < n_samples; ++row) {
        const float sign = (row & 1u) ? -1.0f : 1.0f;
        for (uint32_t col = 0; col < dim; ++col) {
            float value = 0.0f;
            if (col < semantic_span) {
                value = sign * layer_scale * head_scale * (2.4f - 0.045f * float(col));
                value += 0.03f * float(int(row % 3) - 1);
            } else {
                value = 0.0125f * float(int((row + layer + head) % 5) - 2) * float(int(col % 7) - 3);
            }
            if (!is_key) {
                value *= 0.85f;
                value += 0.01f * float(int((row + col) % 4) - 1);
            }
            flat.push_back(value);
        }
    }

    return flat;
}

static std::filesystem::path make_unique_temp_path(const char * stem, const char * ext) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    return std::filesystem::temp_directory_path() /
        (std::string(stem) + "-" + std::to_string(dis(gen)) + ext);
}

static std::filesystem::path write_spectral_sidecar(const llama_model * model) {
    llama_spectral_artifact artifact;
    artifact.source_architecture = model->arch_name();

    const auto & tensor_map = llama_internal_get_tensor_map(model);
    std::vector<const ggml_tensor *> tensors;
    tensors.reserve(tensor_map.size());
    for (const auto & [_, tensor] : tensor_map) {
        tensors.push_back(tensor);
    }
    artifact.source_digest = llama_spectral_compute_tensor_digest(model->arch_name().c_str(), tensors);

    llama_spectral_calibrate_params key_nonuniform_params;
    key_nonuniform_params.kind = LLAMA_SPECTRAL_KIND_K;
    key_nonuniform_params.profile = LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM;
    key_nonuniform_params.base_bits = 4;
    key_nonuniform_params.min_tail_bits = 2;
    key_nonuniform_params.max_semantic_bits = 6;

    llama_spectral_calibrate_params key_selcorr_params = key_nonuniform_params;
    key_selcorr_params.profile = LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM_SELCORR;
    key_selcorr_params.correction_dim = 16;

    llama_spectral_calibrate_params value_params;
    value_params.kind = LLAMA_SPECTRAL_KIND_V;
    value_params.profile = LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM;
    value_params.base_bits = 4;
    value_params.min_tail_bits = 2;
    value_params.max_semantic_bits = 6;

    std::string err;
    const uint32_t n_layer = 2;
    const uint32_t n_head = 2;
    const uint32_t head_dim = 128;

    artifact.entries.reserve(size_t(n_layer) * n_head * 3);

    for (uint32_t il = 0; il < n_layer; ++il) {
        for (uint32_t h = 0; h < n_head; ++h) {
            const std::vector<float> key_samples = make_calibration_samples(head_dim, 12, il, h, true);
            for (const llama_spectral_calibrate_params * key_params : { &key_nonuniform_params, &key_selcorr_params }) {
                llama_spectral_entry key_entry;
                if (!llama_spectral_calibrate_entry(
                            "cache_k_l" + std::to_string(il) + ".h" + std::to_string(h),
                            key_samples,
                            12,
                            head_dim,
                            *key_params,
                            key_entry,
                            &err)) {
                    throw std::runtime_error("failed to calibrate K entry: " + err);
                }
                artifact.entries.push_back(std::move(key_entry));
            }

            llama_spectral_entry value_entry;
            const std::vector<float> value_samples = make_calibration_samples(head_dim, 12, il, h, false);
            if (!llama_spectral_calibrate_entry(
                        "cache_v_l" + std::to_string(il) + ".h" + std::to_string(h),
                        value_samples,
                        12,
                        head_dim,
                        value_params,
                        value_entry,
                        &err)) {
                throw std::runtime_error("failed to calibrate V entry: " + err);
            }
            artifact.entries.push_back(std::move(value_entry));
        }
    }

    const std::filesystem::path sidecar_path = make_unique_temp_path("llama-skv-sidecar", ".gguf");
    if (!llama_spectral_save_gguf(artifact, sidecar_path.string(), &err)) {
        throw std::runtime_error("failed to save spectral sidecar: " + err);
    }

    return sidecar_path;
}

static std::vector<float> extract_tensor_samples(const ggml_tensor * tensor, uint32_t max_rows = 16) {
    const int64_t dim = tensor->ne[0];
    if (dim <= 0 || ggml_n_dims(tensor) < 2) {
        return {};
    }

    const int64_t total_rows = ggml_nelements(tensor) / dim;
    if (total_rows < 2) {
        return {};
    }

    const uint32_t n_rows = std::min<uint32_t>(max_rows, total_rows);
    std::vector<float> samples(size_t(n_rows) * dim);

    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> raw(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, raw.data(), 0, ggml_nbytes(tensor));
        std::copy(raw.begin(), raw.begin() + samples.size(), samples.begin());
        return samples;
    }

    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> raw(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, raw.data(), 0, ggml_nbytes(tensor));
        for (size_t i = 0; i < samples.size(); ++i) {
            samples[i] = ggml_fp16_to_fp32(raw[i]);
        }
        return samples;
    }

    return {};
}

static std::filesystem::path write_weight_spectral_sidecar(const llama_model * model) {
    llama_spectral_artifact artifact;
    artifact.source_architecture = model->arch_name();

    const auto & tensor_map = llama_internal_get_tensor_map(model);
    std::vector<const ggml_tensor *> tensors;
    tensors.reserve(tensor_map.size());
    for (const auto & [_, tensor] : tensor_map) {
        tensors.push_back(tensor);
    }

    llama_spectral_calibrate_params nonuniform_params;
    nonuniform_params.kind = LLAMA_SPECTRAL_KIND_WEIGHT;
    nonuniform_params.profile = LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM;
    nonuniform_params.base_bits = 4;
    nonuniform_params.min_tail_bits = 2;
    nonuniform_params.max_semantic_bits = 6;

    llama_spectral_calibrate_params selcorr_params = nonuniform_params;
    selcorr_params.profile = LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM_SELCORR;
    selcorr_params.correction_dim = 8;

    std::string err;
    for (const auto & [name, tensor] : tensor_map) {
        const std::vector<float> samples = extract_tensor_samples(tensor);
        if (samples.empty()) {
            continue;
        }

        for (const llama_spectral_calibrate_params * params : { &nonuniform_params, &selcorr_params }) {
            llama_spectral_entry entry;
            if (!llama_spectral_calibrate_entry(
                        name,
                        samples,
                        uint32_t(samples.size() / tensor->ne[0]),
                        uint32_t(tensor->ne[0]),
                        *params,
                        entry,
                        &err)) {
                throw std::runtime_error("failed to calibrate weight entry for " + name + ": " + err);
            }
            artifact.entries.push_back(std::move(entry));
        }
    }

    if (artifact.entries.empty()) {
        throw std::runtime_error("weight spectral artifact is empty");
    }

    const std::filesystem::path sidecar_path = make_unique_temp_path("llama-sq-sidecar", ".gguf");
    if (!llama_spectral_save_gguf(artifact, sidecar_path.string(), &err)) {
        throw std::runtime_error("failed to save weight spectral sidecar: " + err);
    }

    return sidecar_path;
}

static void clear_sidecar_digest(const std::filesystem::path & sidecar_path) {
    llama_spectral_artifact artifact;
    std::string err;
    if (!llama_spectral_load_gguf(sidecar_path.string(), artifact, &err)) {
        throw std::runtime_error("failed to reload spectral sidecar: " + err);
    }
    artifact.source_digest.clear();
    if (!llama_spectral_save_gguf(artifact, sidecar_path.string(), &err)) {
        throw std::runtime_error("failed to rewrite spectral sidecar: " + err);
    }
}

static std::filesystem::path write_model_file(const llama_model * model) {
    const std::filesystem::path model_path = make_unique_temp_path("llama-spectral-model", ".gguf");
    llama_model_save_to_file(model, model_path.string().c_str());
    return model_path;
}

static llama_model_ptr load_model_from_file_cpu(
        const std::filesystem::path & model_path,
        const std::filesystem::path * spectral_sidecar = nullptr) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.progress_callback = silent_model_load_progress;
    const std::string spectral_sidecar_str = spectral_sidecar ? spectral_sidecar->string() : std::string();
    model_params.spectral_calibration = spectral_sidecar ? spectral_sidecar_str.c_str() : nullptr;
    model_params.spectral_profile = "auto";

    llama_model_ptr model(llama_model_load_from_file(model_path.string().c_str(), model_params));
    if (!model) {
        throw std::runtime_error("failed to load model from file: " + model_path.string());
    }

    return model;
}

static void require_skv_context_init_failure(
        llama_model * model,
        const std::filesystem::path & sidecar_path,
        bool offload_kqv,
        bool with_sidecar,
        const char * what) {
    llama_context_params ctx_params = llama_context_default_params();
    const std::string sidecar_path_str = sidecar_path.string();
    ctx_params.n_ctx = 64;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;
    ctx_params.type_k = GGML_TYPE_SKV4_0;
    ctx_params.type_v = GGML_TYPE_SKV4_0;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    ctx_params.offload_kqv = offload_kqv;
    ctx_params.spectral_calibration = with_sidecar ? sidecar_path_str.c_str() : nullptr;
    ctx_params.spectral_profile = "auto";

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx != nullptr) {
        llama_free(ctx);
        throw std::runtime_error(std::string("unexpected SKV init success for ") + what);
    }
}

static llama_context_ptr make_skv_context(
        llama_model * model,
        const std::filesystem::path & sidecar_path,
        const char * spectral_profile = "auto") {
    llama_context_params ctx_params = llama_context_default_params();
    const std::string sidecar_path_str = sidecar_path.string();
    ctx_params.n_ctx = 64;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;
    ctx_params.type_k = GGML_TYPE_SKV4_0;
    ctx_params.type_v = GGML_TYPE_SKV4_0;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    ctx_params.offload_kqv = false;
    ctx_params.spectral_calibration = sidecar_path_str.c_str();
    ctx_params.spectral_profile = spectral_profile;

    llama_context_ptr ctx(llama_init_from_model(model, ctx_params));
    if (!ctx) {
        throw std::runtime_error("failed to create SKV llama context");
    }

    return ctx;
}

static void require_sq_model_gpu_offload_rejected(const std::filesystem::path & model_path) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 1;
    model_params.progress_callback = silent_model_load_progress;

    llama_model * model = llama_model_load_from_file(model_path.string().c_str(), model_params);
    if (model != nullptr) {
        llama_model_free(model);
        throw std::runtime_error("unexpected SQ model load success with GPU offload enabled");
    }
}

static llama_context_ptr make_plain_context(llama_model * model) {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 64;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;
    ctx_params.type_k = GGML_TYPE_F16;
    ctx_params.type_v = GGML_TYPE_F16;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    ctx_params.offload_kqv = false;

    llama_context_ptr ctx(llama_init_from_model(model, ctx_params));
    if (!ctx) {
        throw std::runtime_error("failed to create plain llama context");
    }

    return ctx;
}

static void decode_prefix(llama_context * ctx, const std::vector<llama_token> & tokens) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], i, { 0 }, i + 1 == tokens.size());
    }
    batch.n_tokens = tokens.size();

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode prefix");
    }

    llama_batch_free(batch);
}

static std::vector<float> decode_one_and_get_logits(llama_context * ctx, llama_token token, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, token, pos, { 0 }, true);
    batch.n_tokens = 1;

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode continuation token");
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float * logits_ptr = llama_get_logits_ith(ctx, 0);
    std::vector<float> logits(logits_ptr, logits_ptr + n_vocab);

    llama_batch_free(batch);
    return logits;
}

static float max_abs_diff(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    float result = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        result = std::max(result, std::fabs(lhs[i] - rhs[i]));
    }
    return result;
}

static float cosine_similarity(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    double dot = 0.0;
    double lhs_norm = 0.0;
    double rhs_norm = 0.0;

    for (size_t i = 0; i < lhs.size(); ++i) {
        dot += double(lhs[i]) * double(rhs[i]);
        lhs_norm += double(lhs[i]) * double(lhs[i]);
        rhs_norm += double(rhs[i]) * double(rhs[i]);
    }

    if (lhs_norm == 0.0 || rhs_norm == 0.0) {
        return 0.0f;
    }

    return float(dot / std::sqrt(lhs_norm * rhs_norm));
}

static void require_finite_logits(const std::vector<float> & logits, const char * what) {
    for (float value : logits) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(std::string("non-finite logits in ") + what);
        }
    }
}

int main() {
    common_init();

    try {
        const size_t seed = 1337;
        const std::vector<llama_token> prefix = { 1, 2, 3, 4, 5, 6 };
        const std::vector<llama_token> continuation = { 7, 8 };

        llama_model_ptr model = make_test_model(seed);
        const std::filesystem::path fp_model_path = write_model_file(model.get());
        llama_model_ptr fp_file_model = load_model_from_file_cpu(fp_model_path);
        const std::filesystem::path kv_sidecar_path = write_spectral_sidecar(model.get());
        const std::filesystem::path weight_sidecar_path = write_weight_spectral_sidecar(fp_file_model.get());
        clear_sidecar_digest(weight_sidecar_path);
        const std::filesystem::path session_path = make_unique_temp_path("llama-skv-session", ".bin");
        const std::string weight_sidecar_path_str = weight_sidecar_path.string();
        std::vector<std::filesystem::path> sq_model_paths;

        try {
            require_skv_context_init_failure(model.get(), kv_sidecar_path, false, false, "missing sidecar");
            require_skv_context_init_failure(model.get(), kv_sidecar_path, true, true, "KV offload");

            llama_context_ptr ctx_selcorr = make_skv_context(model.get(), kv_sidecar_path, "selcorr");
            if (llama_context_memory_size(ctx_selcorr.get()) == 0 || llama_context_compute_size(ctx_selcorr.get()) == 0) {
                std::fprintf(stderr, "explicit selcorr profile failed to initialize paper-style SKV context\n");
                return 1;
            }

            llama_context_ptr ctx_save = make_skv_context(model.get(), kv_sidecar_path);
            const uint64_t context_size = llama_context_memory_size(ctx_save.get());
            const uint64_t compute_size = llama_context_compute_size(ctx_save.get());
            const uint64_t runtime_size = llama_model_size(model.get()) + context_size + compute_size;
            if (context_size == 0 || compute_size == 0 || runtime_size <= llama_model_size(model.get())) {
                std::fprintf(stderr, "invalid context memory metrics for SKV context\n");
                return 1;
            }
            decode_prefix(ctx_save.get(), prefix);

            const size_t state_size = llama_state_get_size(ctx_save.get());
            if (state_size == 0) {
                std::fprintf(stderr, "empty state size for SKV context\n");
                return 1;
            }

            if (!llama_state_save_file(ctx_save.get(), session_path.string().c_str(), prefix.data(), prefix.size())) {
                std::fprintf(stderr, "failed to save SKV session state\n");
                return 1;
            }

            llama_context_ptr ctx_load = make_skv_context(model.get(), kv_sidecar_path);
            std::vector<llama_token> restored(prefix.size(), 0);
            size_t restored_count = 0;
            if (!llama_state_load_file(ctx_load.get(), session_path.string().c_str(), restored.data(), restored.size(), &restored_count)) {
                std::fprintf(stderr, "failed to load SKV session state\n");
                return 1;
            }

            if (restored_count != prefix.size() || !std::equal(prefix.begin(), prefix.end(), restored.begin())) {
                std::fprintf(stderr, "restored prompt tokens mismatch\n");
                return 1;
            }

            for (size_t i = 0; i < continuation.size(); ++i) {
                const llama_pos pos = prefix.size() + i;
                const std::vector<float> logits_save = decode_one_and_get_logits(ctx_save.get(), continuation[i], pos);
                const std::vector<float> logits_load = decode_one_and_get_logits(ctx_load.get(), continuation[i], pos);

                if (logits_save.size() != logits_load.size()) {
                    std::fprintf(stderr, "logit size mismatch after SKV state restore\n");
                    return 1;
                }

                const float diff = max_abs_diff(logits_save, logits_load);
                if (diff > 1e-5f) {
                    std::fprintf(stderr, "SKV state restore divergence too large at step %zu: %.8f\n", i, diff);
                    return 1;
                }
            }

            llama_context_ptr ctx_fp = make_plain_context(model.get());
            decode_prefix(ctx_fp.get(), prefix);
            const std::vector<float> fp_logits = decode_one_and_get_logits(ctx_fp.get(), continuation.front(), prefix.size());
            require_finite_logits(fp_logits, "fp baseline");
            clear_sidecar_digest(kv_sidecar_path);

            for (const char * profile : { "nonuniform", "selcorr" }) {
                const std::filesystem::path sq_model_path =
                        make_unique_temp_path(std::string("llama-sq4-model-").append(profile).c_str(), ".gguf");
                sq_model_paths.push_back(sq_model_path);

                llama_model_quantize_params qparams = llama_model_quantize_default_params();
                qparams.nthread = 1;
                qparams.ftype = LLAMA_FTYPE_MOSTLY_SQ4_1S;
                qparams.pure = true;
                qparams.spectral_calibration = weight_sidecar_path_str.c_str();
                qparams.spectral_profile = profile;

                if (llama_model_quantize(fp_model_path.string().c_str(), sq_model_path.string().c_str(), &qparams) != 0) {
                    throw std::runtime_error(std::string("failed to quantize SQ model for profile ") + profile);
                }

                require_sq_model_gpu_offload_rejected(sq_model_path);

                llama_model_ptr sq_model = load_model_from_file_cpu(sq_model_path, &weight_sidecar_path);
                if (!sq_model->has_spectral_weights() || sq_model->n_spectral_weights() == 0) {
                    throw std::runtime_error(std::string("loaded SQ model is missing spectral runtime sidecar metadata for profile ") + profile);
                }

                llama_context_ptr ctx_sq_plain = make_plain_context(sq_model.get());
                decode_prefix(ctx_sq_plain.get(), prefix);
                const std::vector<float> sq_logits = decode_one_and_get_logits(ctx_sq_plain.get(), continuation.front(), prefix.size());
                require_finite_logits(sq_logits, profile);

                const float sq_cos = cosine_similarity(fp_logits, sq_logits);
                if (sq_cos < 0.70f) {
                    throw std::runtime_error(std::string("SQ logits cosine too low for profile ") + profile);
                }

                std::filesystem::remove(sq_model_path);
            }
        } catch (...) {
            for (const auto & sq_model_path : sq_model_paths) {
                std::filesystem::remove(sq_model_path);
            }
            std::filesystem::remove(kv_sidecar_path);
            std::filesystem::remove(weight_sidecar_path);
            std::filesystem::remove(fp_model_path);
            std::filesystem::remove(session_path);
            throw;
        }

        for (const auto & sq_model_path : sq_model_paths) {
            std::filesystem::remove(sq_model_path);
        }
        std::filesystem::remove(kv_sidecar_path);
        std::filesystem::remove(weight_sidecar_path);
        std::filesystem::remove(fp_model_path);
        std::filesystem::remove(session_path);
    } catch (const std::exception & err) {
        std::fprintf(stderr, "%s\n", err.what());
        return 1;
    }

    return 0;
}
