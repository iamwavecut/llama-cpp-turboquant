#include "chat.h"
#include "common.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "llama-cpp.h"
#include "llama.h"
#include "llama-spectral.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-model.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct kv_cli_params {
    std::string model;
    std::string output;
    std::string system_prompt;
    std::string chat_template;
    std::vector<std::string> prompt_files;
    uint32_t synthetic_samples = 0;
    uint32_t ctx_size = 8192;
    uint32_t batch_size = 512;
    uint32_t ubatch_size = 512;
    uint32_t base_bits = 4;
    uint32_t correction_dim = 16;
    int32_t n_gpu_layers = 0;
    std::string device;
};

struct sample_bucket {
    std::string name;
    llama_spectral_kind kind = LLAMA_SPECTRAL_KIND_K;
    uint32_t dim = 0;
    uint32_t n_samples = 0;
    std::vector<float> values;
};

void print_usage(const char * argv0) {
    std::printf(
            "usage: %s --model model.gguf --output kv-spectral.gguf --prompt-file prompt.txt [options]\n"
            "options:\n"
            "  --prompt-file <file>                 may be repeated; prompts are accumulated into one sidecar\n"
            "  --synthetic-samples <n>              build a deterministic synthetic sidecar without prompt evaluation\n"
            "  --system-prompt-file <file>          optional system prompt\n"
            "  --chat-template-file <file>          optional chat template override\n"
            "  --ctx-size <n>                       context size used for prompt evaluation\n"
            "  --batch-size <n>                     logical batch size for prompt evaluation\n"
            "  --ubatch-size <n>                    physical batch size for prompt evaluation\n"
            "  --n-gpu-layers <n>                   offload up to n layers during prompt evaluation\n"
            "  --device <dev1,dev2,...>             optional backend device list for prompt evaluation\n"
            "  --base-bits <n>                      base bit budget before semantic/tail split\n"
            "  --correction-dim <n>                 semantic correction width for K selcorr profiles\n",
            argv0);
}

std::string read_text_file(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open file: " + path);
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (!text.empty() && text.back() == '\n') {
        text.pop_back();
    }
    return text;
}

std::vector<ggml_backend_dev_t> parse_device_list(const std::string & value) {
    std::vector<ggml_backend_dev_t> devices;
    if (value.empty() || value == "none") {
        return devices;
    }

    size_t start = 0;
    while (start < value.size()) {
        size_t comma = value.find(',', start);
        const std::string dev_name = value.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (dev_name.empty()) {
            throw std::runtime_error("empty device name in --device list");
        }
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name.c_str());
        if (dev == nullptr) {
            throw std::runtime_error("invalid device in --device: " + dev_name);
        }
        devices.push_back(dev);
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }

    devices.push_back(nullptr);
    return devices;
}

bool parse_args(int argc, char ** argv, kv_cli_params & params) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "-m" || arg == "--model") {
            params.model = require_value("--model");
        } else if (arg == "-o" || arg == "--output") {
            params.output = require_value("--output");
        } else if (arg == "-f" || arg == "--prompt-file") {
            params.prompt_files.emplace_back(require_value("--prompt-file"));
        } else if (arg == "--synthetic-samples") {
            params.synthetic_samples = std::strtoul(require_value("--synthetic-samples"), nullptr, 10);
        } else if (arg == "--system-prompt-file") {
            params.system_prompt = read_text_file(require_value("--system-prompt-file"));
        } else if (arg == "--chat-template-file") {
            params.chat_template = read_text_file(require_value("--chat-template-file"));
        } else if (arg == "--ctx-size") {
            params.ctx_size = std::strtoul(require_value("--ctx-size"), nullptr, 10);
        } else if (arg == "--batch-size") {
            params.batch_size = std::strtoul(require_value("--batch-size"), nullptr, 10);
        } else if (arg == "--ubatch-size") {
            params.ubatch_size = std::strtoul(require_value("--ubatch-size"), nullptr, 10);
        } else if (arg == "--n-gpu-layers") {
            params.n_gpu_layers = std::strtol(require_value("--n-gpu-layers"), nullptr, 10);
        } else if (arg == "--device") {
            params.device = require_value("--device");
        } else if (arg == "--base-bits") {
            params.base_bits = std::strtoul(require_value("--base-bits"), nullptr, 10);
        } else if (arg == "--correction-dim") {
            params.correction_dim = std::strtoul(require_value("--correction-dim"), nullptr, 10);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return false;
        }
    }

    return !params.model.empty() && !params.output.empty() && (!params.prompt_files.empty() || params.synthetic_samples > 0);
}

std::string make_chat_prompt(
        const llama_model * model,
        const std::string & chat_template,
        const std::string & system_prompt,
        const std::string & user_prompt) {
    common_chat_templates_ptr tmpls = common_chat_templates_init(model, chat_template);

    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    inputs.add_generation_prompt = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    inputs.enable_thinking = false;

    if (!system_prompt.empty()) {
        common_chat_msg system_msg;
        system_msg.role = "system";
        system_msg.content = system_prompt;
        inputs.messages.push_back(std::move(system_msg));
    }

    common_chat_msg user_msg;
    user_msg.role = "user";
    user_msg.content = user_prompt;
    inputs.messages.push_back(std::move(user_msg));

    return common_chat_templates_apply(tmpls.get(), inputs).prompt;
}

float read_scalar(const ggml_tensor * tensor, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const char * base = static_cast<const char *>(tensor->data);
    const char * ptr = base + i0 * tensor->nb[0] + i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3];

    switch (tensor->type) {
        case GGML_TYPE_F32:
            return *reinterpret_cast<const float *>(ptr);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(ptr));
        case GGML_TYPE_BF16:
            return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(ptr));
        default:
            throw std::runtime_error("unsupported KV calibration tensor type: " + std::string(ggml_type_name(tensor->type)));
    }
}

std::vector<float> extract_head_samples(
        const ggml_tensor * tensor,
        uint32_t head_idx,
        uint32_t n_tokens,
        uint32_t head_dim) {
    std::vector<float> samples;
    samples.reserve(size_t(n_tokens) * head_dim);

    const bool transposed = tensor->ne[0] == n_tokens;
    for (uint32_t pos = 0; pos < n_tokens; ++pos) {
        for (uint32_t d = 0; d < head_dim; ++d) {
            samples.push_back(transposed
                    ? read_scalar(tensor, pos, head_idx, d, 0)
                    : read_scalar(tensor, d, head_idx, pos, 0));
        }
    }

    return samples;
}

void append_samples(
        std::unordered_map<std::string, sample_bucket> & buckets,
        const std::string & name,
        llama_spectral_kind kind,
        uint32_t dim,
        uint32_t n_samples,
        std::vector<float> && values) {
    auto [it, inserted] = buckets.try_emplace(name);
    sample_bucket & bucket = it->second;

    if (inserted) {
        bucket.name = name;
        bucket.kind = kind;
        bucket.dim = dim;
    } else if (bucket.kind != kind || bucket.dim != dim) {
        throw std::runtime_error("inconsistent calibration bucket for " + name);
    }

    bucket.n_samples += n_samples;
    bucket.values.insert(bucket.values.end(), values.begin(), values.end());
}

std::vector<float> make_synthetic_calibration_samples(
        uint32_t dim,
        uint32_t n_samples,
        uint32_t layer,
        uint32_t head,
        bool is_key) {
    std::vector<float> flat;
    flat.reserve(size_t(dim) * n_samples);

    const uint32_t semantic_span = is_key ? std::min<uint32_t>(dim, 24u) : std::min<uint32_t>(dim, 28u);
    const float layer_scale = 1.0f + 0.06f * float(layer);
    const float head_scale = 1.0f + 0.04f * float(head);

    for (uint32_t row = 0; row < n_samples; ++row) {
        const float sign = (row & 1u) ? -1.0f : 1.0f;
        for (uint32_t col = 0; col < dim; ++col) {
            float value = 0.0f;
            if (col < semantic_span) {
                value = sign * layer_scale * head_scale * (2.1f - 0.03f * float(col));
                value += 0.04f * float(int(row % 5) - 2);
            } else {
                value = 0.015f * float(int((row + layer + head) % 7) - 3) * float(int(col % 9) - 4);
            }
            if (!is_key) {
                value *= 0.8f;
                value += 0.01f * float(int((row + col) % 6) - 2);
            }
            flat.push_back(value);
        }
    }

    return flat;
}

size_t accumulate_synthetic_samples(
        llama_model * model,
        uint32_t synthetic_samples,
        std::unordered_map<std::string, sample_bucket> & buckets) {
    size_t total_rows = 0;

    for (uint32_t il = 0; il < model->hparams.n_layer; ++il) {
        const uint32_t n_head_kv = model->hparams.n_head_kv(il);
        if (n_head_kv == 0) {
            continue;
        }

        const uint32_t head_dim_k = model->hparams.n_embd_head_k(il);
        const uint32_t head_dim_v = model->hparams.n_embd_head_v(il);

        for (uint32_t h = 0; h < n_head_kv; ++h) {
            append_samples(
                    buckets,
                    "cache_k_l" + std::to_string(il) + ".h" + std::to_string(h),
                    LLAMA_SPECTRAL_KIND_K,
                    head_dim_k,
                    synthetic_samples,
                    make_synthetic_calibration_samples(head_dim_k, synthetic_samples, il, h, true));

            append_samples(
                    buckets,
                    "cache_v_l" + std::to_string(il) + ".h" + std::to_string(h),
                    LLAMA_SPECTRAL_KIND_V,
                    head_dim_v,
                    synthetic_samples,
                    make_synthetic_calibration_samples(head_dim_v, synthetic_samples, il, h, false));

            total_rows += size_t(synthetic_samples) * 2;
        }
    }

    return total_rows;
}

void accumulate_prompt_samples_from_cache(
        llama_kv_cache * kv,
        llama_model * internal_model,
        ggml_context * view_ctx,
        uint32_t n_tokens,
        std::unordered_map<std::string, sample_bucket> & buckets) {
    llama_kv_cache::slot_info sinfo;
    sinfo.s0 = 0;
    sinfo.s1 = 0;
    sinfo.resize(1);
    sinfo.strm[0] = 0;
    sinfo.idxs[0].push_back(0);

    for (uint32_t il = 0; il < internal_model->hparams.n_layer; ++il) {
        const uint32_t n_head_kv = internal_model->hparams.n_head_kv(il);
        if (n_head_kv == 0) {
            continue;
        }

        const uint32_t head_dim_k = internal_model->hparams.n_embd_head_k(il);
        const uint32_t head_dim_v = internal_model->hparams.n_embd_head_v(il);

        const ggml_tensor * k = nullptr;
        const ggml_tensor * v = nullptr;
        try {
            k = kv->get_k(view_ctx, il, n_tokens, sinfo);
            v = kv->get_v(view_ctx, il, n_tokens, sinfo);
        } catch (const std::out_of_range &) {
            continue;
        }
        if (k == nullptr || v == nullptr) {
            throw std::runtime_error("failed to create KV views for layer " + std::to_string(il));
        }

        for (uint32_t h = 0; h < n_head_kv; ++h) {
            append_samples(
                    buckets,
                    "cache_k_l" + std::to_string(il) + ".h" + std::to_string(h),
                    LLAMA_SPECTRAL_KIND_K,
                    head_dim_k,
                    n_tokens,
                    extract_head_samples(k, h, n_tokens, head_dim_k));

            append_samples(
                    buckets,
                    "cache_v_l" + std::to_string(il) + ".h" + std::to_string(h),
                    LLAMA_SPECTRAL_KIND_V,
                    head_dim_v,
                    n_tokens,
                    extract_head_samples(v, h, n_tokens, head_dim_v));
        }
    }
}

void accumulate_prompt_samples(
        llama_context * ctx,
        uint32_t n_tokens,
        std::unordered_map<std::string, sample_bucket> & buckets) {
    auto * mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        throw std::runtime_error("current model context has no memory module");
    }

    auto * internal_model = const_cast<llama_model *>(static_cast<const llama_model *>(llama_get_model(ctx)));

    ggml_init_params params = {
        /*.mem_size   =*/ 1024u * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr view_ctx(ggml_init(params));
    if (!view_ctx) {
        throw std::runtime_error("failed to allocate temporary ggml context");
    }

    if (auto * kv = dynamic_cast<llama_kv_cache *>(mem)) {
        accumulate_prompt_samples_from_cache(kv, internal_model, view_ctx.get(), n_tokens, buckets);
        return;
    }

    if (auto * kv_iswa = dynamic_cast<llama_kv_cache_iswa *>(mem)) {
        accumulate_prompt_samples_from_cache(kv_iswa->get_base(), internal_model, view_ctx.get(), n_tokens, buckets);
        accumulate_prompt_samples_from_cache(kv_iswa->get_swa(),  internal_model, view_ctx.get(), n_tokens, buckets);
        return;
    }

    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        accumulate_prompt_samples_from_cache(hybrid->get_mem_attn(), internal_model, view_ctx.get(), n_tokens, buckets);
        return;
    }

    if (auto * hybrid_iswa = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        auto * kv_iswa = hybrid_iswa->get_mem_attn();
        accumulate_prompt_samples_from_cache(kv_iswa->get_base(), internal_model, view_ctx.get(), n_tokens, buckets);
        accumulate_prompt_samples_from_cache(kv_iswa->get_swa(),  internal_model, view_ctx.get(), n_tokens, buckets);
        return;
    }

    throw std::runtime_error("current model context does not expose a supported KV memory layout");
}

} // namespace

int main(int argc, char ** argv) {
    kv_cli_params args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    llama_backend_init();

    std::vector<ggml_backend_dev_t> devices = parse_device_list(args.device);
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;
    if (!devices.empty()) {
        mparams.devices = devices.data();
    }

    llama_model_ptr model(llama_model_load_from_file(args.model.c_str(), mparams));
    if (!model) {
        std::fprintf(stderr, "failed to load model: %s\n", args.model.c_str());
        return 1;
    }

    std::unordered_map<std::string, sample_bucket> buckets;
    size_t total_sample_rows = 0;

    if (args.synthetic_samples > 0) {
        total_sample_rows = accumulate_synthetic_samples(model.get(), args.synthetic_samples, buckets);
        std::printf("accumulated %zu synthetic KV rows\n", total_sample_rows);
    } else {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = args.ctx_size;
        cparams.n_batch = args.batch_size;
        cparams.n_ubatch = std::min(args.batch_size, args.ubatch_size);
        cparams.type_k = GGML_TYPE_F16;
        cparams.type_v = GGML_TYPE_F16;
        cparams.no_perf = true;
        cparams.offload_kqv = false;

        llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
        if (!ctx) {
            std::fprintf(stderr, "failed to create context\n");
            return 1;
        }

        for (const std::string & prompt_file : args.prompt_files) {
            llama_memory_clear(llama_get_memory(ctx.get()), true);

            const std::string user_prompt = read_text_file(prompt_file);
            const std::string prompt = make_chat_prompt(model.get(), args.chat_template, args.system_prompt, user_prompt);
            const std::vector<llama_token> tokens = common_tokenize(ctx.get(), prompt, true, true);
            if (tokens.empty()) {
                continue;
            }

            int n_past = 0;
            if (!common_prompt_batch_decode(ctx.get(), tokens, n_past, args.batch_size, "", false)) {
                std::fprintf(stderr, "failed to decode prompt: %s\n", prompt_file.c_str());
                return 1;
            }
            if (n_past <= 0) {
                continue;
            }

            total_sample_rows += size_t(n_past);
            accumulate_prompt_samples(ctx.get(), uint32_t(n_past), buckets);
            std::printf("accumulated %d tokens from %s\n", n_past, prompt_file.c_str());
        }
    }

    if (buckets.empty()) {
        std::fprintf(stderr, "no KV samples collected\n");
        return 1;
    }

    auto * internal_model = model.get();
    llama_spectral_artifact artifact;
    artifact.source_architecture = internal_model->arch_name();

    const auto & tensor_map = llama_internal_get_tensor_map(internal_model);
    std::vector<const ggml_tensor *> tensors;
    tensors.reserve(tensor_map.size());
    for (const auto & [_, tensor] : tensor_map) {
        tensors.push_back(tensor);
    }
    artifact.source_digest = llama_spectral_compute_tensor_digest(artifact.source_architecture.c_str(), tensors);

    llama_spectral_calibrate_params k_nonuniform;
    k_nonuniform.kind = LLAMA_SPECTRAL_KIND_K;
    k_nonuniform.profile = LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM;
    k_nonuniform.base_bits = args.base_bits;
    k_nonuniform.min_tail_bits = 2;
    k_nonuniform.max_semantic_bits = 6;

    llama_spectral_calibrate_params k_selcorr = k_nonuniform;
    k_selcorr.profile = LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM_SELCORR;
    k_selcorr.correction_dim = args.correction_dim;

    llama_spectral_calibrate_params v_nonuniform;
    v_nonuniform.kind = LLAMA_SPECTRAL_KIND_V;
    v_nonuniform.profile = LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM;
    v_nonuniform.base_bits = args.base_bits;
    v_nonuniform.min_tail_bits = 2;
    v_nonuniform.max_semantic_bits = 6;

    artifact.entries.reserve(buckets.size() * 2);
    for (const auto & [_, bucket] : buckets) {
        std::string err;
        if (bucket.kind == LLAMA_SPECTRAL_KIND_K) {
            for (const llama_spectral_calibrate_params * params : { &k_nonuniform, &k_selcorr }) {
                llama_spectral_entry entry;
                if (!llama_spectral_calibrate_entry(
                            bucket.name,
                            bucket.values,
                            bucket.n_samples,
                            bucket.dim,
                            *params,
                            entry,
                            &err)) {
                    std::fprintf(stderr, "failed to calibrate %s: %s\n", bucket.name.c_str(), err.c_str());
                    return 1;
                }
                artifact.entries.push_back(std::move(entry));
            }
        } else {
            llama_spectral_entry entry;
            if (!llama_spectral_calibrate_entry(
                        bucket.name,
                        bucket.values,
                        bucket.n_samples,
                        bucket.dim,
                        v_nonuniform,
                        entry,
                        &err)) {
                std::fprintf(stderr, "failed to calibrate %s: %s\n", bucket.name.c_str(), err.c_str());
                return 1;
            }
            artifact.entries.push_back(std::move(entry));
        }
    }

    std::string err;
    if (!llama_spectral_save_gguf(artifact, args.output, &err)) {
        std::fprintf(stderr, "failed to save KV spectral sidecar: %s\n", err.c_str());
        return 1;
    }

    std::printf("saved %zu entries from %zu calibration rows into %s\n",
            artifact.entries.size(), total_sample_rows, args.output.c_str());
    llama_backend_free();
    return 0;
}
