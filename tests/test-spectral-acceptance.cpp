#include "common.h"

#include "get-model.h"

#include "ggml-backend.h"
#include "llama-cpp.h"
#include "llama.h"

#include "../src/llama-model.h"
#include "../src/llama-spectral.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static bool silent_model_load_progress(float, void *) {
    return true;
}

static std::filesystem::path make_unique_temp_path(const char * stem, const char * ext) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    return std::filesystem::temp_directory_path() /
        (std::string(stem) + "-" + std::to_string(dis(gen)) + ext);
}

static std::string read_file(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

static size_t count_occurrences(const std::string & haystack, const std::string & needle) {
    size_t count = 0;
    size_t pos = 0;
    while ((pos = haystack.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

static std::string shell_quote(const std::string & text) {
    std::string quoted = "'";
    for (char ch : text) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

static int run_shell_checked(const std::string & command, const std::filesystem::path & log_path) {
    const std::string wrapped = command + " > " + shell_quote(log_path.string()) + " 2>&1";
    const int rc = std::system(wrapped.c_str());
    if (rc != 0) {
        throw std::runtime_error("command failed (" + std::to_string(rc) + "):\n" + command + "\n--- log ---\n" + read_file(log_path));
    }
    return rc;
}

static std::vector<float> make_calibration_samples(uint32_t dim, uint32_t n_samples, uint32_t layer, uint32_t head, bool is_key) {
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

static std::filesystem::path write_kv_spectral_sidecar(const llama_model * model) {
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

    llama_spectral_calibrate_params value_params = key_nonuniform_params;
    value_params.kind = LLAMA_SPECTRAL_KIND_V;

    std::string err;
    for (uint32_t il = 0; il < model->hparams.n_layer; ++il) {
        const uint32_t n_head_kv = model->hparams.n_head_kv(il);
        if (n_head_kv == 0) {
            continue;
        }

        const uint32_t head_dim_k = model->hparams.n_embd_head_k(il);
        const uint32_t head_dim_v = model->hparams.n_embd_head_v(il);

        for (uint32_t h = 0; h < n_head_kv; ++h) {
            const std::vector<float> key_samples = make_calibration_samples(head_dim_k, 16, il, h, true);
            for (const llama_spectral_calibrate_params * key_params : { &key_nonuniform_params, &key_selcorr_params }) {
                llama_spectral_entry key_entry;
                if (!llama_spectral_calibrate_entry(
                            "cache_k_l" + std::to_string(il) + ".h" + std::to_string(h),
                            key_samples,
                            16,
                            head_dim_k,
                            *key_params,
                            key_entry,
                            &err)) {
                    throw std::runtime_error("failed to calibrate key spectral entry: " + err);
                }
                artifact.entries.push_back(std::move(key_entry));
            }

            llama_spectral_entry value_entry;
            const std::vector<float> value_samples = make_calibration_samples(head_dim_v, 16, il, h, false);
            if (!llama_spectral_calibrate_entry(
                        "cache_v_l" + std::to_string(il) + ".h" + std::to_string(h),
                        value_samples,
                        16,
                        head_dim_v,
                        value_params,
                        value_entry,
                        &err)) {
                throw std::runtime_error("failed to calibrate value spectral entry: " + err);
            }
            artifact.entries.push_back(std::move(value_entry));
        }
    }

    if (artifact.entries.empty()) {
        throw std::runtime_error("empty SKV spectral artifact");
    }

    const std::filesystem::path sidecar_path = make_unique_temp_path("llama-skv-acceptance", ".gguf");
    if (!llama_spectral_save_gguf(artifact, sidecar_path.string(), &err)) {
        throw std::runtime_error("failed to save SKV sidecar: " + err);
    }

    return sidecar_path;
}

static std::filesystem::path write_dataset_file() {
    const std::filesystem::path dataset_path = make_unique_temp_path("llama-spectral-dataset", ".txt");
    std::ofstream out(dataset_path);
    if (!out) {
        throw std::runtime_error("failed to create dataset file");
    }

    for (int i = 0; i < 64; ++i) {
        out << "The quick brown fox studies spectral quantization and measures cache compression quality. ";
        out << "TurboQuant remains the legacy baseline, while SpectralQuant uses rotation, regime splitting, and selective correction on keys only. ";
        out << "This tiny dataset exists only for an end-to-end smoke test of benchmark and perplexity tooling.\n";
    }

    return dataset_path;
}

static uintmax_t file_size_checked(const std::filesystem::path & path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) {
        throw std::runtime_error("failed to stat file: " + path.string());
    }
    return size;
}

static llama_model_ptr load_model_cpu(const std::filesystem::path & model_path, const std::filesystem::path * spectral_sidecar = nullptr) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.progress_callback = silent_model_load_progress;

    std::string sidecar_text;
    if (spectral_sidecar != nullptr) {
        sidecar_text = spectral_sidecar->string();
        model_params.spectral_calibration = sidecar_text.c_str();
        model_params.spectral_profile = "auto";
    }

    return llama_model_ptr(llama_model_load_from_file(model_path.string().c_str(), model_params));
}

int main(int argc, char ** argv) {
    common_init();

    try {
        const std::filesystem::path model_path = get_model_or_exit(argc, argv);
        const std::filesystem::path repo_root = LLAMA_PROJECT_SOURCE_DIR;
        const std::filesystem::path build_root = LLAMA_PROJECT_BINARY_DIR;
        const std::filesystem::path bench_bin = build_root / "bin" / "llama-bench";
        const std::filesystem::path perplexity_bin = build_root / "bin" / "llama-perplexity";
        const std::filesystem::path calibrate_bin = build_root / "bin" / "llama-spectral-calibrate";
        const std::filesystem::path quantize_bin = build_root / "bin" / "llama-quantize";
        const std::filesystem::path bench_script = repo_root / "scripts" / "spectralquant-bench-matrix.py";
        const std::filesystem::path perplexity_script = repo_root / "scripts" / "spectralquant-perplexity-matrix.py";

        llama_model_ptr model = load_model_cpu(model_path);
        if (!model) {
            throw std::runtime_error("failed to load real GGUF fixture model");
        }

        const std::filesystem::path sidecar_path = write_kv_spectral_sidecar(model.get());
        const std::filesystem::path dataset_path = write_dataset_file();
        const std::filesystem::path weight_sidecar_path = make_unique_temp_path("llama-sq-weight-sidecar", ".gguf");
        const std::filesystem::path sq_model_path = make_unique_temp_path("llama-sq4-model", ".gguf");
        const std::filesystem::path sq_cal_log_path = make_unique_temp_path("llama-sq-weight-calibrate", ".log");
        const std::filesystem::path sq_quant_log_path = make_unique_temp_path("llama-sq-weight-quantize", ".log");
        const std::filesystem::path bench_json_path = make_unique_temp_path("llama-spectral-bench", ".json");
        const std::filesystem::path bench_log_path = make_unique_temp_path("llama-spectral-bench", ".log");
        const std::filesystem::path ppl_json_path = make_unique_temp_path("llama-spectral-ppl", ".json");
        const std::filesystem::path ppl_log_path = make_unique_temp_path("llama-spectral-ppl", ".log");

        try {
            const uintmax_t baseline_size = file_size_checked(model_path);

            const std::string sq_cal_cmd =
                    shell_quote(calibrate_bin.string()) +
                    " --input " + shell_quote(model_path.string()) +
                    " --output " + shell_quote(weight_sidecar_path.string()) +
                    " --kind weight" +
                    " --profile nonuniform" +
                    " --match " + shell_quote(".*") +
                    " --max-rows 64" +
                    " --threads 1";
            run_shell_checked(sq_cal_cmd, sq_cal_log_path);

            const std::string sq_quant_cmd =
                    shell_quote(quantize_bin.string()) +
                    " --allow-requantize" +
                    " --spectral-calibration " + shell_quote(weight_sidecar_path.string()) +
                    " --spectral-profile nonuniform" +
                    " " + shell_quote(model_path.string()) +
                    " " + shell_quote(sq_model_path.string()) +
                    " SQ4_1S 1";
            run_shell_checked(sq_quant_cmd, sq_quant_log_path);

            const uintmax_t sq_model_size = file_size_checked(sq_model_path);
            if (sq_model_size >= baseline_size) {
                throw std::runtime_error(
                        "SQ4 tiny size gate failed: sq=" + std::to_string(sq_model_size) +
                        " baseline=" + std::to_string(baseline_size));
            }

            if (load_model_cpu(sq_model_path) != nullptr) {
                throw std::runtime_error("SQ4 tiny model unexpectedly loaded without an external spectral sidecar");
            }

            llama_model_ptr sq_model = load_model_cpu(sq_model_path, &weight_sidecar_path);
            if (!sq_model) {
                throw std::runtime_error("failed to load SQ4 tiny model with external spectral sidecar");
            }

            const std::string bench_cmd =
                    "python3 " + shell_quote(bench_script.string()) +
                    " --bench-bin " + shell_quote(bench_bin.string()) +
                    " --legacy-model " + shell_quote(model_path.string()) +
                    " --spectral-calibration " + shell_quote(sidecar_path.string()) +
                    " --cache-type skv4_0" +
                    " --require-context-reduction" +
                    " --prompt 32 --gen 16 --batch 64 --ubatch 32 --threads 1 --reps 1" +
                    " --output-json " + shell_quote(bench_json_path.string());
            run_shell_checked(bench_cmd, bench_log_path);

            const std::string bench_json = read_file(bench_json_path);
            if (count_occurrences(bench_json, "\"scenario\":") != 6) {
                throw std::runtime_error("unexpected spectral bench scenario count");
            }
            if (bench_json.find("legacy_kv_fp_weights") == std::string::npos ||
                bench_json.find("spectral_kv_nonuniform_fp_weights") == std::string::npos ||
                bench_json.find("spectral_kv_selcorr_fp_weights") == std::string::npos) {
                throw std::runtime_error("spectral bench JSON is missing expected paper-style KV scenarios");
            }
            if (bench_json.find("\"context_delta_pct\": -") == std::string::npos) {
                throw std::runtime_error("spectral bench JSON is missing a negative context delta");
            }

            const std::string ppl_cmd =
                    "python3 " + shell_quote(perplexity_script.string()) +
                    " --perplexity-bin " + shell_quote(perplexity_bin.string()) +
                    " --legacy-model " + shell_quote(model_path.string()) +
                    " --spectral-calibration " + shell_quote(sidecar_path.string()) +
                    " --dataset " + shell_quote(dataset_path.string()) +
                    " --cache-type skv4_0" +
                    " --require-kld-metrics" +
                    " --ctx-size 256 --batch 256 --ubatch 64 --chunks 1 --threads 1" +
                    " --output-json " + shell_quote(ppl_json_path.string());
            run_shell_checked(ppl_cmd, ppl_log_path);

            const std::string ppl_json = read_file(ppl_json_path);
            if (count_occurrences(ppl_json, "\"scenario\":") != 3) {
                throw std::runtime_error("unexpected spectral perplexity scenario count");
            }
            if (ppl_json.find("spectral_kv_nonuniform_fp_weights") == std::string::npos ||
                ppl_json.find("spectral_kv_selcorr_fp_weights") == std::string::npos) {
                throw std::runtime_error("spectral perplexity JSON is missing expected paper-style KV scenarios");
            }
            if (ppl_json.find("\"mean_kld_value\":") == std::string::npos) {
                throw std::runtime_error("spectral perplexity JSON is missing KLD metrics");
            }
        } catch (...) {
            std::filesystem::remove(sidecar_path);
            std::filesystem::remove(dataset_path);
            std::filesystem::remove(weight_sidecar_path);
            std::filesystem::remove(sq_model_path);
            std::filesystem::remove(sq_cal_log_path);
            std::filesystem::remove(sq_quant_log_path);
            std::filesystem::remove(bench_json_path);
            std::filesystem::remove(bench_log_path);
            std::filesystem::remove(ppl_json_path);
            std::filesystem::remove(ppl_log_path);
            throw;
        }

        std::filesystem::remove(sidecar_path);
        std::filesystem::remove(dataset_path);
        std::filesystem::remove(weight_sidecar_path);
        std::filesystem::remove(sq_model_path);
        std::filesystem::remove(sq_cal_log_path);
        std::filesystem::remove(sq_quant_log_path);
        std::filesystem::remove(bench_json_path);
        std::filesystem::remove(bench_log_path);
        std::filesystem::remove(ppl_json_path);
        std::filesystem::remove(ppl_log_path);
    } catch (const std::exception & err) {
        std::fprintf(stderr, "%s\n", err.what());
        return 1;
    }

    return 0;
}
