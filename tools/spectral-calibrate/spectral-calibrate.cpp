#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-model-loader.h"
#include "llama.h"
#include "llama-spectral.h"

#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct spectral_cli_params {
    std::string input;
    std::string output;
    std::string match;
    std::string source_digest;
    std::string source_architecture;
    llama_spectral_kind kind = LLAMA_SPECTRAL_KIND_K;
    uint32_t split_heads = 1;
    std::vector<llama_spectral_profile> profiles = {
        LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM,
        LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM_SELCORR,
    };
    uint32_t base_bits = 4;
    uint32_t correction_dim = 0;
    uint32_t max_rows = 0;
    uint32_t n_threads = 0;
};

using spectral_clock = std::chrono::steady_clock;

void llama_null_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

void print_usage(const char * argv0) {
    std::printf(
            "usage: %s --input samples.gguf --output spectral.gguf --match <regex> [options]\n"
            "options:\n"
            "  --kind <k|v|weight>                  calibration target kind\n"
            "  --profile <all|nonuniform|selcorr>   emitted profile set\n"
            "  --split-heads <n>                    emit per-head entries as <tensor>.hN\n"
            "  --base-bits <n>                      base bit budget before semantic/tail split\n"
            "  --correction-dim <n>                 override semantic correction width\n"
            "  --max-rows <n>                       evenly sample up to n rows per tensor before calibration\n"
            "  --threads <n>                        number of tensors to calibrate in parallel\n"
            "  --source-digest <text>               optional compatibility fingerprint\n",
            argv0);
}

bool parse_args(int argc, char ** argv, spectral_cli_params & params) {
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
        } else if (arg == "-i" || arg == "--input") {
            params.input = require_value("--input");
        } else if (arg == "-o" || arg == "--output") {
            params.output = require_value("--output");
        } else if (arg == "-m" || arg == "--match") {
            params.match = require_value("--match");
        } else if (arg == "--kind") {
            if (!llama_spectral_kind_parse(require_value("--kind"), params.kind)) {
                std::fprintf(stderr, "unsupported --kind value\n");
                return false;
            }
        } else if (arg == "--profile") {
            const std::string value = require_value("--profile");
            params.profiles.clear();
            if (value == "all" || value == "auto") {
                params.profiles = {
                    LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM,
                    LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM_SELCORR,
                };
            } else {
                llama_spectral_profile profile;
                if (!llama_spectral_profile_parse(value, profile)) {
                    std::fprintf(stderr, "unsupported --profile value\n");
                    return false;
                }
                params.profiles.push_back(profile);
            }
        } else if (arg == "--split-heads") {
            params.split_heads = std::strtoul(require_value("--split-heads"), nullptr, 10);
        } else if (arg == "--base-bits") {
            params.base_bits = std::strtoul(require_value("--base-bits"), nullptr, 10);
        } else if (arg == "--correction-dim") {
            params.correction_dim = std::strtoul(require_value("--correction-dim"), nullptr, 10);
        } else if (arg == "--max-rows") {
            params.max_rows = std::strtoul(require_value("--max-rows"), nullptr, 10);
        } else if (arg == "--threads") {
            params.n_threads = std::strtoul(require_value("--threads"), nullptr, 10);
        } else if (arg == "--source-digest") {
            params.source_digest = require_value("--source-digest");
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return false;
        }
    }

    return !params.input.empty() && !params.output.empty() && !params.match.empty();
}

double elapsed_ms(const spectral_clock::time_point & start, const spectral_clock::time_point & end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void log_line(std::mutex & log_mutex, const char * fmt, ...) {
    std::lock_guard<std::mutex> guard(log_mutex);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fflush(stderr);
}

std::vector<uint32_t> select_row_indices_evenly(uint32_t n_rows, uint32_t max_rows) {
    if (max_rows == 0 || n_rows <= max_rows) {
        std::vector<uint32_t> rows(n_rows);
        for (uint32_t i = 0; i < n_rows; ++i) {
            rows[i] = i;
        }
        return rows;
    }

    std::vector<uint32_t> rows(max_rows);
    if (max_rows == 1) {
        rows[0] = 0;
        return rows;
    }

    for (uint32_t row = 0; row < max_rows; ++row) {
        rows[row] = (uint64_t) row * (n_rows - 1) / (max_rows - 1);
    }
    return rows;
}

size_t tensor_row_offset_bytes(const ggml_tensor * tensor, uint32_t logical_row) {
    uint64_t remaining = logical_row;
    const uint64_t d1 = tensor->ne[1] > 0 ? uint64_t(tensor->ne[1]) : 1;
    const uint64_t d2 = tensor->ne[2] > 0 ? uint64_t(tensor->ne[2]) : 1;

    const uint64_t i1 = remaining % d1;
    remaining /= d1;
    const uint64_t i2 = remaining % d2;
    remaining /= d2;
    const uint64_t i3 = remaining;

    return size_t(i1) * tensor->nb[1] + size_t(i2) * tensor->nb[2] + size_t(i3) * tensor->nb[3];
}

bool tensor_rows_to_f32(
        const ggml_tensor * tensor,
        const std::vector<uint32_t> & rows,
        std::vector<float> & values) {
    const uint32_t row_dim = tensor->ne[0];
    values.resize(size_t(rows.size()) * row_dim);

    const ggml_type_traits * traits = ggml_is_quantized(tensor->type) ? ggml_get_type_traits(tensor->type) : nullptr;
    if (ggml_is_quantized(tensor->type) && (traits == nullptr || traits->to_float == nullptr)) {
        return false;
    }

    for (size_t i = 0; i < rows.size(); ++i) {
        const char * src = static_cast<const char *>(tensor->data) + tensor_row_offset_bytes(tensor, rows[i]);
        float * dst = values.data() + i * row_dim;

        switch (tensor->type) {
            case GGML_TYPE_F32:
                std::memcpy(dst, src, size_t(row_dim) * sizeof(float));
                break;
            case GGML_TYPE_F16:
                ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(src), dst, row_dim);
                break;
            case GGML_TYPE_BF16:
                ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(src), dst, row_dim);
                break;
            default:
                if (!ggml_is_quantized(tensor->type)) {
                    return false;
                }
                traits->to_float(src, dst, row_dim);
                break;
        }
    }

    return true;
}

bool compute_model_digest_from_file(
        const std::string & model_path,
        const std::string & architecture,
        std::string & digest,
        std::string & err) {
    llama_log_set(llama_null_log_callback, nullptr);
    try {
        std::vector<std::string> splits;
        llama_model_loader ml(
                /*metadata*/ nullptr,
                /*set_tensor_data*/ nullptr,
                /*set_tensor_data_ud*/ nullptr,
                model_path,
                splits,
                /*file*/ nullptr,
                /*use_mmap*/ false,
                /*use_direct_io*/ false,
                /*check_tensors*/ true,
                /*no_alloc*/ false,
                /*param_overrides_p*/ nullptr,
                /*param_tensor_buft_overrides_p*/ nullptr);

        std::vector<const ggml_tensor *> tensors;
        tensors.reserve(ml.weights_map.size());
        for (const auto & [_, weight] : ml.weights_map) {
            tensors.push_back(weight.tensor);
        }

        digest = llama_spectral_compute_tensor_digest(architecture.c_str(), tensors);
        return true;
    } catch (const std::exception & ex) {
        err = ex.what();
        return false;
    }
}

} // namespace

int main(int argc, char ** argv) {
    const auto run_t0 = spectral_clock::now();
    spectral_cli_params params;
    if (!parse_args(argc, argv, params)) {
        print_usage(argv[0]);
        return 1;
    }

    ggml_context * ggml_ctx_raw = nullptr;
    gguf_init_params gguf_params = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &ggml_ctx_raw,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(params.input.c_str(), gguf_params));
    ggml_context_ptr ggml_ctx(ggml_ctx_raw);
    if (!gguf_ctx || !ggml_ctx) {
        std::fprintf(stderr, "failed to open GGUF input: %s\n", params.input.c_str());
        return 1;
    }

    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id >= 0) {
        params.source_architecture = gguf_get_val_str(gguf_ctx.get(), arch_id);
    }
    if (params.split_heads == 0) {
        std::fprintf(stderr, "--split-heads must be >= 1\n");
        return 1;
    }
    if (params.n_threads == 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        params.n_threads = hw == 0 ? 1u : std::min(hw, 6u);
    }

#if defined(__APPLE__)
    if (params.n_threads > 1 && std::getenv("VECLIB_MAXIMUM_THREADS") == nullptr) {
        setenv("VECLIB_MAXIMUM_THREADS", "1", 0);
    }
#endif

    const std::regex match_re(params.match);
    llama_spectral_artifact artifact;
    artifact.source_architecture = params.source_architecture;
    if (params.source_digest.empty()) {
        std::string digest_err;
        if (!compute_model_digest_from_file(params.input, params.source_architecture, artifact.source_digest, digest_err)) {
            std::fprintf(stderr, "%s\n", digest_err.c_str());
            return 1;
        }
    } else {
        artifact.source_digest = params.source_digest;
    }

    std::vector<ggml_tensor *> matched_tensors;
    for (ggml_tensor * tensor = ggml_get_first_tensor(ggml_ctx.get()); tensor; tensor = ggml_get_next_tensor(ggml_ctx.get(), tensor)) {
        if (!std::regex_search(tensor->name, match_re)) {
            continue;
        }
        if (tensor->ne[0] <= 1 || ggml_nelements(tensor) <= tensor->ne[0]) {
            continue;
        }
        matched_tensors.push_back(tensor);
    }

    if (matched_tensors.empty()) {
        std::fprintf(stderr, "no tensors matched regex: %s\n", params.match.c_str());
        return 1;
    }

    const size_t total_jobs = matched_tensors.size() * params.split_heads * params.profiles.size();
    std::mutex log_mutex;
    log_line(log_mutex,
            "spectral-calibrate: matched %zu tensors, %zu jobs, kind=%s, profiles=%zu, split-heads=%u, max-rows=%u, threads=%u\n",
            matched_tensors.size(),
            total_jobs,
            llama_spectral_kind_name(params.kind),
            params.profiles.size(),
            params.split_heads,
            params.max_rows,
            params.n_threads);
#if defined(__APPLE__)
    if (params.n_threads > 1) {
        log_line(log_mutex, "spectral-calibrate: forcing Accelerate to 1 thread per worker via VECLIB_MAXIMUM_THREADS=1\n");
    }
#endif

    std::vector<std::vector<llama_spectral_entry>> tensor_entries(matched_tensors.size());
    std::atomic<size_t> next_tensor{0};
    std::atomic<size_t> started_jobs{0};
    std::atomic<size_t> finished_jobs{0};
    std::atomic<bool> has_error{false};
    std::mutex error_mutex;
    std::string first_error;

    auto fail = [&](const std::string & message) {
        std::lock_guard<std::mutex> guard(error_mutex);
        if (first_error.empty()) {
            first_error = message;
            has_error.store(true);
        }
    };

    auto worker = [&]() {
        while (true) {
            if (has_error.load()) {
                return;
            }
            const size_t tensor_index = next_tensor.fetch_add(1);
            if (tensor_index >= matched_tensors.size()) {
                return;
            }

            ggml_tensor * tensor = matched_tensors[tensor_index];
            const uint32_t dim_full = tensor->ne[0];
            const uint32_t n_rows_full = ggml_nelements(tensor) / dim_full;
            const std::vector<uint32_t> sampled_rows = select_row_indices_evenly(n_rows_full, params.max_rows);
            const uint32_t n_samples = sampled_rows.size();

            std::vector<float> values;
            if (!tensor_rows_to_f32(tensor, sampled_rows, values)) {
                fail(std::string("skipping ") + tensor->name + ": unsupported tensor type " + ggml_type_name(tensor->type));
                return;
            }

            if (dim_full % params.split_heads != 0) {
                fail("tensor " + std::string(tensor->name) + " width " + std::to_string(dim_full) +
                        " is not divisible by --split-heads=" + std::to_string(params.split_heads));
                return;
            }

            const uint32_t head_dim = dim_full / params.split_heads;
            std::vector<llama_spectral_entry> local_entries;

            for (uint32_t head = 0; head < params.split_heads; ++head) {
                std::vector<float> head_values;
                const std::vector<float> * calibration_values = &values;
                std::string entry_name = tensor->name;

                if (params.split_heads > 1) {
                    head_values.resize(size_t(n_samples) * head_dim);
                    for (uint32_t row = 0; row < n_samples; ++row) {
                        const float * src = values.data() + size_t(row) * dim_full + size_t(head) * head_dim;
                        float * dst = head_values.data() + size_t(row) * head_dim;
                        std::memcpy(dst, src, size_t(head_dim) * sizeof(float));
                    }
                    calibration_values = &head_values;
                    entry_name += ".h" + std::to_string(head);
                }

                for (llama_spectral_profile profile : params.profiles) {
                    llama_spectral_calibrate_params cal;
                    cal.kind = params.kind;
                    cal.profile = profile;
                    cal.base_bits = params.base_bits;
                    cal.correction_dim = params.correction_dim;

                    const size_t job_index = started_jobs.fetch_add(1) + 1;
                    log_line(log_mutex,
                            "[%zu/%zu] spectral-calibrate: start %s profile=%s type=%s rows=%u/%u dim=%u\n",
                            job_index,
                            total_jobs,
                            entry_name.c_str(),
                            llama_spectral_profile_name(profile),
                            ggml_type_name(tensor->type),
                            n_samples,
                            n_rows_full,
                            head_dim);

                    llama_spectral_entry entry;
                    llama_spectral_calibration_metrics metrics;
                    std::string err;
                    if (!llama_spectral_calibrate_entry(entry_name, *calibration_values, n_samples, head_dim, cal, entry, &err, &metrics)) {
                        fail("failed to calibrate " + entry_name + ": " + err);
                        return;
                    }
                    const size_t done = finished_jobs.fetch_add(1) + 1;
                    const double elapsed_s = elapsed_ms(run_t0, spectral_clock::now()) / 1000.0;
                    const double avg_s = elapsed_s / double(done);
                    const double eta_s = avg_s * double(total_jobs - done);
                    log_line(log_mutex,
                            "[%zu/%zu] spectral-calibrate: done  %s profile=%s mode=%s rank=%u total=%.1fms cov=%.1fms eig=%.1fms rot=%.1fms codebook=%.1fms split=%u d_eff=%.2f elapsed=%.1fs eta=%.1fs\n",
                            done,
                            total_jobs,
                            entry_name.c_str(),
                            llama_spectral_profile_name(profile),
                            metrics.used_snapshot ? "snapshot" : "dense",
                            metrics.spectral_rank,
                            metrics.total_ms,
                            metrics.covariance_ms,
                            metrics.eigen_ms,
                            metrics.rotate_ms,
                            metrics.codebook_ms,
                            entry.split,
                            entry.d_eff,
                            elapsed_s,
                            eta_s);
                    local_entries.push_back(std::move(entry));
                }
            }

            tensor_entries[tensor_index] = std::move(local_entries);
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(params.n_threads);
    for (uint32_t i = 0; i < params.n_threads; ++i) {
        workers.emplace_back(worker);
    }
    for (auto & thread : workers) {
        thread.join();
    }

    if (!first_error.empty()) {
        std::fprintf(stderr, "%s\n", first_error.c_str());
        return 1;
    }

    for (auto & entries : tensor_entries) {
        for (auto & entry : entries) {
            artifact.entries.push_back(std::move(entry));
        }
    }

    std::string err;
    if (!llama_spectral_save_gguf(artifact, params.output, &err)) {
        std::fprintf(stderr, "failed to write spectral artifact: %s\n", err.c_str());
        return 1;
    }

    const auto run_t1 = spectral_clock::now();
    std::printf("wrote %zu spectral entries from %zu tensors to %s in %.2fs\n",
            artifact.entries.size(), matched_tensors.size(), params.output.c_str(), elapsed_ms(run_t0, run_t1) / 1000.0);
    return 0;
}
