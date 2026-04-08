#pragma once

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;

enum llama_spectral_kind : uint32_t {
    LLAMA_SPECTRAL_KIND_K      = 0,
    LLAMA_SPECTRAL_KIND_V      = 1,
    LLAMA_SPECTRAL_KIND_WEIGHT = 2,
};

enum llama_spectral_profile : uint32_t {
    LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM         = 0,
    LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM_SELCORR = 1,
};

struct llama_spectral_calibrate_params {
    llama_spectral_kind    kind             = LLAMA_SPECTRAL_KIND_K;
    llama_spectral_profile profile          = LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM;
    uint32_t               base_bits        = 4;
    uint32_t               min_tail_bits    = 1;
    uint32_t               max_semantic_bits = 8;
    uint32_t               correction_dim   = 0;
};

struct llama_spectral_calibration_metrics {
    double covariance_ms = 0.0;
    double eigen_ms      = 0.0;
    double rotate_ms     = 0.0;
    double codebook_ms   = 0.0;
    double total_ms      = 0.0;
    bool used_snapshot   = false;
    uint32_t spectral_rank = 0;
};

struct llama_spectral_entry {
    std::string            name;
    llama_spectral_kind    kind             = LLAMA_SPECTRAL_KIND_K;
    llama_spectral_profile profile          = LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM;
    uint32_t               dim              = 0;
    uint32_t               split            = 0;
    uint32_t               semantic_bits    = 0;
    uint32_t               tail_bits        = 0;
    uint32_t               correction_dim   = 0;
    float                  d_eff            = 0.0f;
    std::vector<float>     eigenvalues;
    std::vector<float>     basis;
    std::vector<float>     semantic_codebook;
    std::vector<float>     tail_codebook;
};

struct llama_spectral_artifact {
    uint32_t                        version = 1;
    std::string                     source_architecture;
    std::string                     source_digest;
    std::vector<llama_spectral_entry> entries;
};

struct llama_spectral_encoded_vector {
    std::vector<uint16_t> semantic_codes;
    std::vector<uint16_t> tail_codes;
    std::vector<float>    correction;
};

const char * llama_spectral_kind_name(llama_spectral_kind kind);
const char * llama_spectral_profile_name(llama_spectral_profile profile);

bool llama_spectral_kind_parse(const std::string & text, llama_spectral_kind & kind);
bool llama_spectral_profile_parse(const std::string & text, llama_spectral_profile & profile);

float    llama_spectral_compute_d_eff(const std::vector<float> & eigenvalues);
uint32_t llama_spectral_choose_split(float d_eff, uint32_t dim);

bool llama_spectral_calibrate_entry(
        const std::string & name,
        const std::vector<float> & samples,
        uint32_t n_samples,
        uint32_t dim,
        const llama_spectral_calibrate_params & params,
        llama_spectral_entry & out,
        std::string * err = nullptr,
        llama_spectral_calibration_metrics * metrics = nullptr);

bool llama_spectral_encode_vector(
        const llama_spectral_entry & entry,
        const float * input,
        size_t input_size,
        llama_spectral_encoded_vector & encoded,
        std::string * err = nullptr);

bool llama_spectral_decode_vector(
        const llama_spectral_entry & entry,
        const llama_spectral_encoded_vector & encoded,
        std::vector<float> & output,
        std::string * err = nullptr);

bool llama_spectral_validate_artifact(
        const llama_spectral_artifact & artifact,
        std::string * err = nullptr);

std::string llama_spectral_compute_tensor_digest(
        const char * architecture,
        const ggml_context * ctx);

std::string llama_spectral_compute_tensor_digest(
        const char * architecture,
        const std::vector<const ggml_tensor *> & tensors);

bool llama_spectral_filter_artifact(
        const llama_spectral_artifact & artifact,
        llama_spectral_kind kind,
        const char * profile_filter,
        llama_spectral_artifact & filtered,
        std::string * err = nullptr);

bool llama_spectral_prepare_weight_entry(
        const llama_spectral_entry & entry,
        ggml_type type,
        llama_spectral_entry & prepared,
        std::string * err = nullptr);

bool llama_spectral_prepare_kv_entry(
        const llama_spectral_entry & entry,
        ggml_type type,
        llama_spectral_entry & prepared,
        std::string * err = nullptr);

bool llama_spectral_save_gguf(
        const llama_spectral_artifact & artifact,
        const std::string & path,
        std::string * err = nullptr);

bool llama_spectral_load_gguf(
        const std::string & path,
        llama_spectral_artifact & artifact,
        std::string * err = nullptr);

bool llama_spectral_embed_gguf_metadata(
        const llama_spectral_artifact & artifact,
        struct gguf_context * ctx,
        std::string * err = nullptr);

bool llama_spectral_extract_gguf_metadata(
        const struct gguf_context * ctx,
        llama_spectral_artifact & artifact,
        std::string * err = nullptr);
