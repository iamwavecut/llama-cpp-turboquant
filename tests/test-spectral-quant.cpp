#include "llama-spectral.h"

#include "ggml-cpp.h"
#include "ggml-quants.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

static std::vector<float> make_samples() {
    const std::vector<std::vector<float>> rows = {
        {  3.0f,  2.9f,  0.20f, -0.10f },
        {  2.8f,  3.1f,  0.10f,  0.00f },
        { -3.0f, -3.1f, -0.10f,  0.10f },
        { -2.9f, -2.8f,  0.00f, -0.20f },
        {  3.2f,  3.0f,  0.30f,  0.10f },
        { -3.1f, -3.2f, -0.20f,  0.00f },
        {  2.9f,  2.7f,  0.15f, -0.05f },
        { -2.7f, -2.9f, -0.05f,  0.05f },
    };

    std::vector<float> flat;
    flat.reserve(rows.size() * rows.front().size());
    for (const auto & row : rows) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}

static std::vector<float> make_sq_samples(uint32_t dim, uint32_t n_samples) {
    std::vector<float> flat;
    flat.reserve(size_t(dim) * n_samples);
    for (uint32_t row = 0; row < n_samples; ++row) {
        for (uint32_t col = 0; col < dim; ++col) {
            const float sign = (row & 1) ? -1.0f : 1.0f;
            const float semantic = col < 6 ? sign * (2.5f - 0.15f * col) : 0.0f;
            const float tail = col >= 6 ? 0.02f * float((int(row % 5) - 2)) * float((col % 7) - 3) : 0.0f;
            flat.push_back(semantic + tail);
        }
    }
    return flat;
}

static float mse(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    float total = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const float d = lhs[i] - rhs[i];
        total += d * d;
    }
    return total / lhs.size();
}

static float max_abs_diff(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    float total = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        total = std::max(total, std::fabs(lhs[i] - rhs[i]));
    }
    return total;
}

static float dot_product(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    float total = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        total += lhs[i] * rhs[i];
    }
    return total;
}

int main() {
    const std::vector<float> samples = make_samples();
    const uint32_t dim = 4;
    const uint32_t n_samples = samples.size() / dim;

    llama_spectral_calibrate_params base;
    base.kind = LLAMA_SPECTRAL_KIND_WEIGHT;
    base.profile = LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM;
    base.base_bits = 3;

    llama_spectral_calibrate_params selcorr = base;
    selcorr.profile = LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM_SELCORR;

    llama_spectral_entry base_entry;
    llama_spectral_entry selcorr_entry;
    std::string err;

    if (!llama_spectral_calibrate_entry("toy", samples, n_samples, dim, base, base_entry, &err)) {
        std::fprintf(stderr, "base calibration failed: %s\n", err.c_str());
        return 1;
    }
    if (!llama_spectral_calibrate_entry("toy", samples, n_samples, dim, selcorr, selcorr_entry, &err)) {
        std::fprintf(stderr, "selcorr calibration failed: %s\n", err.c_str());
        return 1;
    }

    if (base_entry.split == 0 || base_entry.split > dim) {
        std::fprintf(stderr, "invalid split\n");
        return 1;
    }
    if (selcorr_entry.correction_dim == 0) {
        std::fprintf(stderr, "selcorr profile must keep a correction path\n");
        return 1;
    }

    const std::vector<float> probe = { 2.95f, 2.85f, 0.12f, -0.08f };
    llama_spectral_encoded_vector base_encoded;
    llama_spectral_encoded_vector selcorr_encoded;
    std::vector<float> base_decoded;
    std::vector<float> selcorr_decoded;

    if (!llama_spectral_encode_vector(base_entry, probe.data(), probe.size(), base_encoded, &err)) {
        std::fprintf(stderr, "base encode failed: %s\n", err.c_str());
        return 1;
    }
    if (!llama_spectral_decode_vector(base_entry, base_encoded, base_decoded, &err)) {
        std::fprintf(stderr, "base decode failed: %s\n", err.c_str());
        return 1;
    }
    if (!llama_spectral_encode_vector(selcorr_entry, probe.data(), probe.size(), selcorr_encoded, &err)) {
        std::fprintf(stderr, "selcorr encode failed: %s\n", err.c_str());
        return 1;
    }
    if (!llama_spectral_decode_vector(selcorr_entry, selcorr_encoded, selcorr_decoded, &err)) {
        std::fprintf(stderr, "selcorr decode failed: %s\n", err.c_str());
        return 1;
    }

    if (mse(selcorr_decoded, probe) > mse(base_decoded, probe) + 1e-6f) {
        std::fprintf(stderr, "selective correction should not be worse than plain nonuniform quantization\n");
        return 1;
    }

    llama_spectral_artifact artifact;
    artifact.source_architecture = "toy-arch";
    artifact.source_digest = "toy-digest";
    artifact.entries.push_back(base_entry);
    artifact.entries.push_back(selcorr_entry);

    if (!llama_spectral_validate_artifact(artifact, &err)) {
        std::fprintf(stderr, "artifact validation failed: %s\n", err.c_str());
        return 1;
    }

    {
        ggml_init_params init = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 8 + 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr digest_ctx(ggml_init(init));
        if (!digest_ctx) {
            std::fprintf(stderr, "failed to allocate digest context\n");
            return 1;
        }
        ggml_tensor * a = ggml_new_tensor_2d(digest_ctx.get(), GGML_TYPE_F32, 4, 8);
        ggml_tensor * b = ggml_new_tensor_1d(digest_ctx.get(), GGML_TYPE_F32, 16);
        ggml_set_name(a, "tensor_a");
        ggml_set_name(b, "tensor_b");

        const std::string digest0 = llama_spectral_compute_tensor_digest("toy-arch", digest_ctx.get());
        const std::string digest1 = llama_spectral_compute_tensor_digest("toy-arch", digest_ctx.get());
        if (digest0 != digest1 || digest0.empty()) {
            std::fprintf(stderr, "tensor digest must be deterministic\n");
            return 1;
        }

        ggml_init_params init_changed = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 8 + 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr digest_ctx_changed(ggml_init(init_changed));
        ggml_tensor * a_changed = ggml_new_tensor_2d(digest_ctx_changed.get(), GGML_TYPE_F32, 5, 8);
        ggml_tensor * b_changed = ggml_new_tensor_1d(digest_ctx_changed.get(), GGML_TYPE_F32, 16);
        ggml_set_name(a_changed, "tensor_a");
        ggml_set_name(b_changed, "tensor_b");

        const std::string digest_changed = llama_spectral_compute_tensor_digest("toy-arch", digest_ctx_changed.get());
        if (digest_changed == digest0) {
            std::fprintf(stderr, "tensor digest must change when tensor shapes change\n");
            return 1;
        }
    }

    llama_spectral_artifact filtered;
    if (!llama_spectral_filter_artifact(artifact, LLAMA_SPECTRAL_KIND_WEIGHT, "selcorr", filtered, &err)) {
        std::fprintf(stderr, "artifact filter failed: %s\n", err.c_str());
        return 1;
    }
    if (filtered.entries.size() != 1 || filtered.entries[0].profile != LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM_SELCORR) {
        std::fprintf(stderr, "artifact filter mismatch\n");
        return 1;
    }

    const auto path = std::filesystem::temp_directory_path() / "llama-spectral-test.gguf";
    if (!llama_spectral_save_gguf(artifact, path.string(), &err)) {
        std::fprintf(stderr, "save failed: %s\n", err.c_str());
        return 1;
    }

    llama_spectral_artifact loaded;
    if (!llama_spectral_load_gguf(path.string(), loaded, &err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }

    if (loaded.entries.size() != 2 || loaded.entries[1].name != selcorr_entry.name) {
        std::fprintf(stderr, "artifact roundtrip mismatch\n");
        return 1;
    }
    if (loaded.entries[1].correction_dim != selcorr_entry.correction_dim) {
        std::fprintf(stderr, "correction metadata mismatch\n");
        return 1;
    }
    if (loaded.entries[1].eigenvalues.size() != selcorr_entry.eigenvalues.size()) {
        std::fprintf(stderr, "eigenvalue tensor mismatch\n");
        return 1;
    }

    llama_spectral_entry loaded_prepared;
    llama_spectral_entry selcorr_prepared_ref;
    if (!llama_spectral_prepare_weight_entry(loaded.entries[1], GGML_TYPE_SQ4_1S, loaded_prepared, &err)) {
        std::fprintf(stderr, "loaded prepare failed: %s\n", err.c_str());
        return 1;
    }
    if (!llama_spectral_prepare_weight_entry(selcorr_entry, GGML_TYPE_SQ4_1S, selcorr_prepared_ref, &err)) {
        std::fprintf(stderr, "reference prepare failed: %s\n", err.c_str());
        return 1;
    }
    if (max_abs_diff(loaded_prepared.basis, selcorr_prepared_ref.basis) > 1e-6f) {
        std::fprintf(stderr, "prepared basis mismatch after roundtrip\n");
        return 1;
    }

    gguf_context_ptr metadata(gguf_init_empty());
    if (!metadata) {
        std::fprintf(stderr, "failed to allocate metadata gguf\n");
        return 1;
    }
    if (!llama_spectral_embed_gguf_metadata(artifact, metadata.get(), &err)) {
        std::fprintf(stderr, "metadata embed failed: %s\n", err.c_str());
        return 1;
    }

    llama_spectral_artifact embedded;
    if (!llama_spectral_extract_gguf_metadata(metadata.get(), embedded, &err)) {
        std::fprintf(stderr, "metadata extract failed: %s\n", err.c_str());
        return 1;
    }
    if (embedded.entries.size() != artifact.entries.size()) {
        std::fprintf(stderr, "embedded metadata entry count mismatch\n");
        return 1;
    }

    const uint32_t sq_dim = 32;
    const uint32_t sq_n_samples = 10;
    const std::vector<float> sq_samples = make_sq_samples(sq_dim, sq_n_samples);

    llama_spectral_calibrate_params sq_base;
    sq_base.kind = LLAMA_SPECTRAL_KIND_WEIGHT;
    sq_base.profile = LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM;
    sq_base.base_bits = 4;
    sq_base.correction_dim = 8;

    llama_spectral_calibrate_params sq_sel = sq_base;
    sq_sel.profile = LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM_SELCORR;

    llama_spectral_entry sq_base_entry;
    llama_spectral_entry sq_sel_entry;
    llama_spectral_calibration_metrics sq_base_metrics;
    llama_spectral_calibration_metrics sq_sel_metrics;
    if (!llama_spectral_calibrate_entry("sq_toy", sq_samples, sq_n_samples, sq_dim, sq_base, sq_base_entry, &err, &sq_base_metrics)) {
        std::fprintf(stderr, "sq base calibration failed: %s\n", err.c_str());
        return 1;
    }
    if (!llama_spectral_calibrate_entry("sq_toy", sq_samples, sq_n_samples, sq_dim, sq_sel, sq_sel_entry, &err, &sq_sel_metrics)) {
        std::fprintf(stderr, "sq selcorr calibration failed: %s\n", err.c_str());
        return 1;
    }
    if (!sq_base_metrics.used_snapshot || !sq_sel_metrics.used_snapshot) {
        std::fprintf(stderr, "sq calibration should use snapshot path when n_samples < dim\n");
        return 1;
    }
    if (sq_base_metrics.spectral_rank == 0 || sq_base_metrics.spectral_rank > sq_n_samples) {
        std::fprintf(stderr, "unexpected snapshot rank for sq calibration\n");
        return 1;
    }

    llama_spectral_entry sq_base_prepared;
    llama_spectral_entry sq_sel_prepared;
    if (!llama_spectral_prepare_weight_entry(sq_base_entry, GGML_TYPE_SQ4_1S, sq_base_prepared, &err)) {
        std::fprintf(stderr, "sq base prepare failed: %s\n", err.c_str());
        return 1;
    }
    if (!llama_spectral_prepare_weight_entry(sq_sel_entry, GGML_TYPE_SQ4_1S, sq_sel_prepared, &err)) {
        std::fprintf(stderr, "sq selcorr prepare failed: %s\n", err.c_str());
        return 1;
    }

    std::vector<float> sq_probe(sq_dim, 0.0f);
    for (uint32_t i = 0; i < sq_dim; ++i) {
        sq_probe[i] = i < 6 ? 2.0f - 0.1f * i : 0.01f * float((int(i % 9) - 4));
    }

    const ggml_spectral_weight_meta sq_base_meta = {
        /*.dim                    =*/ sq_base_prepared.dim,
        /*.split                  =*/ sq_base_prepared.split,
        /*.correction_dim         =*/ sq_base_prepared.correction_dim,
        /*.semantic_codebook_size =*/ (uint32_t) sq_base_prepared.semantic_codebook.size(),
        /*.tail_codebook_size     =*/ (uint32_t) sq_base_prepared.tail_codebook.size(),
        /*.basis                  =*/ sq_base_prepared.basis.data(),
        /*.semantic_codebook      =*/ sq_base_prepared.semantic_codebook.data(),
        /*.tail_codebook          =*/ sq_base_prepared.tail_codebook.data(),
    };
    const ggml_spectral_weight_meta sq_sel_meta = {
        /*.dim                    =*/ sq_sel_prepared.dim,
        /*.split                  =*/ sq_sel_prepared.split,
        /*.correction_dim         =*/ sq_sel_prepared.correction_dim,
        /*.semantic_codebook_size =*/ (uint32_t) sq_sel_prepared.semantic_codebook.size(),
        /*.tail_codebook_size     =*/ (uint32_t) sq_sel_prepared.tail_codebook.size(),
        /*.basis                  =*/ sq_sel_prepared.basis.data(),
        /*.semantic_codebook      =*/ sq_sel_prepared.semantic_codebook.data(),
        /*.tail_codebook          =*/ sq_sel_prepared.tail_codebook.data(),
    };

    std::vector<uint8_t> sq_base_bytes(ggml_row_size(GGML_TYPE_SQ4_1S, sq_dim));
    std::vector<uint8_t> sq_sel_bytes(ggml_row_size(GGML_TYPE_SQ4_1S, sq_dim));
    if (ggml_quantize_spectral_weight(GGML_TYPE_SQ4_1S, sq_probe.data(), sq_base_bytes.data(), 1, sq_dim, &sq_base_meta) != sq_base_bytes.size()) {
        std::fprintf(stderr, "sq base quantization failed\n");
        return 1;
    }
    if (ggml_quantize_spectral_weight(GGML_TYPE_SQ4_1S, sq_probe.data(), sq_sel_bytes.data(), 1, sq_dim, &sq_sel_meta) != sq_sel_bytes.size()) {
        std::fprintf(stderr, "sq selcorr quantization failed\n");
        return 1;
    }

    if (!ggml_spectral_register_tensor(sq_base_bytes.data(), sq_base_bytes.data(), sq_base_bytes.size(), GGML_TYPE_SQ4_1S, &sq_base_meta)) {
        std::fprintf(stderr, "sq base registry failed\n");
        return 1;
    }
    std::vector<float> sq_base_decoded(sq_dim, 0.0f);
    ggml_get_type_traits(GGML_TYPE_SQ4_1S)->to_float(sq_base_bytes.data(), sq_base_decoded.data(), sq_dim);
    ggml_spectral_unregister_owner(sq_base_bytes.data());

    if (!ggml_spectral_register_tensor(sq_sel_bytes.data(), sq_sel_bytes.data(), sq_sel_bytes.size(), GGML_TYPE_SQ4_1S, &sq_sel_meta)) {
        std::fprintf(stderr, "sq selcorr registry failed\n");
        return 1;
    }
    std::vector<float> sq_sel_decoded(sq_dim, 0.0f);
    ggml_get_type_traits(GGML_TYPE_SQ4_1S)->to_float(sq_sel_bytes.data(), sq_sel_decoded.data(), sq_dim);
    ggml_spectral_unregister_owner(sq_sel_bytes.data());

    if (mse(sq_sel_decoded, sq_probe) > mse(sq_base_decoded, sq_probe) + 1e-6f) {
        std::fprintf(stderr, "SQ selective correction should not be worse than SQ nonuniform quantization\n");
        return 1;
    }

    {
        llama_spectral_artifact sq_artifact;
        sq_artifact.source_architecture = "sq-toy";
        sq_artifact.entries.push_back(sq_sel_entry);

        const auto compact_path = std::filesystem::temp_directory_path() / "llama-spectral-compact-test.gguf";
        if (!llama_spectral_save_gguf(sq_artifact, compact_path.string(), &err)) {
            std::fprintf(stderr, "compact save failed: %s\n", err.c_str());
            return 1;
        }

        llama_spectral_artifact compact_loaded;
        if (!llama_spectral_load_gguf(compact_path.string(), compact_loaded, &err)) {
            std::fprintf(stderr, "compact load failed: %s\n", err.c_str());
            return 1;
        }
        if (compact_loaded.entries.size() != 1) {
            std::fprintf(stderr, "compact roundtrip entry count mismatch\n");
            return 1;
        }
        if (compact_loaded.entries[0].basis.size() >= sq_sel_entry.basis.size()) {
            std::fprintf(stderr, "compact roundtrip did not shrink basis payload\n");
            return 1;
        }

        llama_spectral_entry compact_prepared;
        if (!llama_spectral_prepare_weight_entry(compact_loaded.entries[0], GGML_TYPE_SQ4_1S, compact_prepared, &err)) {
            std::fprintf(stderr, "compact prepare failed: %s\n", err.c_str());
            return 1;
        }
        if (max_abs_diff(compact_prepared.basis, sq_sel_prepared.basis) > 1e-5f) {
            std::fprintf(stderr, "compact basis reconstruction mismatch\n");
            return 1;
        }

        std::filesystem::remove(compact_path);
    }

    llama_spectral_calibrate_params sk_base = sq_base;
    sk_base.kind = LLAMA_SPECTRAL_KIND_K;
    llama_spectral_calibrate_params sk_sel = sk_base;
    sk_sel.profile = LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM_SELCORR;
    llama_spectral_calibrate_params sv_sel = sk_sel;
    sv_sel.kind = LLAMA_SPECTRAL_KIND_V;

    llama_spectral_entry sk_base_entry;
    llama_spectral_entry sk_sel_entry;
    llama_spectral_entry sv_sel_entry;
    if (!llama_spectral_calibrate_entry("cache_k_l0.h0", sq_samples, sq_n_samples, sq_dim, sk_base, sk_base_entry, &err)) {
        std::fprintf(stderr, "sk base calibration failed: %s\n", err.c_str());
        return 1;
    }
    if (!llama_spectral_calibrate_entry("cache_k_l0.h0", sq_samples, sq_n_samples, sq_dim, sk_sel, sk_sel_entry, &err)) {
        std::fprintf(stderr, "sk selcorr calibration failed: %s\n", err.c_str());
        return 1;
    }
    if (!llama_spectral_calibrate_entry("cache_v_l0.h0", sq_samples, sq_n_samples, sq_dim, sv_sel, sv_sel_entry, &err)) {
        std::fprintf(stderr, "sv selcorr calibration failed: %s\n", err.c_str());
        return 1;
    }

    llama_spectral_entry sk_base_prepared;
    llama_spectral_entry sk_sel_prepared;
    llama_spectral_entry sv_sel_prepared;
    if (!llama_spectral_prepare_kv_entry(sk_base_entry, GGML_TYPE_SKV4_0, sk_base_prepared, &err)) {
        std::fprintf(stderr, "sk base prepare failed: %s\n", err.c_str());
        return 1;
    }
    if (!llama_spectral_prepare_kv_entry(sk_sel_entry, GGML_TYPE_SKV4_0, sk_sel_prepared, &err)) {
        std::fprintf(stderr, "sk selcorr prepare failed: %s\n", err.c_str());
        return 1;
    }
    if (!llama_spectral_prepare_kv_entry(sv_sel_entry, GGML_TYPE_SKV4_0, sv_sel_prepared, &err)) {
        std::fprintf(stderr, "sv selcorr prepare failed: %s\n", err.c_str());
        return 1;
    }
    if (sv_sel_prepared.correction_dim != 0) {
        std::fprintf(stderr, "V spectral path must not keep selective correction\n");
        return 1;
    }

    std::vector<float> sk_probe(sq_dim, 0.0f);
    std::vector<float> sk_query(sq_dim, 0.0f);
    for (uint32_t i = 0; i < sq_dim; ++i) {
        sk_probe[i] = i < 8 ? 1.8f - 0.11f * i : 0.015f * float((int(i % 7) - 3));
        sk_query[i] = i < 8 ? 1.3f - 0.09f * i : 0.01f * float((int(i % 5) - 2));
    }

    std::vector<float> identity_qjl(size_t(sk_sel_prepared.split) * sk_sel_prepared.split, 0.0f);
    for (uint32_t i = 0; i < sk_sel_prepared.split; ++i) {
        identity_qjl[size_t(i) * sk_sel_prepared.split + i] = 1.0f;
    }

    const ggml_spectral_kv_head_meta sk_base_head = {
        /*.dim                    =*/ sk_base_prepared.dim,
        /*.split                  =*/ sk_base_prepared.split,
        /*.semantic_codebook_size =*/ (uint32_t) sk_base_prepared.semantic_codebook.size(),
        /*.tail_codebook_size     =*/ (uint32_t) sk_base_prepared.tail_codebook.size(),
        /*.basis                  =*/ sk_base_prepared.basis.data(),
        /*.semantic_codebook      =*/ sk_base_prepared.semantic_codebook.data(),
        /*.tail_codebook          =*/ sk_base_prepared.tail_codebook.data(),
        /*.qjl_matrix             =*/ nullptr,
    };
    const ggml_spectral_kv_head_meta sk_sel_head = {
        /*.dim                    =*/ sk_sel_prepared.dim,
        /*.split                  =*/ sk_sel_prepared.split,
        /*.semantic_codebook_size =*/ (uint32_t) sk_sel_prepared.semantic_codebook.size(),
        /*.tail_codebook_size     =*/ (uint32_t) sk_sel_prepared.tail_codebook.size(),
        /*.basis                  =*/ sk_sel_prepared.basis.data(),
        /*.semantic_codebook      =*/ sk_sel_prepared.semantic_codebook.data(),
        /*.tail_codebook          =*/ sk_sel_prepared.tail_codebook.data(),
        /*.qjl_matrix             =*/ identity_qjl.data(),
    };

    std::vector<float> sk_base_vec_norms(1, 0.0f);
    std::vector<float> sk_sel_vec_norms(1, 0.0f);
    std::vector<float> sk_sel_residual_norms(1, 0.0f);
    std::vector<uint8_t> sk_sel_signs((sk_sel_prepared.split + 7u) / 8u, 0);

    const ggml_spectral_kv_meta sk_base_meta = {
        /*.is_key            =*/ true,
        /*.use_correction    =*/ false,
        /*.n_head            =*/ 1,
        /*.head_dim          =*/ sk_base_prepared.dim,
        /*.head_dim_padded   =*/ sk_base_prepared.dim,
        /*.n_rows            =*/ 1,
        /*.qjl_bytes_per_head=*/ 0,
        /*.heads             =*/ &sk_base_head,
        /*.vec_norms         =*/ sk_base_vec_norms.data(),
        /*.residual_norms    =*/ nullptr,
        /*.qjl_signs         =*/ nullptr,
    };
    const ggml_spectral_kv_meta sk_sel_meta = {
        /*.is_key            =*/ true,
        /*.use_correction    =*/ true,
        /*.n_head            =*/ 1,
        /*.head_dim          =*/ sk_sel_prepared.dim,
        /*.head_dim_padded   =*/ sk_sel_prepared.dim,
        /*.n_rows            =*/ 1,
        /*.qjl_bytes_per_head=*/ (uint32_t) sk_sel_signs.size(),
        /*.heads             =*/ &sk_sel_head,
        /*.vec_norms         =*/ sk_sel_vec_norms.data(),
        /*.residual_norms    =*/ sk_sel_residual_norms.data(),
        /*.qjl_signs         =*/ sk_sel_signs.data(),
    };

    std::vector<uint8_t> sk_base_bytes(ggml_row_size(GGML_TYPE_SKV4_0, sq_dim));
    std::vector<uint8_t> sk_sel_bytes(ggml_row_size(GGML_TYPE_SKV4_0, sq_dim));

    if (!ggml_spectral_kv_register_tensor(sk_base_bytes.data(), sk_base_bytes.data(), sk_base_bytes.size(), GGML_TYPE_SKV4_0, &sk_base_meta)) {
        std::fprintf(stderr, "sk base registry failed\n");
        return 1;
    }
    ggml_get_type_traits(GGML_TYPE_SKV4_0)->from_float_ref(sk_probe.data(), sk_base_bytes.data(), sq_dim);
    std::vector<float> sk_base_decoded(sq_dim, 0.0f);
    ggml_get_type_traits(GGML_TYPE_SKV4_0)->to_float(sk_base_bytes.data(), sk_base_decoded.data(), sq_dim);
    float sk_base_dot = 0.0f;
    ggml_vec_dot_skv4_0_f32((int) sq_dim, &sk_base_dot, 0, sk_base_bytes.data(), 0, sk_query.data(), 0, 1);
    ggml_spectral_kv_unregister_owner(sk_base_bytes.data());

    if (!ggml_spectral_kv_register_tensor(sk_sel_bytes.data(), sk_sel_bytes.data(), sk_sel_bytes.size(), GGML_TYPE_SKV4_0, &sk_sel_meta)) {
        std::fprintf(stderr, "sk sel registry failed\n");
        return 1;
    }
    ggml_get_type_traits(GGML_TYPE_SKV4_0)->from_float_ref(sk_probe.data(), sk_sel_bytes.data(), sq_dim);
    std::vector<float> sk_sel_decoded(sq_dim, 0.0f);
    ggml_get_type_traits(GGML_TYPE_SKV4_0)->to_float(sk_sel_bytes.data(), sk_sel_decoded.data(), sq_dim);
    float sk_sel_dot = 0.0f;
    ggml_vec_dot_skv4_0_f32((int) sq_dim, &sk_sel_dot, 0, sk_sel_bytes.data(), 0, sk_query.data(), 0, 1);
    ggml_spectral_kv_unregister_owner(sk_sel_bytes.data());

    for (float value : sk_sel_decoded) {
        if (!std::isfinite(value)) {
            std::fprintf(stderr, "sk decode produced non-finite value\n");
            return 1;
        }
    }

    const float sk_ref_dot = dot_product(sk_probe, sk_query);
    const float sk_base_err = std::fabs(sk_base_dot - sk_ref_dot);
    const float sk_sel_err = std::fabs(sk_sel_dot - sk_ref_dot);
    if (sk_sel_err > sk_base_err + 1e-5f) {
        std::fprintf(stderr, "SKV selective correction should not be worse than plain nonuniform dot estimation\n");
        return 1;
    }

    std::filesystem::remove(path);
    return 0;
}
