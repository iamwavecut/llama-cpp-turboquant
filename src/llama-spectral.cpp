#include "llama-spectral.h"

#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <string>
#include <utility>
#include <vector>

#if defined(__APPLE__) && __has_include(<Accelerate/Accelerate.h>)
#include <Accelerate/Accelerate.h>
#define LLAMA_SPECTRAL_USE_ACCELERATE
#endif

namespace {

using spectral_clock = std::chrono::steady_clock;

constexpr const char * LLAMA_SPECTRAL_GENERAL_TYPE       = "general.type";
constexpr const char * LLAMA_SPECTRAL_GENERAL_TYPE_VALUE = "spectral_calibration";
constexpr const char * LLAMA_SPECTRAL_VERSION            = "spectral.version";
constexpr const char * LLAMA_SPECTRAL_SOURCE_ARCH        = "spectral.source_architecture";
constexpr const char * LLAMA_SPECTRAL_SOURCE_DIGEST      = "spectral.source_digest";
constexpr const char * LLAMA_SPECTRAL_ENTRY_NAMES        = "spectral.entry_names";
constexpr const char * LLAMA_SPECTRAL_ENTRY_KINDS        = "spectral.entry_kinds";
constexpr const char * LLAMA_SPECTRAL_ENTRY_PROFILES     = "spectral.entry_profiles";
constexpr const char * LLAMA_SPECTRAL_ENTRY_DIMS         = "spectral.entry_dims";
constexpr const char * LLAMA_SPECTRAL_ENTRY_SPLITS       = "spectral.entry_splits";
constexpr const char * LLAMA_SPECTRAL_ENTRY_D_EFF        = "spectral.entry_d_eff";
constexpr const char * LLAMA_SPECTRAL_ENTRY_SEM_BITS     = "spectral.entry_semantic_bits";
constexpr const char * LLAMA_SPECTRAL_ENTRY_TAIL_BITS    = "spectral.entry_tail_bits";
constexpr const char * LLAMA_SPECTRAL_ENTRY_CORR_DIMS    = "spectral.entry_correction_dims";

void set_error(std::string * err, const std::string & message) {
    if (err) {
        *err = message;
    }
}

bool check(bool cond, std::string * err, const std::string & message) {
    if (!cond) {
        set_error(err, message);
        return false;
    }
    return true;
}

bool parse_profile_filter(
        const char * profile_filter,
        bool & use_all_profiles,
        llama_spectral_profile & requested_profile,
        std::string * err) {
    use_all_profiles = true;
    requested_profile = LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM;

    if (profile_filter == nullptr || profile_filter[0] == '\0') {
        return true;
    }
    if (std::strcmp(profile_filter, "all") == 0 || std::strcmp(profile_filter, "auto") == 0) {
        return true;
    }
    if (!llama_spectral_profile_parse(profile_filter, requested_profile)) {
        set_error(err, std::string("unsupported spectral profile filter: ") + profile_filter);
        return false;
    }

    use_all_profiles = false;
    return true;
}

bool spectral_type_bits(ggml_type type, uint32_t & bits) {
    switch (type) {
        case GGML_TYPE_SQ2_0:  bits = 2; return true;
        case GGML_TYPE_SQ3_1S: bits = 3; return true;
        case GGML_TYPE_SQ4_1S: bits = 4; return true;
        default:               return false;
    }
}

bool spectral_kv_type_bits(ggml_type type, uint32_t & bits) {
    switch (type) {
        case GGML_TYPE_SKV2_0: bits = 2; return true;
        case GGML_TYPE_SKV3_0: bits = 3; return true;
        case GGML_TYPE_SKV4_0: bits = 4; return true;
        default:               return false;
    }
}

uint32_t clamp_bits(uint32_t bits) {
    return std::max<uint32_t>(1, std::min<uint32_t>(bits, 8));
}

std::vector<float> resample_codebook(const std::vector<float> & codebook, uint32_t target_levels) {
    if (codebook.empty() || target_levels == 0 || codebook.size() <= target_levels) {
        return codebook;
    }

    std::vector<float> out;
    out.reserve(target_levels);
    for (uint32_t i = 0; i < target_levels; ++i) {
        const size_t src = target_levels == 1 ? 0 : size_t(std::lround(double(i) * double(codebook.size() - 1) / double(target_levels - 1)));
        out.push_back(codebook[src]);
    }
    return out;
}

std::string tensor_name_for(size_t index, const char * suffix) {
    return "spectral." + std::to_string(index) + "." + suffix;
}

std::string metadata_name_for(size_t index, const char * suffix) {
    return "spectral." + std::to_string(index) + "." + suffix;
}

void spectral_digest_update(uint64_t & hash, const void * data, size_t size) {
    constexpr uint64_t fnv_prime = 0x100000001b3ULL;
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= fnv_prime;
    }
}

template<typename T>
void spectral_digest_update_pod(uint64_t & hash, const T & value) {
    spectral_digest_update(hash, &value, sizeof(value));
}

void spectral_digest_update_str(uint64_t & hash, const std::string & value) {
    const uint64_t size = value.size();
    spectral_digest_update_pod(hash, size);
    if (!value.empty()) {
        spectral_digest_update(hash, value.data(), value.size());
    }
}

bool complete_basis_from_columns(
        const std::vector<float> & semantic_basis,
        uint32_t dim,
        uint32_t rank,
        std::vector<float> & basis);

bool validate_entry(const llama_spectral_entry & entry, std::string * err) {
    if (!check(entry.dim > 0, err, "spectral entry has zero dimension")) {
        return false;
    }
    if (!check(entry.split > 0 && entry.split <= entry.dim, err, "spectral entry has invalid split")) {
        return false;
    }
    if (!check(entry.semantic_bits > 0 && entry.tail_bits > 0, err, "spectral entry has invalid bit allocation")) {
        return false;
    }
    if (!check(entry.basis.size() % entry.dim == 0, err, "spectral entry basis has wrong size")) {
        return false;
    }
    const uint32_t basis_cols = uint32_t(entry.basis.size() / entry.dim);
    if (!check(basis_cols > 0 && basis_cols <= entry.dim, err, "spectral entry basis column count is invalid")) {
        return false;
    }
    if (!check(basis_cols >= entry.split, err, "spectral entry basis is missing semantic columns")) {
        return false;
    }
    if (!entry.eigenvalues.empty() &&
        !check(entry.eigenvalues.size() == entry.dim, err, "spectral entry eigenvalues have wrong size")) {
        return false;
    }
    if (!check(!entry.semantic_codebook.empty(), err, "spectral semantic codebook is empty")) {
        return false;
    }
    if (!check(!entry.tail_codebook.empty(), err, "spectral tail codebook is empty")) {
        return false;
    }
    if (!check(entry.correction_dim <= entry.split, err, "spectral correction dim exceeds split")) {
        return false;
    }
    return true;
}

uint32_t basis_column_count(const llama_spectral_entry & entry) {
    if (entry.dim == 0 || entry.basis.empty() || entry.basis.size() % entry.dim != 0) {
        return 0;
    }
    return uint32_t(entry.basis.size() / entry.dim);
}

uint32_t nonzero_eigen_rank(const std::vector<float> & eigenvalues) {
    if (eigenvalues.empty()) {
        return 0;
    }

    const float max_eigenvalue = *std::max_element(eigenvalues.begin(), eigenvalues.end());
    const float tol = std::max(1e-8f, max_eigenvalue * 1e-5f);
    uint32_t rank = 0;
    while (rank < eigenvalues.size() && eigenvalues[rank] > tol) {
        ++rank;
    }
    return rank;
}

uint32_t compact_basis_columns(const llama_spectral_entry & entry) {
    uint32_t cols = nonzero_eigen_rank(entry.eigenvalues);
    if (cols == 0) {
        cols = basis_column_count(entry);
    }
    cols = std::max<uint32_t>(cols, entry.split);
    cols = std::max<uint32_t>(cols, entry.correction_dim);
    return std::min<uint32_t>(cols, entry.dim);
}

std::vector<float> trim_basis_columns(const llama_spectral_entry & entry, uint32_t cols) {
    const uint32_t src_cols = basis_column_count(entry);
    if (cols == 0 || src_cols == 0) {
        return {};
    }
    if (cols >= src_cols) {
        return entry.basis;
    }

    std::vector<float> trimmed(size_t(entry.dim) * cols, 0.0f);
    for (uint32_t row = 0; row < entry.dim; ++row) {
        std::memcpy(
                trimmed.data() + size_t(row) * cols,
                entry.basis.data() + size_t(row) * src_cols,
                size_t(cols) * sizeof(float));
    }
    return trimmed;
}

bool materialize_full_basis(
        const llama_spectral_entry & entry,
        std::vector<float> & basis,
        std::string * err) {
    const uint32_t cols = basis_column_count(entry);
    if (!check(cols > 0 && cols <= entry.dim, err, "spectral entry basis column count is invalid")) {
        return false;
    }
    if (cols == entry.dim) {
        basis = entry.basis;
        return true;
    }
    if (!check(cols >= entry.split, err, "spectral entry basis is missing semantic columns")) {
        return false;
    }
    if (!complete_basis_from_columns(entry.basis, entry.dim, cols, basis)) {
        set_error(err, "failed to reconstruct full spectral basis");
        return false;
    }
    return true;
}

double elapsed_ms(const spectral_clock::time_point & start, const spectral_clock::time_point & end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<float> center_samples(const std::vector<float> & samples, uint32_t n_samples, uint32_t dim) {
    std::vector<float> centered(samples);
    if (n_samples == 0 || dim == 0) {
        return centered;
    }

    std::vector<float> mean(dim, 0.0f);
    for (uint32_t row = 0; row < n_samples; ++row) {
        const float * sample = samples.data() + size_t(row) * dim;
        for (uint32_t col = 0; col < dim; ++col) {
            mean[col] += sample[col];
        }
    }
    for (float & value : mean) {
        value /= n_samples;
    }

    for (uint32_t row = 0; row < n_samples; ++row) {
        float * sample = centered.data() + size_t(row) * dim;
        for (uint32_t col = 0; col < dim; ++col) {
            sample[col] -= mean[col];
        }
    }

    return centered;
}

void set_identity_matrix(std::vector<float> & matrix, uint32_t dim) {
    matrix.assign(size_t(dim) * dim, 0.0f);
    for (uint32_t i = 0; i < dim; ++i) {
        matrix[size_t(i) * dim + i] = 1.0f;
    }
}

std::vector<float> compute_covariance_from_centered(const std::vector<float> & centered, uint32_t n_samples, uint32_t dim) {
    std::vector<float> cov(size_t(dim) * dim, 0.0f);

    if (n_samples == 0 || dim == 0) {
        return cov;
    }

#ifdef LLAMA_SPECTRAL_USE_ACCELERATE
    const float inv = n_samples > 1 ? 1.0f / float(n_samples - 1) : 1.0f;
    cblas_sgemm(
            CblasRowMajor,
            CblasTrans,
            CblasNoTrans,
            (int) dim,
            (int) dim,
            (int) n_samples,
            inv,
            centered.data(),
            (int) dim,
            centered.data(),
            (int) dim,
            0.0f,
            cov.data(),
            (int) dim);
#else
    for (uint32_t row = 0; row < n_samples; ++row) {
        const float * sample = centered.data() + size_t(row) * dim;
        for (uint32_t i = 0; i < dim; ++i) {
            const float xi = sample[i];
            for (uint32_t j = i; j < dim; ++j) {
                cov[size_t(i) * dim + j] += xi * sample[j];
            }
        }
    }

    const float inv = n_samples > 1 ? 1.0f / float(n_samples - 1) : 1.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        for (uint32_t j = i; j < dim; ++j) {
            const float value = cov[size_t(i) * dim + j] * inv;
            cov[size_t(i) * dim + j] = value;
            cov[size_t(j) * dim + i] = value;
        }
    }
#endif

    return cov;
}

std::vector<float> compute_gram_matrix_from_centered(const std::vector<float> & centered, uint32_t n_samples, uint32_t dim) {
    std::vector<float> gram(size_t(n_samples) * n_samples, 0.0f);

    if (n_samples == 0 || dim == 0) {
        return gram;
    }

#ifdef LLAMA_SPECTRAL_USE_ACCELERATE
    const float inv = n_samples > 1 ? 1.0f / float(n_samples - 1) : 1.0f;
    cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            (int) n_samples,
            (int) n_samples,
            (int) dim,
            inv,
            centered.data(),
            (int) dim,
            centered.data(),
            (int) dim,
            0.0f,
            gram.data(),
            (int) n_samples);
#else
    for (uint32_t row = 0; row < n_samples; ++row) {
        const float * lhs = centered.data() + size_t(row) * dim;
        for (uint32_t col = row; col < n_samples; ++col) {
            const float * rhs = centered.data() + size_t(col) * dim;
            float acc = 0.0f;
            for (uint32_t i = 0; i < dim; ++i) {
                acc += lhs[i] * rhs[i];
            }
            const float value = acc * (n_samples > 1 ? 1.0f / float(n_samples - 1) : 1.0f);
            gram[size_t(row) * n_samples + col] = value;
            gram[size_t(col) * n_samples + row] = value;
        }
    }
#endif

    return gram;
}

bool accelerate_eigendecomposition(
        const std::vector<float> & matrix,
        uint32_t dim,
        std::vector<float> & eigenvalues,
        std::vector<float> & eigenvectors) {
#ifdef LLAMA_SPECTRAL_USE_ACCELERATE
    if (dim == 0) {
        eigenvalues.clear();
        eigenvectors.clear();
        return true;
    }

    eigenvalues.assign(dim, 0.0f);
    eigenvectors = matrix;

    const char jobz = 'V';
    const char uplo = 'U';
    const __LAPACK_int n = (__LAPACK_int) dim;
    const __LAPACK_int lda = n;

    float work_query = 0.0f;
    __LAPACK_int iwork_query = 0;
    __LAPACK_int lwork = -1;
    __LAPACK_int liwork = -1;
    __LAPACK_int info = 0;

    ssyevd_(
            &jobz,
            &uplo,
            &n,
            eigenvectors.data(),
            &lda,
            eigenvalues.data(),
            &work_query,
            &lwork,
            &iwork_query,
            &liwork,
            &info);
    if (info != 0 || !std::isfinite(work_query)) {
        return false;
    }

    lwork = std::max<__LAPACK_int>(1, (__LAPACK_int) std::ceil(work_query));
    liwork = std::max<__LAPACK_int>(1, iwork_query);

    std::vector<float> work((size_t) lwork, 0.0f);
    std::vector<__LAPACK_int> iwork((size_t) liwork, 0);
    info = 0;

    ssyevd_(
            &jobz,
            &uplo,
            &n,
            eigenvectors.data(),
            &lda,
            eigenvalues.data(),
            work.data(),
            &lwork,
            iwork.data(),
            &liwork,
            &info);
    if (info != 0) {
        return false;
    }

    for (float & value : eigenvalues) {
        value = std::max(0.0f, value);
    }
    return true;
#else
    (void) matrix;
    (void) dim;
    (void) eigenvalues;
    (void) eigenvectors;
    return false;
#endif
}

void jacobi_eigendecomposition(
        const std::vector<float> & matrix,
        uint32_t dim,
        std::vector<float> & eigenvalues,
        std::vector<float> & eigenvectors) {
    eigenvalues.assign(dim, 0.0f);
    eigenvectors.assign(size_t(dim) * dim, 0.0f);

    if (dim == 0) {
        return;
    }

    std::vector<float> a = matrix;
    for (uint32_t i = 0; i < dim; ++i) {
        eigenvectors[size_t(i) * dim + i] = 1.0f;
    }

    const size_t max_iter = std::max<size_t>(64, size_t(dim) * dim * 16);
    for (size_t iter = 0; iter < max_iter; ++iter) {
        uint32_t p = 0;
        uint32_t q = 1 % dim;
        float max_offdiag = 0.0f;

        for (uint32_t i = 0; i < dim; ++i) {
            for (uint32_t j = i + 1; j < dim; ++j) {
                const float offdiag = std::fabs(a[size_t(i) * dim + j]);
                if (offdiag > max_offdiag) {
                    max_offdiag = offdiag;
                    p = i;
                    q = j;
                }
            }
        }

        if (max_offdiag < 1e-6f) {
            break;
        }

        const float app = a[size_t(p) * dim + p];
        const float aqq = a[size_t(q) * dim + q];
        const float apq = a[size_t(p) * dim + q];
        const float tau = (aqq - app) / (2.0f * apq);
        const float t = (tau >= 0.0f ? 1.0f : -1.0f) / (std::fabs(tau) + std::sqrt(1.0f + tau * tau));
        const float c = 1.0f / std::sqrt(1.0f + t * t);
        const float s = t * c;

        for (uint32_t k = 0; k < dim; ++k) {
            if (k == p || k == q) {
                continue;
            }

            const float akp = a[size_t(k) * dim + p];
            const float akq = a[size_t(k) * dim + q];

            a[size_t(k) * dim + p] = c * akp - s * akq;
            a[size_t(p) * dim + k] = a[size_t(k) * dim + p];
            a[size_t(k) * dim + q] = s * akp + c * akq;
            a[size_t(q) * dim + k] = a[size_t(k) * dim + q];
        }

        const float new_app = c * c * app - 2.0f * s * c * apq + s * s * aqq;
        const float new_aqq = s * s * app + 2.0f * s * c * apq + c * c * aqq;

        a[size_t(p) * dim + p] = new_app;
        a[size_t(q) * dim + q] = new_aqq;
        a[size_t(p) * dim + q] = 0.0f;
        a[size_t(q) * dim + p] = 0.0f;

        for (uint32_t k = 0; k < dim; ++k) {
            const float vip = eigenvectors[size_t(k) * dim + p];
            const float viq = eigenvectors[size_t(k) * dim + q];
            eigenvectors[size_t(k) * dim + p] = c * vip - s * viq;
            eigenvectors[size_t(k) * dim + q] = s * vip + c * viq;
        }
    }

    for (uint32_t i = 0; i < dim; ++i) {
        eigenvalues[i] = std::max(0.0f, a[size_t(i) * dim + i]);
    }
}

void eigendecomposition_symmetric(
        const std::vector<float> & matrix,
        uint32_t dim,
        std::vector<float> & eigenvalues,
        std::vector<float> & eigenvectors) {
    if (!accelerate_eigendecomposition(matrix, dim, eigenvalues, eigenvectors)) {
        jacobi_eigendecomposition(matrix, dim, eigenvalues, eigenvectors);
    }
}

void sort_eigensystem(std::vector<float> & eigenvalues, std::vector<float> & eigenvectors, uint32_t dim) {
    std::vector<uint32_t> order(dim);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t lhs, uint32_t rhs) {
        return eigenvalues[lhs] > eigenvalues[rhs];
    });

    std::vector<float> sorted_values(dim, 0.0f);
    std::vector<float> sorted_vectors(size_t(dim) * dim, 0.0f);

    for (uint32_t dst = 0; dst < dim; ++dst) {
        const uint32_t src = order[dst];
        sorted_values[dst] = eigenvalues[src];
        for (uint32_t row = 0; row < dim; ++row) {
            sorted_vectors[size_t(row) * dim + dst] = eigenvectors[size_t(row) * dim + src];
        }

        uint32_t anchor_row = 0;
        float anchor_abs = 0.0f;
        for (uint32_t row = 0; row < dim; ++row) {
            const float value = sorted_vectors[size_t(row) * dim + dst];
            const float abs_value = std::fabs(value);
            if (abs_value > anchor_abs) {
                anchor_abs = abs_value;
                anchor_row = row;
            }
        }
        if (anchor_abs > 0.0f && sorted_vectors[size_t(anchor_row) * dim + dst] < 0.0f) {
            for (uint32_t row = 0; row < dim; ++row) {
                sorted_vectors[size_t(row) * dim + dst] = -sorted_vectors[size_t(row) * dim + dst];
            }
        }
    }

    eigenvalues.swap(sorted_values);
    eigenvectors.swap(sorted_vectors);
}

void normalize_columns(std::vector<float> & matrix, uint32_t n_rows, uint32_t n_cols) {
    if (n_rows == 0 || n_cols == 0) {
        return;
    }

    for (uint32_t col = 0; col < n_cols; ++col) {
        for (int pass = 0; pass < 2; ++pass) {
            for (uint32_t prev = 0; prev < col; ++prev) {
                float dot = 0.0f;
                for (uint32_t row = 0; row < n_rows; ++row) {
                    dot += matrix[size_t(row) * n_cols + col] * matrix[size_t(row) * n_cols + prev];
                }
                for (uint32_t row = 0; row < n_rows; ++row) {
                    matrix[size_t(row) * n_cols + col] -= dot * matrix[size_t(row) * n_cols + prev];
                }
            }
        }

        float norm_sq = 0.0f;
        for (uint32_t row = 0; row < n_rows; ++row) {
            const float value = matrix[size_t(row) * n_cols + col];
            norm_sq += value * value;
        }
        const float norm = std::sqrt(std::max(0.0f, norm_sq));
        if (norm <= 1e-12f) {
            for (uint32_t row = 0; row < n_rows; ++row) {
                matrix[size_t(row) * n_cols + col] = 0.0f;
            }
            continue;
        }
        const float inv_norm = 1.0f / norm;
        for (uint32_t row = 0; row < n_rows; ++row) {
            matrix[size_t(row) * n_cols + col] *= inv_norm;
        }
    }
}

bool accelerate_complete_basis_from_columns(
        const std::vector<float> & semantic_basis,
        uint32_t dim,
        uint32_t rank,
        std::vector<float> & basis) {
#ifdef LLAMA_SPECTRAL_USE_ACCELERATE
    if (rank == 0) {
        set_identity_matrix(basis, dim);
        return true;
    }

    const __LAPACK_int m = (__LAPACK_int) dim;
    const __LAPACK_int n = (__LAPACK_int) rank;
    const __LAPACK_int lda = m;

    std::vector<float> qr(size_t(dim) * rank, 0.0f);
    for (uint32_t row = 0; row < dim; ++row) {
        for (uint32_t col = 0; col < rank; ++col) {
            qr[size_t(col) * dim + row] = semantic_basis[size_t(row) * rank + col];
        }
    }

    std::vector<float> tau(rank, 0.0f);
    float work_query = 0.0f;
    __LAPACK_int lwork = -1;
    __LAPACK_int info = 0;
    sgeqrf_(&m, &n, qr.data(), &lda, tau.data(), &work_query, &lwork, &info);
    if (info != 0 || !std::isfinite(work_query)) {
        return false;
    }

    lwork = std::max<__LAPACK_int>(1, (__LAPACK_int) std::ceil(work_query));
    std::vector<float> work((size_t) lwork, 0.0f);
    sgeqrf_(&m, &n, qr.data(), &lda, tau.data(), work.data(), &lwork, &info);
    if (info != 0) {
        return false;
    }

    std::vector<float> q(size_t(dim) * dim, 0.0f);
    for (uint32_t col = 0; col < rank; ++col) {
        std::memcpy(q.data() + size_t(col) * dim, qr.data() + size_t(col) * dim, size_t(dim) * sizeof(float));
    }

    const __LAPACK_int full_n = (__LAPACK_int) dim;
    const __LAPACK_int k = (__LAPACK_int) rank;
    work_query = 0.0f;
    lwork = -1;
    info = 0;
    sorgqr_(&m, &full_n, &k, q.data(), &lda, tau.data(), &work_query, &lwork, &info);
    if (info != 0 || !std::isfinite(work_query)) {
        return false;
    }

    lwork = std::max<__LAPACK_int>(1, (__LAPACK_int) std::ceil(work_query));
    work.assign((size_t) lwork, 0.0f);
    sorgqr_(&m, &full_n, &k, q.data(), &lda, tau.data(), work.data(), &lwork, &info);
    if (info != 0) {
        return false;
    }

    for (uint32_t col = 0; col < rank; ++col) {
        const float sign = qr[size_t(col) * dim + col] < 0.0f ? -1.0f : 1.0f;
        if (sign < 0.0f) {
            for (uint32_t row = 0; row < dim; ++row) {
                q[size_t(col) * dim + row] = -q[size_t(col) * dim + row];
            }
        }
    }

    basis.resize(size_t(dim) * dim);
    for (uint32_t row = 0; row < dim; ++row) {
        for (uint32_t col = 0; col < dim; ++col) {
            basis[size_t(row) * dim + col] = q[size_t(col) * dim + row];
        }
    }
    return true;
#else
    (void) semantic_basis;
    (void) dim;
    (void) rank;
    (void) basis;
    return false;
#endif
}

bool complete_basis_with_canonical_fallback(
        const std::vector<float> & semantic_basis,
        uint32_t dim,
        uint32_t rank,
        std::vector<float> & basis) {
    if (rank == 0) {
        set_identity_matrix(basis, dim);
        return true;
    }
    if (dim > 512) {
        return false;
    }

    basis.assign(size_t(dim) * dim, 0.0f);
    for (uint32_t row = 0; row < dim; ++row) {
        for (uint32_t col = 0; col < rank; ++col) {
            basis[size_t(row) * dim + col] = semantic_basis[size_t(row) * rank + col];
        }
    }

    uint32_t next_col = rank;
    std::vector<float> candidate(dim, 0.0f);
    for (uint32_t axis = 0; axis < dim && next_col < dim; ++axis) {
        std::fill(candidate.begin(), candidate.end(), 0.0f);
        candidate[axis] = 1.0f;

        for (int pass = 0; pass < 2; ++pass) {
            for (uint32_t prev = 0; prev < next_col; ++prev) {
                float dot = 0.0f;
                for (uint32_t row = 0; row < dim; ++row) {
                    dot += candidate[row] * basis[size_t(row) * dim + prev];
                }
                for (uint32_t row = 0; row < dim; ++row) {
                    candidate[row] -= dot * basis[size_t(row) * dim + prev];
                }
            }
        }

        float norm_sq = 0.0f;
        for (float value : candidate) {
            norm_sq += value * value;
        }
        const float norm = std::sqrt(std::max(0.0f, norm_sq));
        if (norm <= 1e-8f) {
            continue;
        }
        const float inv_norm = 1.0f / norm;
        for (uint32_t row = 0; row < dim; ++row) {
            basis[size_t(row) * dim + next_col] = candidate[row] * inv_norm;
        }
        ++next_col;
    }

    return next_col == dim;
}

bool complete_basis_from_columns(
        const std::vector<float> & semantic_basis,
        uint32_t dim,
        uint32_t rank,
        std::vector<float> & basis) {
    if (accelerate_complete_basis_from_columns(semantic_basis, dim, rank, basis)) {
        return true;
    }
    return complete_basis_with_canonical_fallback(semantic_basis, dim, rank, basis);
}

bool compute_snapshot_eigensystem(
        const std::vector<float> & centered,
        uint32_t n_samples,
        uint32_t dim,
        std::vector<float> & eigenvalues,
        std::vector<float> & basis,
        uint32_t & spectral_rank) {
    if (n_samples == 0 || dim == 0 || n_samples >= dim) {
        return false;
    }

    std::vector<float> gram = compute_gram_matrix_from_centered(centered, n_samples, dim);
    std::vector<float> gram_eigenvalues;
    std::vector<float> gram_eigenvectors;
    eigendecomposition_symmetric(gram, n_samples, gram_eigenvalues, gram_eigenvectors);
    sort_eigensystem(gram_eigenvalues, gram_eigenvectors, n_samples);

    const float max_eigenvalue = gram_eigenvalues.empty() ? 0.0f : gram_eigenvalues[0];
    const float tol = std::max(1e-8f, max_eigenvalue * 1e-5f);

    spectral_rank = 0;
    while (spectral_rank < gram_eigenvalues.size() && gram_eigenvalues[spectral_rank] > tol) {
        ++spectral_rank;
    }

    eigenvalues.assign(dim, 0.0f);
    if (spectral_rank == 0) {
        set_identity_matrix(basis, dim);
        return true;
    }

    const uint32_t semantic_cols = spectral_rank;
    std::vector<float> semantic_basis(size_t(dim) * semantic_cols, 0.0f);
#ifdef LLAMA_SPECTRAL_USE_ACCELERATE
    cblas_sgemm(
            CblasRowMajor,
            CblasTrans,
            CblasNoTrans,
            (int) dim,
            (int) semantic_cols,
            (int) n_samples,
            1.0f,
            centered.data(),
            (int) dim,
            gram_eigenvectors.data(),
            (int) n_samples,
            0.0f,
            semantic_basis.data(),
            (int) semantic_cols);
#else
    for (uint32_t row = 0; row < dim; ++row) {
        for (uint32_t col = 0; col < semantic_cols; ++col) {
            float acc = 0.0f;
            for (uint32_t sample = 0; sample < n_samples; ++sample) {
                acc += centered[size_t(sample) * dim + row] * gram_eigenvectors[size_t(sample) * n_samples + col];
            }
            semantic_basis[size_t(row) * semantic_cols + col] = acc;
        }
    }
#endif

    const float denom_scale = n_samples > 1 ? float(n_samples - 1) : 1.0f;
    for (uint32_t col = 0; col < semantic_cols; ++col) {
        eigenvalues[col] = gram_eigenvalues[col];
        const float denom = std::sqrt(std::max(0.0f, gram_eigenvalues[col]) * denom_scale);
        if (denom <= 1e-12f) {
            spectral_rank = col;
            break;
        }
        const float inv_denom = 1.0f / denom;
        for (uint32_t row = 0; row < dim; ++row) {
            semantic_basis[size_t(row) * semantic_cols + col] *= inv_denom;
        }
    }

    if (spectral_rank == 0) {
        eigenvalues.assign(dim, 0.0f);
        set_identity_matrix(basis, dim);
        return true;
    }

    if (spectral_rank < semantic_cols) {
        std::vector<float> trimmed(size_t(dim) * spectral_rank, 0.0f);
        for (uint32_t row = 0; row < dim; ++row) {
            std::memcpy(trimmed.data() + size_t(row) * spectral_rank,
                    semantic_basis.data() + size_t(row) * semantic_cols,
                    size_t(spectral_rank) * sizeof(float));
        }
        semantic_basis.swap(trimmed);
    }

    normalize_columns(semantic_basis, dim, spectral_rank);
    if (!complete_basis_from_columns(semantic_basis, dim, spectral_rank, basis)) {
        return false;
    }

    sort_eigensystem(eigenvalues, basis, dim);
    return true;
}

std::vector<float> forward_rotate(const llama_spectral_entry & entry, const float * input) {
    std::vector<float> rotated(entry.dim, 0.0f);
#ifdef LLAMA_SPECTRAL_USE_ACCELERATE
    cblas_sgemv(
            CblasRowMajor,
            CblasTrans,
            (int) entry.dim,
            (int) entry.dim,
            1.0f,
            entry.basis.data(),
            (int) entry.dim,
            input,
            1,
            0.0f,
            rotated.data(),
            1);
#else
    for (uint32_t col = 0; col < entry.dim; ++col) {
        float acc = 0.0f;
        for (uint32_t row = 0; row < entry.dim; ++row) {
            acc += entry.basis[size_t(row) * entry.dim + col] * input[row];
        }
        rotated[col] = acc;
    }
#endif
    return rotated;
}

std::vector<float> inverse_rotate(const llama_spectral_entry & entry, const std::vector<float> & rotated) {
    std::vector<float> restored(entry.dim, 0.0f);
#ifdef LLAMA_SPECTRAL_USE_ACCELERATE
    cblas_sgemv(
            CblasRowMajor,
            CblasNoTrans,
            (int) entry.dim,
            (int) entry.dim,
            1.0f,
            entry.basis.data(),
            (int) entry.dim,
            rotated.data(),
            1,
            0.0f,
            restored.data(),
            1);
#else
    for (uint32_t row = 0; row < entry.dim; ++row) {
        float acc = 0.0f;
        for (uint32_t col = 0; col < entry.dim; ++col) {
            acc += entry.basis[size_t(row) * entry.dim + col] * rotated[col];
        }
        restored[row] = acc;
    }
#endif
    return restored;
}

std::vector<float> rotate_samples(
        const llama_spectral_entry & entry,
        const std::vector<float> & samples,
        uint32_t n_samples) {
    std::vector<float> rotated(size_t(n_samples) * entry.dim, 0.0f);
    if (n_samples == 0 || entry.dim == 0) {
        return rotated;
    }

#ifdef LLAMA_SPECTRAL_USE_ACCELERATE
    cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            (int) n_samples,
            (int) entry.dim,
            (int) entry.dim,
            1.0f,
            samples.data(),
            (int) entry.dim,
            entry.basis.data(),
            (int) entry.dim,
            0.0f,
            rotated.data(),
            (int) entry.dim);
#else
    for (uint32_t sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
        const float * sample = samples.data() + size_t(sample_idx) * entry.dim;
        float * dst = rotated.data() + size_t(sample_idx) * entry.dim;
        for (uint32_t col = 0; col < entry.dim; ++col) {
            float acc = 0.0f;
            for (uint32_t row = 0; row < entry.dim; ++row) {
                acc += entry.basis[size_t(row) * entry.dim + col] * sample[row];
            }
            dst[col] = acc;
        }
    }
#endif

    return rotated;
}

std::vector<float> lloyd_max(std::vector<float> values, uint32_t levels) {
    if (values.empty()) {
        return { 0.0f };
    }

    levels = std::max<uint32_t>(1, levels);
    const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    const float min_value = *min_it;
    const float max_value = *max_it;

    std::vector<float> centers(levels, min_value);
    if (levels == 1 || min_value == max_value) {
        centers[0] = std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
        return centers;
    }

    for (uint32_t i = 0; i < levels; ++i) {
        const float alpha = levels == 1 ? 0.0f : float(i) / float(levels - 1);
        centers[i] = min_value + (max_value - min_value) * alpha;
    }

    std::vector<uint32_t> assignment(values.size(), 0);
    for (int iter = 0; iter < 24; ++iter) {
        for (size_t i = 0; i < values.size(); ++i) {
            uint32_t best = 0;
            float best_dist = std::numeric_limits<float>::max();
            for (uint32_t c = 0; c < levels; ++c) {
                const float dist = std::fabs(values[i] - centers[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = c;
                }
            }
            assignment[i] = best;
        }

        std::vector<float> sums(levels, 0.0f);
        std::vector<uint32_t> counts(levels, 0);
        for (size_t i = 0; i < values.size(); ++i) {
            sums[assignment[i]] += values[i];
            counts[assignment[i]]++;
        }

        for (uint32_t c = 0; c < levels; ++c) {
            if (counts[c] > 0) {
                centers[c] = sums[c] / counts[c];
            }
        }
    }

    std::sort(centers.begin(), centers.end());
    return centers;
}

uint16_t quantize_scalar(float value, const std::vector<float> & codebook) {
    uint16_t best = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (uint16_t i = 0; i < codebook.size(); ++i) {
        const float dist = std::fabs(value - codebook[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best = i;
        }
    }
    return best;
}

const float * require_arr_data(const gguf_context * ctx, const char * key, gguf_type expected, size_t * n, std::string * err) {
    const int64_t id = gguf_find_key(ctx, key);
    if (!check(id >= 0, err, std::string("missing GGUF key: ") + key)) {
        return nullptr;
    }
    if (!check(gguf_get_kv_type(ctx, id) == GGUF_TYPE_ARRAY, err, std::string("GGUF key is not an array: ") + key)) {
        return nullptr;
    }
    if (!check(gguf_get_arr_type(ctx, id) == expected, err, std::string("GGUF array has wrong type: ") + key)) {
        return nullptr;
    }
    *n = gguf_get_arr_n(ctx, id);
    return reinterpret_cast<const float *>(gguf_get_arr_data(ctx, id));
}

template<typename T>
const T * require_typed_arr_data(const gguf_context * ctx, const char * key, gguf_type expected, size_t * n, std::string * err) {
    const int64_t id = gguf_find_key(ctx, key);
    if (!check(id >= 0, err, std::string("missing GGUF key: ") + key)) {
        return nullptr;
    }
    if (!check(gguf_get_kv_type(ctx, id) == GGUF_TYPE_ARRAY, err, std::string("GGUF key is not an array: ") + key)) {
        return nullptr;
    }
    if (!check(gguf_get_arr_type(ctx, id) == expected, err, std::string("GGUF array has wrong type: ") + key)) {
        return nullptr;
    }
    *n = gguf_get_arr_n(ctx, id);
    return reinterpret_cast<const T *>(gguf_get_arr_data(ctx, id));
}

} // namespace

const char * llama_spectral_kind_name(llama_spectral_kind kind) {
    switch (kind) {
        case LLAMA_SPECTRAL_KIND_K:      return "k";
        case LLAMA_SPECTRAL_KIND_V:      return "v";
        case LLAMA_SPECTRAL_KIND_WEIGHT: return "weight";
    }
    return "unknown";
}

const char * llama_spectral_profile_name(llama_spectral_profile profile) {
    switch (profile) {
        case LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM:         return "rotation_nonuniform";
        case LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM_SELCORR: return "rotation_nonuniform_selcorr";
    }
    return "unknown";
}

bool llama_spectral_kind_parse(const std::string & text, llama_spectral_kind & kind) {
    if (text == "k" || text == "key") {
        kind = LLAMA_SPECTRAL_KIND_K;
        return true;
    }
    if (text == "v" || text == "value") {
        kind = LLAMA_SPECTRAL_KIND_V;
        return true;
    }
    if (text == "weight" || text == "weights") {
        kind = LLAMA_SPECTRAL_KIND_WEIGHT;
        return true;
    }
    return false;
}

bool llama_spectral_profile_parse(const std::string & text, llama_spectral_profile & profile) {
    if (text == "rotation_nonuniform" || text == "nonuniform") {
        profile = LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM;
        return true;
    }
    if (text == "rotation_nonuniform_selcorr" || text == "selcorr") {
        profile = LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM_SELCORR;
        return true;
    }
    return false;
}

float llama_spectral_compute_d_eff(const std::vector<float> & eigenvalues) {
    double sum = 0.0;
    double sum_sq = 0.0;
    for (float value : eigenvalues) {
        const double clamped = std::max(0.0f, value);
        sum += clamped;
        sum_sq += clamped * clamped;
    }
    if (sum <= 0.0 || sum_sq <= 0.0) {
        return eigenvalues.empty() ? 0.0f : 1.0f;
    }
    return float((sum * sum) / sum_sq);
}

uint32_t llama_spectral_choose_split(float d_eff, uint32_t dim) {
    if (dim == 0) {
        return 0;
    }
    const uint32_t rounded = uint32_t(std::lround(std::max(1.0f, std::min(d_eff, float(dim)))));
    return std::max<uint32_t>(1, std::min<uint32_t>(rounded, dim));
}

bool llama_spectral_calibrate_entry(
        const std::string & name,
        const std::vector<float> & samples,
        uint32_t n_samples,
        uint32_t dim,
        const llama_spectral_calibrate_params & params,
        llama_spectral_entry & out,
        std::string * err,
        llama_spectral_calibration_metrics * metrics) {
    const auto total_t0 = spectral_clock::now();
    if (!check(dim > 0, err, "spectral calibration requires dim > 0")) {
        return false;
    }
    if (!check(n_samples > 0, err, "spectral calibration requires at least one sample")) {
        return false;
    }
    if (!check(samples.size() == size_t(n_samples) * dim, err, "spectral calibration input size mismatch")) {
        return false;
    }

    const std::vector<float> centered = center_samples(samples, n_samples, dim);

    const auto cov_t0 = spectral_clock::now();
    std::vector<float> cov_or_gram;
    if (n_samples < dim) {
        cov_or_gram = compute_gram_matrix_from_centered(centered, n_samples, dim);
    } else {
        cov_or_gram = compute_covariance_from_centered(centered, n_samples, dim);
    }
    const auto cov_t1 = spectral_clock::now();

    const auto eigen_t0 = cov_t1;
    std::vector<float> eigenvalues;
    std::vector<float> basis;
    uint32_t spectral_rank = dim;
    bool used_snapshot = false;
    if (n_samples < dim) {
        used_snapshot = compute_snapshot_eigensystem(centered, n_samples, dim, eigenvalues, basis, spectral_rank);
    }
    if (!used_snapshot) {
        if (n_samples < dim) {
            cov_or_gram = compute_covariance_from_centered(centered, n_samples, dim);
        }
        eigendecomposition_symmetric(cov_or_gram, dim, eigenvalues, basis);
        sort_eigensystem(eigenvalues, basis, dim);
        spectral_rank = dim;
    }
    const auto eigen_t1 = spectral_clock::now();

    const float d_eff = std::min<float>(float(dim), std::max<float>(1.0f, llama_spectral_compute_d_eff(eigenvalues)));
    const uint32_t split = llama_spectral_choose_split(d_eff, dim);
    const uint32_t semantic_bits = clamp_bits(std::min<uint32_t>(params.base_bits + (split < dim ? 1 : 0), params.max_semantic_bits));
    const uint32_t tail_bits = clamp_bits(split < dim ? std::max<uint32_t>(params.min_tail_bits, params.base_bits > 1 ? params.base_bits - 1 : 1) : semantic_bits);
    const uint32_t correction_dim =
            (params.profile == LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM_SELCORR && params.kind != LLAMA_SPECTRAL_KIND_V)
            ? std::min<uint32_t>(split, params.correction_dim == 0 ? split : params.correction_dim)
            : 0;

    llama_spectral_entry entry;
    entry.name = name;
    entry.kind = params.kind;
    entry.profile = params.profile;
    entry.dim = dim;
    entry.split = split;
    entry.semantic_bits = semantic_bits;
    entry.tail_bits = tail_bits;
    entry.correction_dim = correction_dim;
    entry.d_eff = d_eff;
    entry.eigenvalues = eigenvalues;
    entry.basis = basis;

    std::vector<float> semantic_values;
    std::vector<float> tail_values;
    semantic_values.reserve(size_t(n_samples) * split);
    tail_values.reserve(size_t(n_samples) * (dim - split));

    const auto rotate_t0 = eigen_t1;
    const std::vector<float> rotated_samples = rotate_samples(entry, samples, n_samples);
    const auto rotate_t1 = spectral_clock::now();

    for (uint32_t sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
        const float * rotated = rotated_samples.data() + size_t(sample_idx) * dim;
        semantic_values.insert(semantic_values.end(), rotated, rotated + split);
        tail_values.insert(tail_values.end(), rotated + split, rotated + dim);
    }

    const auto codebook_t0 = rotate_t1;
    entry.semantic_codebook = lloyd_max(std::move(semantic_values), 1u << semantic_bits);
    entry.tail_codebook = lloyd_max(std::move(tail_values), 1u << tail_bits);
    const auto codebook_t1 = spectral_clock::now();

    if (!validate_entry(entry, err)) {
        return false;
    }

    out = std::move(entry);
    if (metrics != nullptr) {
        metrics->covariance_ms = elapsed_ms(cov_t0, cov_t1);
        metrics->eigen_ms      = elapsed_ms(eigen_t0, eigen_t1);
        metrics->rotate_ms     = elapsed_ms(rotate_t0, rotate_t1);
        metrics->codebook_ms   = elapsed_ms(codebook_t0, codebook_t1);
        metrics->total_ms      = elapsed_ms(total_t0, codebook_t1);
        metrics->used_snapshot = used_snapshot;
        metrics->spectral_rank = spectral_rank;
    }
    return true;
}

bool llama_spectral_encode_vector(
        const llama_spectral_entry & entry,
        const float * input,
        size_t input_size,
        llama_spectral_encoded_vector & encoded,
        std::string * err) {
    llama_spectral_entry prepared = entry;
    if (!validate_entry(prepared, err)) {
        return false;
    }
    if (!check(input != nullptr, err, "spectral encode received null input")) {
        return false;
    }
    if (!materialize_full_basis(entry, prepared.basis, err)) {
        return false;
    }
    if (!validate_entry(prepared, err)) {
        return false;
    }
    if (!check(input_size == prepared.dim, err, "spectral encode input size mismatch")) {
        return false;
    }

    std::vector<float> rotated = forward_rotate(prepared, input);

    encoded.semantic_codes.resize(prepared.split);
    encoded.tail_codes.resize(prepared.dim - prepared.split);
    encoded.correction.clear();
    encoded.correction.reserve(prepared.correction_dim);

    for (uint32_t i = 0; i < prepared.split; ++i) {
        encoded.semantic_codes[i] = quantize_scalar(rotated[i], prepared.semantic_codebook);
    }
    for (uint32_t i = prepared.split; i < prepared.dim; ++i) {
        encoded.tail_codes[i - prepared.split] = quantize_scalar(rotated[i], prepared.tail_codebook);
    }
    for (uint32_t i = 0; i < prepared.correction_dim; ++i) {
        encoded.correction.push_back(rotated[i]);
    }

    return true;
}

bool llama_spectral_decode_vector(
        const llama_spectral_entry & entry,
        const llama_spectral_encoded_vector & encoded,
        std::vector<float> & output,
        std::string * err) {
    llama_spectral_entry prepared = entry;
    if (!validate_entry(prepared, err)) {
        return false;
    }
    if (!materialize_full_basis(entry, prepared.basis, err)) {
        return false;
    }
    if (!validate_entry(prepared, err)) {
        return false;
    }
    if (!check(encoded.semantic_codes.size() == prepared.split, err, "spectral decode semantic code size mismatch")) {
        return false;
    }
    if (!check(encoded.tail_codes.size() == prepared.dim - prepared.split, err, "spectral decode tail code size mismatch")) {
        return false;
    }
    if (!check(encoded.correction.size() == prepared.correction_dim, err, "spectral decode correction size mismatch")) {
        return false;
    }

    std::vector<float> rotated(prepared.dim, 0.0f);
    for (uint32_t i = 0; i < prepared.split; ++i) {
        const uint16_t code = encoded.semantic_codes[i];
        if (!check(code < prepared.semantic_codebook.size(), err, "spectral semantic code out of range")) {
            return false;
        }
        rotated[i] = prepared.semantic_codebook[code];
    }
    for (uint32_t i = prepared.split; i < prepared.dim; ++i) {
        const uint16_t code = encoded.tail_codes[i - prepared.split];
        if (!check(code < prepared.tail_codebook.size(), err, "spectral tail code out of range")) {
            return false;
        }
        rotated[i] = prepared.tail_codebook[code];
    }
    for (uint32_t i = 0; i < prepared.correction_dim; ++i) {
        rotated[i] = encoded.correction[i];
    }

    output = inverse_rotate(prepared, rotated);
    return true;
}

bool llama_spectral_validate_artifact(
        const llama_spectral_artifact & artifact,
        std::string * err) {
    std::unordered_set<std::string> seen;
    seen.reserve(artifact.entries.size());

    for (const auto & entry : artifact.entries) {
        if (!validate_entry(entry, err)) {
            return false;
        }

        const std::string key = entry.name + '\x1f' + std::to_string(entry.kind) + '\x1f' + std::to_string(entry.profile);
        if (!seen.insert(key).second) {
            set_error(err, "duplicate spectral entry for the same name/kind/profile");
            return false;
        }
    }

    return true;
}

std::string llama_spectral_compute_tensor_digest(
        const char * architecture,
        const ggml_context * ctx) {
    std::vector<const ggml_tensor *> tensors;
    if (ctx != nullptr) {
        for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor != nullptr; tensor = ggml_get_next_tensor(ctx, tensor)) {
            tensors.push_back(tensor);
        }
    }
    return llama_spectral_compute_tensor_digest(architecture, tensors);
}

std::string llama_spectral_compute_tensor_digest(
        const char * architecture,
        const std::vector<const ggml_tensor *> & tensors) {
    std::vector<const ggml_tensor *> sorted;
    sorted.reserve(tensors.size());
    for (const ggml_tensor * tensor : tensors) {
        if (tensor != nullptr) {
            sorted.push_back(tensor);
        }
    }

    std::sort(sorted.begin(), sorted.end(), [](const ggml_tensor * lhs, const ggml_tensor * rhs) {
        return std::strcmp(ggml_get_name(lhs), ggml_get_name(rhs)) < 0;
    });

    uint64_t hash = 0xcbf29ce484222325ULL;
    spectral_digest_update_str(hash, "spectral.tensor_digest.v1");
    spectral_digest_update_str(hash, architecture ? architecture : "");

    const uint64_t n_tensors = sorted.size();
    spectral_digest_update_pod(hash, n_tensors);

    for (const ggml_tensor * tensor : sorted) {
        spectral_digest_update_str(hash, ggml_get_name(tensor));
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            const int64_t dim = tensor->ne[i];
            spectral_digest_update_pod(hash, dim);
        }
    }

    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "fnv1a64:%016llx", (unsigned long long) hash);
    return buffer;
}

bool llama_spectral_filter_artifact(
        const llama_spectral_artifact & artifact,
        llama_spectral_kind kind,
        const char * profile_filter,
        llama_spectral_artifact & filtered,
        std::string * err) {
    if (!llama_spectral_validate_artifact(artifact, err)) {
        return false;
    }

    bool use_all_profiles = true;
    llama_spectral_profile requested_profile = LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM;
    if (!parse_profile_filter(profile_filter, use_all_profiles, requested_profile, err)) {
        return false;
    }

    filtered = {};
    filtered.version = artifact.version;
    filtered.source_architecture = artifact.source_architecture;
    filtered.source_digest = artifact.source_digest;

    for (const auto & entry : artifact.entries) {
        if (entry.kind != kind) {
            continue;
        }
        if (!use_all_profiles && entry.profile != requested_profile) {
            continue;
        }
        filtered.entries.push_back(entry);
    }

    return true;
}

bool llama_spectral_prepare_weight_entry(
        const llama_spectral_entry & entry,
        ggml_type type,
        llama_spectral_entry & prepared,
        std::string * err) {
    uint32_t bits = 0;
    if (!spectral_type_bits(type, bits)) {
        set_error(err, "unsupported spectral weight type");
        return false;
    }
    if (entry.kind != LLAMA_SPECTRAL_KIND_WEIGHT) {
        set_error(err, "spectral entry is not a weight profile");
        return false;
    }
    if (!validate_entry(entry, err)) {
        return false;
    }

    prepared = entry;
    if (!materialize_full_basis(entry, prepared.basis, err)) {
        return false;
    }
    const uint32_t max_levels = 1u << bits;
    prepared.semantic_codebook = resample_codebook(entry.semantic_codebook, max_levels);
    prepared.tail_codebook = resample_codebook(entry.tail_codebook, max_levels);
    prepared.semantic_bits = std::min<uint32_t>(bits, entry.semantic_bits);
    prepared.tail_bits = std::min<uint32_t>(bits, entry.tail_bits);
    prepared.correction_dim = entry.profile == LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM_SELCORR
            ? std::min<uint32_t>(entry.correction_dim, 8u)
            : 0u;
    prepared.eigenvalues.clear();
    prepared.eigenvalues.shrink_to_fit();
    prepared.name.clear();
    prepared.name.shrink_to_fit();

    if (!validate_entry(prepared, err)) {
        return false;
    }
    return true;
}

bool llama_spectral_prepare_kv_entry(
        const llama_spectral_entry & entry,
        ggml_type type,
        llama_spectral_entry & prepared,
        std::string * err) {
    uint32_t bits = 0;
    if (!spectral_kv_type_bits(type, bits)) {
        set_error(err, "unsupported spectral KV type");
        return false;
    }
    if (entry.kind != LLAMA_SPECTRAL_KIND_K && entry.kind != LLAMA_SPECTRAL_KIND_V) {
        set_error(err, "spectral entry is not a KV profile");
        return false;
    }
    if (!validate_entry(entry, err)) {
        return false;
    }

    prepared = entry;
    if (!materialize_full_basis(entry, prepared.basis, err)) {
        return false;
    }

    uint32_t semantic_bits = std::min<uint32_t>(bits, entry.semantic_bits);
    if (entry.kind == LLAMA_SPECTRAL_KIND_K &&
        entry.profile == LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM_SELCORR &&
        semantic_bits > 1) {
        semantic_bits -= 1;
    }

    prepared.semantic_codebook = resample_codebook(entry.semantic_codebook, 1u << semantic_bits);
    prepared.tail_codebook = resample_codebook(entry.tail_codebook, 1u << std::min<uint32_t>(bits, entry.tail_bits));
    prepared.semantic_bits = std::max<uint32_t>(1, semantic_bits);
    prepared.tail_bits = std::min<uint32_t>(bits, entry.tail_bits);
    prepared.correction_dim =
            entry.kind == LLAMA_SPECTRAL_KIND_K &&
            entry.profile == LLAMA_SPECTRAL_PROFILE_ROTATION_NONUNIFORM_SELCORR
            ? entry.split
            : 0u;

    if (!validate_entry(prepared, err)) {
        return false;
    }
    return true;
}

bool llama_spectral_save_gguf(
        const llama_spectral_artifact & artifact,
        const std::string & path,
        std::string * err) {
    if (!llama_spectral_validate_artifact(artifact, err)) {
        return false;
    }

    size_t mem_size = 0;
    for (const auto & entry : artifact.entries) {
        mem_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * size_t(entry.dim) * compact_basis_columns(entry), GGML_MEM_ALIGN);
        mem_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * entry.eigenvalues.size(), GGML_MEM_ALIGN);
        mem_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * entry.semantic_codebook.size(), GGML_MEM_ALIGN);
        mem_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * entry.tail_codebook.size(), GGML_MEM_ALIGN);
    }
    if (mem_size == 0) {
        mem_size = GGML_MEM_ALIGN;
    }

    ggml_init_params ggml_params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_context_ptr ggml_ctx(ggml_init(ggml_params));
    gguf_context_ptr gguf_ctx(gguf_init_empty());
    if (!check(ggml_ctx != nullptr && gguf_ctx != nullptr, err, "failed to allocate spectral GGUF context")) {
        return false;
    }

    gguf_set_val_str(gguf_ctx.get(), LLAMA_SPECTRAL_GENERAL_TYPE, LLAMA_SPECTRAL_GENERAL_TYPE_VALUE);
    gguf_set_val_u32(gguf_ctx.get(), LLAMA_SPECTRAL_VERSION, artifact.version);
    if (!artifact.source_architecture.empty()) {
        gguf_set_val_str(gguf_ctx.get(), LLAMA_SPECTRAL_SOURCE_ARCH, artifact.source_architecture.c_str());
    }
    if (!artifact.source_digest.empty()) {
        gguf_set_val_str(gguf_ctx.get(), LLAMA_SPECTRAL_SOURCE_DIGEST, artifact.source_digest.c_str());
    }

    std::vector<const char *> names;
    std::vector<uint32_t> kinds;
    std::vector<uint32_t> profiles;
    std::vector<uint32_t> dims;
    std::vector<uint32_t> splits;
    std::vector<float> d_effs;
    std::vector<uint32_t> semantic_bits;
    std::vector<uint32_t> tail_bits;
    std::vector<uint32_t> correction_dims;

    names.reserve(artifact.entries.size());
    kinds.reserve(artifact.entries.size());
    profiles.reserve(artifact.entries.size());
    dims.reserve(artifact.entries.size());
    splits.reserve(artifact.entries.size());
    d_effs.reserve(artifact.entries.size());
    semantic_bits.reserve(artifact.entries.size());
    tail_bits.reserve(artifact.entries.size());
    correction_dims.reserve(artifact.entries.size());

    for (size_t i = 0; i < artifact.entries.size(); ++i) {
        const auto & entry = artifact.entries[i];
        const uint32_t basis_cols = compact_basis_columns(entry);
        const std::vector<float> compact_basis = trim_basis_columns(entry, basis_cols);

        names.push_back(entry.name.c_str());
        kinds.push_back(entry.kind);
        profiles.push_back(entry.profile);
        dims.push_back(entry.dim);
        splits.push_back(entry.split);
        d_effs.push_back(entry.d_eff);
        semantic_bits.push_back(entry.semantic_bits);
        tail_bits.push_back(entry.tail_bits);
        correction_dims.push_back(entry.correction_dim);

        ggml_tensor * basis = ggml_new_tensor_2d(ggml_ctx.get(), GGML_TYPE_F32, entry.dim, basis_cols);
        ggml_tensor * eigenvalues = ggml_new_tensor_1d(ggml_ctx.get(), GGML_TYPE_F32, entry.eigenvalues.size());
        ggml_tensor * semantic = ggml_new_tensor_1d(ggml_ctx.get(), GGML_TYPE_F32, entry.semantic_codebook.size());
        ggml_tensor * tail = ggml_new_tensor_1d(ggml_ctx.get(), GGML_TYPE_F32, entry.tail_codebook.size());

        if (!check(basis != nullptr && eigenvalues != nullptr && semantic != nullptr && tail != nullptr, err, "failed to allocate spectral tensors")) {
            return false;
        }

        ggml_format_name(basis, "%s", tensor_name_for(i, "basis").c_str());
        ggml_format_name(eigenvalues, "%s", tensor_name_for(i, "eigenvalues").c_str());
        ggml_format_name(semantic, "%s", tensor_name_for(i, "semantic_codebook").c_str());
        ggml_format_name(tail, "%s", tensor_name_for(i, "tail_codebook").c_str());

        std::copy(compact_basis.begin(), compact_basis.end(), static_cast<float *>(basis->data));
        std::copy(entry.eigenvalues.begin(), entry.eigenvalues.end(), static_cast<float *>(eigenvalues->data));
        std::copy(entry.semantic_codebook.begin(), entry.semantic_codebook.end(), static_cast<float *>(semantic->data));
        std::copy(entry.tail_codebook.begin(), entry.tail_codebook.end(), static_cast<float *>(tail->data));

        gguf_add_tensor(gguf_ctx.get(), basis);
        gguf_add_tensor(gguf_ctx.get(), eigenvalues);
        gguf_add_tensor(gguf_ctx.get(), semantic);
        gguf_add_tensor(gguf_ctx.get(), tail);
    }

    gguf_set_arr_str(gguf_ctx.get(), LLAMA_SPECTRAL_ENTRY_NAMES, names.data(), names.size());
    gguf_set_arr_data(gguf_ctx.get(), LLAMA_SPECTRAL_ENTRY_KINDS, GGUF_TYPE_UINT32, kinds.data(), kinds.size());
    gguf_set_arr_data(gguf_ctx.get(), LLAMA_SPECTRAL_ENTRY_PROFILES, GGUF_TYPE_UINT32, profiles.data(), profiles.size());
    gguf_set_arr_data(gguf_ctx.get(), LLAMA_SPECTRAL_ENTRY_DIMS, GGUF_TYPE_UINT32, dims.data(), dims.size());
    gguf_set_arr_data(gguf_ctx.get(), LLAMA_SPECTRAL_ENTRY_SPLITS, GGUF_TYPE_UINT32, splits.data(), splits.size());
    gguf_set_arr_data(gguf_ctx.get(), LLAMA_SPECTRAL_ENTRY_D_EFF, GGUF_TYPE_FLOAT32, d_effs.data(), d_effs.size());
    gguf_set_arr_data(gguf_ctx.get(), LLAMA_SPECTRAL_ENTRY_SEM_BITS, GGUF_TYPE_UINT32, semantic_bits.data(), semantic_bits.size());
    gguf_set_arr_data(gguf_ctx.get(), LLAMA_SPECTRAL_ENTRY_TAIL_BITS, GGUF_TYPE_UINT32, tail_bits.data(), tail_bits.size());
    gguf_set_arr_data(gguf_ctx.get(), LLAMA_SPECTRAL_ENTRY_CORR_DIMS, GGUF_TYPE_UINT32, correction_dims.data(), correction_dims.size());

    if (!gguf_write_to_file(gguf_ctx.get(), path.c_str(), false)) {
        set_error(err, "failed to write spectral GGUF artifact");
        return false;
    }

    return true;
}

bool llama_spectral_embed_gguf_metadata(
        const llama_spectral_artifact & artifact,
        struct gguf_context * ctx,
        std::string * err) {
    if (!check(ctx != nullptr, err, "cannot embed spectral metadata into null GGUF context")) {
        return false;
    }

    if (!llama_spectral_validate_artifact(artifact, err)) {
        return false;
    }

    gguf_set_val_u32(ctx, LLAMA_SPECTRAL_VERSION, artifact.version);
    if (!artifact.source_architecture.empty()) {
        gguf_set_val_str(ctx, LLAMA_SPECTRAL_SOURCE_ARCH, artifact.source_architecture.c_str());
    }
    if (!artifact.source_digest.empty()) {
        gguf_set_val_str(ctx, LLAMA_SPECTRAL_SOURCE_DIGEST, artifact.source_digest.c_str());
    }

    std::vector<const char *> names;
    std::vector<uint32_t> kinds;
    std::vector<uint32_t> profiles;
    std::vector<uint32_t> dims;
    std::vector<uint32_t> splits;
    std::vector<float> d_effs;
    std::vector<uint32_t> semantic_bits;
    std::vector<uint32_t> tail_bits;
    std::vector<uint32_t> correction_dims;

    names.reserve(artifact.entries.size());
    kinds.reserve(artifact.entries.size());
    profiles.reserve(artifact.entries.size());
    dims.reserve(artifact.entries.size());
    splits.reserve(artifact.entries.size());
    d_effs.reserve(artifact.entries.size());
    semantic_bits.reserve(artifact.entries.size());
    tail_bits.reserve(artifact.entries.size());
    correction_dims.reserve(artifact.entries.size());

    for (size_t i = 0; i < artifact.entries.size(); ++i) {
        const auto & entry = artifact.entries[i];
        const std::vector<float> compact_basis = trim_basis_columns(entry, compact_basis_columns(entry));
        names.push_back(entry.name.c_str());
        kinds.push_back(entry.kind);
        profiles.push_back(entry.profile);
        dims.push_back(entry.dim);
        splits.push_back(entry.split);
        d_effs.push_back(entry.d_eff);
        semantic_bits.push_back(entry.semantic_bits);
        tail_bits.push_back(entry.tail_bits);
        correction_dims.push_back(entry.correction_dim);

        gguf_set_arr_data(ctx, metadata_name_for(i, "basis").c_str(), GGUF_TYPE_FLOAT32, compact_basis.data(), compact_basis.size());
        gguf_set_arr_data(ctx, metadata_name_for(i, "eigenvalues").c_str(), GGUF_TYPE_FLOAT32, entry.eigenvalues.data(), entry.eigenvalues.size());
        gguf_set_arr_data(ctx, metadata_name_for(i, "semantic_codebook").c_str(), GGUF_TYPE_FLOAT32, entry.semantic_codebook.data(), entry.semantic_codebook.size());
        gguf_set_arr_data(ctx, metadata_name_for(i, "tail_codebook").c_str(), GGUF_TYPE_FLOAT32, entry.tail_codebook.data(), entry.tail_codebook.size());
    }

    gguf_set_arr_str(ctx, LLAMA_SPECTRAL_ENTRY_NAMES, names.data(), names.size());
    gguf_set_arr_data(ctx, LLAMA_SPECTRAL_ENTRY_KINDS, GGUF_TYPE_UINT32, kinds.data(), kinds.size());
    gguf_set_arr_data(ctx, LLAMA_SPECTRAL_ENTRY_PROFILES, GGUF_TYPE_UINT32, profiles.data(), profiles.size());
    gguf_set_arr_data(ctx, LLAMA_SPECTRAL_ENTRY_DIMS, GGUF_TYPE_UINT32, dims.data(), dims.size());
    gguf_set_arr_data(ctx, LLAMA_SPECTRAL_ENTRY_SPLITS, GGUF_TYPE_UINT32, splits.data(), splits.size());
    gguf_set_arr_data(ctx, LLAMA_SPECTRAL_ENTRY_D_EFF, GGUF_TYPE_FLOAT32, d_effs.data(), d_effs.size());
    gguf_set_arr_data(ctx, LLAMA_SPECTRAL_ENTRY_SEM_BITS, GGUF_TYPE_UINT32, semantic_bits.data(), semantic_bits.size());
    gguf_set_arr_data(ctx, LLAMA_SPECTRAL_ENTRY_TAIL_BITS, GGUF_TYPE_UINT32, tail_bits.data(), tail_bits.size());
    gguf_set_arr_data(ctx, LLAMA_SPECTRAL_ENTRY_CORR_DIMS, GGUF_TYPE_UINT32, correction_dims.data(), correction_dims.size());

    return true;
}

bool llama_spectral_load_gguf(
        const std::string & path,
        llama_spectral_artifact & artifact,
        std::string * err) {
    ggml_context * ggml_ctx_raw = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &ggml_ctx_raw,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(path.c_str(), params));
    ggml_context_ptr ggml_ctx(ggml_ctx_raw);
    if (!check(gguf_ctx != nullptr && ggml_ctx != nullptr, err, "failed to open spectral GGUF artifact")) {
        return false;
    }

    const int64_t type_id = gguf_find_key(gguf_ctx.get(), LLAMA_SPECTRAL_GENERAL_TYPE);
    if (!check(type_id >= 0, err, "GGUF artifact is missing general.type")) {
        return false;
    }
    if (!check(std::string(gguf_get_val_str(gguf_ctx.get(), type_id)) == LLAMA_SPECTRAL_GENERAL_TYPE_VALUE,
               err, "GGUF artifact is not a spectral calibration file")) {
        return false;
    }

    artifact = {};
    {
        const int64_t version_id = gguf_find_key(gguf_ctx.get(), LLAMA_SPECTRAL_VERSION);
        artifact.version = version_id >= 0 ? gguf_get_val_u32(gguf_ctx.get(), version_id) : 1;
    }
    {
        const int64_t arch_id = gguf_find_key(gguf_ctx.get(), LLAMA_SPECTRAL_SOURCE_ARCH);
        if (arch_id >= 0) {
            artifact.source_architecture = gguf_get_val_str(gguf_ctx.get(), arch_id);
        }
    }
    {
        const int64_t digest_id = gguf_find_key(gguf_ctx.get(), LLAMA_SPECTRAL_SOURCE_DIGEST);
        if (digest_id >= 0) {
            artifact.source_digest = gguf_get_val_str(gguf_ctx.get(), digest_id);
        }
    }

    const int64_t names_id = gguf_find_key(gguf_ctx.get(), LLAMA_SPECTRAL_ENTRY_NAMES);
    if (!check(names_id >= 0, err, "spectral artifact is missing entry names")) {
        return false;
    }
    if (!check(gguf_get_kv_type(gguf_ctx.get(), names_id) == GGUF_TYPE_ARRAY &&
               gguf_get_arr_type(gguf_ctx.get(), names_id) == GGUF_TYPE_STRING,
               err, "spectral entry names have wrong type")) {
        return false;
    }

    const size_t n_entries = gguf_get_arr_n(gguf_ctx.get(), names_id);
    size_t n = 0;

    const uint32_t * kinds = require_typed_arr_data<uint32_t>(gguf_ctx.get(), LLAMA_SPECTRAL_ENTRY_KINDS, GGUF_TYPE_UINT32, &n, err);
    if (!kinds || n != n_entries) {
        set_error(err, "spectral kinds array size mismatch");
        return false;
    }
    const uint32_t * profiles = require_typed_arr_data<uint32_t>(gguf_ctx.get(), LLAMA_SPECTRAL_ENTRY_PROFILES, GGUF_TYPE_UINT32, &n, err);
    if (!profiles || n != n_entries) {
        set_error(err, "spectral profiles array size mismatch");
        return false;
    }
    const uint32_t * dims = require_typed_arr_data<uint32_t>(gguf_ctx.get(), LLAMA_SPECTRAL_ENTRY_DIMS, GGUF_TYPE_UINT32, &n, err);
    if (!dims || n != n_entries) {
        set_error(err, "spectral dims array size mismatch");
        return false;
    }
    const uint32_t * splits = require_typed_arr_data<uint32_t>(gguf_ctx.get(), LLAMA_SPECTRAL_ENTRY_SPLITS, GGUF_TYPE_UINT32, &n, err);
    if (!splits || n != n_entries) {
        set_error(err, "spectral splits array size mismatch");
        return false;
    }
    const float * d_effs = require_typed_arr_data<float>(gguf_ctx.get(), LLAMA_SPECTRAL_ENTRY_D_EFF, GGUF_TYPE_FLOAT32, &n, err);
    if (!d_effs || n != n_entries) {
        set_error(err, "spectral d_eff array size mismatch");
        return false;
    }
    const uint32_t * semantic_bits = require_typed_arr_data<uint32_t>(gguf_ctx.get(), LLAMA_SPECTRAL_ENTRY_SEM_BITS, GGUF_TYPE_UINT32, &n, err);
    if (!semantic_bits || n != n_entries) {
        set_error(err, "spectral semantic bits array size mismatch");
        return false;
    }
    const uint32_t * tail_bits = require_typed_arr_data<uint32_t>(gguf_ctx.get(), LLAMA_SPECTRAL_ENTRY_TAIL_BITS, GGUF_TYPE_UINT32, &n, err);
    if (!tail_bits || n != n_entries) {
        set_error(err, "spectral tail bits array size mismatch");
        return false;
    }
    const uint32_t * correction_dims = require_typed_arr_data<uint32_t>(gguf_ctx.get(), LLAMA_SPECTRAL_ENTRY_CORR_DIMS, GGUF_TYPE_UINT32, &n, err);
    if (!correction_dims || n != n_entries) {
        set_error(err, "spectral correction dims array size mismatch");
        return false;
    }

    artifact.entries.reserve(n_entries);
    for (size_t i = 0; i < n_entries; ++i) {
        llama_spectral_entry entry;
        entry.name = gguf_get_arr_str(gguf_ctx.get(), names_id, i);
        entry.kind = static_cast<llama_spectral_kind>(kinds[i]);
        entry.profile = static_cast<llama_spectral_profile>(profiles[i]);
        entry.dim = dims[i];
        entry.split = splits[i];
        entry.d_eff = d_effs[i];
        entry.semantic_bits = semantic_bits[i];
        entry.tail_bits = tail_bits[i];
        entry.correction_dim = correction_dims[i];

        ggml_tensor * basis = ggml_get_tensor(ggml_ctx.get(), tensor_name_for(i, "basis").c_str());
        ggml_tensor * eigenvalues = ggml_get_tensor(ggml_ctx.get(), tensor_name_for(i, "eigenvalues").c_str());
        ggml_tensor * semantic = ggml_get_tensor(ggml_ctx.get(), tensor_name_for(i, "semantic_codebook").c_str());
        ggml_tensor * tail = ggml_get_tensor(ggml_ctx.get(), tensor_name_for(i, "tail_codebook").c_str());

        if (!check(basis != nullptr && eigenvalues != nullptr && semantic != nullptr && tail != nullptr, err, "spectral artifact is missing tensor payloads")) {
            return false;
        }
        if (!check(ggml_nelements(basis) % entry.dim == 0, err, "spectral basis tensor size mismatch")) {
            return false;
        }
        if (!check(ggml_nelements(eigenvalues) == entry.dim, err, "spectral eigenvalues tensor size mismatch")) {
            return false;
        }

        entry.basis.assign(static_cast<const float *>(basis->data), static_cast<const float *>(basis->data) + ggml_nelements(basis));
        entry.eigenvalues.assign(static_cast<const float *>(eigenvalues->data), static_cast<const float *>(eigenvalues->data) + ggml_nelements(eigenvalues));
        entry.semantic_codebook.assign(static_cast<const float *>(semantic->data), static_cast<const float *>(semantic->data) + ggml_nelements(semantic));
        entry.tail_codebook.assign(static_cast<const float *>(tail->data), static_cast<const float *>(tail->data) + ggml_nelements(tail));

        if (!validate_entry(entry, err)) {
            return false;
        }

        artifact.entries.push_back(std::move(entry));
    }

    return true;
}

bool llama_spectral_extract_gguf_metadata(
        const struct gguf_context * ctx,
        llama_spectral_artifact & artifact,
        std::string * err) {
    if (!check(ctx != nullptr, err, "cannot extract spectral metadata from null GGUF context")) {
        return false;
    }

    const int64_t names_id = gguf_find_key(ctx, LLAMA_SPECTRAL_ENTRY_NAMES);
    if (names_id < 0) {
        artifact = {};
        return true;
    }

    if (!check(gguf_get_kv_type(ctx, names_id) == GGUF_TYPE_ARRAY &&
               gguf_get_arr_type(ctx, names_id) == GGUF_TYPE_STRING,
               err, "spectral entry names have wrong type")) {
        return false;
    }

    artifact = {};
    {
        const int64_t version_id = gguf_find_key(ctx, LLAMA_SPECTRAL_VERSION);
        artifact.version = version_id >= 0 ? gguf_get_val_u32(ctx, version_id) : 1;
    }
    {
        const int64_t arch_id = gguf_find_key(ctx, LLAMA_SPECTRAL_SOURCE_ARCH);
        if (arch_id >= 0) {
            artifact.source_architecture = gguf_get_val_str(ctx, arch_id);
        }
    }
    {
        const int64_t digest_id = gguf_find_key(ctx, LLAMA_SPECTRAL_SOURCE_DIGEST);
        if (digest_id >= 0) {
            artifact.source_digest = gguf_get_val_str(ctx, digest_id);
        }
    }

    const size_t n_entries = gguf_get_arr_n(ctx, names_id);
    size_t n = 0;

    const uint32_t * kinds = require_typed_arr_data<uint32_t>(ctx, LLAMA_SPECTRAL_ENTRY_KINDS, GGUF_TYPE_UINT32, &n, err);
    if (!kinds || n != n_entries) {
        set_error(err, "spectral kinds array size mismatch");
        return false;
    }
    const uint32_t * profiles = require_typed_arr_data<uint32_t>(ctx, LLAMA_SPECTRAL_ENTRY_PROFILES, GGUF_TYPE_UINT32, &n, err);
    if (!profiles || n != n_entries) {
        set_error(err, "spectral profiles array size mismatch");
        return false;
    }
    const uint32_t * dims = require_typed_arr_data<uint32_t>(ctx, LLAMA_SPECTRAL_ENTRY_DIMS, GGUF_TYPE_UINT32, &n, err);
    if (!dims || n != n_entries) {
        set_error(err, "spectral dims array size mismatch");
        return false;
    }
    const uint32_t * splits = require_typed_arr_data<uint32_t>(ctx, LLAMA_SPECTRAL_ENTRY_SPLITS, GGUF_TYPE_UINT32, &n, err);
    if (!splits || n != n_entries) {
        set_error(err, "spectral splits array size mismatch");
        return false;
    }
    const float * d_effs = require_typed_arr_data<float>(ctx, LLAMA_SPECTRAL_ENTRY_D_EFF, GGUF_TYPE_FLOAT32, &n, err);
    if (!d_effs || n != n_entries) {
        set_error(err, "spectral d_eff array size mismatch");
        return false;
    }
    const uint32_t * semantic_bits = require_typed_arr_data<uint32_t>(ctx, LLAMA_SPECTRAL_ENTRY_SEM_BITS, GGUF_TYPE_UINT32, &n, err);
    if (!semantic_bits || n != n_entries) {
        set_error(err, "spectral semantic bits array size mismatch");
        return false;
    }
    const uint32_t * tail_bits = require_typed_arr_data<uint32_t>(ctx, LLAMA_SPECTRAL_ENTRY_TAIL_BITS, GGUF_TYPE_UINT32, &n, err);
    if (!tail_bits || n != n_entries) {
        set_error(err, "spectral tail bits array size mismatch");
        return false;
    }
    const uint32_t * correction_dims = require_typed_arr_data<uint32_t>(ctx, LLAMA_SPECTRAL_ENTRY_CORR_DIMS, GGUF_TYPE_UINT32, &n, err);
    if (!correction_dims || n != n_entries) {
        set_error(err, "spectral correction dims array size mismatch");
        return false;
    }

    artifact.entries.reserve(n_entries);
    for (size_t i = 0; i < n_entries; ++i) {
        llama_spectral_entry entry;
        entry.name = gguf_get_arr_str(ctx, names_id, i);
        entry.kind = static_cast<llama_spectral_kind>(kinds[i]);
        entry.profile = static_cast<llama_spectral_profile>(profiles[i]);
        entry.dim = dims[i];
        entry.split = splits[i];
        entry.d_eff = d_effs[i];
        entry.semantic_bits = semantic_bits[i];
        entry.tail_bits = tail_bits[i];
        entry.correction_dim = correction_dims[i];

        size_t basis_n = 0;
        const float * basis = require_typed_arr_data<float>(ctx, metadata_name_for(i, "basis").c_str(), GGUF_TYPE_FLOAT32, &basis_n, err);
        if (!basis) {
            return false;
        }
        size_t eigen_n = 0;
        const float * eigenvalues = nullptr;
        const int64_t eigen_id = gguf_find_key(ctx, metadata_name_for(i, "eigenvalues").c_str());
        if (eigen_id >= 0) {
            eigenvalues = require_typed_arr_data<float>(ctx, metadata_name_for(i, "eigenvalues").c_str(), GGUF_TYPE_FLOAT32, &eigen_n, err);
            if (!eigenvalues) {
                return false;
            }
        }
        size_t semantic_n = 0;
        const float * semantic = require_typed_arr_data<float>(ctx, metadata_name_for(i, "semantic_codebook").c_str(), GGUF_TYPE_FLOAT32, &semantic_n, err);
        if (!semantic) {
            return false;
        }
        size_t tail_n = 0;
        const float * tail = require_typed_arr_data<float>(ctx, metadata_name_for(i, "tail_codebook").c_str(), GGUF_TYPE_FLOAT32, &tail_n, err);
        if (!tail) {
            return false;
        }

        entry.basis.assign(basis, basis + basis_n);
        if (eigenvalues != nullptr) {
            entry.eigenvalues.assign(eigenvalues, eigenvalues + eigen_n);
        }
        entry.semantic_codebook.assign(semantic, semantic + semantic_n);
        entry.tail_codebook.assign(tail, tail + tail_n);

        if (!validate_entry(entry, err)) {
            return false;
        }

        artifact.entries.push_back(std::move(entry));
    }

    return true;
}
