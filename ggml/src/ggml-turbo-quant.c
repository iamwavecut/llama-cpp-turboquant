/*
 * TurboQuant: KV cache compression via PolarQuant + QJL
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements GGML_TYPE_TURBO2_0 (2-bit), GGML_TYPE_TURBO3_0 (3-bit) and
 * GGML_TYPE_TURBO4_0 (4-bit) for use as --cache-type-k turboN in llama-server.
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <float.h>

#if defined(GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

#if defined(_MSC_VER)
#define GGML_THREAD_LOCAL __declspec(thread)
#else
#define GGML_THREAD_LOCAL __thread
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Global: WHT group size for CPU quantize path (set by CPU SET_ROWS handler) */
GGML_API int turbo3_cpu_wht_group_size = 0;

/* ---------- constants ---------- */

#define TURBO_SEED_ROTATION 42
#define TURBO_SEED_QJL      1042
#define TURBO_D             128  /* rotation group size = head_dim (independent of block size) */
#define TURBO_QJL_CONST     1.2533141373155003f  /* sqrt(pi/2) */

/* Optimal centroids from paper (scaled by 1/sqrt(d)) */
/* 2-bit: {±0.453, ±1.51} / sqrt(d) */
static const float CENTROIDS_2BIT[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };

/* 3-bit: Lloyd-Max for N(0, 1/128), pre-computed */
static const float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

/* ---------- rotation matrix (lazy init) ---------- */

static float turbo_rotation[TURBO_D * TURBO_D];
static float turbo_rotation_t[TURBO_D * TURBO_D]; /* transpose */
static int   turbo_rotation_initialized = 0;

/* Simple LCG PRNG for deterministic rotation generation */
static uint64_t turbo_prng_state;

static void turbo_prng_seed(uint64_t seed) {
    turbo_prng_state = seed;
}

static double turbo_prng_normal(void) {
    /* Box-Muller transform from uniform LCG */
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void turbo_init_rotation(void) {
    if (turbo_rotation_initialized) return;

    const int d = TURBO_D;

    /* Generate random Gaussian matrix */
    turbo_prng_seed(TURBO_SEED_ROTATION);
    float G[TURBO_D * TURBO_D];
    for (int i = 0; i < d * d; i++) {
        G[i] = (float)turbo_prng_normal();
    }

    /* QR decomposition via modified Gram-Schmidt */
    /* Q stored column-major in turbo_rotation */
    memcpy(turbo_rotation, G, d * d * sizeof(float));

    for (int j = 0; j < d; j++) {
        /* Normalize column j */
        float norm = 0.0f;
        for (int i = 0; i < d; i++) {
            norm += turbo_rotation[i * d + j] * turbo_rotation[i * d + j];
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + j] /= norm;
            }
        }

        /* Orthogonalize remaining columns against j */
        for (int k = j + 1; k < d; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) {
                dot += turbo_rotation[i * d + j] * turbo_rotation[i * d + k];
            }
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + k] -= dot * turbo_rotation[i * d + j];
            }
        }
    }

    /* Compute transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_rotation_t[i * d + j] = turbo_rotation[j * d + i];
        }
    }

    turbo_rotation_initialized = 1;
}

/* ---------- QJL projection matrix (lazy init, seed-based) ---------- */

static float turbo_qjl_matrix[TURBO_D * TURBO_D];
static float turbo_qjl_matrix_t[TURBO_D * TURBO_D];
static int   turbo_qjl_initialized = 0;

static void turbo_init_qjl(void) {
    if (turbo_qjl_initialized) return;

    const int d = TURBO_D;
    turbo_prng_seed(TURBO_SEED_QJL);

    for (int i = 0; i < d * d; i++) {
        turbo_qjl_matrix[i] = (float)turbo_prng_normal();
    }

    /* Transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_qjl_matrix_t[i * d + j] = turbo_qjl_matrix[j * d + i];
        }
    }

    turbo_qjl_initialized = 1;
}

/* ---------- helper: matrix-vector multiply ---------- */

static void matvec(const float * M, const float * x, float * y, int d) {
    /* y = M @ x, M is row-major d×d */
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += M[i * d + j] * x[j];
        }
        y[i] = sum;
    }
}

/* ---------- nearest centroid ---------- */

static int nearest_centroid_2bit(float val) {
    /* Binary search on midpoints: {-0.133, -0.040, 0.040, 0.133} */
    if (val < -0.086728f) return 0;       /* midpoint(-0.133, -0.040) */
    if (val <  0.000000f) return 1;       /* midpoint(-0.040, 0.040) */
    if (val <  0.086728f) return 2;       /* midpoint(0.040, 0.133) */
    return 3;
}

static int nearest_centroid_3bit(float val) {
    /* 8 centroids, find nearest via midpoints */
    if (val < -0.154259f) return 0;
    if (val < -0.091775f) return 1;
    if (val < -0.043589f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.043589f) return 4;
    if (val <  0.091775f) return 5;
    if (val <  0.154259f) return 6;
    return 7;
}

static int nearest_centroid_4bit(float val) {
    /* 16 centroids, optimal for N(0, 1/sqrt(128)), find nearest via midpoints */
    if (val < -0.145560f) return 0;
    if (val < -0.103361f) return 1;
    if (val < -0.079142f) return 2;
    if (val < -0.060009f) return 3;
    if (val < -0.043430f) return 4;
    if (val < -0.028293f) return 5;
    if (val < -0.013963f) return 6;
    if (val <  0.000000f) return 7;
    if (val <  0.013963f) return 8;
    if (val <  0.028293f) return 9;
    if (val <  0.043430f) return 10;
    if (val <  0.060009f) return 11;
    if (val <  0.079142f) return 12;
    if (val <  0.103361f) return 13;
    if (val <  0.145560f) return 14;
    return 15;
}

/* ---------- WHT sign arrays (must match CUDA/Metal, seed=42) ---------- */

static const float turbo_cpu_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static const float turbo_cpu_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

/* ---------- CPU forward WHT (in-place, group_size elements) ---------- */

static void turbo_cpu_fwht(float * x, int group_size) {
    const float * s1 = turbo_cpu_s1;
    const float * s2 = turbo_cpu_s2;
    const float inv_sqrt = (group_size == 128) ? 0.08838834764831845f : 0.125f;

    // signs1
    for (int i = 0; i < group_size; i++) x[i] *= s1[i];

    // butterfly stages
    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // normalize + signs2
    for (int i = 0; i < group_size; i++) x[i] *= inv_sqrt * s2[i];
}

/* ---------- TURBO3_0: 3-bit PolarQuant with WHT rotation ---------- */

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);

    // Read WHT group size from global (set by CPU SET_ROWS handler before each call).
    // Fallback: 128 if row is 128-aligned, else 64.
    extern int turbo3_cpu_wht_group_size;
    int group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO3;

    for (int g = 0; g < n_groups; g++) {
        const float * grp_src = x + g * group_size;
        block_turbo3_0 * grp_dst = y + g * blocks_per_group;

        // 1. L2 norm over the group
        float norm_sq = 0.0f;
        float buf[128];  // max group_size
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        // 2. Normalize
        for (int j = 0; j < group_size; j++) buf[j] *= inv_norm;

        // 3. Forward WHT rotation
        turbo_cpu_fwht(buf, group_size);

        // 4. Quantize + pack into sub-blocks
        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turbo3_0 * blk = &grp_dst[b];
            const int off = b * QK_TURBO3;

            memset(blk->qs, 0, QK_TURBO3 / 4);
            memset(blk->signs, 0, QK_TURBO3 / 8);

            for (int j = 0; j < QK_TURBO3; j++) {
                int idx = nearest_centroid_3bit(buf[off + j]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                if (idx & 0x4) {
                    blk->signs[j / 8] |= (1 << (j % 8));
                }
                recon_sq += CENTROIDS_3BIT[idx] * CENTROIDS_3BIT[idx];
            }
        }

        // 5. Corrected norm: grp_norm / recon_norm (matching CUDA kernel)
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    // Stub — Metal shader handles dequant on GPU.
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t low2 = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            uint8_t hi1 = (x[block].signs[j/8] >> (j%8)) & 0x1;
            uint8_t idx = low2 | (hi1 << 2);
            y[block * QK_TURBO3 + j] = CENTROIDS_3BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_ref(
            src + row * n_per_row,
            (block_turbo3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO2_0: 2-bit PolarQuant (no QJL) ---------- */

void quantize_row_turbo2_0_ref(const float * GGML_RESTRICT x, block_turbo2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);

    extern int turbo3_cpu_wht_group_size;
    int group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO2;

    for (int g = 0; g < n_groups; g++) {
        const float * grp_src = x + g * group_size;
        block_turbo2_0 * grp_dst = y + g * blocks_per_group;

        /* 1. L2 norm over the group */
        float norm_sq = 0.0f;
        float buf[128];
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        /* 2. Normalize */
        for (int j = 0; j < group_size; j++) buf[j] *= inv_norm;

        /* 3. Forward WHT rotation */
        turbo_cpu_fwht(buf, group_size);

        /* 4. Quantize + pack into sub-blocks */
        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turbo2_0 * blk = &grp_dst[b];
            const int off = b * QK_TURBO2;

            memset(blk->qs, 0, QK_TURBO2 / 4);

            for (int j = 0; j < QK_TURBO2; j++) {
                int idx = nearest_centroid_2bit(buf[off + j]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                recon_sq += CENTROIDS_2BIT[idx] * CENTROIDS_2BIT[idx];
            }
        }

        /* 5. Corrected norm */
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void dequantize_row_turbo2_0(const block_turbo2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);
    const int nb = k / QK_TURBO2;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO2; j++) {
            uint8_t idx = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            y[block * QK_TURBO2 + j] = CENTROIDS_2BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO2 == 0);

    size_t row_size = (n_per_row / QK_TURBO2) * sizeof(block_turbo2_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo2_0_ref(
            src + row * n_per_row,
            (block_turbo2_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO4_0: 3-bit PolarQuant + 1-bit QJL ---------- */

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();
    turbo_init_qjl();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        /* Step 1: Extract norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[TURBO_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, d * sizeof(float));
        }

        /* Step 2: Forward WHT rotation (matches CUDA set_rows) */
        float rotated[TURBO_D];
        memcpy(rotated, normalized, d * sizeof(float));
        turbo_cpu_fwht(rotated, d);

#if TURBO4_USE_4BIT
        /* Step 3: 4-bit quantization (16 centroids) */
        static const float CENTROIDS_4BIT[16] = {
            -0.173926f, -0.117195f, -0.089527f, -0.068756f,
            -0.051262f, -0.035597f, -0.020989f, -0.006938f,
             0.006938f,  0.020989f,  0.035597f,  0.051262f,
             0.068756f,  0.089527f,  0.117195f,  0.173926f
        };
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_4bit(rotated[i]);
        }

        /* Norm correction */
        float recon_norm_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            recon_norm_sq += CENTROIDS_4BIT[indices[i]] * CENTROIDS_4BIT[indices[i]];
        }
        float recon_norm = sqrtf(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? norm / recon_norm : norm;
        y[block].norm = GGML_FP32_TO_FP16(corrected_norm);
#else
        /* Step 3: 3-bit quantization (8 centroids) */
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_3bit(rotated[i]);
        }

        /* Step 4: Residual */
        float reconstructed[TURBO_D];
        for (int i = 0; i < d; i++) {
            reconstructed[i] = CENTROIDS_3BIT[indices[i]];
        }
        float mse_recon[TURBO_D];
        matvec(turbo_rotation_t, reconstructed, mse_recon, d);

        float residual[TURBO_D];
        for (int i = 0; i < d; i++) {
            residual[i] = normalized[i] - mse_recon[i];
        }

        /* Step 5: QJL */
        float projected[TURBO_D];
        matvec(turbo_qjl_matrix, residual, projected, d);
#endif

        /* Pack */
#if !TURBO4_USE_4BIT
        y[block].norm  = GGML_FP32_TO_FP16(norm);
#endif

#if TURBO4_USE_4BIT
        /* 4-bit PolarQuant: nibble pack into qs[64] */
        memset(y[block].qs, 0, d / 2);
        for (int i = 0; i < d; i++) {
            y[block].qs[i / 2] |= (uint8_t)((indices[i] & 0xF) << ((i % 2) * 4));
        }
        y[block].rnorm = GGML_FP32_TO_FP16(0.0f);
#else
        /* Legacy 3-bit + QJL: pack 3-bit indices + QJL signs */
        memset(y[block].qs, 0, d * 3 / 8);
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_pos    = bit_offset % 8;
            uint16_t val   = (uint16_t)(indices[i] & 0x7);
            y[block].qs[byte_idx] |= (uint8_t)(val << bit_pos);
            if (bit_pos > 5 && byte_idx + 1 < d * 3 / 8) {
                y[block].qs[byte_idx + 1] |= (uint8_t)(val >> (8 - bit_pos));
            }
        }
        memset(y[block].signs, 0, d / 8);
        for (int i = 0; i < d; i++) {
            if (projected[i] >= 0.0f) {
                y[block].signs[i / 8] |= (1 << (i % 8));
            }
        }
#endif
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

#if TURBO4_USE_4BIT
    /* 4-bit PolarQuant: nibble unpack → centroid → inverse rotate → scale */
    /* TODO: add proper 4-bit centroid table to C code (currently only in Metal) */
    static const float CENTROIDS_4BIT[16] = {
        -0.173926f, -0.117195f, -0.089527f, -0.068756f,
        -0.051262f, -0.035597f, -0.020989f, -0.006938f,
         0.006938f,  0.020989f,  0.035597f,  0.051262f,
         0.068756f,  0.089527f,  0.117195f,  0.173926f
    };
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        float * dst = y + block * d;
        for (int i = 0; i < d; i++) {
            uint8_t idx = (x[block].qs[i / 2] >> ((i % 2) * 4)) & 0xF;
            dst[i] = CENTROIDS_4BIT[idx] * norm;
        }
        /* No inverse WHT, dequant stays in the rotated domain.
        * Q is WHT-rotated by the graph, so <Q_rot, K_rot> gives correct attention scores.
        * The inverse WHT is applied to the attention output via GGML_OP_TURBO_WHT (direction=1) in the graph. 
        */
    }
#else
    /* Legacy 3-bit + QJL dequant */
    turbo_init_qjl();
    for (int block = 0; block < nb; block++) {
        float norm  = GGML_FP16_TO_FP32(x[block].norm);

        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_pos    = bit_offset % 8;
            uint16_t raw   = (uint16_t)x[block].qs[byte_idx];
            if (byte_idx + 1 < d * 3 / 8) {
                raw |= (uint16_t)x[block].qs[byte_idx + 1] << 8;
            }
            indices[i] = (uint8_t)((raw >> bit_pos) & 0x7);
        }

        float signs[TURBO_D];
        for (int i = 0; i < d; i++) {
            signs[i] = (x[block].signs[i / 8] & (1 << (i % 8))) ? 1.0f : -1.0f;
        }

        float rnorm = GGML_FP16_TO_FP32(x[block].rnorm);
        const float qjl_scale = TURBO_QJL_CONST / (float)d * rnorm;

        float rotated_recon[TURBO_D];
        for (int i = 0; i < d; i++) {
            rotated_recon[i] = CENTROIDS_3BIT[indices[i]];
        }
        float mse_recon[TURBO_D];
        matvec(turbo_rotation_t, rotated_recon, mse_recon, d);

        float qjl_recon[TURBO_D];
        matvec(turbo_qjl_matrix_t, signs, qjl_recon, d);
        for (int i = 0; i < d; i++) {
            qjl_recon[i] *= qjl_scale;
        }

        float * dst = y + block * d;
        for (int i = 0; i < d; i++) {
            dst[i] = (mse_recon[i] + qjl_recon[i]) * norm;
        }
    }
#endif
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);

    size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(
            src + row * n_per_row,
            (block_turbo4_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ================================================================== */
/* TQ3_1S / TQ4_1S: WHT-rotated weight quantization                  */
/* ================================================================== */

/* Lloyd-Max centroids for N(0,1) — shared with Metal shaders */
static const float TQ3_0_CENTROIDS[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f,
     0.230106f,  0.725222f,  1.277503f,  1.988943f
};

static const float TQ4_0_CENTROIDS[16] = {
    -2.732590f, -2.069017f, -1.618046f, -1.256231f,
    -0.942340f, -0.656759f, -0.388048f, -0.128395f,
     0.128395f,  0.388048f,  0.656759f,  0.942340f,
     1.256231f,  1.618046f,  2.069017f,  2.732590f,
};

/* WHT sign pattern (golden ratio hash, 32-element blocks) — shared by TQ3 and TQ4 */
static const float TQ3_0_SIGNS[32] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
};

#define TQ_BLOCK_SIZE 32
#define TQ_INV_SQRT32 0.17677669529663688f  /* 1/sqrt(32) */

/* Forward RHT: sign flips -> WHT butterfly -> normalize */
static void tq3_0_rht_forward(float * buf) {
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) buf[i] *= TQ3_0_SIGNS[i];
    for (int step = 1; step < TQ_BLOCK_SIZE; step <<= 1) {
        for (int i = 0; i < TQ_BLOCK_SIZE; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j], b = buf[j + step];
                buf[j]     = a + b;
                buf[j + step] = a - b;
            }
        }
    }
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) buf[i] *= TQ_INV_SQRT32;
}

/* Inverse RHT: WHT butterfly -> normalize + unsign */
static void tq3_0_rht_inverse(float * buf) {
    for (int step = 1; step < TQ_BLOCK_SIZE; step <<= 1) {
        for (int i = 0; i < TQ_BLOCK_SIZE; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j], b = buf[j + step];
                buf[j]     = a + b;
                buf[j + step] = a - b;
            }
        }
    }
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) buf[i] *= TQ_INV_SQRT32 * TQ3_0_SIGNS[i];
}

/* Nearest centroid for TQ3 (8 centroids) */
static int tq3_0_choose_index(float val) {
    /* Binary search on midpoints of TQ3_0_CENTROIDS */
    if (val < -1.644041f) return 0;
    if (val < -1.015870f) return 1;
    if (val < -0.493925f) return 2;
    if (val < -0.008701f) return 3;
    if (val <  0.477664f) return 4;
    if (val <  1.001363f) return 5;
    if (val <  1.633223f) return 6;
    return 7;
}

/* Nearest centroid for TQ4 (16 centroids) */
static int tq4_0_choose_index(float val) {
    /* Binary search on midpoints of TQ4_0_CENTROIDS */
    if (val < -2.400804f) return 0;
    if (val < -1.843532f) return 1;
    if (val < -1.437139f) return 2;
    if (val < -1.099286f) return 3;
    if (val < -0.799550f) return 4;
    if (val < -0.522404f) return 5;
    if (val < -0.258222f) return 6;
    if (val <  0.000000f) return 7;
    if (val <  0.258222f) return 8;
    if (val <  0.522404f) return 9;
    if (val <  0.799550f) return 10;
    if (val <  1.099286f) return 11;
    if (val <  1.437139f) return 12;
    if (val <  1.843532f) return 13;
    if (val <  2.400804f) return 14;
    return 15;
}

/* ---------- TQ3_1S quantization ---------- */

void quantize_row_tq3_1s_ref(const float * GGML_RESTRICT x, block_tq3_1s * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ3_0 == 0);
    const int nb = k / QK_TQ3_0;

    for (int block = 0; block < nb; block++) {
        const float * src_blk = x + block * QK_TQ3_0;
        block_tq3_1s * blk = &y[block];

        /* 1. Forward RHT */
        float buf[TQ_BLOCK_SIZE];
        memcpy(buf, src_blk, TQ_BLOCK_SIZE * sizeof(float));
        tq3_0_rht_forward(buf);

        /* 2. Split into two halves, compute RMS per half */
        float rms0 = 0.0f, rms1 = 0.0f;
        for (int j = 0; j < 16; j++) rms0 += buf[j] * buf[j];
        for (int j = 16; j < 32; j++) rms1 += buf[j] * buf[j];
        rms0 = sqrtf(rms0 / 16.0f);
        rms1 = sqrtf(rms1 / 16.0f);

        /* 3. Scale search (9 points) */
        static const float scales[] = { 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.35f, 1.5f };
        float best_d0 = rms0, best_d1 = rms1;
        float best_err = 1e30f;

        for (int si = 0; si < 9; si++) {
            float d0 = rms0 * scales[si];
            float d1 = rms1 * scales[si];
            float inv0 = (d0 > 1e-10f) ? 1.0f / d0 : 0.0f;
            float inv1 = (d1 > 1e-10f) ? 1.0f / d1 : 0.0f;

            float err = 0.0f;
            for (int j = 0; j < 16; j++) {
                int idx = tq3_0_choose_index(buf[j] * inv0);
                float diff = buf[j] - TQ3_0_CENTROIDS[idx] * d0;
                err += diff * diff;
            }
            for (int j = 16; j < 32; j++) {
                int idx = tq3_0_choose_index(buf[j] * inv1);
                float diff = buf[j] - TQ3_0_CENTROIDS[idx] * d1;
                err += diff * diff;
            }
            if (err < best_err) {
                best_err = err;
                best_d0 = d0;
                best_d1 = d1;
            }
        }

        /* 4. Iterative refinement (6 iterations) */
        for (int iter = 0; iter < 6; iter++) {
            float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
            float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

            float num0 = 0.0f, den0 = 0.0f;
            float num1 = 0.0f, den1 = 0.0f;
            for (int j = 0; j < 16; j++) {
                int idx = tq3_0_choose_index(buf[j] * inv0);
                float c = TQ3_0_CENTROIDS[idx];
                num0 += buf[j] * c;
                den0 += c * c;
            }
            for (int j = 16; j < 32; j++) {
                int idx = tq3_0_choose_index(buf[j] * inv1);
                float c = TQ3_0_CENTROIDS[idx];
                num1 += buf[j] * c;
                den1 += c * c;
            }
            if (den0 > 1e-10f) best_d0 = num0 / den0;
            if (den1 > 1e-10f) best_d1 = num1 / den1;
        }

        /* 5. Final quantize + pack */
        float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
        float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

        blk->d0 = GGML_FP32_TO_FP16(best_d0);
        blk->d1 = GGML_FP32_TO_FP16(best_d1);
        memset(blk->qs, 0, QK_TQ3_0 * 3 / 8);

        /* TQ3 packing: 4 groups of 8 indices packed into 3 bytes each */
        for (int g = 0; g < 4; g++) {
            uint8_t indices[8];
            for (int i = 0; i < 8; i++) {
                int j = g * 8 + i;
                float inv = (j < 16) ? inv0 : inv1;
                indices[i] = (uint8_t)tq3_0_choose_index(buf[j] * inv);
            }
            uint8_t * qp = blk->qs + g * 3;
            qp[0] = (indices[0] & 7) | ((indices[1] & 7) << 3) | ((indices[2] & 3) << 6);
            qp[1] = ((indices[2] >> 2) & 1) | ((indices[3] & 7) << 1) | ((indices[4] & 7) << 4) | ((indices[5] & 1) << 7);
            qp[2] = ((indices[5] >> 1) & 3) | ((indices[6] & 7) << 2) | ((indices[7] & 7) << 5);
        }
    }
}

void dequantize_row_tq3_1s(const block_tq3_1s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ3_0 == 0);
    const int nb = k / QK_TQ3_0;

    for (int blk_i = 0; blk_i < nb; blk_i++) {
        float d0 = GGML_FP16_TO_FP32(x[blk_i].d0);
        float d1 = GGML_FP16_TO_FP32(x[blk_i].d1);

        /* Unpack 3-bit indices */
        float buf[32];
        for (int g = 0; g < 4; g++) {
            const uint8_t * qp = x[blk_i].qs + g * 3;
            uint8_t idx[8];
            idx[0] =  qp[0]       & 7;
            idx[1] = (qp[0] >> 3) & 7;
            idx[2] = ((qp[0] >> 6) | (qp[1] << 2)) & 7;
            idx[3] = (qp[1] >> 1) & 7;
            idx[4] = (qp[1] >> 4) & 7;
            idx[5] = ((qp[1] >> 7) | (qp[2] << 1)) & 7;
            idx[6] = (qp[2] >> 2) & 7;
            idx[7] = (qp[2] >> 5) & 7;

            for (int i = 0; i < 8; i++) {
                int j = g * 8 + i;
                float d = (j < 16) ? d0 : d1;
                buf[j] = TQ3_0_CENTROIDS[idx[i]] * d;
            }
        }

        /* Inverse RHT */
        tq3_0_rht_inverse(buf);

        memcpy(y + blk_i * QK_TQ3_0, buf, QK_TQ3_0 * sizeof(float));
    }
}

size_t quantize_tq3_1s(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TQ3_0 == 0);

    size_t row_size = (n_per_row / QK_TQ3_0) * sizeof(block_tq3_1s);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_tq3_1s_ref(
            src + row * n_per_row,
            (block_tq3_1s *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TQ4_1S quantization ---------- */

void quantize_row_tq4_1s_ref(const float * GGML_RESTRICT x, block_tq4_1s * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ4_1S == 0);
    const int nb = k / QK_TQ4_1S;

    for (int block = 0; block < nb; block++) {
        const float * src_blk = x + block * QK_TQ4_1S;
        block_tq4_1s * blk = &y[block];

        /* 1. Forward RHT */
        float buf[TQ_BLOCK_SIZE];
        memcpy(buf, src_blk, TQ_BLOCK_SIZE * sizeof(float));
        tq3_0_rht_forward(buf);

        /* 2. Split into two halves, compute RMS per half */
        float rms0 = 0.0f, rms1 = 0.0f;
        for (int j = 0; j < 16; j++) rms0 += buf[j] * buf[j];
        for (int j = 16; j < 32; j++) rms1 += buf[j] * buf[j];
        rms0 = sqrtf(rms0 / 16.0f);
        rms1 = sqrtf(rms1 / 16.0f);

        /* 3. Scale search (9 points) */
        static const float scales[] = { 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.35f, 1.5f };
        float best_d0 = rms0, best_d1 = rms1;
        float best_err = 1e30f;

        for (int si = 0; si < 9; si++) {
            float d0 = rms0 * scales[si];
            float d1 = rms1 * scales[si];
            float inv0 = (d0 > 1e-10f) ? 1.0f / d0 : 0.0f;
            float inv1 = (d1 > 1e-10f) ? 1.0f / d1 : 0.0f;

            float err = 0.0f;
            for (int j = 0; j < 16; j++) {
                int idx = tq4_0_choose_index(buf[j] * inv0);
                float diff = buf[j] - TQ4_0_CENTROIDS[idx] * d0;
                err += diff * diff;
            }
            for (int j = 16; j < 32; j++) {
                int idx = tq4_0_choose_index(buf[j] * inv1);
                float diff = buf[j] - TQ4_0_CENTROIDS[idx] * d1;
                err += diff * diff;
            }
            if (err < best_err) {
                best_err = err;
                best_d0 = d0;
                best_d1 = d1;
            }
        }

        /* 4. Iterative refinement (6 iterations) */
        for (int iter = 0; iter < 6; iter++) {
            float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
            float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

            float num0 = 0.0f, den0 = 0.0f;
            float num1 = 0.0f, den1 = 0.0f;
            for (int j = 0; j < 16; j++) {
                int idx = tq4_0_choose_index(buf[j] * inv0);
                float c = TQ4_0_CENTROIDS[idx];
                num0 += buf[j] * c;
                den0 += c * c;
            }
            for (int j = 16; j < 32; j++) {
                int idx = tq4_0_choose_index(buf[j] * inv1);
                float c = TQ4_0_CENTROIDS[idx];
                num1 += buf[j] * c;
                den1 += c * c;
            }
            if (den0 > 1e-10f) best_d0 = num0 / den0;
            if (den1 > 1e-10f) best_d1 = num1 / den1;
        }

        /* 5. Final quantize + pack (nibble packing) */
        float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
        float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

        blk->d0 = GGML_FP32_TO_FP16(best_d0);
        blk->d1 = GGML_FP32_TO_FP16(best_d1);
        memset(blk->qs, 0, QK_TQ4_1S / 2);

        for (int j = 0; j < QK_TQ4_1S; j++) {
            float inv = (j < 16) ? inv0 : inv1;
            int idx = tq4_0_choose_index(buf[j] * inv);
            blk->qs[j / 2] |= (uint8_t)((idx & 0xF) << ((j & 1) * 4));
        }
    }
}

void dequantize_row_tq4_1s(const block_tq4_1s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ4_1S == 0);
    const int nb = k / QK_TQ4_1S;

    for (int blk_i = 0; blk_i < nb; blk_i++) {
        float d0 = GGML_FP16_TO_FP32(x[blk_i].d0);
        float d1 = GGML_FP16_TO_FP32(x[blk_i].d1);

        float buf[32];
        for (int j = 0; j < 32; j++) {
            uint8_t idx = (x[blk_i].qs[j / 2] >> ((j & 1) * 4)) & 0xF;
            float d = (j < 16) ? d0 : d1;
            buf[j] = TQ4_0_CENTROIDS[idx] * d;
        }

        /* Inverse RHT */
        tq3_0_rht_inverse(buf);

        memcpy(y + blk_i * QK_TQ4_1S, buf, QK_TQ4_1S * sizeof(float));
    }
}

size_t quantize_tq4_1s(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TQ4_1S == 0);

    size_t row_size = (n_per_row / QK_TQ4_1S) * sizeof(block_tq4_1s);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_tq4_1s_ref(
            src + row * n_per_row,
            (block_tq4_1s *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ----------------------------------------------------------------------- */
/* SpectralQuant SQ2_0 / SQ3_1S / SQ4_1S                                   */
/* ----------------------------------------------------------------------- */

struct ggml_spectral_registry_entry {
    const void * owner;
    const uint8_t * data;
    size_t size;
    enum ggml_type type;
    struct ggml_spectral_weight_meta meta;
};

static struct ggml_spectral_registry_entry * ggml_spectral_registry = NULL;
static size_t ggml_spectral_registry_size = 0;
static size_t ggml_spectral_registry_capacity = 0;

bool ggml_is_spectral_weight_type(enum ggml_type type) {
    return type == GGML_TYPE_SQ2_0 || type == GGML_TYPE_SQ3_1S || type == GGML_TYPE_SQ4_1S;
}

static int ggml_sq_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_SQ2_0:  return 2;
        case GGML_TYPE_SQ3_1S: return 3;
        case GGML_TYPE_SQ4_1S: return 4;
        default:               return 0;
    }
}

static size_t ggml_sq_row_size(enum ggml_type type, uint32_t dim) {
    return ggml_row_size(type, dim);
}

static size_t ggml_sq_qs_bytes(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_SQ2_0:  return sizeof(((block_sq2_0 *) 0)->qs);
        case GGML_TYPE_SQ3_1S: return sizeof(((block_sq3_1s *) 0)->qs);
        case GGML_TYPE_SQ4_1S: return sizeof(((block_sq4_1s *) 0)->qs);
        default:               return 0;
    }
}

static ggml_half * ggml_sq_corr_scale_ptr(void * block, enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_SQ2_0:  return &((block_sq2_0 *) block)->dc;
        case GGML_TYPE_SQ3_1S: return &((block_sq3_1s *) block)->dc;
        case GGML_TYPE_SQ4_1S: return &((block_sq4_1s *) block)->dc;
        default:               return NULL;
    }
}

static const ggml_half * ggml_sq_corr_scale_ptr_const(const void * block, enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_SQ2_0:  return &((const block_sq2_0 *) block)->dc;
        case GGML_TYPE_SQ3_1S: return &((const block_sq3_1s *) block)->dc;
        case GGML_TYPE_SQ4_1S: return &((const block_sq4_1s *) block)->dc;
        default:               return NULL;
    }
}

static uint8_t * ggml_sq_corr_ptr(void * block, enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_SQ2_0:  return ((block_sq2_0 *) block)->corr;
        case GGML_TYPE_SQ3_1S: return ((block_sq3_1s *) block)->corr;
        case GGML_TYPE_SQ4_1S: return ((block_sq4_1s *) block)->corr;
        default:               return NULL;
    }
}

static const uint8_t * ggml_sq_corr_ptr_const(const void * block, enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_SQ2_0:  return ((const block_sq2_0 *) block)->corr;
        case GGML_TYPE_SQ3_1S: return ((const block_sq3_1s *) block)->corr;
        case GGML_TYPE_SQ4_1S: return ((const block_sq4_1s *) block)->corr;
        default:               return NULL;
    }
}

static uint8_t * ggml_sq_qs_ptr(void * block, enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_SQ2_0:  return ((block_sq2_0 *) block)->qs;
        case GGML_TYPE_SQ3_1S: return ((block_sq3_1s *) block)->qs;
        case GGML_TYPE_SQ4_1S: return ((block_sq4_1s *) block)->qs;
        default:               return NULL;
    }
}

static const uint8_t * ggml_sq_qs_ptr_const(const void * block, enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_SQ2_0:  return ((const block_sq2_0 *) block)->qs;
        case GGML_TYPE_SQ3_1S: return ((const block_sq3_1s *) block)->qs;
        case GGML_TYPE_SQ4_1S: return ((const block_sq4_1s *) block)->qs;
        default:               return NULL;
    }
}

static bool ggml_sq_meta_valid(const struct ggml_spectral_weight_meta * meta, enum ggml_type type) {
    const int bits = ggml_sq_bits(type);
    if (bits == 0 || meta == NULL) {
        return false;
    }
    if (meta->dim == 0 || meta->dim % QK_SQ != 0) {
        return false;
    }
    if (meta->split == 0 || meta->split > meta->dim) {
        return false;
    }
    if (meta->correction_dim > GGML_SQ_CORR_MAX || meta->correction_dim > meta->split) {
        return false;
    }
    if (meta->semantic_codebook_size == 0 || meta->tail_codebook_size == 0) {
        return false;
    }
    if (meta->semantic_codebook_size > (uint32_t) (1u << bits) || meta->tail_codebook_size > (uint32_t) (1u << bits)) {
        return false;
    }
    return meta->basis != NULL && meta->semantic_codebook != NULL && meta->tail_codebook != NULL;
}

static uint32_t ggml_sq_quantize_scalar(float value, const float * codebook, uint32_t n_levels) {
    uint32_t best = 0;
    float best_dist = FLT_MAX;
    for (uint32_t i = 0; i < n_levels; ++i) {
        const float dist = fabsf(value - codebook[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best = i;
        }
    }
    return best;
}

static void ggml_sq_pack_index(uint8_t * qs, int idx, int bits, uint32_t code) {
    const int bit_pos = idx * bits;
    const int byte_pos = bit_pos / 8;
    const int bit_off = bit_pos % 8;
    const uint32_t mask = (uint32_t) ((1u << bits) - 1u);

    uint32_t v = (code & mask) << bit_off;
    qs[byte_pos] |= (uint8_t) (v & 0xFFu);
    if (bit_off + bits > 8) {
        qs[byte_pos + 1] |= (uint8_t) ((v >> 8) & 0xFFu);
    }
}

static uint32_t ggml_sq_unpack_index(const uint8_t * qs, int idx, int bits) {
    const int bit_pos = idx * bits;
    const int byte_pos = bit_pos / 8;
    const int bit_off = bit_pos % 8;
    uint32_t v = (uint32_t) (qs[byte_pos] >> bit_off);
    if (bit_off + bits > 8) {
        v |= (uint32_t) qs[byte_pos + 1] << (8 - bit_off);
    }
    return v & ((1u << bits) - 1u);
}

static void ggml_sq_store_corr(uint8_t * corr, int idx, int8_t q) {
    const uint8_t nibble = (uint8_t) (q & 0x0F);
    corr[idx / 2] &= (uint8_t) ~(0x0Fu << ((idx & 1) * 4));
    corr[idx / 2] |= (uint8_t) (nibble << ((idx & 1) * 4));
}

static int8_t ggml_sq_load_corr(const uint8_t * corr, int idx) {
    const uint8_t nibble = (corr[idx / 2] >> ((idx & 1) * 4)) & 0x0F;
    return (int8_t) (nibble >= 8 ? (int) nibble - 16 : nibble);
}

static void ggml_sq_forward_rotate(const struct ggml_spectral_weight_meta * meta, const float * input, float * rotated) {
#if defined(GGML_USE_ACCELERATE)
    cblas_sgemv(
            CblasRowMajor,
            CblasTrans,
            (int) meta->dim,
            (int) meta->dim,
            1.0f,
            meta->basis,
            (int) meta->dim,
            input,
            1,
            0.0f,
            rotated,
            1);
#else
    for (uint32_t col = 0; col < meta->dim; ++col) {
        float acc = 0.0f;
        for (uint32_t row = 0; row < meta->dim; ++row) {
            acc += meta->basis[(size_t) row * meta->dim + col] * input[row];
        }
        rotated[col] = acc;
    }
#endif
}

static void ggml_sq_inverse_rotate(const struct ggml_spectral_weight_meta * meta, const float * rotated, float * output) {
#if defined(GGML_USE_ACCELERATE)
    cblas_sgemv(
            CblasRowMajor,
            CblasNoTrans,
            (int) meta->dim,
            (int) meta->dim,
            1.0f,
            meta->basis,
            (int) meta->dim,
            rotated,
            1,
            0.0f,
            output,
            1);
#else
    for (uint32_t row = 0; row < meta->dim; ++row) {
        float acc = 0.0f;
        for (uint32_t col = 0; col < meta->dim; ++col) {
            acc += meta->basis[(size_t) row * meta->dim + col] * rotated[col];
        }
        output[row] = acc;
    }
#endif
}

static void ggml_sq_quantize_row(
        enum ggml_type type,
        const struct ggml_spectral_weight_meta * meta,
        const float * x,
        void * y) {
    const int bits = ggml_sq_bits(type);
    const size_t block_size = ggml_type_size(type);
    const uint32_t correction_dim = meta->correction_dim;
    float * rotated = (float *) malloc((size_t) meta->dim * sizeof(float));
    GGML_ASSERT(rotated != NULL);

    ggml_sq_forward_rotate(meta, x, rotated);

    for (uint32_t block = 0; block < meta->dim / QK_SQ; ++block) {
        uint8_t * block_ptr = (uint8_t *) y + block * block_size;
        uint8_t * qs = ggml_sq_qs_ptr(block_ptr, type);
        uint8_t * corr = ggml_sq_corr_ptr(block_ptr, type);
        ggml_half * dc = ggml_sq_corr_scale_ptr(block_ptr, type);

        memset(qs, 0, ggml_sq_qs_bytes(type));
        memset(corr, 0, GGML_SQ_CORR_BYTES);
        *dc = GGML_FP32_TO_FP16(0.0f);

        float residuals[GGML_SQ_CORR_MAX] = { 0 };
        float max_abs_residual = 0.0f;

        for (uint32_t j = 0; j < QK_SQ; ++j) {
            const uint32_t idx = block * QK_SQ + j;
            const bool semantic = idx < meta->split;
            const float * codebook = semantic ? meta->semantic_codebook : meta->tail_codebook;
            const uint32_t n_levels = semantic ? meta->semantic_codebook_size : meta->tail_codebook_size;
            const uint32_t code = ggml_sq_quantize_scalar(rotated[idx], codebook, n_levels);
            ggml_sq_pack_index(qs, (int) j, bits, code);

            if (idx < correction_dim) {
                const float residual = rotated[idx] - codebook[code];
                residuals[idx] = residual;
                if (fabsf(residual) > max_abs_residual) {
                    max_abs_residual = fabsf(residual);
                }
            }
        }

        if (max_abs_residual > 0.0f) {
            const float corr_scale = max_abs_residual / 7.0f;
            *dc = GGML_FP32_TO_FP16(corr_scale);
            for (uint32_t j = 0; j < correction_dim; ++j) {
                const uint32_t idx = block * QK_SQ + j;
                if (idx >= correction_dim) {
                    break;
                }
                int q = corr_scale > 0.0f ? (int) lrintf(residuals[idx] / corr_scale) : 0;
                if (q < -7) q = -7;
                if (q >  7) q =  7;
                ggml_sq_store_corr(corr, (int) j, (int8_t) q);
            }
        }
    }

    free(rotated);
}

static void ggml_sq_dequantize_rows(
        enum ggml_type type,
        const struct ggml_spectral_registry_entry * reg,
        const uint8_t * src,
        float * dst,
        int64_t n) {
    const struct ggml_spectral_weight_meta * meta = &reg->meta;
    const int bits = ggml_sq_bits(type);
    const size_t row_size = ggml_sq_row_size(type, meta->dim);
    const size_t row_offset = (size_t) (src - reg->data);
    GGML_ASSERT(row_offset % row_size == 0);
    GGML_ASSERT(n % meta->dim == 0);

    const int64_t nrows = n / meta->dim;
    float * rotated = (float *) malloc((size_t) meta->dim * sizeof(float));
    GGML_ASSERT(rotated != NULL);

    for (int64_t row = 0; row < nrows; ++row) {
        const uint8_t * row_ptr = src + row * row_size;
        memset(rotated, 0, (size_t) meta->dim * sizeof(float));

        for (uint32_t block = 0; block < meta->dim / QK_SQ; ++block) {
            const uint8_t * block_ptr = row_ptr + block * ggml_type_size(type);
            const uint8_t * qs = ggml_sq_qs_ptr_const(block_ptr, type);
            const uint8_t * corr = ggml_sq_corr_ptr_const(block_ptr, type);
            const float corr_scale = GGML_FP16_TO_FP32(*ggml_sq_corr_scale_ptr_const(block_ptr, type));

            for (uint32_t j = 0; j < QK_SQ; ++j) {
                const uint32_t idx = block * QK_SQ + j;
                const bool semantic = idx < meta->split;
                const float * codebook = semantic ? meta->semantic_codebook : meta->tail_codebook;
                const uint32_t n_levels = semantic ? meta->semantic_codebook_size : meta->tail_codebook_size;
                uint32_t code = ggml_sq_unpack_index(qs, (int) j, bits);
                if (code >= n_levels) {
                    code = n_levels - 1;
                }
                rotated[idx] = codebook[code];
            }

            if (block == 0 && meta->correction_dim > 0 && corr_scale > 0.0f) {
                for (uint32_t j = 0; j < meta->correction_dim; ++j) {
                    rotated[j] += corr_scale * ggml_sq_load_corr(corr, (int) j);
                }
            }
        }

        ggml_sq_inverse_rotate(meta, rotated, dst + row * meta->dim);
    }

    free(rotated);
}

static const struct ggml_spectral_registry_entry * ggml_sq_lookup_registry(const void * data, enum ggml_type type) {
    const uint8_t * ptr = (const uint8_t *) data;
    for (size_t i = 0; i < ggml_spectral_registry_size; ++i) {
        const struct ggml_spectral_registry_entry * entry = &ggml_spectral_registry[i];
        if (entry->type != type) {
            continue;
        }
        if (ptr >= entry->data && ptr < entry->data + entry->size) {
            return entry;
        }
    }
    return NULL;
}

static void ggml_sq_dequantize(
        enum ggml_type type,
        const void * x,
        float * y,
        int64_t k) {
    const struct ggml_spectral_registry_entry * reg = ggml_sq_lookup_registry(x, type);
    GGML_ASSERT(reg != NULL && "missing spectral tensor runtime metadata");
    ggml_sq_dequantize_rows(type, reg, (const uint8_t *) x, y, k);
}

struct ggml_sq_rot_cache {
    const struct ggml_spectral_weight_meta * meta;
    const float * input_ptr;
    int64_t n;
    size_t capacity;
    float * rotated;
    float * lut;
};

static GGML_THREAD_LOCAL struct ggml_sq_rot_cache ggml_sq_rot_cache = { 0 };

#define GGML_SQ_LUT_LEVELS 16

void ggml_sq_vec_cache_reset(void) {
    ggml_sq_rot_cache.meta = NULL;
    ggml_sq_rot_cache.input_ptr = NULL;
    ggml_sq_rot_cache.n = 0;
}

static bool ggml_sq_rot_cache_ensure(size_t n) {
    if (ggml_sq_rot_cache.capacity >= n) {
        return true;
    }

    float * new_rotated = (float *) malloc(n * sizeof(float));
    if (new_rotated == NULL) {
        return false;
    }

    float * new_lut = (float *) malloc(n * GGML_SQ_LUT_LEVELS * sizeof(float));
    if (new_lut == NULL) {
        free(new_rotated);
        return false;
    }

    free(ggml_sq_rot_cache.rotated);
    free(ggml_sq_rot_cache.lut);
    ggml_sq_rot_cache.rotated = new_rotated;
    ggml_sq_rot_cache.lut = new_lut;
    ggml_sq_rot_cache.capacity = n;
    ggml_sq_rot_cache.meta = NULL;
    ggml_sq_rot_cache.input_ptr = NULL;
    ggml_sq_rot_cache.n = 0;
    return true;
}

static void ggml_sq_build_lookup_table(
        const struct ggml_spectral_weight_meta * meta,
        const float * rotated,
        float * lut) {
    for (uint32_t idx = 0; idx < meta->dim; ++idx) {
        const bool semantic = idx < meta->split;
        const float * codebook = semantic ? meta->semantic_codebook : meta->tail_codebook;
        const uint32_t n_levels = semantic ? meta->semantic_codebook_size : meta->tail_codebook_size;
        const uint32_t last = n_levels - 1;
        float * row_lut = lut + (size_t) idx * GGML_SQ_LUT_LEVELS;
        const float v = rotated[idx];

        for (uint32_t code = 0; code < GGML_SQ_LUT_LEVELS; ++code) {
            const uint32_t clamped = code < n_levels ? code : last;
            row_lut[code] = v * codebook[clamped];
        }
    }
}

static bool ggml_sq_prepare_rotated_input(
        const struct ggml_spectral_registry_entry * reg,
        const float * GGML_RESTRICT vy,
        int64_t n,
        const float ** rotated_out,
        const float ** lut_out) {
    if (reg == NULL || vy == NULL || rotated_out == NULL || lut_out == NULL) {
        return false;
    }
    if (n <= 0 || n != (int64_t) reg->meta.dim) {
        return false;
    }
    if (!ggml_sq_rot_cache_ensure((size_t) n)) {
        return false;
    }

    if (ggml_sq_rot_cache.meta != &reg->meta ||
            ggml_sq_rot_cache.n != n ||
            ggml_sq_rot_cache.input_ptr != vy) {
        ggml_sq_forward_rotate(&reg->meta, vy, ggml_sq_rot_cache.rotated);
        ggml_sq_build_lookup_table(&reg->meta, ggml_sq_rot_cache.rotated, ggml_sq_rot_cache.lut);
        ggml_sq_rot_cache.meta = &reg->meta;
        ggml_sq_rot_cache.input_ptr = vy;
        ggml_sq_rot_cache.n = n;
    }

    *rotated_out = ggml_sq_rot_cache.rotated;
    *lut_out = ggml_sq_rot_cache.lut;
    return true;
}

static float ggml_sq_dot_row_correction(
        enum ggml_type type,
        const struct ggml_spectral_weight_meta * meta,
        const uint8_t * row_ptr,
        const float * rotated) {
    if (meta->correction_dim == 0) {
        return 0.0f;
    }

    const uint8_t * corr = ggml_sq_corr_ptr_const(row_ptr, type);
    const float corr_scale = GGML_FP16_TO_FP32(*ggml_sq_corr_scale_ptr_const(row_ptr, type));
    if (corr_scale <= 0.0f) {
        return 0.0f;
    }

    float sum = 0.0f;
    for (uint32_t j = 0; j < meta->correction_dim; ++j) {
        sum += corr_scale * ggml_sq_load_corr(corr, (int) j) * rotated[j];
    }
    return sum;
}

static float ggml_sq_dot_row_sq2_0(
        const struct ggml_spectral_weight_meta * meta,
        const uint8_t * row_ptr,
        const float * lut,
        const float * rotated) {
    float sum = 0.0f;

    for (uint32_t block = 0; block < meta->dim / QK_SQ; ++block) {
        const uint8_t * block_ptr = row_ptr + block * ggml_type_size(GGML_TYPE_SQ2_0);
        const uint8_t * qs = ggml_sq_qs_ptr_const(block_ptr, GGML_TYPE_SQ2_0);
        const float * block_lut = lut + (size_t) block * QK_SQ * GGML_SQ_LUT_LEVELS;

        for (uint32_t j = 0; j < QK_SQ / 4; ++j) {
            const uint8_t packed = qs[j];
            sum += block_lut[(size_t) (4 * j + 0) * GGML_SQ_LUT_LEVELS + ((packed >> 0) & 0x03)];
            sum += block_lut[(size_t) (4 * j + 1) * GGML_SQ_LUT_LEVELS + ((packed >> 2) & 0x03)];
            sum += block_lut[(size_t) (4 * j + 2) * GGML_SQ_LUT_LEVELS + ((packed >> 4) & 0x03)];
            sum += block_lut[(size_t) (4 * j + 3) * GGML_SQ_LUT_LEVELS + ((packed >> 6) & 0x03)];
        }
    }

    return sum + ggml_sq_dot_row_correction(GGML_TYPE_SQ2_0, meta, row_ptr, rotated);
}

static float ggml_sq_dot_row_sq3_1s(
        const struct ggml_spectral_weight_meta * meta,
        const uint8_t * row_ptr,
        const float * lut,
        const float * rotated) {
    float sum = 0.0f;

    for (uint32_t block = 0; block < meta->dim / QK_SQ; ++block) {
        const uint8_t * block_ptr = row_ptr + block * ggml_type_size(GGML_TYPE_SQ3_1S);
        const uint8_t * qs = ggml_sq_qs_ptr_const(block_ptr, GGML_TYPE_SQ3_1S);
        const float * block_lut = lut + (size_t) block * QK_SQ * GGML_SQ_LUT_LEVELS;
        uint32_t bitbuf = 0;
        int bits_in_buf = 0;
        const uint8_t * p = qs;

        for (uint32_t j = 0; j < QK_SQ; ++j) {
            while (bits_in_buf < 3) {
                bitbuf |= (uint32_t) (*p++) << bits_in_buf;
                bits_in_buf += 8;
            }
            const uint32_t code = bitbuf & 0x07u;
            bitbuf >>= 3;
            bits_in_buf -= 3;
            sum += block_lut[(size_t) j * GGML_SQ_LUT_LEVELS + code];
        }
    }

    return sum + ggml_sq_dot_row_correction(GGML_TYPE_SQ3_1S, meta, row_ptr, rotated);
}

static float ggml_sq_dot_row_sq4_1s(
        const struct ggml_spectral_weight_meta * meta,
        const uint8_t * row_ptr,
        const float * lut,
        const float * rotated) {
    float sum = 0.0f;

    for (uint32_t block = 0; block < meta->dim / QK_SQ; ++block) {
        const uint8_t * block_ptr = row_ptr + block * ggml_type_size(GGML_TYPE_SQ4_1S);
        const uint8_t * qs = ggml_sq_qs_ptr_const(block_ptr, GGML_TYPE_SQ4_1S);
        const float * block_lut = lut + (size_t) block * QK_SQ * GGML_SQ_LUT_LEVELS;

        for (uint32_t j = 0; j < QK_SQ / 2; ++j) {
            const uint8_t packed = qs[j];
            sum += block_lut[(size_t) (2 * j + 0) * GGML_SQ_LUT_LEVELS + (packed & 0x0Fu)];
            sum += block_lut[(size_t) (2 * j + 1) * GGML_SQ_LUT_LEVELS + (packed >> 4)];
        }
    }

    return sum + ggml_sq_dot_row_correction(GGML_TYPE_SQ4_1S, meta, row_ptr, rotated);
}

static float ggml_sq_dot_row(
        enum ggml_type type,
        const struct ggml_spectral_weight_meta * meta,
        const uint8_t * row_ptr,
        const float * lut,
        const float * rotated) {
    switch (type) {
        case GGML_TYPE_SQ2_0:
            return ggml_sq_dot_row_sq2_0(meta, row_ptr, lut, rotated);
        case GGML_TYPE_SQ3_1S:
            return ggml_sq_dot_row_sq3_1s(meta, row_ptr, lut, rotated);
        case GGML_TYPE_SQ4_1S:
            return ggml_sq_dot_row_sq4_1s(meta, row_ptr, lut, rotated);
        default:
            return 0.0f;
    }
}

bool ggml_sq_vec_dot_f32(
        enum ggml_type type,
        const void * GGML_RESTRICT vx,
        const float * GGML_RESTRICT vy,
        int64_t n,
        float * GGML_RESTRICT s) {
    const struct ggml_spectral_registry_entry * reg = ggml_sq_lookup_registry(vx, type);
    if (reg == NULL || s == NULL || vy == NULL) {
        return false;
    }

    const float * rotated = NULL;
    const float * lut = NULL;
    if (!ggml_sq_prepare_rotated_input(reg, vy, n, &rotated, &lut)) {
        return false;
    }

    *s = ggml_sq_dot_row(type, &reg->meta, (const uint8_t *) vx, lut, rotated);
    return true;
}

void dequantize_row_sq2_0(const block_sq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    ggml_sq_dequantize(GGML_TYPE_SQ2_0, x, y, k);
}

void dequantize_row_sq3_1s(const block_sq3_1s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    ggml_sq_dequantize(GGML_TYPE_SQ3_1S, x, y, k);
}

void dequantize_row_sq4_1s(const block_sq4_1s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    ggml_sq_dequantize(GGML_TYPE_SQ4_1S, x, y, k);
}

size_t ggml_quantize_spectral_weight(
        enum ggml_type type,
        const float * src,
        void * dst,
        int64_t nrows,
        int64_t n_per_row,
        const struct ggml_spectral_weight_meta * meta) {
    if (!ggml_is_spectral_weight_type(type) || !ggml_sq_meta_valid(meta, type)) {
        return 0;
    }
    if (nrows <= 0 || n_per_row <= 0 || n_per_row != (int64_t) meta->dim) {
        return 0;
    }

    const size_t row_size = ggml_sq_row_size(type, meta->dim);
    for (int64_t row = 0; row < nrows; ++row) {
        ggml_sq_quantize_row(type, meta, src + row * n_per_row, (uint8_t *) dst + row * row_size);
    }
    return (size_t) nrows * row_size;
}

bool ggml_spectral_register_tensor(
        const void * owner,
        const void * data,
        size_t size,
        enum ggml_type type,
        const struct ggml_spectral_weight_meta * meta) {
    if (!ggml_is_spectral_weight_type(type) || !ggml_sq_meta_valid(meta, type) || data == NULL || size == 0) {
        return false;
    }

    for (size_t i = 0; i < ggml_spectral_registry_size; ++i) {
        if (ggml_spectral_registry[i].data == (const uint8_t *) data) {
            ggml_spectral_registry[i].owner = owner;
            ggml_spectral_registry[i].size = size;
            ggml_spectral_registry[i].type = type;
            ggml_spectral_registry[i].meta = *meta;
            return true;
        }
    }

    if (ggml_spectral_registry_size == ggml_spectral_registry_capacity) {
        const size_t new_capacity = ggml_spectral_registry_capacity == 0 ? 16 : ggml_spectral_registry_capacity * 2;
        struct ggml_spectral_registry_entry * new_entries = (struct ggml_spectral_registry_entry *) realloc(
                ggml_spectral_registry, new_capacity * sizeof(struct ggml_spectral_registry_entry));
        if (new_entries == NULL) {
            return false;
        }
        ggml_spectral_registry = new_entries;
        ggml_spectral_registry_capacity = new_capacity;
    }

    ggml_spectral_registry[ggml_spectral_registry_size++] = (struct ggml_spectral_registry_entry) {
        .owner = owner,
        .data = (const uint8_t *) data,
        .size = size,
        .type = type,
        .meta = *meta,
    };
    return true;
}

void ggml_spectral_unregister_owner(const void * owner) {
    if (owner == NULL || ggml_spectral_registry_size == 0) {
        return;
    }

    size_t dst = 0;
    for (size_t src = 0; src < ggml_spectral_registry_size; ++src) {
        if (ggml_spectral_registry[src].owner != owner) {
            if (dst != src) {
                ggml_spectral_registry[dst] = ggml_spectral_registry[src];
            }
            ++dst;
        }
    }
    ggml_spectral_registry_size = dst;
}

/* ----------------------------------------------------------------------- */
/* SpectralQuant SKV2_0 / SKV3_0 / SKV4_0                                   */
/* ----------------------------------------------------------------------- */

struct ggml_spectral_kv_registry_entry {
    const void * owner;
    uint8_t * data;
    size_t size;
    enum ggml_type type;
    struct ggml_spectral_kv_meta meta;
};

static struct ggml_spectral_kv_registry_entry * ggml_spectral_kv_registry = NULL;
static size_t ggml_spectral_kv_registry_size = 0;
static size_t ggml_spectral_kv_registry_capacity = 0;

bool ggml_is_spectral_kv_type(enum ggml_type type) {
    return type == GGML_TYPE_SKV2_0 || type == GGML_TYPE_SKV3_0 || type == GGML_TYPE_SKV4_0;
}

static int ggml_skv_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_SKV2_0: return 2;
        case GGML_TYPE_SKV3_0: return 3;
        case GGML_TYPE_SKV4_0: return 4;
        default:               return 0;
    }
}

static size_t ggml_skv_qs_bytes(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_SKV2_0: return sizeof(((block_skv2_0 *) 0)->qs);
        case GGML_TYPE_SKV3_0: return sizeof(((block_skv3_0 *) 0)->qs);
        case GGML_TYPE_SKV4_0: return sizeof(((block_skv4_0 *) 0)->qs);
        default:               return 0;
    }
}

static uint8_t * ggml_skv_qs_ptr(void * block, enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_SKV2_0: return ((block_skv2_0 *) block)->qs;
        case GGML_TYPE_SKV3_0: return ((block_skv3_0 *) block)->qs;
        case GGML_TYPE_SKV4_0: return ((block_skv4_0 *) block)->qs;
        default:               return NULL;
    }
}

static const uint8_t * ggml_skv_qs_ptr_const(const void * block, enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_SKV2_0: return ((const block_skv2_0 *) block)->qs;
        case GGML_TYPE_SKV3_0: return ((const block_skv3_0 *) block)->qs;
        case GGML_TYPE_SKV4_0: return ((const block_skv4_0 *) block)->qs;
        default:               return NULL;
    }
}

static bool ggml_skv_meta_valid(const struct ggml_spectral_kv_meta * meta, enum ggml_type type) {
    if (meta == NULL || !ggml_is_spectral_kv_type(type)) {
        return false;
    }
    if (meta->n_head == 0 || meta->head_dim == 0 || meta->head_dim_padded < meta->head_dim) {
        return false;
    }
    if (meta->head_dim_padded % QK_SKV != 0 || meta->n_rows == 0 || meta->heads == NULL || meta->vec_norms == NULL) {
        return false;
    }
    if (meta->use_correction && meta->is_key) {
        if (meta->residual_norms == NULL || meta->qjl_signs == NULL || meta->qjl_bytes_per_head == 0) {
            return false;
        }
    }
    for (uint32_t h = 0; h < meta->n_head; ++h) {
        const struct ggml_spectral_kv_head_meta * head = &meta->heads[h];
        const uint32_t max_levels = (uint32_t) (1u << ggml_skv_bits(type));
        if (head->dim != meta->head_dim || head->split == 0 || head->split > head->dim) {
            return false;
        }
        if (head->semantic_codebook_size == 0 || head->tail_codebook_size == 0) {
            return false;
        }
        if (head->semantic_codebook_size > max_levels || head->tail_codebook_size > max_levels) {
            return false;
        }
        if (head->basis == NULL || head->semantic_codebook == NULL || head->tail_codebook == NULL) {
            return false;
        }
        if (meta->use_correction && meta->is_key && head->qjl_matrix == NULL) {
            return false;
        }
    }
    return true;
}

static void ggml_skv_store_sign(uint8_t * dst, uint32_t idx, int positive) {
    const uint8_t bit = positive ? 1u : 0u;
    dst[idx / 8] &= (uint8_t) ~(1u << (idx & 7));
    dst[idx / 8] |= (uint8_t) (bit << (idx & 7));
}

static int ggml_skv_load_sign(const uint8_t * src, uint32_t idx) {
    return ((src[idx / 8] >> (idx & 7)) & 1u) ? 1 : -1;
}

static const struct ggml_spectral_kv_registry_entry * ggml_skv_lookup_registry(
        const void * data,
        enum ggml_type type,
        uint32_t * row_index,
        uint32_t * head_index) {
    const uint8_t * ptr = (const uint8_t *) data;
    for (size_t i = 0; i < ggml_spectral_kv_registry_size; ++i) {
        const struct ggml_spectral_kv_registry_entry * entry = &ggml_spectral_kv_registry[i];
        if (entry->type != type) {
            continue;
        }
        if (ptr < entry->data || ptr >= entry->data + entry->size) {
            continue;
        }
        const size_t row_size_token = ggml_row_size(type, (int64_t) entry->meta.head_dim_padded * entry->meta.n_head);
        const size_t row_size_head = ggml_row_size(type, entry->meta.head_dim_padded);
        const size_t offset = (size_t) (ptr - entry->data);
        const size_t in_row = offset % row_size_token;
        GGML_ASSERT(in_row % row_size_head == 0);
        if (row_index) {
            *row_index = (uint32_t) (offset / row_size_token);
        }
        if (head_index) {
            *head_index = (uint32_t) (in_row / row_size_head);
        }
        return entry;
    }
    return NULL;
}

static struct ggml_spectral_kv_registry_entry * ggml_skv_lookup_registry_mut(
        void * data,
        enum ggml_type type,
        uint32_t * row_index) {
    uint32_t unused_head = 0;
    return (struct ggml_spectral_kv_registry_entry *) ggml_skv_lookup_registry(data, type, row_index, &unused_head);
}

static void ggml_skv_forward_rotate(
        const struct ggml_spectral_kv_head_meta * meta,
        const float * input,
        float * rotated) {
    for (uint32_t col = 0; col < meta->dim; ++col) {
        float acc = 0.0f;
        for (uint32_t row = 0; row < meta->dim; ++row) {
            acc += meta->basis[(size_t) row * meta->dim + col] * input[row];
        }
        rotated[col] = acc;
    }
}

static void ggml_skv_inverse_rotate(
        const struct ggml_spectral_kv_head_meta * meta,
        const float * rotated,
        float * output) {
    for (uint32_t row = 0; row < meta->dim; ++row) {
        float acc = 0.0f;
        for (uint32_t col = 0; col < meta->dim; ++col) {
            acc += meta->basis[(size_t) row * meta->dim + col] * rotated[col];
        }
        output[row] = acc;
    }
}

static void ggml_skv_quantize_head(
        enum ggml_type type,
        const struct ggml_spectral_kv_head_meta * head,
        uint32_t head_dim_padded,
        int is_key,
        int use_correction,
        const float * src,
        uint8_t * dst,
        float * vec_norm_slot,
        float * residual_norm_slot,
        uint8_t * sign_slot,
        uint32_t sign_bytes) {
    const int bits = ggml_skv_bits(type);
    const size_t block_size = ggml_type_size(type);
    float * normalized = (float *) malloc((size_t) head->dim * sizeof(float));
    float * rotated = (float *) malloc((size_t) head->dim * sizeof(float));
    float * residual_sem = (float *) malloc((size_t) head->split * sizeof(float));
    GGML_ASSERT(normalized != NULL && rotated != NULL && residual_sem != NULL);

    float vec_norm = 0.0f;
    for (uint32_t i = 0; i < head->dim; ++i) {
        vec_norm += src[i] * src[i];
    }
    vec_norm = sqrtf(vec_norm);
    const float inv_norm = vec_norm > 1e-8f ? 1.0f / vec_norm : 0.0f;

    for (uint32_t i = 0; i < head->dim; ++i) {
        normalized[i] = src[i] * inv_norm;
    }
    ggml_skv_forward_rotate(head, normalized, rotated);

    float residual_ss = 0.0f;
    for (uint32_t i = 0; i < head->split; ++i) {
        residual_sem[i] = 0.0f;
    }

    for (uint32_t block = 0; block < head_dim_padded / QK_SKV; ++block) {
        uint8_t * block_ptr = dst + block * block_size;
        uint8_t * qs = ggml_skv_qs_ptr(block_ptr, type);
        memset(qs, 0, ggml_skv_qs_bytes(type));

        for (uint32_t j = 0; j < QK_SKV; ++j) {
            const uint32_t idx = block * QK_SKV + j;
            const int semantic = idx < head->split;
            const float * codebook = semantic ? head->semantic_codebook : head->tail_codebook;
            const uint32_t n_levels = semantic ? head->semantic_codebook_size : head->tail_codebook_size;
            const float value = idx < head->dim ? rotated[idx] : 0.0f;
            const uint32_t code = ggml_sq_quantize_scalar(value, codebook, n_levels);
            ggml_sq_pack_index(qs, (int) j, bits, code);

            if (idx < head->dim) {
                const float residual = value - codebook[code];
                residual_ss += residual * residual;
                if (semantic && use_correction && is_key) {
                    residual_sem[idx] = residual * vec_norm;
                }
            }
        }
    }

    *vec_norm_slot = vec_norm;
    if (residual_norm_slot) {
        *residual_norm_slot = 0.0f;
    }
    if (sign_slot && sign_bytes > 0) {
        memset(sign_slot, 0, sign_bytes);
    }

    if (is_key && use_correction && residual_norm_slot && sign_slot && sign_bytes > 0) {
        const float residual_norm = vec_norm * sqrtf(residual_ss);
        *residual_norm_slot = residual_norm;
        if (residual_norm > 0.0f) {
            for (uint32_t proj = 0; proj < head->split; ++proj) {
                float acc = 0.0f;
                for (uint32_t i = 0; i < head->split; ++i) {
                    acc += residual_sem[i] * head->qjl_matrix[(size_t) proj * head->split + i];
                }
                ggml_skv_store_sign(sign_slot, proj, acc >= 0.0f);
            }
        }
    }

    free(residual_sem);
    free(rotated);
    free(normalized);
}

static void ggml_skv_quantize_row(
        enum ggml_type type,
        const float * x,
        void * y,
        int64_t k) {
    uint32_t row_index = 0;
    struct ggml_spectral_kv_registry_entry * reg = ggml_skv_lookup_registry_mut(y, type, &row_index);
    GGML_ASSERT(reg != NULL && "missing spectral KV tensor runtime metadata");
    GGML_ASSERT(k == (int64_t) reg->meta.head_dim_padded * reg->meta.n_head);

    const size_t row_size_head = ggml_row_size(type, reg->meta.head_dim_padded);
    for (uint32_t h = 0; h < reg->meta.n_head; ++h) {
        float * vec_norm_slot = &reg->meta.vec_norms[(size_t) row_index * reg->meta.n_head + h];
        float * residual_norm_slot = reg->meta.residual_norms ? &reg->meta.residual_norms[(size_t) row_index * reg->meta.n_head + h] : NULL;
        uint8_t * sign_slot = reg->meta.qjl_signs ? reg->meta.qjl_signs +
                ((size_t) row_index * reg->meta.n_head + h) * reg->meta.qjl_bytes_per_head : NULL;
        ggml_skv_quantize_head(
                type,
                &reg->meta.heads[h],
                reg->meta.head_dim_padded,
                reg->meta.is_key,
                reg->meta.use_correction,
                x + (size_t) h * reg->meta.head_dim_padded,
                (uint8_t *) y + (size_t) h * row_size_head,
                vec_norm_slot,
                residual_norm_slot,
                sign_slot,
                reg->meta.qjl_bytes_per_head);
    }
}

static void ggml_skv_decode_head(
        enum ggml_type type,
        const struct ggml_spectral_kv_registry_entry * reg,
        uint32_t row_index,
        uint32_t head_index,
        const uint8_t * src,
        float * dst) {
    const struct ggml_spectral_kv_head_meta * head = &reg->meta.heads[head_index];
    const int bits = ggml_skv_bits(type);
    float * rotated = (float *) malloc((size_t) head->dim * sizeof(float));
    float * decoded = (float *) malloc((size_t) head->dim * sizeof(float));
    GGML_ASSERT(rotated != NULL && decoded != NULL);

    for (uint32_t i = 0; i < head->dim; ++i) {
        rotated[i] = 0.0f;
    }

    for (uint32_t block = 0; block < reg->meta.head_dim_padded / QK_SKV; ++block) {
        const uint8_t * block_ptr = src + block * ggml_type_size(type);
        const uint8_t * qs = ggml_skv_qs_ptr_const(block_ptr, type);

        for (uint32_t j = 0; j < QK_SKV; ++j) {
            const uint32_t idx = block * QK_SKV + j;
            if (idx >= head->dim) {
                continue;
            }
            const int semantic = idx < head->split;
            const float * codebook = semantic ? head->semantic_codebook : head->tail_codebook;
            const uint32_t n_levels = semantic ? head->semantic_codebook_size : head->tail_codebook_size;
            uint32_t code = ggml_sq_unpack_index(qs, (int) j, bits);
            if (code >= n_levels) {
                code = n_levels - 1;
            }
            rotated[idx] = codebook[code] * reg->meta.vec_norms[(size_t) row_index * reg->meta.n_head + head_index];
        }
    }

    ggml_skv_inverse_rotate(head, rotated, decoded);
    memcpy(dst, decoded, (size_t) head->dim * sizeof(float));
    if (reg->meta.head_dim_padded > head->dim) {
        memset(dst + head->dim, 0, (size_t) (reg->meta.head_dim_padded - head->dim) * sizeof(float));
    }

    free(decoded);
    free(rotated);
}

static void ggml_skv_dequantize(
        enum ggml_type type,
        const void * x,
        float * y,
        int64_t k) {
    uint32_t row_index = 0;
    uint32_t head_index = 0;
    const struct ggml_spectral_kv_registry_entry * reg = ggml_skv_lookup_registry(x, type, &row_index, &head_index);
    GGML_ASSERT(reg != NULL && "missing spectral KV tensor runtime metadata");

    if (k == reg->meta.head_dim_padded) {
        ggml_skv_decode_head(type, reg, row_index, head_index, (const uint8_t *) x, y);
        return;
    }

    if (k == (int64_t) reg->meta.head_dim_padded * reg->meta.n_head && head_index == 0) {
        const size_t row_size_head = ggml_row_size(type, reg->meta.head_dim_padded);
        for (uint32_t h = 0; h < reg->meta.n_head; ++h) {
            ggml_skv_decode_head(type, reg, row_index, h, (const uint8_t *) x + (size_t) h * row_size_head,
                    y + (size_t) h * reg->meta.head_dim_padded);
        }
        return;
    }

    GGML_ABORT("%s: unsupported SKV dequantize width %" PRId64, __func__, k);
}

static void ggml_skv_vec_dot_f32(
        enum ggml_type type,
        int n,
        float * GGML_RESTRICT s,
        const void * GGML_RESTRICT vx,
        const void * GGML_RESTRICT vy) {
    uint32_t row_index = 0;
    uint32_t head_index = 0;
    const struct ggml_spectral_kv_registry_entry * reg = ggml_skv_lookup_registry(vx, type, &row_index, &head_index);
    GGML_ASSERT(reg != NULL && "missing spectral KV tensor runtime metadata");
    GGML_ASSERT(n == (int) reg->meta.head_dim_padded);

    const struct ggml_spectral_kv_head_meta * head = &reg->meta.heads[head_index];
    const int bits = ggml_skv_bits(type);
    const float * query = (const float *) vy;
    const uint8_t * row_ptr = (const uint8_t *) vx;
    const float vec_norm = reg->meta.vec_norms[(size_t) row_index * reg->meta.n_head + head_index];

    float * q_rot = (float *) malloc((size_t) head->dim * sizeof(float));
    GGML_ASSERT(q_rot != NULL);
    ggml_skv_forward_rotate(head, query, q_rot);

    float term1 = 0.0f;
    for (uint32_t block = 0; block < reg->meta.head_dim_padded / QK_SKV; ++block) {
        const uint8_t * block_ptr = row_ptr + block * ggml_type_size(type);
        const uint8_t * qs = ggml_skv_qs_ptr_const(block_ptr, type);

        for (uint32_t j = 0; j < QK_SKV; ++j) {
            const uint32_t idx = block * QK_SKV + j;
            if (idx >= head->dim) {
                continue;
            }
            const int semantic = idx < head->split;
            const float * codebook = semantic ? head->semantic_codebook : head->tail_codebook;
            const uint32_t n_levels = semantic ? head->semantic_codebook_size : head->tail_codebook_size;
            uint32_t code = ggml_sq_unpack_index(qs, (int) j, bits);
            if (code >= n_levels) {
                code = n_levels - 1;
            }
            term1 += q_rot[idx] * (codebook[code] * vec_norm);
        }
    }

    float term2 = 0.0f;
    if (reg->meta.is_key && reg->meta.use_correction && reg->meta.residual_norms && reg->meta.qjl_signs && head->split > 0) {
        const float residual_norm = reg->meta.residual_norms[(size_t) row_index * reg->meta.n_head + head_index];
        if (residual_norm > 0.0f) {
            const uint8_t * sign_slot = reg->meta.qjl_signs +
                    ((size_t) row_index * reg->meta.n_head + head_index) * reg->meta.qjl_bytes_per_head;
            for (uint32_t proj = 0; proj < head->split; ++proj) {
                float acc = 0.0f;
                for (uint32_t i = 0; i < head->split; ++i) {
                    acc += q_rot[i] * head->qjl_matrix[(size_t) proj * head->split + i];
                }
                term2 += acc * (float) ggml_skv_load_sign(sign_slot, proj);
            }
            term2 *= TURBO_QJL_CONST * residual_norm / (float) head->split;
        }
    }

    *s = term1 + term2;
    free(q_rot);
}

void quantize_row_skv2_0_ref(const float * GGML_RESTRICT x, block_skv2_0 * GGML_RESTRICT y, int64_t k) {
    ggml_skv_quantize_row(GGML_TYPE_SKV2_0, x, y, k);
}

void quantize_row_skv3_0_ref(const float * GGML_RESTRICT x, block_skv3_0 * GGML_RESTRICT y, int64_t k) {
    ggml_skv_quantize_row(GGML_TYPE_SKV3_0, x, y, k);
}

void quantize_row_skv4_0_ref(const float * GGML_RESTRICT x, block_skv4_0 * GGML_RESTRICT y, int64_t k) {
    ggml_skv_quantize_row(GGML_TYPE_SKV4_0, x, y, k);
}

void dequantize_row_skv2_0(const block_skv2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    ggml_skv_dequantize(GGML_TYPE_SKV2_0, x, y, k);
}

void dequantize_row_skv3_0(const block_skv3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    ggml_skv_dequantize(GGML_TYPE_SKV3_0, x, y, k);
}

void dequantize_row_skv4_0(const block_skv4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    ggml_skv_dequantize(GGML_TYPE_SKV4_0, x, y, k);
}

void ggml_vec_dot_skv2_0_f32(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(nrc);
    ggml_skv_vec_dot_f32(GGML_TYPE_SKV2_0, n, s, vx, vy);
}

void ggml_vec_dot_skv3_0_f32(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(nrc);
    ggml_skv_vec_dot_f32(GGML_TYPE_SKV3_0, n, s, vx, vy);
}

void ggml_vec_dot_skv4_0_f32(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(nrc);
    ggml_skv_vec_dot_f32(GGML_TYPE_SKV4_0, n, s, vx, vy);
}

bool ggml_spectral_kv_register_tensor(
        const void * owner,
        void * data,
        size_t size,
        enum ggml_type type,
        const struct ggml_spectral_kv_meta * meta) {
    if (!ggml_skv_meta_valid(meta, type) || data == NULL || size == 0) {
        return false;
    }

    for (size_t i = 0; i < ggml_spectral_kv_registry_size; ++i) {
        if (ggml_spectral_kv_registry[i].data == (uint8_t *) data) {
            ggml_spectral_kv_registry[i].owner = owner;
            ggml_spectral_kv_registry[i].size = size;
            ggml_spectral_kv_registry[i].type = type;
            ggml_spectral_kv_registry[i].meta = *meta;
            return true;
        }
    }

    if (ggml_spectral_kv_registry_size == ggml_spectral_kv_registry_capacity) {
        const size_t new_capacity = ggml_spectral_kv_registry_capacity == 0 ? 16 : ggml_spectral_kv_registry_capacity * 2;
        struct ggml_spectral_kv_registry_entry * new_entries = (struct ggml_spectral_kv_registry_entry *) realloc(
                ggml_spectral_kv_registry, new_capacity * sizeof(struct ggml_spectral_kv_registry_entry));
        if (new_entries == NULL) {
            return false;
        }
        ggml_spectral_kv_registry = new_entries;
        ggml_spectral_kv_registry_capacity = new_capacity;
    }

    ggml_spectral_kv_registry[ggml_spectral_kv_registry_size++] = (struct ggml_spectral_kv_registry_entry) {
        .owner = owner,
        .data = (uint8_t *) data,
        .size = size,
        .type = type,
        .meta = *meta,
    };
    return true;
}

void ggml_spectral_kv_unregister_owner(const void * owner) {
    if (owner == NULL || ggml_spectral_kv_registry_size == 0) {
        return;
    }

    size_t dst = 0;
    for (size_t src = 0; src < ggml_spectral_kv_registry_size; ++src) {
        if (ggml_spectral_kv_registry[src].owner != owner) {
            if (dst != src) {
                ggml_spectral_kv_registry[dst] = ggml_spectral_kv_registry[src];
            }
            ++dst;
        }
    }
    ggml_spectral_kv_registry_size = dst;
}
