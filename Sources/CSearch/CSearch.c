#include "CSearch.h"

#include <limits.h>
#include <stddef.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

static inline void rinha_init_neighbors(size_t k, rinha_neighbor_t *out_neighbors) {
    for (size_t i = 0; i < k; i++) {
        out_neighbors[i].record_index = -1;
        out_neighbors[i].distance_squared = INT64_MAX;
    }
}

static inline void rinha_insert_neighbor(
    rinha_neighbor_t candidate,
    size_t k,
    rinha_neighbor_t *top
) {
    if (k == 0) {
        return;
    }
    if (candidate.distance_squared >= top[k - 1].distance_squared) {
        return;
    }
    size_t i = k - 1;
    while (i > 0 && top[i - 1].distance_squared > candidate.distance_squared) {
        top[i] = top[i - 1];
        i--;
    }
    top[i] = candidate;
}

static inline int64_t rinha_distance_squared_scalar(
    const int16_t *query,
    const int16_t *record,
    size_t dim
) {
    int64_t sum = 0;
    for (size_t lane = 0; lane < dim; lane++) {
        int32_t diff = (int32_t)query[lane] - (int32_t)record[lane];
        sum += (int64_t)diff * (int64_t)diff;
    }
    return sum;
}

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2"), always_inline))
static inline int64_t rinha_distance_squared_avx2_16(
    __m256i q,
    const int16_t *record
) {
    __m256i r = _mm256_loadu_si256((const __m256i *)record);
    __m256i diff = _mm256_sub_epi16(q, r);
    __m256i pair_sums = _mm256_madd_epi16(diff, diff);
    // Horizontal sum of 8 int32s in pair_sums.
    __m128i lo = _mm256_castsi256_si128(pair_sums);
    __m128i hi = _mm256_extracti128_si256(pair_sums, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return (int64_t)_mm_cvtsi128_si32(s);
}

__attribute__((target("avx2")))
static void rinha_topk_avx2_contiguous(
    const int16_t *query,
    const int16_t *vectors,
    size_t count,
    size_t stride,
    size_t k,
    rinha_neighbor_t *out_neighbors
) {
    __m256i q = _mm256_loadu_si256((const __m256i *)query);
    const size_t prefetch_distance = 8;
    for (size_t record_index = 0; record_index < count; record_index++) {
        const int16_t *record = vectors + (record_index * stride);
        if (record_index + prefetch_distance < count) {
            _mm_prefetch((const char *)(vectors + ((record_index + prefetch_distance) * stride)), _MM_HINT_T0);
        }
        int64_t distance_squared = rinha_distance_squared_avx2_16(q, record);
        rinha_neighbor_t candidate = {
            .record_index = (int32_t)record_index,
            .distance_squared = distance_squared,
        };
        rinha_insert_neighbor(candidate, k, out_neighbors);
    }
}

__attribute__((target("avx2")))
static void rinha_topk_avx2_indexed(
    const int16_t *query,
    const int16_t *vectors,
    const uint32_t *record_indices,
    size_t candidate_count,
    size_t stride,
    size_t k,
    rinha_neighbor_t *out_neighbors
) {
    __m256i q = _mm256_loadu_si256((const __m256i *)query);
    const size_t prefetch_distance = 8;
    for (size_t i = 0; i < candidate_count; i++) {
        const uint32_t record_index = record_indices[i];
        const int16_t *record = vectors + ((size_t)record_index * stride);
        if (i + prefetch_distance < candidate_count) {
            const uint32_t pf_idx = record_indices[i + prefetch_distance];
            _mm_prefetch((const char *)(vectors + ((size_t)pf_idx * stride)), _MM_HINT_T0);
        }
        int64_t distance_squared = rinha_distance_squared_avx2_16(q, record);
        rinha_neighbor_t candidate = {
            .record_index = (int32_t)record_index,
            .distance_squared = distance_squared,
        };
        rinha_insert_neighbor(candidate, k, out_neighbors);
    }
}

static int rinha_supports_avx2(void) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2");
#else
    return 0;
#endif
}
#endif

void rinha_topk_exact_i16(
    const int16_t *query,
    const int16_t *vectors,
    size_t count,
    size_t dim,
    size_t stride,
    size_t k,
    rinha_neighbor_t *out_neighbors
) {
    if (k == 0 || count == 0) {
        return;
    }

    rinha_init_neighbors(k, out_neighbors);

#if defined(__x86_64__) || defined(__i386__)
    if (stride == 16 && dim <= 16 && rinha_supports_avx2()) {
        rinha_topk_avx2_contiguous(query, vectors, count, stride, k, out_neighbors);
        return;
    }
#endif

    for (size_t record_index = 0; record_index < count; record_index++) {
        const int16_t *record = vectors + (record_index * stride);
        int64_t distance_squared = rinha_distance_squared_scalar(query, record, dim);
        rinha_neighbor_t candidate = {
            .record_index = (int32_t)record_index,
            .distance_squared = distance_squared,
        };
        rinha_insert_neighbor(candidate, k, out_neighbors);
    }
}

void rinha_topk_exact_i16_indexed(
    const int16_t *query,
    const int16_t *vectors,
    const uint32_t *record_indices,
    size_t candidate_count,
    size_t dim,
    size_t stride,
    size_t k,
    rinha_neighbor_t *out_neighbors
) {
    if (k == 0 || candidate_count == 0) {
        return;
    }

    rinha_init_neighbors(k, out_neighbors);

#if defined(__x86_64__) || defined(__i386__)
    if (stride == 16 && dim <= 16 && rinha_supports_avx2()) {
        rinha_topk_avx2_indexed(query, vectors, record_indices, candidate_count, stride, k, out_neighbors);
        return;
    }
#endif

    for (size_t i = 0; i < candidate_count; i++) {
        const uint32_t record_index = record_indices[i];
        const int16_t *record = vectors + ((size_t)record_index * stride);
        int64_t distance_squared = rinha_distance_squared_scalar(query, record, dim);
        rinha_neighbor_t candidate = {
            .record_index = (int32_t)record_index,
            .distance_squared = distance_squared,
        };
        rinha_insert_neighbor(candidate, k, out_neighbors);
    }
}
