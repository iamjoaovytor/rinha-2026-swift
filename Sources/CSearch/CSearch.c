#include "CSearch.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

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

__attribute__((target("avx2"), always_inline))
static inline int64_t rinha_hsum_epi32(__m256i pair_sums) {
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
    size_t i = 0;
    size_t pair_end = count >= 2 ? count - 1 : 0;
    for (; i < pair_end; i += 2) {
        const int16_t *r0 = vectors + (i * stride);
        const int16_t *r1 = vectors + ((i + 1) * stride);
        if (i + prefetch_distance < count) {
            _mm_prefetch((const char *)(vectors + ((i + prefetch_distance) * stride)), _MM_HINT_T0);
        }
        if (i + prefetch_distance + 1 < count) {
            _mm_prefetch((const char *)(vectors + ((i + prefetch_distance + 1) * stride)), _MM_HINT_T0);
        }
        __m256i v0 = _mm256_loadu_si256((const __m256i *)r0);
        __m256i v1 = _mm256_loadu_si256((const __m256i *)r1);
        __m256i d0 = _mm256_sub_epi16(q, v0);
        __m256i d1 = _mm256_sub_epi16(q, v1);
        __m256i p0 = _mm256_madd_epi16(d0, d0);
        __m256i p1 = _mm256_madd_epi16(d1, d1);
        int64_t dist0 = rinha_hsum_epi32(p0);
        int64_t dist1 = rinha_hsum_epi32(p1);
        rinha_neighbor_t c0 = { .record_index = (int32_t)i, .distance_squared = dist0 };
        rinha_neighbor_t c1 = { .record_index = (int32_t)(i + 1), .distance_squared = dist1 };
        rinha_insert_neighbor(c0, k, out_neighbors);
        rinha_insert_neighbor(c1, k, out_neighbors);
    }
    for (; i < count; i++) {
        const int16_t *record = vectors + (i * stride);
        int64_t distance_squared = rinha_distance_squared_avx2_16(q, record);
        rinha_neighbor_t candidate = {
            .record_index = (int32_t)i,
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
    size_t i = 0;
    size_t pair_end = candidate_count >= 2 ? candidate_count - 1 : 0;
    for (; i < pair_end; i += 2) {
        const uint32_t idx0 = record_indices[i];
        const uint32_t idx1 = record_indices[i + 1];
        const int16_t *r0 = vectors + ((size_t)idx0 * stride);
        const int16_t *r1 = vectors + ((size_t)idx1 * stride);
        if (i + prefetch_distance < candidate_count) {
            const uint32_t pf0 = record_indices[i + prefetch_distance];
            _mm_prefetch((const char *)(vectors + ((size_t)pf0 * stride)), _MM_HINT_T0);
        }
        if (i + prefetch_distance + 1 < candidate_count) {
            const uint32_t pf1 = record_indices[i + prefetch_distance + 1];
            _mm_prefetch((const char *)(vectors + ((size_t)pf1 * stride)), _MM_HINT_T0);
        }
        __m256i v0 = _mm256_loadu_si256((const __m256i *)r0);
        __m256i v1 = _mm256_loadu_si256((const __m256i *)r1);
        __m256i d0 = _mm256_sub_epi16(q, v0);
        __m256i d1 = _mm256_sub_epi16(q, v1);
        __m256i p0 = _mm256_madd_epi16(d0, d0);
        __m256i p1 = _mm256_madd_epi16(d1, d1);
        int64_t dist0 = rinha_hsum_epi32(p0);
        int64_t dist1 = rinha_hsum_epi32(p1);
        rinha_neighbor_t c0 = { .record_index = (int32_t)idx0, .distance_squared = dist0 };
        rinha_neighbor_t c1 = { .record_index = (int32_t)idx1, .distance_squared = dist1 };
        rinha_insert_neighbor(c0, k, out_neighbors);
        rinha_insert_neighbor(c1, k, out_neighbors);
    }
    for (; i < candidate_count; i++) {
        const uint32_t record_index = record_indices[i];
        const int16_t *record = vectors + ((size_t)record_index * stride);
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

// Filtered AVX2 variants: skip records where lanes 9/10/11 (binary features:
// isOnline, cardPresent, knownMerchant) don't match the query. Binary lane
// diff contributes scale² (= 1e8 with scale=10000) to L2 squared, dominating
// the typical top-5 distance (~ few million). Safe to skip mismatches.

__attribute__((target("avx2")))
static void rinha_topk_avx2_contiguous_filtered(
    const int16_t *query,
    const int16_t *vectors,
    size_t count,
    size_t stride,
    size_t k,
    rinha_neighbor_t *out_neighbors
) {
    __m256i q = _mm256_loadu_si256((const __m256i *)query);
    // Lanes 9..11 occupy bytes 18..23 of the int16[16] record. Load 6 bytes
    // (mask onto a 64-bit register) so a single 48-bit compare gates the
    // distance compute instead of three separate int16 compares.
    uint64_t q_mask = 0;
    memcpy(&q_mask, ((const char *)query) + 18, 6);
    const size_t prefetch_distance = 8;
    for (size_t i = 0; i < count; i++) {
        const int16_t *r = vectors + (i * stride);
        if (i + prefetch_distance < count) {
            _mm_prefetch((const char *)(vectors + ((i + prefetch_distance) * stride)), _MM_HINT_T0);
        }
        uint64_t r_mask = 0;
        memcpy(&r_mask, ((const char *)r) + 18, 6);
        if (r_mask != q_mask) continue;
        int64_t distance_squared = rinha_distance_squared_avx2_16(q, r);
        rinha_neighbor_t candidate = {
            .record_index = (int32_t)i,
            .distance_squared = distance_squared,
        };
        rinha_insert_neighbor(candidate, k, out_neighbors);
    }
}

__attribute__((target("avx2")))
static void rinha_topk_avx2_indexed_filtered(
    const int16_t *query,
    const int16_t *vectors,
    const uint32_t *record_indices,
    size_t candidate_count,
    size_t stride,
    size_t k,
    rinha_neighbor_t *out_neighbors
) {
    __m256i q = _mm256_loadu_si256((const __m256i *)query);
    uint64_t q_mask = 0;
    memcpy(&q_mask, ((const char *)query) + 18, 6);
    const size_t prefetch_distance = 8;
    for (size_t i = 0; i < candidate_count; i++) {
        const uint32_t record_index = record_indices[i];
        const int16_t *r = vectors + ((size_t)record_index * stride);
        if (i + prefetch_distance < candidate_count) {
            const uint32_t pf = record_indices[i + prefetch_distance];
            _mm_prefetch((const char *)(vectors + ((size_t)pf * stride)), _MM_HINT_T0);
        }
        uint64_t r_mask = 0;
        memcpy(&r_mask, ((const char *)r) + 18, 6);
        if (r_mask != q_mask) continue;
        int64_t distance_squared = rinha_distance_squared_avx2_16(q, r);
        rinha_neighbor_t candidate = {
            .record_index = (int32_t)record_index,
            .distance_squared = distance_squared,
        };
        rinha_insert_neighbor(candidate, k, out_neighbors);
    }
}
#endif

void rinha_topk_exact_i16_filtered(
    const int16_t *query,
    const int16_t *vectors,
    size_t count,
    size_t dim,
    size_t stride,
    size_t k,
    rinha_neighbor_t *out_neighbors
) {
    if (k == 0 || count == 0) return;
    rinha_init_neighbors(k, out_neighbors);

#if defined(__x86_64__) || defined(__i386__)
    if (stride == 16 && dim <= 16 && rinha_supports_avx2()) {
        rinha_topk_avx2_contiguous_filtered(query, vectors, count, stride, k, out_neighbors);
        return;
    }
#endif

    const int16_t q9 = query[9], q10 = query[10], q11 = query[11];
    for (size_t i = 0; i < count; i++) {
        const int16_t *r = vectors + (i * stride);
        if (r[9] != q9 || r[10] != q10 || r[11] != q11) continue;
        int64_t distance_squared = rinha_distance_squared_scalar(query, r, dim);
        rinha_neighbor_t candidate = {
            .record_index = (int32_t)i,
            .distance_squared = distance_squared,
        };
        rinha_insert_neighbor(candidate, k, out_neighbors);
    }
}

void rinha_topk_exact_i16_indexed_filtered(
    const int16_t *query,
    const int16_t *vectors,
    const uint32_t *record_indices,
    size_t candidate_count,
    size_t dim,
    size_t stride,
    size_t k,
    rinha_neighbor_t *out_neighbors
) {
    if (k == 0 || candidate_count == 0) return;
    rinha_init_neighbors(k, out_neighbors);

#if defined(__x86_64__) || defined(__i386__)
    if (stride == 16 && dim <= 16 && rinha_supports_avx2()) {
        rinha_topk_avx2_indexed_filtered(query, vectors, record_indices, candidate_count, stride, k, out_neighbors);
        return;
    }
#endif

    const int16_t q9 = query[9], q10 = query[10], q11 = query[11];
    for (size_t i = 0; i < candidate_count; i++) {
        const uint32_t record_index = record_indices[i];
        const int16_t *r = vectors + ((size_t)record_index * stride);
        if (r[9] != q9 || r[10] != q10 || r[11] != q11) continue;
        int64_t distance_squared = rinha_distance_squared_scalar(query, r, dim);
        rinha_neighbor_t candidate = {
            .record_index = (int32_t)record_index,
            .distance_squared = distance_squared,
        };
        rinha_insert_neighbor(candidate, k, out_neighbors);
    }
}

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
