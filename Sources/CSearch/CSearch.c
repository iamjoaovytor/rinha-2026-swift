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
__attribute__((target("avx2")))
static int64_t rinha_distance_squared_avx2_16(
    const int16_t *query,
    const int16_t *record
) {
    __m256i q = _mm256_loadu_si256((const __m256i *)query);
    __m256i r = _mm256_loadu_si256((const __m256i *)record);
    __m256i diff = _mm256_sub_epi16(q, r);
    __m256i pair_sums = _mm256_madd_epi16(diff, diff);

    int32_t lanes[8];
    _mm256_storeu_si256((__m256i *)lanes, pair_sums);

    int64_t sum = 0;
    for (size_t i = 0; i < 8; i++) {
        sum += lanes[i];
    }
    return sum;
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
    const int use_avx2 = (stride == 16 && dim <= 16 && rinha_supports_avx2());
#endif

    for (size_t record_index = 0; record_index < count; record_index++) {
        const int16_t *record = vectors + (record_index * stride);
        int64_t distance_squared;

#if defined(__x86_64__) || defined(__i386__)
        if (use_avx2) {
            distance_squared = rinha_distance_squared_avx2_16(query, record);
        } else {
            distance_squared = rinha_distance_squared_scalar(query, record, dim);
        }
#else
        distance_squared = rinha_distance_squared_scalar(query, record, dim);
#endif

        rinha_neighbor_t candidate = {
            .record_index = (int32_t)record_index,
            .distance_squared = distance_squared,
        };
        rinha_insert_neighbor(candidate, k, out_neighbors);
    }
}
