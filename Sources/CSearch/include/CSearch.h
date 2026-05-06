#ifndef CSEARCH_H
#define CSEARCH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t record_index;
    int64_t distance_squared;
} rinha_neighbor_t;

void rinha_topk_exact_i16(
    const int16_t *query,
    const int16_t *vectors,
    size_t count,
    size_t dim,
    size_t stride,
    size_t k,
    rinha_neighbor_t *out_neighbors
);

void rinha_topk_exact_i16_indexed(
    const int16_t *query,
    const int16_t *vectors,
    const uint32_t *record_indices,
    size_t candidate_count,
    size_t dim,
    size_t stride,
    size_t k,
    rinha_neighbor_t *out_neighbors
);

#ifdef __cplusplus
}
#endif

#endif
