#pragma once

#include "playground/system.hpp"

namespace playground
{

template <typename DType, uint8_t Version>
void matmul(const size_t M, const size_t N, const size_t K,
            const DType* const A, const DType* const B,
            DType* const C) = delete;

// Playground Matmul Signature.
#define PLAYGROUND_MATMUL_SIG(DType, Version, M, N, K, A, B, C)                \
    template <>                                                                \
    void matmul<DType, Version>(const size_t M, const size_t N,                \
                                const size_t K, const DType* const A,          \
                                const DType* const B, DType* const C)

// =============================================================================
// Declaration of library matmul functions.
// -----------------------------------------------------------------------------
/**
 * @brief Matrix multiplication, fp16-v0, cBLAS.
 */
PLAYGROUND_MATMUL_SIG(float16_t, 0, M, N, K, A, B, C);

/**
 * @brief Matrix multiplication, fp32-v0, cBLAS.
 */
PLAYGROUND_MATMUL_SIG(float32_t, 0, M, N, K, A, B, C);

/**
 * @brief Matrix multiplication, fp16-v1, cuBLAS.
 */
PLAYGROUND_MATMUL_SIG(float16_t, 1, M, N, K, A, B, C);

/**
 * @brief Matrix multiplication, fp32-v1, cuBLAS.
 */
PLAYGROUND_MATMUL_SIG(float32_t, 1, M, N, K, A, B, C);

}  // namespace playground