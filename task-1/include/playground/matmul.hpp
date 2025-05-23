#pragma once

#include <cmath>
#include <cstdio>
#include <ctime>

#include "playground/system.hpp"

namespace playground
{
template <typename DType, uint8_t Version>
void matmul(size_t M, size_t N, size_t K, const DType* A, const DType* B,
            DType* C) = delete;

// Playground Matmul Declaration
#define PLAYGROUND_MATMUL_DEC(DType, Version, M, N, K, A, B, C)               \
    template <>                                                               \
    void matmul<DType, Version>(size_t M, size_t N, size_t K, const DType* A, \
                                const DType* B, DType* C)

#define PLAYGOUND_MATMUL_CALL(Version, M, N, K, A, B, C)                      \
    ::playground::matmul<::std::remove_cvref_t<decltype(*A)>, Version>(       \
        M, N, K, A, B, C)

// ============================================================================
// Declaration of library matmul functions.
// ----------------------------------------------------------------------------
constexpr auto PG_MATMUL_FP16_CBLAS = 0;
constexpr auto PG_MATMUL_FP16_CUBLAS = 1;
constexpr auto PG_MATMUL_FP32_CBLAS = 0;
constexpr auto PG_MATMUL_FP32_CUBLAS = 1;

/**
 * @brief Matrix multiplication, fp16-v0, cBLAS.
 */
PLAYGROUND_MATMUL_DEC(float16_t, PG_MATMUL_FP16_CBLAS, M, N, K, A, B, C);

/**
 * @brief Matrix multiplication, fp32-v0, cBLAS.
 */
PLAYGROUND_MATMUL_DEC(float32_t, PG_MATMUL_FP32_CBLAS, M, N, K, A, B, C);

/**
 * @brief Matrix multiplication, fp16-v1, cuBLAS.
 */
PLAYGROUND_MATMUL_DEC(float16_t, PG_MATMUL_FP16_CUBLAS, M, N, K, A, B, C);

/**
 * @brief Matrix multiplication, fp32-v1, cuBLAS.
 */
PLAYGROUND_MATMUL_DEC(float32_t, PG_MATMUL_FP32_CUBLAS, M, N, K, A, B, C);

}  // namespace playground
