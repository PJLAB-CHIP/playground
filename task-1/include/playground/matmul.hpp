#pragma once

#include <cmath>
#include <cstdio>
#include <ctime>

#include "playground/system.hpp"

namespace playground
{
template <typename DType, uint8_t Version>
void matmul(size_t m, size_t n, size_t k, const DType* A, const DType* B,
            DType* C) = delete;

// Playground Matmul Declaration
#define PLAYGROUND_MATMUL_DEC(DType, Version, m, n, k, A, B, C)               \
    template <>                                                               \
    void matmul<DType, Version>(size_t m, size_t n, size_t k, const DType* A, \
                                const DType* B, DType* C)

#define PLAYGOUND_MATMUL_CALL(Version, m, n, k, A, B, C)                      \
    ::playground::matmul<::std::remove_cvref_t<decltype(*A)>, Version>(       \
        m, n, k, A, B, C)

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
PLAYGROUND_MATMUL_DEC(float16_t, PG_MATMUL_FP16_CBLAS, m, n, k, A, B, C);

/**
 * @brief Matrix multiplication, fp32-v0, cBLAS.
 */
PLAYGROUND_MATMUL_DEC(float32_t, PG_MATMUL_FP32_CBLAS, m, n, k, A, B, C);

/**
 * @brief Matrix multiplication, fp16-v1, cuBLAS.
 */
PLAYGROUND_MATMUL_DEC(float16_t, PG_MATMUL_FP16_CUBLAS, m, n, k, A, B, C);

/**
 * @brief Matrix multiplication, fp32-v1, cuBLAS.
 */
PLAYGROUND_MATMUL_DEC(float32_t, PG_MATMUL_FP32_CUBLAS, m, n, k, A, B, C);

}  // namespace playground
