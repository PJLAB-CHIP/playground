#pragma once

#include "playground/system.hpp"

namespace playground
{

template <typename DType, uint8_t Version>
void matmul(const size_t M, const size_t N, const size_t K,
            const DType* const A, const DType* const B,
            DType* const C) = delete;

// =============================================================================
// Declaration of library matmul functions.
// -----------------------------------------------------------------------------
/**
 * @brief Matrix multiplication, fp16, cBLAS.
 */
template <>
void matmul<float16_t, 0>(const size_t M, const size_t N, const size_t K,
                          const float16_t* const A, const float16_t* const B,
                          float16_t* const C);
/**
 * @brief Matrix multiplication, fp32, cBLAS.
 */
template <>
void matmul<float32_t, 0>(const size_t M, const size_t N, const size_t K,
                          const float32_t* const A, const float32_t* const B,
                          float32_t* const C);
/**
 * @brief Matrix multiplication, fp16, cuBLAS.
 */
template <>
void matmul<float16_t, 1>(const size_t M, const size_t N, const size_t K,
                          const float16_t* const A, const float16_t* const B,
                          float16_t* const C);
/**
 * @brief Matrix multiplication, fp32, cuBLAS.
 */
template <>
void matmul<float32_t, 1>(const size_t M, const size_t N, const size_t K,
                          const float32_t* const A, const float32_t* const B,
                          float32_t* const C);

// =============================================================================
// Declaration of self-implemented matmul functions.
// -----------------------------------------------------------------------------

// fp32 ------------------------------------------------------------------------

template <>
void matmul<float32_t, 2>(const size_t M, const size_t N, const size_t K,
                          const float32_t* const A, const float32_t* const B,
                          float32_t* const C);
template <>
void matmul<float32_t, 3>(const size_t M, const size_t N, const size_t K,
                          const float32_t* const A, const float32_t* const B,
                          float32_t* const C);
template <>
void matmul<float32_t, 4>(const size_t M, const size_t N, const size_t K,
                          const float32_t* const A, const float32_t* const B,
                          float32_t* const C);
template <>
void matmul<float32_t, 5>(const size_t M, const size_t N, const size_t K,
                          const float32_t* const A, const float32_t* const B,
                          float32_t* const C);

// fp16 ------------------------------------------------------------------------

template <>
void matmul<float16_t, 2>(const size_t M, const size_t N, const size_t K,
                          const float16_t* const A, const float16_t* const B,
                          float16_t* const C);
template <>
void matmul<float16_t, 3>(const size_t M, const size_t N, const size_t K,
                          const float16_t* const A, const float16_t* const B,
                          float16_t* const C);
template <>
void matmul<float16_t, 4>(const size_t M, const size_t N, const size_t K,
                          const float16_t* const A, const float16_t* const B,
                          float16_t* const C);
template <>
void matmul<float16_t, 5>(const size_t M, const size_t N, const size_t K,
                          const float16_t* const A, const float16_t* const B,
                          float16_t* const C);
template <>
void matmul<float16_t, 6>(const size_t M, const size_t N, const size_t K,
                          const float16_t* const A, const float16_t* const B,
                          float16_t* const C);
template <>
void matmul<float16_t, 7>(const size_t M, const size_t N, const size_t K,
                          const float16_t* const A, const float16_t* const B,
                          float16_t* const C);

}  // namespace playground