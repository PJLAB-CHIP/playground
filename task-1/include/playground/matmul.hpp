#pragma once

#include "playground/system.hpp"

namespace playground
{

template <typename DType, uint8_t Version>
void matmul(const size_t m, const size_t n, const size_t k, const DType* const A,
            const DType* const B, DType* const C) = delete;

// =================================================================================================
// Declaration of library matmul functions.
// -------------------------------------------------------------------------------------------------
/**
 * @brief Matrix multiplication, fp16, cBLAS.
 */
template <>
void matmul<float16_t, 0>(const size_t m, const size_t n, const size_t k, const float16_t* const A,
                        const float16_t* const B, float16_t* const C);
/**
 * @brief Matrix multiplication, fp32, cBLAS.
 */
template <>
void matmul<float32_t, 0>(const size_t m, const size_t n, const size_t k, const float32_t* const A,
                        const float32_t* const B, float32_t* const C);
/**
 * @brief Matrix multiplication, fp16, cuBLAS.
 */
template <>
void matmul<float16_t, 1>(const size_t m, const size_t n, const size_t k, const float16_t* const A,
                        const float16_t* const B, float16_t* const C);
/**
 * @brief Matrix multiplication, fp32, cuBLAS.
 */
template <>
void matmul<float32_t, 1>(const size_t m, const size_t n, const size_t k, const float32_t* const A,
                        const float32_t* const B, float32_t* const C);

// =================================================================================================
// Declaration of self-implemented matmul functions.
// -------------------------------------------------------------------------------------------------

// ...

}  // namespace playground