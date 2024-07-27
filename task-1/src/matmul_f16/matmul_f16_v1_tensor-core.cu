#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"
#include "playground/static.hpp"
#include "playground/utils.hpp"

namespace playground
{
template <>
void matmul<float16_t, 1>(const size_t m, const size_t n, const size_t k, const float16_t* const A,
                          const float16_t* const B, float16_t* const C)
{
    const float16_t alpha = 1.0f;
    const float16_t beta = 0.0f;
    cublasGemmEx(s_getCublasHandle<float16_t>(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B,
                 CUDA_R_16F, n, A, CUDA_R_16F, k, &beta, C, CUDA_R_16F, n, CUDA_R_16F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
}  // namespace playground
