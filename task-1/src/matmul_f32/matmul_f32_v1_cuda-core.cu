#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"
#include "playground/static.hpp"
#include "playground/utils.hpp"

namespace playground
{
template <>
void matmul<float32_t, 1>(const size_t m, const size_t n, const size_t k, const float32_t* const A,
                          const float32_t* const B, float32_t* const C)
{
    const float32_t alpha = 1.0f;
    const float32_t beta = 0.0f;
    cublasSgemm(s_getCublasHandle<float32_t>(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k,
                &beta, C, n);
}
}  // namespace playground