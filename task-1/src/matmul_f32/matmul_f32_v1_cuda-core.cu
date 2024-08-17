#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"
#include "playground/static.hpp"
#include "playground/system.hpp"
#include "playground/utils.hpp"

namespace playground
{
template <>
void matmul<float32_t, 1>(const size_t M, const size_t N, const size_t K, const float32_t* const A,
                          const float32_t* const B, float32_t* const C)
{
    const float32_t Alpha = 1.0f;
    const float32_t Beta = 0.0f;
    cublasSgemm(s_getCublasHandle<float32_t>(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &Alpha, B, N, A, K,
                &Beta, C, N);
}
}  // namespace playground