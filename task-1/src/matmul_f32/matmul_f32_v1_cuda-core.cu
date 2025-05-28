#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "playground/cublas_handle.hpp"
#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
PLAYGROUND_MATMUL_DEC(float32_t, 1, m, n, k, A, B, C)
{
    const float32_t Alpha = 1.0F;
    const float32_t Beta = 0.0F;
    cublasSgemm(s_getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &Alpha,
                B, n, A, k, &Beta, C, n);
}
}  // namespace playground