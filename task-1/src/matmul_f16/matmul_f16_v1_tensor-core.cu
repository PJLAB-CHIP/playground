#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <library_types.h>

#include "playground/cublas_handle.hpp"
#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
PLAYGROUND_MATMUL_DEC(float16_t, 1, M, N, K, A, B, C)
{
    const float16_t Alpha = 1.0F;
    const float16_t Beta = 0.0F;
    cublasGemmEx(s_getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                 &Alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, K, &Beta, C,
                 CUDA_R_16F, N, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
}  // namespace playground
