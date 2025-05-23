#include <cblas.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
PLAYGROUND_MATMUL_DEC(float32_t, 0, M, N, K, A, B, C)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0F, A, K,
                B, N, 0.0F, C, N);
}
}  // namespace playground