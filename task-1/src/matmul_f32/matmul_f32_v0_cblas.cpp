#include <cblas.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
PLAYGROUND_MATMUL_DEC(float32_t, 0, m, n, k, A, B, C)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0F, A, k,
                B, n, 0.0F, C, n);
}
}  // namespace playground