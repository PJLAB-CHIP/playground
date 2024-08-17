#include "playground/system.hpp"
#include "playground/matmul.hpp"
#include <cblas.h>

namespace playground
{
template <>
void matmul<float32_t, 0>(const size_t M, const size_t N, const size_t K, const float32_t* const A,
                          const float32_t* const B, float32_t* const C)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}
}  // namespace playground