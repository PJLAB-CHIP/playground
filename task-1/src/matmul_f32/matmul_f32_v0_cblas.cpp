#include "playground/matmul.hpp"
#include <cblas.h>

namespace playground
{
template <>
void matmul<float32_t, 0>(const size_t m, const size_t n, const size_t k, const float32_t* const A,
                          const float32_t* const B, float32_t* const C)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, A, k, B, n, 0.0f, C, n);
}
}  // namespace playground