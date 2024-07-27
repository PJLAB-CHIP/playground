#include <algorithm>
#include <cblas.h>
#include <vector>

#include "playground/matmul.hpp"

namespace playground
{
template <>
void matmul<float16_t, 0>(const size_t m, const size_t n, const size_t k, const float16_t* const A,
                          const float16_t* const B, float16_t* const C)
{
    std::vector<float32_t> Af32, Bf32, Cf32;
    // Convert float16_t to float32_t, storing in Af32, Bf32, Cf32
    std::transform(A, A + m * k, std::back_inserter(Af32),
                   [](float16_t a) { return float32_t(a); });
    std::transform(B, B + n * k, std::back_inserter(Bf32),
                   [](float16_t b) { return float32_t(b); });
    std::transform(C, C + m * n, std::back_inserter(Cf32),
                   [](float16_t c) { return float32_t(c); });
    // Cf32 = Cf32 + Af32 * Bf32
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, Af32.data(), k,
                Bf32.data(), n, 0.0f, Cf32.data(), n);
    // Convert float32_t to float16_t, storing in C
    std::transform(Cf32.begin(), Cf32.end(), C, [](float32_t c) { return float16_t(c); });
}

}  // namespace playground
