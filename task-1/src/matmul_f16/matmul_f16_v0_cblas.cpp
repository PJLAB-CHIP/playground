#include <algorithm>
#include <cblas.h>
#include <iterator>
#include <vector>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
template <>
void matmul<float16_t, 0>(const size_t M, const size_t N, const size_t K,
                          const float16_t* const A, const float16_t* const B,
                          float16_t* const C)
{
    std::vector<float32_t> Af32, Bf32, Cf32;
    // Convert float16_t to float32_t, storing in Af32, Bf32, Cf32
    std::transform(A, A + M * K, std::back_inserter(Af32),
                   [](float16_t a) { return float32_t(a); });
    std::transform(B, B + N * K, std::back_inserter(Bf32),
                   [](float16_t b) { return float32_t(b); });
    std::transform(C, C + M * N, std::back_inserter(Cf32),
                   [](float16_t c) { return float32_t(c); });
    // Cf32 = Cf32 + Af32 * Bf32
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
                Af32.data(), K, Bf32.data(), N, 0.0f, Cf32.data(), N);
    // Convert float32_t to float16_t, storing in C
    std::transform(Cf32.begin(), Cf32.end(), C,
                   [](float32_t c) { return float16_t(c); });
}

}  // namespace playground
