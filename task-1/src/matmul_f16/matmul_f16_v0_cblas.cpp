#include <algorithm>
#include <cblas.h>
#include <iterator>
#include <vector>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
PLAYGROUND_MATMUL_SIG(float16_t, 0, M, N, K, A, B, C)
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
