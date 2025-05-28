#include <algorithm>
#include <cblas.h>
#include <iterator>
#include <vector>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
PLAYGROUND_MATMUL_DEC(float16_t, 0, m, n, k, A, B, C)
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
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0F,
                Af32.data(), k, Bf32.data(), n, 0.0F, Cf32.data(), n);
    // Convert float32_t to float16_t, storing in C
    std::ranges::transform(Cf32, C, [](float32_t c) { return float16_t(c); });
}

}  // namespace playground
