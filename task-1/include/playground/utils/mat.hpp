#include <cmath>
#include <cstdio>
#include <ctime>
#include <stdexcept>

#include "playground/system.hpp"

namespace playground
{

template <typename T1, typename T2>
constexpr auto compareMat(size_t m, size_t n, T1 A, T2 B) -> float32_t
{
    float32_t gap = 0.0;
    float32_t err_sum = 0.0;
    float32_t avg_err = 0.0;

    for (size_t i = 0; i < m * n; ++i) {
        gap = std::abs(float32_t(A[i]) - float32_t(B[i]));
        err_sum += gap / float32_t(A[i]);

        if (std::isinf(err_sum)) {
            ::printf("%zu/%zu err_sum: %f, gap: %f, divider: %f\n", i, m * n,
                     err_sum, gap, float32_t(A[i]));
            throw std::runtime_error("Error sum is inf");
        }
    }

    avg_err = err_sum / float32_t(m * n);

    return avg_err;
}

template <typename T>
void initRandMat(std::size_t m, std::size_t n, T* mat)
{
    ::srand(::time(nullptr));
    for (std::size_t cnt = 0; cnt < m * n; cnt++) {
        mat[cnt] = T{float32_t(::rand()) / float32_t(RAND_MAX) +
                     std::numeric_limits<float32_t>::min()};
    }
}

}  // namespace playground
