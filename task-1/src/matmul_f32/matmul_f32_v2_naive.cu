#include <vector_types.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"
#include "playground/utils.hpp"

namespace playground
{

__global__ void sgemmV2_kernel(const float32_t* const A, const float32_t* const B,
                               float32_t* const C, const size_t M, const size_t N, const size_t K)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        float psum = 0.0;
#pragma unroll
        for (int k = 0; k < K; k++) {
            psum += A[idx2To1(m, k, K)] * B[idx2To1(k, n, N)];
        }
        C[idx2To1(m, n, N)] = psum;
    }
}

template <>
void matmul<float32_t, 2>(const size_t M, const size_t N, const size_t K, const float32_t* const A,
                          const float32_t* const B, float32_t* const C)
{
    const dim3 BlockSize(32, 32);
    const dim3 GridSize(128, 128);
    sgemmV2_kernel<<<GridSize, BlockSize>>>(A, B, C, M, N, K);
}
}  // namespace playground