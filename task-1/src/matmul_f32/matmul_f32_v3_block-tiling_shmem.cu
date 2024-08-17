#include <vector_types.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"
#include "playground/utils.hpp"

namespace playground
{

static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr int BK = 8;
static constexpr int TM = 8;
static constexpr int TN = 8;

__global__ void sgemmV3_kernel(const float32_t* const A, const float32_t* const B,
                               float32_t* const C, const size_t M, const size_t N, const size_t K)
{

    const size_t Bx = blockIdx.x;
    const size_t By = blockIdx.y;
    const size_t Tx = threadIdx.x;
    const size_t Ty = threadIdx.y;
    const size_t Tid = Ty * blockDim.x + Tx;

    __shared__ float32_t s_a[BM][BK];
    __shared__ float32_t s_b[BK][BN];

    float r_c[TM][TN] = {0.0};

    size_t load_a_smem_m = Tid >> 1;
    size_t load_a_smem_k = (Tid & 1) << 2;
    size_t load_b_smem_k = Tid >> 5;
    size_t load_b_smem_n = (Tid & 31) << 2;

    size_t load_a_gmem_m = By * BM + load_a_smem_m;
    size_t load_b_gmem_n = Bx * BN + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        size_t load_a_gmem_k = bk * BK + load_a_smem_k;
        size_t load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        rCast<float4>(s_a[load_a_smem_m][load_a_smem_k]) = rCast<const float4>(A[load_a_gmem_addr]);
        size_t load_b_gmem_k = bk * BK + load_b_smem_k;
        size_t load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        rCast<float4>(s_b[load_b_smem_k][load_b_smem_n]) = rCast<const float4>(B[load_b_gmem_addr]);

        __syncthreads();

#pragma unroll
        for (size_t k = 0; k < BK; k++) {
#pragma unroll
            for (size_t m = 0; m < TM; m++) {
#pragma unroll
                for (size_t n = 0; n < TN; n++) {
                    size_t comp_a_smem_m = Ty * TM + m;
                    size_t comp_b_smem_n = Tx * TN + n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (size_t i = 0; i < TM; i++) {
        size_t store_c_gmem_m = By * BM + Ty * TM + i;
#pragma unroll
        for (size_t j = 0; j < TN; j += 4) {
            size_t store_c_gmem_n = Bx * BN + Tx * TN + j;
            size_t store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
            rCast<float4>(C[store_c_gmem_addr]) = rCast<float4>(r_c[i][j]);
        }
    }
}

template <>
void matmul<float32_t, 3>(const size_t M, const size_t N, const size_t K, const float32_t* const A,
                          const float32_t* const B, float32_t* const C)
{
    const dim3 BlockSize(16, 16);
    const dim3 GridSize(32, 32);
    sgemmV3_kernel<<<GridSize, BlockSize>>>(A, B, C, M, N, K);
}
}  // namespace playground