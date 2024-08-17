#include <vector_types.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"
#include "playground/utils.hpp"

namespace playground
{

__global__ void sgemmV4_kernel(const float32_t* const A, const float32_t* const B,
                               float32_t* const C, const size_t M, const size_t N, const size_t K)
{
    const size_t BM = 128;
    const size_t BN = 128;
    const size_t BK = 8;
    const size_t TM = 8;
    const size_t TN = 8;

    const size_t Bx = blockIdx.x;
    const size_t By = blockIdx.y;
    const size_t Tx = threadIdx.x;
    const size_t Ty = threadIdx.y;
    const size_t Tid = Ty * blockDim.x + Tx;

    __shared__ float32_t s_a[BK][BM];
    __shared__ float32_t s_b[BK][BN];

    float32_t r_load_a[4];
    float32_t r_load_b[4];
    float32_t r_comp_a[TM];
    float32_t r_comp_b[TN];
    float32_t r_c[TM][TN] = {0.0};

    size_t load_a_smem_m = Tid >> 1;
    size_t load_a_smem_k = (Tid & 1) << 2;
    size_t load_b_smem_k = Tid >> 5;
    size_t load_b_smem_n = (Tid & 31) << 2;

    size_t load_a_gmem_m = By * BM + load_a_smem_m;
    size_t load_b_gmem_n = Bx * BN + load_b_smem_n;

    for (size_t bk = 0; bk < (K + BK - 1) / BK; bk++) {

        size_t load_a_gmem_k = bk * BK + load_a_smem_k;
        size_t load_a_gmem_addr = idx2To1(load_a_gmem_m, load_a_gmem_k, K);
        size_t load_b_gmem_k = bk * BK + load_b_smem_k;
        size_t load_b_gmem_addr = idx2To1(load_b_gmem_k, load_b_gmem_n, N);
        rCast<float4>(r_load_a[0]) = rCast<const float4>(A[load_a_gmem_addr]);
        rCast<float4>(r_load_b[0]) = rCast<const float4>(B[load_b_gmem_addr]);

        s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        rCast<float4>(s_b[load_b_smem_k][load_b_smem_n]) = rCast<float4>(r_load_b[0]);

        __syncthreads();

#pragma unroll
        for (size_t tk = 0; tk < BK; tk++) {
            rCast<float4>(r_comp_a[0]) = rCast<float4>(s_a[tk][Ty * TM / 2]);
            rCast<float4>(r_comp_a[4]) = rCast<float4>(s_a[tk][Ty * TM / 2 + BM / 2]);
            rCast<float4>(r_comp_b[0]) = rCast<float4>(s_b[tk][Tx * TN / 2]);
            rCast<float4>(r_comp_b[4]) = rCast<float4>(s_b[tk][Tx * TN / 2 + BN / 2]);

#pragma unroll
            for (size_t tm = 0; tm < TM; tm++) {
#pragma unroll
                for (size_t tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (size_t i = 0; i < TM / 2; i++) {
        size_t store_c_gmem_m = By * BM + Ty * TM / 2 + i;
        size_t store_c_gmem_n = Bx * BN + Tx * TN / 2;
        size_t store_c_gmem_addr = idx2To1(store_c_gmem_m, store_c_gmem_n, N);
        rCast<float4>(C[store_c_gmem_addr]) = rCast<float4>(r_c[i][0]);
        rCast<float4>(C[store_c_gmem_addr + BN / 2]) = rCast<float4>(r_c[i][4]);
    }
#pragma unroll
    for (size_t i = 0; i < TM / 2; i++) {
        size_t store_c_gmem_m = By * BM + BM / 2 + Ty * TM / 2 + i;
        size_t store_c_gmem_n = Bx * BN + Tx * TN / 2;
        size_t store_c_gmem_addr = idx2To1(store_c_gmem_m, store_c_gmem_n, N);
        rCast<float4>(C[store_c_gmem_addr]) = rCast<float4>(r_c[i + TM / 2][0]);
        rCast<float4>(C[store_c_gmem_addr + BN / 2]) = rCast<float4>(r_c[i + TM / 2][4]);
    }
}

template <>
void matmul<float32_t, 4>(const size_t M, const size_t N, const size_t K, const float32_t* const A,
                          const float32_t* const B, float32_t* const C)
{
    const dim3 BlockSize(16, 16);
    const dim3 GridSize(32, 32);
    sgemmV4_kernel<<<GridSize, BlockSize>>>(A, B, C, M, N, K);
}
}  // namespace playground