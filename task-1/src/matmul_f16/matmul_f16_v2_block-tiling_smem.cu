#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "playground/common.hpp"
#include "playground/hyperparams.hpp"
#include "playground/matmul.hpp"
#include "playground/ptx.hpp"
#include "playground/system.hpp"
#include "playground/utils.hpp"

namespace playground
{
#ifdef USE_WMMA
static constexpr size_t FragmentM = 16;
static constexpr size_t FragmentN = 16;
static constexpr size_t FragmentK = 16;
#else
static constexpr size_t FragmentM = 16;
static constexpr size_t FragmentN = 8;
static constexpr size_t FragmentK = 16;
#endif

static constexpr size_t WarpSize = device::WRAP_SIZE;

__global__ void hgemmV2_kernel(const float16_t* const A, const float16_t* const B,
                               float16_t* const C, const size_t M, const size_t N, const size_t K)
{
#ifdef USE_WMMA
    using namespace nvcuda;

    const size_t K_tiles = ceilDivide(K, FRAGMENT_K);

    const size_t warp_row = blockIdx.y * FRAGMENT_M;
    const size_t warp_col = blockIdx.x * FRAGMENT_N;

    if (warp_row >= M && warp_col >= N) {
        return;
    }

    wmma::fragment<wmma::accumulator, FRAGMENT_M, FRAGMENT_N, FRAGMENT_K, float16_t> C_frag;

    wmma::fill_fragment(C_frag, 0.0);

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        wmma::fragment<wmma::matrix_a, FRAGMENT_M, FRAGMENT_N, FRAGMENT_K, float16_t,
                       wmma::row_major>
            A_frag;
        wmma::fragment<wmma::matrix_b, FRAGMENT_M, FRAGMENT_N, FRAGMENT_K, float16_t,
                       wmma::col_major>
            B_frag;

        wmma::load_matrix_sync(A_frag, A + warp_row * K + i * FRAGMENT_K, K);
        wmma::load_matrix_sync(B_frag, B + i * FRAGMENT_K + warp_col * K, K);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    wmma::store_matrix_sync(C + warp_row * N + warp_col, C_frag, N, wmma::mem_row_major);

#else
    const size_t K_tiles = ceilDivide(K, FragmentK);

    const size_t warp_row = blockIdx.y * FragmentM;
    const size_t warp_col = blockIdx.x * FragmentN;

    if (warp_row >= M || warp_col >= N) {
        return;
    }

    __shared__ float16_t A_smem[FragmentM][FragmentK];
    __shared__ float16_t B_smem[FragmentN][FragmentK];
    __shared__ float16_t C_smem[FragmentM][FragmentN];

    static_assert((FRAGMENT_M * FRAGMENT_K + FRAGMENT_N * FRAGMENT_K + FRAGMENT_M * FRAGMENT_N) *
                          sizeof(float16_t) <=
                      device::SM_PER_BLOCK,
                  "Not enough shared memory");

    const size_t lane_id = threadIdx.x % WarpSize;

    uint32_t RC[2] = {0, 0};

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        // Load A tiles to shared memory
        *(rCast<int4*>(&A_smem[lane_id / 2][0]) + lane_id % 2) =
            *(rCast<const int4*>(&A[(warp_row + lane_id / 2) * K + i * FragmentK]) + lane_id % 2);
        // Load B tiles to shared memory
        if (lane_id < FragmentN * 2) {
            *(rCast<int4*>(&B_smem[lane_id / 2][0]) + lane_id % 2) =
                *(rCast<const int4*>(&B[i * FragmentK + (warp_col + lane_id / 2) * K]) +
                  lane_id % 2);
        }
        __syncthreads();

        uint32_t RA[4];  // 4*32b -> 8 float16_t
        uint32_t RB[2];  // 2*32b -> 4 float16_t

        uint32_t A_smem_lane_addr =
            __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

        uint32_t B_smem_lane_addr =
            __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
        LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);

        __syncthreads();
    }

    *(rCast<uint32_t*>(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0];
    *(rCast<uint32_t*>(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];

    __syncthreads();

    if (lane_id < FragmentM) {
        rCast<int4>(C[(warp_row + lane_id) * N + warp_col]) = rCast<int4>(C_smem[lane_id][0]);
    }
#endif
}

template <>
void matmul<float16_t, 2>(const size_t m, const size_t n, const size_t k, const float16_t* const A,
                          const float16_t* const B, float16_t* const C)
{
    dim3 blockSize(WarpSize);
    dim3 gridSize(ceilDivide(n, FragmentN), ceilDivide(m, FragmentM));
    hgemmV2_kernel<<<gridSize, blockSize>>>(A, B, C, m, n, k);
}
}  // namespace playground