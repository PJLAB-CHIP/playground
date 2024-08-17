#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "playground/common.hpp"
#include "playground/hyperparams.hpp"
#include "playground/matmul.hpp"
#include "playground/ptx.hpp"
#include "playground/utils.hpp"

using namespace nvcuda;

namespace playground
{

#ifdef USE_WMMA
static constexpr size_t FRAGMENT_M = 16;
static constexpr size_t FRAGMENT_N = 16;
static constexpr size_t FRAGMENT_K = 16;
#else
static constexpr size_t FRAGMENT_M = 16;
static constexpr size_t FRAGMENT_N = 8;
static constexpr size_t FRAGMENT_K = 16;
#endif

static constexpr size_t BLOCK_ROWS = 256;
static constexpr size_t BLOCK_COLS = 128;

static constexpr size_t WARP_ROWS = 64;
static constexpr size_t WARP_COLS = 64;

static constexpr size_t BLOCK_ROW_WARPS = BLOCK_COLS / WARP_COLS;
static constexpr size_t BLOCK_COL_WARPS = BLOCK_ROWS / WARP_ROWS;

static constexpr size_t BLOCK_ROW_TILES = BLOCK_COLS / FRAGMENT_N;
static constexpr size_t BLOCK_COL_TILES = BLOCK_ROWS / FRAGMENT_M;

static constexpr size_t WARP_ROW_TILES = WARP_COLS / FRAGMENT_N;
static constexpr size_t WARP_COL_TILES = WARP_ROWS / FRAGMENT_M;

static constexpr size_t WARP_SIZE = 32;
static constexpr size_t WARPS_PER_BLOCK = BLOCK_ROW_WARPS * BLOCK_COL_WARPS;
static constexpr size_t THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;

static constexpr size_t CHUNK_K = 32 / FRAGMENT_K;

static constexpr size_t CHUNK_LINE_BYTES = CHUNK_K * FRAGMENT_K * sizeof(float16_t);
static constexpr size_t CHUNK_COPY_LINES_PER_WARP = WARP_SIZE * sizeof(int4) / CHUNK_LINE_BYTES;
static constexpr size_t CHUNK_COPY_LINE_LANES = WARP_SIZE / CHUNK_COPY_LINES_PER_WARP;

static constexpr size_t AB_SMEM_STRIDE = CHUNK_K * FRAGMENT_K;

static constexpr size_t C_SMEM_STRIDE = BLOCK_COLS;
static constexpr size_t C_SMEM_OFFSET = WARP_COLS;

static constexpr size_t BLOCK_STRIDE = 16;

__global__ void hgemmV3_kernel(const float16_t* const A, const float16_t* const B,
                               float16_t* const C, const size_t M, const size_t N, const size_t K)
{
#ifdef USE_WMMA
    using namespace nvcuda;

    const size_t M_tiles = ceilDivide(M, FRAGMENT_M);
    const size_t N_tiles = ceilDivide(N, FRAGMENT_N);
    const size_t K_tiles = ceilDivide(K, FRAGMENT_K);

    const size_t block_tile_i = (blockIdx.z % 2) ? ((gridDim.y - blockIdx.y - 1) * BLOCK_COL_TILES)
                                                 : (blockIdx.y * BLOCK_COL_TILES);
    const size_t block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x) * BLOCK_ROW_TILES;

    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) {
        return;
    }

    extern __shared__ float16_t smem[][AB_SMEM_STRIDE];

    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    constexpr size_t B_smem_idx_off = BLOCK_ROWS;

    float16_t* smem_warp_tile_ptr = &smem[0][0] +
                                    (warp_id / BLOCK_ROW_WARPS) * C_SMEM_STRIDE * WARP_ROWS +
                                    (warp_id % BLOCK_ROW_WARPS) * C_SMEM_OFFSET;

    float16_t* smem_warp_stream_ptr = &smem[0][0] + warp_id * FRAGMENT_M * 2 * C_SMEM_STRIDE;

    const size_t gmem_idx =
        (block_tile_i + warp_id * 2) * FRAGMENT_M * N + block_tile_j * FRAGMENT_N;
    float16_t* src_gmem_warp_stream_ptr = &C[gmem_idx];

    wmma::fragment<wmma::accumulator, FRAGMENT_M, FRAGMENT_N, FRAGMENT_K, float16_t>
        C_frag[WARP_COL_TILES][WARP_ROW_TILES];

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    const float16_t* A_warp_ptr =
        &A[block_tile_i * FRAGMENT_M * K] + BLOCK_ROWS / WARPS_PER_BLOCK * K * warp_id;
    const float16_t* B_warp_ptr =
        &B[block_tile_j * FRAGMENT_N * K] + BLOCK_COLS / WARPS_PER_BLOCK * K * warp_id;

    constexpr size_t A_smem_iters = BLOCK_ROWS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
    constexpr size_t B_smem_iters = BLOCK_COLS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);

#pragma unroll
    for (size_t tile_k = 0; tile_k < K_tiles; tile_k += CHUNK_K) {
        size_t A_smem_idx = BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
        const int4* A_lane_ptr = rCast<const int4*>(A_warp_ptr + tile_k * FRAGMENT_K +
                                                    (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                                 (lane_id % CHUNK_COPY_LINE_LANES);
        A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < A_smem_iters; ++i) {
            *(rCast<int4*>(&smem[A_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES)) = *A_lane_ptr;

            A_lane_ptr = rCast<const int4*>(rCast<const float16_t*>(A_lane_ptr) +
                                            CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        size_t B_smem_idx = B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
        const int4* B_lane_ptr = rCast<const int4*>(B_warp_ptr + tile_k * FRAGMENT_K +
                                                    (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                                 (lane_id % CHUNK_COPY_LINE_LANES);
        B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < B_smem_iters; ++i) {
            *(rCast<int4*>(&smem[B_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES)) = *B_lane_ptr;

            B_lane_ptr = rCast<const int4*>(rCast<const float16_t*>(B_lane_ptr) +
                                            CHUNK_COPY_LINES_PER_WARP * K);
            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        __syncthreads();

#pragma unroll
        for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
            wmma::fragment<wmma::matrix_a, FRAGMENT_M, FRAGMENT_N, FRAGMENT_K, float16_t,
                           wmma::row_major>
                A_frag[WARP_COL_TILES];
            wmma::fragment<wmma::matrix_b, FRAGMENT_M, FRAGMENT_N, FRAGMENT_K, float16_t,
                           wmma::col_major>
                B_frag[WARP_ROW_TILES];

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
                size_t A_smem_idx = (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * FRAGMENT_M;
                const float16_t* A_tile_ptr = &smem[A_smem_idx][k_step * FRAGMENT_K];

                wmma::load_matrix_sync(A_frag[i], A_tile_ptr, FRAGMENT_K * CHUNK_K);
            }

#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t B_smem_idx =
                    B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * FRAGMENT_N;
                const float16_t* B_tile_ptr = &smem[B_smem_idx][k_step * FRAGMENT_K];

                wmma::load_matrix_sync(B_frag[j], B_tile_ptr, FRAGMENT_K * CHUNK_K);
            }

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
                for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                    size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                    wmma::mma_sync(C_frag[i][j_s], A_frag[i], B_frag[j_s], C_frag[i][j_s]);
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            float16_t* C_tile_ptr =
                smem_warp_tile_ptr + i * C_SMEM_STRIDE * FRAGMENT_M + j * FRAGMENT_N;

            wmma::store_matrix_sync(C_tile_ptr, C_frag[i][j], C_SMEM_STRIDE, wmma::mem_row_major);
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = 0; i < FRAGMENT_M; ++i) {
        *(rCast<int4*>(src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N) + lane_id % 16) =
            *(rCast<int4*>(smem_warp_stream_ptr + (i * 2 + lane_id / 16) * C_SMEM_STRIDE) +
              lane_id % 16);
    }
#else  // ==========================================================================================
    const size_t M_tiles = ceilDivide(M, FRAGMENT_M);
    const size_t N_tiles = ceilDivide(N, FRAGMENT_N);
    const size_t K_tiles = ceilDivide(K, FRAGMENT_K);

    const size_t block_tile_i = (blockIdx.z % 2) ? ((gridDim.y - blockIdx.y - 1) * BLOCK_COL_TILES)
                                                 : (blockIdx.y * BLOCK_COL_TILES);
    const size_t block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x) * BLOCK_ROW_TILES;

    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) {
        return;
    }

    extern __shared__ float16_t smem[][AB_SMEM_STRIDE];

    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    constexpr size_t B_smem_idx_off = BLOCK_ROWS;

    float16_t* smem_warp_tile_row_ptr =
        &smem[0][0] + (warp_id / BLOCK_ROW_WARPS) * C_SMEM_STRIDE * WARP_ROWS;
    const float16_t* smem_warp_stream_ptr = &smem[0][0] + warp_id * FRAGMENT_M * 2 * C_SMEM_STRIDE;

    const size_t gmem_idx =
        (block_tile_i + warp_id * 2) * FRAGMENT_M * N + block_tile_j * FRAGMENT_N;
    float16_t* src_gmem_warp_stream_ptr = &C[gmem_idx];

    uint32_t RC[WARP_COL_TILES][WARP_ROW_TILES][2];

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }

    const float16_t* A_warp_ptr =
        &A[block_tile_i * FRAGMENT_M * K] + BLOCK_ROWS / WARPS_PER_BLOCK * K * warp_id;
    const float16_t* B_warp_ptr =
        &B[block_tile_j * FRAGMENT_N * K] + BLOCK_COLS / WARPS_PER_BLOCK * K * warp_id;

    constexpr size_t A_smem_iters = BLOCK_ROWS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
    constexpr size_t B_smem_iters = BLOCK_COLS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);

#pragma unroll
    for (size_t tile_k = 0; tile_k < K_tiles; tile_k += CHUNK_K) {
        size_t A_smem_idx = BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
        const int4* A_lane_ptr = rCast<const int4*>(A_warp_ptr + tile_k * FRAGMENT_K +
                                                    (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                                 (lane_id % CHUNK_COPY_LINE_LANES);
        A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < A_smem_iters; ++i) {
            *(rCast<int4*>(&smem[A_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES)) = *A_lane_ptr;

            A_lane_ptr = rCast<const int4*>(rCast<const float16_t*>(A_lane_ptr) +
                                            CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        size_t B_smem_idx = B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
        const int4* B_lane_ptr = rCast<const int4*>(B_warp_ptr + tile_k * FRAGMENT_K +
                                                    (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                                 (lane_id % CHUNK_COPY_LINE_LANES);
        B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < B_smem_iters; ++i) {
            *(rCast<int4*>(&smem[B_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES)) = *B_lane_ptr;

            B_lane_ptr = rCast<const int4*>(rCast<const float16_t*>(B_lane_ptr) +
                                            CHUNK_COPY_LINES_PER_WARP * K);
            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        __syncthreads();

#pragma unroll
        for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
            uint32_t RA[WARP_COL_TILES][4];
            uint32_t RB[WARP_ROW_TILES][2];

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
                size_t A_smem_idx = (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * FRAGMENT_M;
                uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
                    &smem[A_smem_idx + lane_id % 16][k_step * FRAGMENT_K + (lane_id / 16) * 8]);

                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], A_smem_lane_addr);
            }

#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t B_smem_idx =
                    B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * FRAGMENT_N;
                uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
                    &smem[B_smem_idx + lane_id % 8][k_step * FRAGMENT_K + ((lane_id / 8) % 2) * 8]);

                LDMATRIX_X2(RB[j][0], RB[j][1], B_smem_lane_addr);
            }

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
                for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                    size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                    HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[i][0], RA[i][1], RA[i][2], RA[i][3],
                              RB[j_s][0], RB[j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            float16_t* lane_ptr0 = smem_warp_tile_row_ptr +
                                   (i * FRAGMENT_M + lane_id / 4) * C_SMEM_STRIDE +
                                   (warp_id % BLOCK_ROW_WARPS) * C_SMEM_OFFSET + j * FRAGMENT_N +
                                   (lane_id % 4) * sizeof(uint32_t) / sizeof(float16_t);
            float16_t* lane_ptr1 = smem_warp_tile_row_ptr +
                                   (i * FRAGMENT_M + lane_id / 4 + 8) * C_SMEM_STRIDE +
                                   (warp_id % BLOCK_ROW_WARPS) * C_SMEM_OFFSET + j * FRAGMENT_N +
                                   (lane_id % 4) * sizeof(uint32_t) / sizeof(float16_t);

            rCast<uint32_t>(*lane_ptr0) = RC[i][j][0];
            rCast<uint32_t>(*lane_ptr1) = RC[i][j][1];
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = 0; i < FRAGMENT_M; ++i) {
        *(rCast<int4*>(src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N) + lane_id % 16) =
            *(rCast<const int4*>(smem_warp_stream_ptr + (i * 2 + lane_id / 16) * C_SMEM_STRIDE) +
              lane_id % 16);
    }
#endif
}

size_t initV3Kernel()
{
    int dev_id = 0;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));

    cudaDeviceProp dev_prop;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

    size_t smem_max_size = std::max((BLOCK_ROWS + BLOCK_COLS) * AB_SMEM_STRIDE * sizeof(float16_t),
                                    BLOCK_ROWS * C_SMEM_STRIDE * sizeof(float16_t));
    HLOG("smem_max_size: %.0f KBytes (%zu Bytes)", static_cast<double>(smem_max_size) / 1024,
         smem_max_size);

    HGEMM_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, smem_max_size);
    HGEMM_CHECK_CUDART_ERROR(cudaFuncSetAttribute(
        hgemmV3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    return smem_max_size;
}

template <>
void matmul<float16_t, 3>(const size_t m, const size_t n, const size_t k, const float16_t* const A,
                          const float16_t* const B, float16_t* const C)
{
    static size_t smem_max_size = initV3Kernel();

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCK_STRIDE, ceilDivide(m, BLOCK_ROWS), ceilDivide(n, BLOCK_COLS * BLOCK_STRIDE));

    hgemmV3_kernel<<<grid, block, smem_max_size>>>(A, B, C, m, n, k);
}
}  // namespace playground
