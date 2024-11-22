#include "playground/system.hpp"

namespace playground
{

namespace device
{
constexpr size_t SM_PER_BLOCK = 49152;      // Shared memory per block (bytes)
constexpr size_t MAX_THR_PER_BLOCK = 1024;  // Max number of threads per block
constexpr size_t WRAP_SIZE = 32;            // Number of threads in a warp
}  // namespace device

namespace hgemm
{
constexpr size_t WMMA_M = 16;
constexpr size_t WMMA_N = 16;
constexpr size_t WMMA_K = 16;

constexpr size_t MMA_M = 16;
constexpr size_t MMA_N = 8;
constexpr size_t MMA_K = 16;

constexpr size_t BLOCK_ROWS = 256;
constexpr size_t BLOCK_COLS = 128;

constexpr size_t WARP_ROWS = 64;
constexpr size_t WARP_COLS = 64;

constexpr size_t BLOCK_ROW_WARPS = BLOCK_COLS / WARP_COLS;
constexpr size_t BLOCK_COL_WARPS = BLOCK_ROWS / WARP_ROWS;

constexpr size_t BLOCK_ROW_TILES = BLOCK_COLS / WMMA_N;
constexpr size_t BLOCK_COL_TILES = BLOCK_ROWS / WMMA_M;

constexpr size_t WARP_ROW_TILES = WARP_COLS / WMMA_N;
constexpr size_t WARP_COL_TILES = WARP_ROWS / WMMA_M;

constexpr size_t WARP_SIZE = 32;
constexpr size_t WARPS_PER_BLOCK = BLOCK_ROW_WARPS * BLOCK_COL_WARPS;
constexpr size_t THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;

constexpr size_t CHUNK_K = 32 / WMMA_K;

constexpr size_t THREAD_COPY_BYTES = 16;

constexpr size_t CHUNK_LINE_BYTES = CHUNK_K * WMMA_K * sizeof(float16_t);
constexpr size_t CHUNK_COPY_LINES_PER_WARP = WARP_SIZE * THREAD_COPY_BYTES / CHUNK_LINE_BYTES;
constexpr size_t CHUNK_COPY_LINE_LANES = WARP_SIZE / CHUNK_COPY_LINES_PER_WARP;

constexpr size_t SMEM_PADDING = 8;
constexpr size_t AB_SMEM_STRIDE = CHUNK_K * WMMA_K + SMEM_PADDING;
constexpr size_t C_SMEM_STRIDE = BLOCK_COLS + SMEM_PADDING;
constexpr size_t C_SMEM_OFFSET = WARP_COLS;

constexpr size_t BLOCK_STRIDE = 16;

constexpr size_t SMEM_BANK_ROWS = 32 * 4 / (AB_SMEM_STRIDE * sizeof(float16_t));

constexpr size_t PERMUTED_OFFSET = 8;
constexpr size_t PERMUTED_COLS = 4;

constexpr size_t K_STAGE = 4;
}  // namespace hgemm
}  // namespace playground