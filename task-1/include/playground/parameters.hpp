#include <cuda_fp16.h>
#include <string_view>

#include "playground/system.hpp"

namespace playground::params
{

// =============================================================================
// @Note `DataType` and `MatmulVersion` are managed by CMake automatically.
// -----------------------------------------------------------------------------
#ifdef TEST_FLOAT16
using DataType = float16_t;
constexpr std::string_view DataTypeName = "f16";
#else
using DataType = playground::float32_t;
constexpr std::string_view DataTypeName = "f32";
#endif
#ifndef MATMUL_VERSION
    #define MATMUL_VERSION 0  // cBLAS
#endif
constexpr auto MatmulVersion = uint8_t(MATMUL_VERSION);
// -----------------------------------------------------------------------------

// Size of Matrices
// mat(M,K) @ mat(K,N) = mat(M,N)
constexpr uint32_t M = 4096, N = 4096, K = 4096;

// Repeated Times
constexpr size_t NumRep = 100;

// Warmup Times
constexpr size_t NumWarmup = 10;

}  // namespace playground::params
