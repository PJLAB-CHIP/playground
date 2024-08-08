#include <cstdint>
#include <cuda_fp16.h>
#include <string_view>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace params
{

// =============================================================================
// @Note
//   `DataType` and `gemmVersion` are managed by CMake automatically.
// -----------------------------------------------------------------------------
// This is managed automatically by CMake.
#ifdef TEST_FLOAT16
using DataType = playground::float16_t;
constexpr std::string_view DataTypeName = "fp16";
#else
using DataType = playground::float32_t;
constexpr std::string_view DataTypeName = "fp32";
#endif
// This is managed automatically by CMake.
#ifndef TEST_KERNEL_VERSION
    #define TEST_KERNEL_VERSION 0
#endif
constexpr auto GemmVersion = playground::uint8_t(TEST_KERNEL_VERSION);
// -----------------------------------------------------------------------------

// Size of Matrices
// mat(M,K)@mat(K,N)=mat(M,N)
constexpr playground::size_t M = 4096, N = 4096, K = 4096;

// Repeated Times
constexpr playground::size_t NRep = 100;

// Warmup Times
constexpr playground::size_t NWarmup = 10;

// Range of Elements in the Matrices
constexpr playground::size_t ElemMin = 0, ElemMax = 1;

}  // namespace params
