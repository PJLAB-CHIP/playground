#include <cstdint>
#include <cuda_fp16.h>
#include <string>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace params
{

// =================================================================================================
// @Note 
//   `DataType` and `gemmVersion` are managed by CMake automatically.
// -------------------------------------------------------------------------------------------------
// This is managed automatically by CMake.
#ifdef TEST_FLOAT16
using DataType = playground::float16_t;
std::string dataTypeName = "fp16";
#else
using DataType = playground::float32_t;
std::string dataTypeName = "fp32";
#endif
// This is managed automatically by CMake.
#ifndef TEST_KERNEL_VERSION
#define TEST_KERNEL_VERSION 0
#endif
constexpr auto gemmVersion = playground::uint8_t(TEST_KERNEL_VERSION);
// -------------------------------------------------------------------------------------------------

// Size of Matrices
// mat(M,K)@mat(K,N)=mat(M,N)
constexpr playground::size_t M = 4096, N = 4096, K = 4096;

// Repeated Times
constexpr playground::size_t N_REP = 100;

// Warmup Times
constexpr playground::size_t N_WARMUP = 10;

// Range of Elements in the Matrices
constexpr playground::size_t ELE_MIN = 0, ELE_MAX = 1;

}  // namespace params
