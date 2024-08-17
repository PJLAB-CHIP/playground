// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: util function

#pragma once

#include <cmath>
#include <concepts>
#include <cstdlib>
#include <ctime>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_types.h>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector_types.h>

#include "playground/logging.hpp"
#include "playground/system.hpp"

namespace playground
{

constexpr __device__ __host__ size_t ceilDivide(size_t a, size_t b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// Beginning of GPU Architecture definitions
PJ_FINLINE int convert_SM_to_cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the
    // # of cores per SM)
    typedef struct
    {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version, and m =
                 // SM minor version
        int cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128},
        {0x52, 128}, {0x53, 128}, {0x60, 64},  {0x61, 128}, {0x62, 128},
        {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},  {0x86, 128},
        {0x87, 128}, {0x89, 128}, {0x90, 128}, {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run
    // properly
    HLOG("MapSMtoCores for SM %d.%d is undefined. Default to use %d cores/SM",
         major, minor, nGpuArchCoresPerSM[index - 1].cores);

    return nGpuArchCoresPerSM[index - 1].cores;
}

PJ_FINLINE const char* convert_SM_to_arch_name(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the
    // GPU Arch name)
    typedef struct
    {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version, and m =
                 // SM minor version
        const char* name;
    } sSMtoArchName;

    sSMtoArchName nGpuArchNameSM[] = {
        {0x30, "Kepler"},       {0x32, "Kepler"},  {0x35, "Kepler"},
        {0x37, "Kepler"},       {0x50, "Maxwell"}, {0x52, "Maxwell"},
        {0x53, "Maxwell"},      {0x60, "Pascal"},  {0x61, "Pascal"},
        {0x62, "Pascal"},       {0x70, "Volta"},   {0x72, "Xavier"},
        {0x75, "Turing"},       {0x80, "Ampere"},  {0x86, "Ampere"},
        {0x87, "Ampere"},       {0x89, "Ada"},     {0x90, "Hopper"},
        {-1, "Graphics Device"}};

    int index = 0;

    while (nGpuArchNameSM[index].SM != -1) {
        if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchNameSM[index].name;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run
    // properly
    HLOG("MapSMtoArchName for SM %d.%d is undefined. Default to use %s", major,
         minor, nGpuArchNameSM[index - 1].name);

    return nGpuArchNameSM[index - 1].name;
}

template <typename T1, typename T2>
float32_t compareMat(size_t m, size_t n, T1 A, T2 B)
{
    float32_t gap = 0.0;
    float32_t err_sum = 0.0;
    float32_t avg_err = 0.0;

    for (size_t i = 0; i < m * n; ++i) {
        gap = std::abs(float32_t(A[i]) - float32_t(B[i]));
        err_sum += gap / float32_t(A[i]);

        if (std::isinf(err_sum)) {
            printf("%zu/%zu err_sum: %f, gap: %f, divider: %f\n", i, m * n,
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
    srand(time(nullptr));
    for (std::size_t cnt = 0; cnt < m * n; cnt++) {
        mat[cnt] = T(float32_t(rand()) / RAND_MAX +
                     std::numeric_limits<float32_t>::min());
    }
}

/**
 * @brief Convert 2D index to 1D index.
 * `result` = `row` * `col_size` + `col`.
 *
 * @param row Row index.
 * @param col Column index.
 * @param col_size Column size.
 */
template <typename T1, typename T2, typename T3>
    requires std::integral<T1> && std::integral<T2> && std::integral<T3>
PJ_FINLINE __host__ __device__ size_t idx2To1(T1 row, T2 col, T3 col_size)
{
    return size_t(row) * size_t(col_size) + size_t(col);
}

PJ_FINLINE __host__ __device__ float4* castToCuFloat4(auto* x)
{
    return (float4*) x;
}

PJ_FINLINE __host__ __device__ float4& castToCuFloat4(auto& x)
{
    return *((float4*) &x);
}

PJ_FINLINE __host__ __device__ int4* castToCuInt4(auto* x)
{
    return (int4*) x;
}

PJ_FINLINE __host__ __device__ int4& castToCuInt4(auto& x)
{
    return *((int4*) &x);
}

/**
 * @brief Reinterpret cast.
 */
template <typename T1, typename T2>
    requires(!std::is_pointer_v<T2>) &&
            (std::is_const_v<T1> || !std::is_const_v<T2>)
PJ_FINLINE __host__ __device__ T1& rCast(T2& x)
{
    return *(reinterpret_cast<T1*>(&x));
}

/**
 * @brief Reinterpret cast.
 */
template <typename T1, typename T2>
    requires std::is_pointer_v<T1> &&
             (std::is_const_v<std::remove_pointer_t<T1>> ||
              (!std::is_const_v<T2>) )
PJ_FINLINE __host__ __device__ T1 rCast(T2* x)
{
    return reinterpret_cast<T1>(x);
}

}  // namespace playground