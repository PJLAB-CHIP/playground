#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector>

#include "playground/matmul.hpp"
#include "playground/parameters.hpp"
#include "playground/static.hpp"
#include "playground/utils/mat.hpp"

namespace playground
{
PLAYGROUND_MATMUL_SIG(params::DataType, params::MatmulVersion, M, N, K, A, B,
                      C);
}

auto main() -> int
{

    auto A = std::vector<playground::params::DataType>(playground::params::M *
                                                       playground::params::K);
    playground::initRandMat(playground::params::M, playground::params::K,
                            A.data());
    auto B = std::vector<playground::params::DataType>(playground::params::K *
                                                       playground::params::N);
    playground::initRandMat(playground::params::K, playground::params::N,
                            B.data());
    auto C = std::vector<playground::params::DataType>(playground::params::M *
                                                       playground::params::N);

    // Gound Truth of C
    auto C_gt = C;
    ::printf("[Playground] Start Calculating Ground Truth ... ");
    fflush(stdout);
    playground::matmul<playground::params::DataType, 0>(
        playground::params::M, playground::params::N, playground::params::K,
        A.data(), B.data(), C_gt.data());
    ::printf("Finished!\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    playground::params::DataType *d_A, *d_B, *d_C;
    cudaMalloc((void**) &d_A, A.size() * sizeof(playground::params::DataType));
    cudaMalloc((void**) &d_B, B.size() * sizeof(playground::params::DataType));
    cudaMalloc((void**) &d_C, C.size() * sizeof(playground::params::DataType));

    cudaMemcpy(d_A, A.data(), A.size() * sizeof(playground::params::DataType),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B.size() * sizeof(playground::params::DataType),
               cudaMemcpyHostToDevice);

    float runtime = 0.0f, sumRuntime = 0.0f;

    ::printf(
        "[Playgounrd] Start Testing for GEMM Version %d with DType %s ... \n",
        playground::params::MatmulVersion,
        playground::params::DataTypeName.data());

    // If not using cblas, execute the function multiple times to get average
    // runtime
    if constexpr (playground::params::MatmulVersion != 0) {
        for (auto i = 0ULL;
             i < playground::params::NumRep + playground::params::NumWarmup;
             ++i) {
            // Warm Up
            if (i < playground::params::NumWarmup) {
                playground::matmul<playground::params::DataType,
                                   playground::params::MatmulVersion>(
                    playground::params::M, playground::params::N,
                    playground::params::K, d_A, d_B, d_C);
                continue;
            }
            if (i == playground::params::NumWarmup) {
                printf("[Playground] Warming Up Finished!\n");
            }

            cudaEventRecord(start, nullptr);
            playground::matmul<playground::params::DataType,
                               playground::params::MatmulVersion>(
                playground::params::M, playground::params::N,
                playground::params::K, d_A, d_B, d_C);
            cudaEventRecord(stop, nullptr);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&runtime, start, stop);
            sumRuntime += runtime;
        }
        cudaMemcpy(C.data(), d_C,
                   C.size() * sizeof(playground::params::DataType),
                   cudaMemcpyDeviceToHost);
    }
    // If using cblas, run the function only once
    else {
        cudaEventRecord(start, nullptr);
        playground::matmul<playground::params::DataType,
                           playground::params::MatmulVersion>(
            playground::params::M, playground::params::N, playground::params::K,
            A.data(), B.data(), C.data());
        cudaEventRecord(stop, nullptr);
        cudaEventElapsedTime(&runtime, start, stop);
        sumRuntime += runtime;
    }

    cudaDeviceSynchronize();
    printf("[Playground] Calculating Finished\n");

    auto avgErr = playground::compareMat(
        playground::params::M, playground::params::N, C_gt.data(), C.data());

    // Calculate tflops and average error
    float msecPerMatrixMul = sumRuntime / playground::params::NumRep;
    double flopsPerMatrixMul = 2.0 * playground::params::M *
                               playground::params::N * playground::params::K;
    double tflops =
        (flopsPerMatrixMul * 1.0e-12f) / (msecPerMatrixMul / 1000.0f);

    printf("[Playground] Result >>> TFLOPS: %lf; Average Error: %f\n", tflops,
           avgErr);

    // Free memories in device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}