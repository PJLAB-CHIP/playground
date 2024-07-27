#include <cuda_runtime.h>
#include <vector>

#include "parameters.hpp"
#include "playground/matmul.hpp"
#include "playground/static.hpp"
#include "playground/utils.hpp"

int main()
{
    auto A = std::vector<params::DataType>(params::M * params::K);
    playground::initRandMat(params::M, params::K, A.data());
    auto B = std::vector<params::DataType>(params::K * params::N);
    playground::initRandMat(params::K, params::N, B.data());
    auto C = std::vector<params::DataType>(params::M * params::N);

    // Gound Truth of C
    auto C_gt = C;
    printf("[Playground] Start Calculating Ground Truth ... ");
    fflush(stdout);
    playground::matmul<params::DataType, 0>(params::M, params::N, params::K, A.data(), B.data(),
                                            C_gt.data());
    printf("Finished!\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    params::DataType *d_A, *d_B, *d_C;
    cudaMalloc((void**) &d_A, A.size() * sizeof(params::DataType));
    cudaMalloc((void**) &d_B, B.size() * sizeof(params::DataType));
    cudaMalloc((void**) &d_C, C.size() * sizeof(params::DataType));

    cudaMemcpy(d_A, A.data(), A.size() * sizeof(params::DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B.size() * sizeof(params::DataType), cudaMemcpyHostToDevice);

    float runtime = 0.0f, sumRuntime = 0.0f;

    printf("[Playgounrd] Start Testing for GEMM Version %d with DType %s ... \n",
           params::gemmVersion, params::dataTypeName.c_str());
    // If not using cblas, execute the function multiple times to get average runtime
    if constexpr (params::gemmVersion != 0) {
        for (auto i = 0ULL; i < params::N_REP + params::N_WARMUP; ++i) {
            // Warm Up
            if (i < params::N_WARMUP) {
                playground::matmul<params::DataType, uint8_t(params::gemmVersion)>(
                    params::M, params::N, params::K, d_A, d_B, d_C);
                continue;
            }
            if (i == params::N_WARMUP) {
                printf("[Playground] Warming Up Finished!\n");
            }

            cudaEventRecord(start, nullptr);
            playground::matmul<params::DataType, int8_t(params::gemmVersion)>(
                params::M, params::N, params::K, d_A, d_B, d_C);
            cudaEventRecord(stop, nullptr);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&runtime, start, stop);
            sumRuntime += runtime;
        }
        cudaMemcpy(C.data(), d_C, C.size() * sizeof(params::DataType), cudaMemcpyDeviceToHost);
    }
    // If using cblas, run the function only once
    else {
        cudaEventRecord(start, nullptr);
        playground::matmul<params::DataType, uint8_t(params::gemmVersion)>(
            params::M, params::N, params::K, A.data(), B.data(), C.data());
        cudaEventRecord(stop, nullptr);
        cudaEventElapsedTime(&runtime, start, stop);
        sumRuntime += runtime;
    }

    cudaDeviceSynchronize();
    printf("[Playground] Calculating Finished\n");

    auto avgErr = playground::compareMat(params::M, params::N, C_gt.data(), C.data());

    // calculate tflops and average error
    float msecPerMatrixMul = sumRuntime / params::N_REP;
    double flopsPerMatrixMul = 2.0 * params::M * params::N * params::K;
    double tflops = (flopsPerMatrixMul * 1.0e-12f) / (msecPerMatrixMul / 1000.0f);

    printf("[Playground] Result >>> TFLOPS: %lf; Average Error: %f\n", tflops, avgErr);

    // free memories in device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}