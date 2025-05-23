#include <cuda_runtime.h>
#include <cxxopts.hpp>

#include "playground/matmul.hpp"
#include "playground/parameters.hpp"
#include "playground/system.hpp"
#include "playground/test_data.hpp"

// If CPU matmul is used; Otherwise CUDA matmul.
constexpr bool USING_CPU_MATMUL =
    playground::params::MatmulVersion == playground::PG_MATMUL_FP32_CBLAS ||
    playground::params::MatmulVersion == playground::PG_MATMUL_FP16_CBLAS;

namespace playground
{
PLAYGROUND_MATMUL_DEC(params::DataType, params::MatmulVersion, M, N, K, A, B,
                      C);
}

void test(uint32_t m, uint32_t n, uint32_t k, uint32_t nWarmupRound,
          uint32_t nTestRound)
{
    using namespace playground;

    auto testData = TestData(m, n, k);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float32_t runtime = 0.0F, totalRuntime = 0.0F;

    ::printf(
        "[Playgounrd] Start Testing for GEMM Version %d with DType %s ... \n",
        params::MatmulVersion, params::DataTypeName.data());

    params::DataType* Aptr = nullptr;
    params::DataType* Bptr = nullptr;
    params::DataType* Cptr = nullptr;

    auto matmulFn = [&Aptr, &Bptr, &Cptr, m, n, k]() {
        PLAYGOUND_MATMUL_CALL(params::MatmulVersion, m, n, k, Aptr, Bptr,
                              Cptr);
    };

    if constexpr (!USING_CPU_MATMUL) {
        testData.initDeviceData();
        Aptr = testData.getdAptr();
        Bptr = testData.getdBptr();
        Cptr = testData.getdCptr();
        for (size_t i = 0; i < nWarmupRound; ++i) {
            matmulFn();
        }
        for (auto i = 0ULL; i < nTestRound; ++i) {
            cudaEventRecord(start, nullptr);
            matmulFn();
            cudaEventRecord(stop, nullptr);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&runtime, start, stop);
            totalRuntime += runtime;
        }
        testData.copyResultD2H();
    } else {
        Aptr = testData.getAptr();
        Bptr = testData.getBptr();
        Cptr = testData.getCptr();
        cudaEventRecord(start, nullptr);
        matmulFn();
        cudaEventRecord(stop, nullptr);
        cudaEventElapsedTime(&runtime, start, stop);
        totalRuntime += runtime;
    }
    cudaDeviceSynchronize();
    ::printf("[Playground] Calculating Finished\n");

    float32_t avgErr = testData.calculateAvgErr();

    float msecPerMatrixMul = totalRuntime / nTestRound;
    double flopsPerMatrixMul = 2.0 * m * n * k;
    double tflops =
        (flopsPerMatrixMul * 1.0e-12F) / (msecPerMatrixMul / 1000.0F);

    ::printf("[Playground] Result >>> TFLOPS: %lf; Average Error: %f\n",
             tflops, avgErr);
}

auto main(int argc, const char* argv[]) -> int
{
    auto options = cxxopts::Options("Playground", "Matrix Multiplication");
    // clang-format off
    options.add_options()
        ("m", "Num of rows of A and C", 
            cxxopts::value<uint32_t>()->default_value("4096"))
        ("n", "Num of columns of B and C",
            cxxopts::value<uint32_t>()->default_value("4096"))
        ("k", "Num of columns of A and rows of B",
            cxxopts::value<uint32_t>()->default_value("4096"))
        ("w,n_warmup", "Num of warmup rounds",
            cxxopts::value<uint32_t>()->default_value("10"))
        ("t,n_test", "Num of test rounds",
            cxxopts::value<uint32_t>()->default_value("100"))
        ("h,help", "Print usage");
    // clang-format on
    auto results = options.parse(argc, argv);

    uint32_t m = results["m"].as<uint32_t>();
    uint32_t n = results["n"].as<uint32_t>();
    uint32_t k = results["k"].as<uint32_t>();
    uint32_t nWarmupRound = results["w"].as<uint32_t>();
    uint32_t nTestRound = results["t"].as<uint32_t>();
    bool showHelp = results["h"].as<bool>();

    if (showHelp) {
        ::puts(options.help().c_str());
        return 0;
    }

    test(m, n, k, nWarmupRound, nTestRound);
    return 0;
}