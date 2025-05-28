#include <chrono>
#include <cuda_runtime.h>
#include <cxxopts.hpp>
#include <format>

#include "playground/matmul.hpp"
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

    float32_t totalMilliSecs = 0.0F;

    std::cout << std::format(
        "[Playground] Start Testing for GEMM Version {} with DType {} ...\n",
        params::MatmulVersion, params::DataTypeName.data());

    params::DataType* Aptr = nullptr;
    params::DataType* Bptr = nullptr;
    params::DataType* Cptr = nullptr;

    auto matmulFn = [&Aptr, &Bptr, &Cptr, m, n, k]() {
        PLAYGOUND_MATMUL_CALL(params::MatmulVersion, m, n, k, Aptr, Bptr,
                              Cptr);
    };

    if constexpr (!USING_CPU_MATMUL) {
        ::cudaEvent_t start, stop;
        ::cudaEventCreate(&start);
        ::cudaEventCreate(&stop);
        float32_t runtime = 0.0F;
        testData.initDeviceData();
        Aptr = testData.get_mutable_d_A_ptr();
        Bptr = testData.get_mutable_d_B_ptr();
        Cptr = testData.get_mutable_d_C_ptr();
        for (size_t i = 0; i < nWarmupRound; ++i) {
            matmulFn();
        }
        for (auto i = 0ULL; i < nTestRound; ++i) {
            ::cudaEventRecord(start, nullptr);
            matmulFn();
            ::cudaEventRecord(stop, nullptr);
            ::cudaEventSynchronize(stop);
            ::cudaEventElapsedTime(&runtime, start, stop);
            totalMilliSecs += runtime;
        }
        testData.copyResultD2H();
    } else {
        Aptr = testData.get_mutable_A_ptr();
        Bptr = testData.get_mutable_B_ptr();
        Cptr = testData.get_mutable_C_ptr();
        nTestRound = 10;  // 10 rounds for CPU matmul to avoid long runtime.
        for (auto i = 0ULL; i < nTestRound; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            matmulFn();
            auto end = std::chrono::high_resolution_clock::now();
            totalMilliSecs +=
                std::chrono::duration<float32_t, std::milli>(end - start)
                    .count();
        }
    }
    ::cudaDeviceSynchronize();
    std::cout << "[Playground] Calculating Finished\n";

    float32_t avgErr = testData.calculateAvgErr();

    float32_t msecPerMatrixMul = totalMilliSecs / nTestRound;
    float64_t flopsPerMatrixMul = 2.0 * m * n * k;
    float64_t tflops =
        (flopsPerMatrixMul * 1.0e-12F) / (msecPerMatrixMul / 1000.0F);

    std::cout << std::format(
        "[Playground] Result >>> TFLOPS: {}; Average Error: {}\n", tflops,
        avgErr);
}

auto main(int argc, const char* argv[]) -> int
{
    auto options =
        cxxopts::Options(TARGET_BIN_OUTPUT_NAME,
                         "Playground Task1: General Matrix Multiplication");
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

    if (results["h"].as<bool>()) {
        std::cout << options.help() + "\n";
        return 0;
    }

    test(m, n, k, nWarmupRound, nTestRound);
    return 0;
}