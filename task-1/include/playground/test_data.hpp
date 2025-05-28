#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "playground/matmul.hpp"
#include "playground/system.hpp"
#include "playground/utils.hpp"

namespace playground
{

template <typename DType>
class CudaDeviceMemPtr
{
public:
    explicit CudaDeviceMemPtr() : ptr(nullptr)
    {
    }

    explicit CudaDeviceMemPtr(size_t size) : ptr(nullptr)
    {
        cudaMalloc((void**) &ptr, size * sizeof(DType));
    }

    CudaDeviceMemPtr(const CudaDeviceMemPtr&) = delete;

    auto operator=(const CudaDeviceMemPtr&) -> CudaDeviceMemPtr& = delete;

    CudaDeviceMemPtr(CudaDeviceMemPtr&& other) noexcept : ptr(other.ptr)
    {
        other.ptr = nullptr;
    }

    auto operator=(CudaDeviceMemPtr&& other) noexcept -> CudaDeviceMemPtr&
    {
        if (this != &other) {
            if (ptr != nullptr) {
                cudaFree(ptr);
            }
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    ~CudaDeviceMemPtr()
    {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }

    [[nodiscard]]
    auto get_const_raw_ptr() const -> const DType*
    {
        return ptr;
    }

    [[nodiscard]]
    auto get_mutable_raw_ptr() -> DType*
    {
        return ptr;
    }

    [[nodiscard]]

    explicit operator bool() const
    {
        return ptr != nullptr;
    }

private:
    DType* ptr;
};

class TestData
{
public:
    explicit TestData(uint32_t m, uint32_t n, uint32_t k) : _m(m), _n(n), _k(k)
    {
        this->initHostData();
        this->calculateGroundTruth();
    }

public:
    [[nodiscard]]
    auto get_mutable_A_ptr() -> params::DataType*
    {
        return _A.data();
    }

    [[nodiscard]]
    auto get_mutable_B_ptr() -> params::DataType*
    {
        return _B.data();
    }

    [[nodiscard]]
    auto get_mutable_C_ptr() -> params::DataType*
    {
        return _C.data();
    }

    [[nodiscard]]
    auto get_mutable_GT_ptr() -> params::DataType*
    {
        return _GT.data();
    }

    [[nodiscard]]
    auto get_mutable_d_A_ptr() -> params::DataType*
    {
        return _d_A.get_mutable_raw_ptr();
    }

    [[nodiscard]]
    auto get_mutable_d_B_ptr() -> params::DataType*
    {
        return _d_B.get_mutable_raw_ptr();
    }

    [[nodiscard]]
    auto get_mutable_d_C_ptr() -> params::DataType*
    {
        return _d_C.get_mutable_raw_ptr();
    }

    void initHostData()
    {
        _A.resize(_m * _k);
        _B.resize(_k * _n);
        _C.resize(_m * _n);
        _GT.resize(_m * _n);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float64_t> distrib(0.0, 1.0);
        std::ranges::generate(_A, [&]() { return distrib(gen); });
        std::ranges::generate(_B, [&]() { return distrib(gen); });
    }

    auto calculateAvgErr() -> float32_t
    {
        float32_t gap = 0.0;
        float32_t errSum = 0.0;
        float32_t avgErr = 0.0;

        for (size_t i = 0; i < _GT.size(); ++i) {
            gap = float32_t(_GT[i]) - float32_t(_C[i]);
            errSum += ::std::abs(gap / float32_t(_GT[i]));
            PLAYGROUND_CHECK(!::std::isinf(errSum));
            PLAYGROUND_CHECK(!::std::isnan(errSum));
        }

        avgErr = errSum / float32_t(_GT.size());

        return avgErr;
    }

    void calculateGroundTruth()
    {
        if constexpr (std::is_same_v<params::DataType, float32_t>) {
            PLAYGOUND_MATMUL_CALL(PG_MATMUL_FP32_CBLAS, _m, _n, _k, _A.data(),
                                  _B.data(), _GT.data());
        } else if constexpr (std::is_same_v<params::DataType, float16_t>) {
            PLAYGOUND_MATMUL_CALL(PG_MATMUL_FP16_CBLAS, _m, _n, _k, _A.data(),
                                  _B.data(), _GT.data());
        } else {
            throw std::runtime_error("Unsupported data type");
        }
    }

    void initDeviceData()
    {
        _d_A = CudaDeviceMemPtr<params::DataType>(_A.size());
        _d_B = CudaDeviceMemPtr<params::DataType>(_B.size());
        _d_C = CudaDeviceMemPtr<params::DataType>(_C.size());

        cudaMemcpy(_d_A.get_mutable_raw_ptr(), _A.data(),
                   _A.size() * sizeof(params::DataType),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(_d_B.get_mutable_raw_ptr(), _B.data(),
                   _B.size() * sizeof(params::DataType),
                   cudaMemcpyHostToDevice);
    }

    void copyResultD2H()
    {
        cudaMemcpy(_C.data(), _d_C.get_mutable_raw_ptr(),
                   _C.size() * sizeof(params::DataType),
                   cudaMemcpyDeviceToHost);
    }

private:
    uint32_t _m;
    uint32_t _n;
    uint32_t _k;

    std::vector<params::DataType> _A;
    std::vector<params::DataType> _B;
    std::vector<params::DataType> _C;
    std::vector<params::DataType> _GT;
    CudaDeviceMemPtr<params::DataType> _d_A;
    CudaDeviceMemPtr<params::DataType> _d_B;
    CudaDeviceMemPtr<params::DataType> _d_C;
};
}  // namespace playground