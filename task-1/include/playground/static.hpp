#include <cublas_v2.h>

#include "playground/system.hpp"

namespace playground
{

template <typename DType>
class CuBLASHandle
{
public:
    explicit CuBLASHandle()
    {
        cublasCreate(&handle);
        if constexpr (std::is_same_v<DType, float16_t>) {
            cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
        }
    }

    ~CuBLASHandle()
    {
        cublasDestroy(handle);
    }

public:
    cublasHandle_t handle = {};
};

template <typename DType>
inline auto s_getCublasHandle() -> cublasHandle_t&
{
    static CuBLASHandle<DType> handle;
    return handle.handle;
}

}  // namespace playground