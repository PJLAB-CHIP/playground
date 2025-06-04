#include <cublas_v2.h>

namespace playground
{

class CuBLASHandle
{
public:
    explicit CuBLASHandle()
    {
        cublasCreate(&handle);
    }

    ~CuBLASHandle()
    {
        cublasDestroy(handle);
    }

public:
    cublasHandle_t handle = {};
};

[[nodiscard]]
inline auto s_getCublasHandle() -> cublasHandle_t&
{
    static CuBLASHandle handle;
    return handle.handle;
}

}  // namespace playground