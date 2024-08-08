# Task 1: CUDA Programming

High performance gemm implementation on Nvidia A100.

## 1. Target

Implement a high performance gemm (General Matrix Multiply) function with CUDA on Nvidia A100 for float32 and float16 data types.

The implementation should be able to achieve at least 90% of the performance of cuBLAS, with the given benchmarking structure.


## 2. Benchmark cBlas and cuBlas

```bash
# Build gemm implemented with cblas with float32 as dtype:
bash scripts/build-task1.sh -v0 -f32
# Build gemm implemented with cblas with float16 as dtype:
bash scripts/build-task1.sh -v0 -f16
# Build gemm implemented with cublas with float32 as dtype:
bash scripts/build-task1.sh -v1 -f32
# Build gemm implemented with cublas with float16 as dtype:
bash scripts/build-task1.sh -v1 -f16
```

Run the executables in "[./bin](../bin)" directory to get the benchmark results.

## 4. Add Your Own Implementation

Go to "[./task-1/include/playground/matmul.hpp](./include/playground/matmul.hpp)", add a new declaration of function `matmul` inside namespace `playground`.

For example, if you want to implement a new `matmul` with `DType=float16` and `Version=2`, you can add the following lines to the file:

```cpp
template <>
void matmul<float16_t, 2>(const size_t m, const size_t n, const size_t k, const float16_t* const A, const float16_t* const B, float16_t* const C);
```

Then create a `.cu` file in "[./src](./src)" directory with any name you like, and implement the function `matmul` with the signature you just declared.

For example, add following lines in "./src/matmul_f16/v2.cu" to implement the function `matmul<float16_t, 2>`:

```cpp
#include "playground/matmul.hpp"

namespace playground {
template <>
void matmul<float16_t, 2>(const size_t m, const size_t n, const size_t k, const float16_t* const A, const float16_t* const B, float16_t* const C)
{
    // ......
}
}
```

Now you can build an executable to test your implementation with following command:

```bash
bash scripts/build.sh -v2 -f16
```