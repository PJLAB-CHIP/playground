#pragma once

#include <cstdio>
#include <cuda_runtime_api.h>

/**
 * @brief Check the given cuda error. Exit with `EXIT_FAILURE` if not success.
 *        The error message is printed to `stderr`.
 */
#define PLAYGOUND_CUDA_ERR_CHECK(err)                                         \
    do {                                                                      \
        cudaError_t err_ = (err);                                             \
        if (err_ != cudaSuccess) {                                            \
            ::fprintf(stderr,                                                 \
                      "[Playground] CUDA error at %s:%d; Error code: %d(%s) " \
                      "\"%s\"",                                               \
                      __FILE__, __LINE__, err, ::cudaGetErrorString(err_),    \
                      #err);                                                  \
            ::cudaDeviceReset();                                              \
            ::std::exit(EXIT_FAILURE);                                        \
        }                                                                     \
    } while (0)

#ifdef NDEBUG
    /**
     * @brief Cuda error check is turned off on Release mode.
     */
    #define PLAYGOUND_DEBUG_CUDA_ERR_CHECK(err) ((void) 0)
#else
    /**
     * @brief Check the given cuda error. Exit with `EXIT_FAILURE` if not
     *        success.
     *        The error message is printed to `stderr`.
     */
    #define PLAYGOUND_DEBUG_CUDA_ERR_CHECK(err) PLAYGOUND_CUDA_ERR_CHECK(err)
#endif
