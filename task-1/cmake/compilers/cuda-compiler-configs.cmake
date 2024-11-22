# ==================================================================================================
# @file compiler-configs-cuda.cmake
# @brief Compiler configurations for cuda.
#
# @note Several parameters SHOULD be set BEFORE including this file:
#         - `ENV{NVCC_CCBIN}`: CUDA Compiler bindir. Default: auto-detected.
#         - `CMAKE_CUDA_STANDARD`: CUDA Standard. Default: 20.
# ==================================================================================================

include(${PROJECT_SOURCE_DIR}/cmake/utils/logging.cmake)

enable_language(CUDA)

if(WIN32)
    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        log_fatal("You have to use MSVC for CUDA on Windows")
    endif()
endif()

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES native)
log_info("CMAKE_CUDA_STANDARD: ${CMAKE_CUDA_STANDARD}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -lineinfo")