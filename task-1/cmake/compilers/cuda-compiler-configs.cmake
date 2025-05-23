# ==================================================================================================
# @file compiler-configs-cuda.cmake
# @brief Compiler configurations for cuda.
#
# @note Values SHOULD be set BEFORE including this file:
#         - `ENV{NVCC_CCBIN}`: CUDA Compiler bindir. Default: auto-detected.
#         - `CMAKE_CUDA_STANDARD`: CUDA Standard. Default: 20.
# ==================================================================================================

include(${CMAKE_CURRENT_LIST_DIR}/../utils/logging.cmake)

enable_language(CUDA)

if(WIN32)
    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        log_fatal("You have to use MSVC for CUDA on Windows")
    endif()
endif()

set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES native)
log_info("CMAKE_CUDA_STANDARD: ${CMAKE_CUDA_STANDARD}")

string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr")
string(APPEND CMAKE_CUDA_FLAGS_RELWITHDEBINFO " -lineinfo")
string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -lineinfo")