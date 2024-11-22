# ==================================================================================================
# @brief Enable CUDA support.
# [SEE] "https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html"
# 
# @note Several parameters should be set before including this file:
#         - `ENV{CUDA_HOME}`: The path to the CUDA toolkit. For example, `/usr/local/cuda`.
# ==================================================================================================

# Include "./Logging.cmake"
include(${PROJECT_SOURCE_DIR}/cmake/utils/logging.cmake)

if (NOT DEFINED ENV{CUDA_HOME})
    log_error("ENV:CUDA_HOME is not set.")
endif()

# Append the path to the CMAKE_PREFIX_PATH
set(CUDA_CMAKE_PREFIX_PATH "$ENV{CUDA_HOME}/lib64/cmake")
list(APPEND CMAKE_PREFIX_PATH ${CUDA_CMAKE_PREFIX_PATH})

# Find the CUDA package
# [SEE] 
# |- All imported targets: 
# |- "https://cmake.org/cmake/help/git-stage/module/FindCUDAToolkit.html#imported-targets"
find_package(CUDAToolkit REQUIRED)
