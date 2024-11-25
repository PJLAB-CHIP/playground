# ==================================================================================================
# @brief Enable CUDA support.
# @see "https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html"
# 
# @note Several parameters should be set before including this file:
#       - [ENV]{CUDA_HOME}/[ENV]{CUDA_DIR}:
#             Path to spdlog libaray installation path.
# ==================================================================================================

include(${PROJECT_SOURCE_DIR}/cmake/utils/common.cmake)

try_get_value(CUDA_HOME HINTS CUDA_HOME CUDA_DIR)
if (NOT CUDA_HOME_FOUND)
    log_error("ENV:CUDA_HOME is not set.")
endif()

# Append the path to the CMAKE_PREFIX_PATH
set(CUDA_CMAKE_PREFIX_PATH "$CUDA_HOME/lib64/cmake")
list(APPEND CMAKE_PREFIX_PATH ${CUDA_CMAKE_PREFIX_PATH})

# Find the CUDA package
# @see 
# |- All imported targets: 
# |- "https://cmake.org/cmake/help/git-stage/module/FindCUDAToolkit.html#imported-targets"
find_package(CUDAToolkit REQUIRED)