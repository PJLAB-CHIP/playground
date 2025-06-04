# Minimum required CMake version
cmake_minimum_required(VERSION 3.30)
# Project Name
project(Playground-Task1)

include(${PROJECT_SOURCE_DIR}/cmake/utils/common.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set_default_values(
    PROJECT_NAMESPACE "playground::"
    STDOUT_IS_TERMINAL OFF
    CMAKE_EXECUTABLE_SUFFIX ""
)

# Compiler configurations for cpp, e.g., g++, clang++, msvc(cl), etc.
include(${PROJECT_SOURCE_DIR}/cmake/compilers/cxx-compiler-configs.cmake)
# Compiler configurations for CUDA 
include(${PROJECT_SOURCE_DIR}/cmake/compilers/cuda-compiler-configs.cmake)
# Some preparational steps for linking CUDA libraries:
include(${PROJECT_SOURCE_DIR}/cmake/libraries/libcuda.cmake)

find_package(OpenBLAS CONFIG REQUIRED)
find_package(cxxopts CONFIG REQUIRED)
find_package(NvidiaCutlass CONFIG REQUIRED)

add_subdirectory(src)
