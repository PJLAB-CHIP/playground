# Minimum required CMake version
cmake_minimum_required(VERSION 3.30)

# Project Name
project(Playground-Task1)

# Compiler configurations for cpp, e.g., g++, clang++, msvc(cl), etc.
include(${PROJECT_SOURCE_DIR}/cmake/compilers/cxx-compiler-configs.cmake)
# Compiler configurations for CUDA 
include(${PROJECT_SOURCE_DIR}/cmake/compilers/cuda-compiler-configs.cmake)
# Some preparational steps for linking CUDA libraries:
include(${PROJECT_SOURCE_DIR}/cmake/libraries/libcuda.cmake)
# Some preparational steps for linking OpenBLAS libraries:
include(${PROJECT_SOURCE_DIR}/cmake/libraries/libopenblas.cmake)

add_subdirectory(src)
