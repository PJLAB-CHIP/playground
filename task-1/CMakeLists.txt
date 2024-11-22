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

# Set the Target Name and Add Executable
set(TARGET_NAME task1)
file(GLOB_RECURSE CXX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
add_executable(${TARGET_NAME} ${CXX_SOURCES} ${CUDA_SOURCES})

# Find openblas libraries. 
# [NOTE] Install OpenBlas by running:
#        |- $ apt install libopenblas-dev
find_library(OPENBLAS_LIBRARAY openblas REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE ${OPENBLAS_LIBRARAY})

target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(
    ${TARGET_NAME} PRIVATE 
    CUDA::cudart 
    CUDA::cublas
    ${OPENBLAS_LIBRARAY}
)

# Generate preprocessed file and assembly file
target_compile_options(${TARGET_NAME} PRIVATE -save-temps)

# Define Macros for compiling the target
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Macros.cmake)