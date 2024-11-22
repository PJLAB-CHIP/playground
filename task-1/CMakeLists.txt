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

# Set the Target Name and Add Executable
set(TARGET_NAME task1)
file(GLOB_RECURSE CXX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
add_executable(${TARGET_NAME} ${CXX_SOURCES} ${CUDA_SOURCES})

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

set(TARGET_BIN_OUTPUT_NAME "${TARGET_NAME}_${TEST_DATA_TYPE}_v${TEST_KERNEL_VERSION}")
set_target_properties(
    ${TARGET_NAME} PROPERTIES 
    OUTPUT_NAME 
        ${TARGET_BIN_OUTPUT_NAME}
)
