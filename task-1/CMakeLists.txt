# Minimum required CMake version
cmake_minimum_required(VERSION 3.21)

# Project Name
project(Playground-Task1)

# Compiler Configurations
include(${CMAKE_CURRENT_SOURCE_DIR}/Compiler.cmake)

# Set the Target Name and Add Executable
set(TARGET_NAME task1)
file(GLOB_RECURSE CXX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
add_executable(${TARGET_NAME} ${CXX_SOURCES} ${CUDA_SOURCES})
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Generate preprocessed file and assembly file
target_compile_options(${TARGET_NAME} PRIVATE -save-temps)

# Define Macros for compiling the target
include(${CMAKE_CURRENT_SOURCE_DIR}/Macros.cmake)

# Include and Link to Libraries
include(${CMAKE_CURRENT_SOURCE_DIR}/Libraries.cmake)

# Set Binary Output Path
if(NOT DEFINED TARGET_BIN_OUTPUT_DIR)
    set(TARGET_BIN_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()
set(TARGET_BIN_OUTPUT_NAME "${TARGET_NAME}_${TEST_DATA_TYPE}_v${TEST_KERNEL_VERSION}.exe")
set_target_properties(
    ${TARGET_NAME} PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY 
        ${TARGET_BIN_OUTPUT_DIR}/bin/${CMAKE_BUILD_TYPE}/${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}
    OUTPUT_NAME 
        ${TARGET_BIN_OUTPUT_NAME}
)