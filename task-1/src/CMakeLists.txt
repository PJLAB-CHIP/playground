# Set the Target Name and Add Executable
set(TARGET_NAME task1)

# Determine source files based on data type and version
if(TEST_DATA_TYPE STREQUAL "float32")
    set(DATA_TYPE_DIR "matmul_f32")
elseif(TEST_DATA_TYPE STREQUAL "float16")
    set(DATA_TYPE_DIR "matmul_f16")
else()
    message(FATAL_ERROR "Unsupported TEST_DATA_TYPE: ${TEST_DATA_TYPE}")
endif()

# Always include main.cpp
set(CXX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
set(CUDA_SOURCES "")

# Include ALL version files for the specified data type
# This is necessary because the main function may call different versions
file(GLOB VERSION_CXX_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${DATA_TYPE_DIR}/*.cpp")
file(GLOB VERSION_CUDA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${DATA_TYPE_DIR}/*.cu")

# Combine all version-specific files
list(APPEND CXX_SOURCES ${VERSION_CXX_FILES})
list(APPEND CUDA_SOURCES ${VERSION_CUDA_FILES})

# Log included files for debugging
log_info("Including source files for ${TEST_DATA_TYPE} version ${MATMUL_VERSION}:")
foreach(source ${CXX_SOURCES})
    log_info("  CXX: ${source}")
endforeach()
foreach(source ${CUDA_SOURCES})
    log_info("  CUDA: ${source}")
endforeach()

add_executable(${TARGET_NAME} ${CXX_SOURCES} ${CUDA_SOURCES})

# [NOTE] Uncomment this to generate preprocessed file and assembly file
# target_compile_options(${TARGET_NAME} PRIVATE -save-temps)

log_info("Matmul Version: ${MATMUL_VERSION}")
target_compile_definitions(
    ${TARGET_NAME} PRIVATE 
    MATMUL_VERSION=${MATMUL_VERSION}
)

log_info("Test Data Type: ${TEST_DATA_TYPE}")
target_compile_definitions(
    ${TARGET_NAME} PRIVATE 
    $<IF:$<STREQUAL:${TEST_DATA_TYPE},float32>,TEST_FLOAT32,
    $<IF:$<STREQUAL:${TEST_DATA_TYPE},float16>,TEST_FLOAT16,
    $<ERROR:Unsupported data type: ${TEST_DATA_TYPE}>>>
)

target_include_directories(
    ${TARGET_NAME} PRIVATE 
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(
    ${TARGET_NAME} PRIVATE 
    CUDA::cudart 
    CUDA::cublas
    OpenBLAS::OpenBLAS
    cxxopts::cxxopts
    nvidia::cutlass::cutlass
)

set(TARGET_BIN_OUTPUT_NAME "${TARGET_NAME}_${TEST_DATA_TYPE}_v${MATMUL_VERSION}")
set_target_properties(
    ${TARGET_NAME} PROPERTIES 
    OUTPUT_NAME 
        ${TARGET_BIN_OUTPUT_NAME}
)

# Add TARGET_BIN_OUTPUT_NAME to compile definitions
target_compile_definitions(
    ${TARGET_NAME} PRIVATE 
    TARGET_BIN_OUTPUT_NAME="${TARGET_BIN_OUTPUT_NAME}"
)
