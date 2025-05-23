# Set the Target Name and Add Executable
set(TARGET_NAME task1)
file(GLOB_RECURSE CXX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
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
    ${OPENBLAS_LIBRARAY}
    cxxopts::cxxopts
    spdlog::spdlog
)

set(TARGET_BIN_OUTPUT_NAME "${TARGET_NAME}_${TEST_DATA_TYPE}_v${MATMUL_VERSION}")
set_target_properties(
    ${TARGET_NAME} PROPERTIES 
    OUTPUT_NAME 
        ${TARGET_BIN_OUTPUT_NAME}
)
