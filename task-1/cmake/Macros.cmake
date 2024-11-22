# ==================================================================================================
# @brief Macros for building the target.
# @note Create target ${TARGET_NAME} before including this file.
# ==================================================================================================

# Matmul Version
if(NOT DEFINED MATMUL_VERSION)
    set(MATMUL_VERSION 0)
endif()
target_compile_definitions(${TARGET_NAME} PRIVATE MATMUL_VERSION=${MATMUL_VERSION})
log_info("Matmul Version: ${MATMUL_VERSION}")

# Data Type
if(NOT DEFINED TEST_DATA_TYPE OR TEST_DATA_TYPE STREQUAL "float32")
    target_compile_definitions(${TARGET_NAME} PRIVATE TEST_FLOAT32)
elseif(TEST_DATA_TYPE STREQUAL "float16")
    target_compile_definitions(${TARGET_NAME} PRIVATE TEST_FLOAT16)
else()
    message(FATAL_ERROR "Unsupported data type: ${TEST_DATA_TYPE}.")
endif()
log_info("Test Data Type: ${TEST_DATA_TYPE}")