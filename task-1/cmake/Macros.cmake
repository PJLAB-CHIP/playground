# ==================================================================================================
# @brief Macros for building the target.
# @note Create target ${TARGET_NAME} before including this file.
# ==================================================================================================

# Test Kernel Version
if(NOT DEFINED TEST_KERNEL_VERSION)
    set(TEST_KERNEL_VERSION 0)
endif()
target_compile_definitions(${TARGET_NAME} PRIVATE TEST_KERNEL_VERSION=${TEST_KERNEL_VERSION})
message(STATUS "[playground] Test Kernel Version: ${TEST_KERNEL_VERSION}")

# Test Data Type
if(NOT DEFINED TEST_DATA_TYPE OR TEST_DATA_TYPE STREQUAL "float32")
    target_compile_definitions(${TARGET_NAME} PRIVATE TEST_FLOAT32)
elseif(TEST_DATA_TYPE STREQUAL "float16")
    target_compile_definitions(${TARGET_NAME} PRIVATE TEST_FLOAT16)
else()
    message(FATAL_ERROR "Unsupported data type: ${TEST_DATA_TYPE}.")
endif()
message(STATUS "[playground] Test Data Type: ${TEST_DATA_TYPE}")