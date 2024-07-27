# ==================================================================================================
# @brief Libraries for building the target.
# @note Create target ${TARGET_NAME} before including this file.
# ==================================================================================================

# CUDA
enable_language(CUDA)
set_target_properties(
    ${TARGET_NAME}
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80"  # A100
)
target_include_directories(${TARGET_NAME} PRIVATE $ENV{CUDA_HOME}/include)
target_link_libraries(${TARGET_NAME} PRIVATE cuda)

# cuBLAS
link_directories(${TARGET_NAME} PRIVATE $ENV{CUDA_HOME}/lib64)
find_library(CUBLAS_LIBRARY cublas HINTS $ENV{CUDA_HOME}/lib64)
target_link_libraries(${TARGET_NAME} PRIVATE ${CUBLAS_LIBRARY})

# cBLAS
find_library(CBLAS_LIBRARAY cblas)
target_link_libraries(${TARGET_NAME} PRIVATE ${CBLAS_LIBRARAY})

