# ==================================================================================================
# @file compiler-configs-cpp.cmake
# @brief Compiler configurations for the host.
#
# @note Values MUST be set BEFORE including this file:
#         - `STDOUT_IS_TERMINAL`: Whether stdout is a terminal.
#       Values SHOULD be set BEFORE including this file:
#         - `ENV{CXX}`: C++ Compiler. Default: auto-detected.
#         - `CMAKE_CXX_STANDARD`: C++ Standard. Default: 20.
#         - `CMAKE_CXX_SCAN_FOR_MODULES`: Whether to use modules. Default: OFF.
#         - `STACK_SIZE`: Stack size for the executable. Default: 1048576 (1MB).
# ==================================================================================================

include(${CMAKE_CURRENT_LIST_DIR}/../utils/common.cmake)

enable_language(CXX)

set(CMAKE_CXX_STANDARD_REQUIRED   ON)

set_default_values(
    CMAKE_CXX_STANDARD            20
    CMAKE_CXX_SCAN_FOR_MODULES    OFF
    STACK_SIZE                    1048576
)

log_info("CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
log_info("CMAKE_CXX_SCAN_FOR_MODULES: ${CMAKE_CXX_SCAN_FOR_MODULES}")
log_info("STACK_SIZE: ${STACK_SIZE}")

# MSVC ----------------------------------------------------------------------------------------------------------------
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    string(APPEND CMAKE_CXX_FLAGS " /permissive- /Zc:forScope /openmp /Zc:__cplusplus")
    string(APPEND CMAKE_EXE_LINKER_FLAGS " /STACK:${STACK_SIZE}")
# Clang ---------------------------------------------------------------------------------------------------------------
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    string(APPEND CMAKE_CXX_FLAGS " -fopenmp -Wall -Wextra -Werror")
    if (WIN32)
        string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wl,-stack,${STACK_SIZE}")
    else()
        string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wl,-zstack-size=${STACK_SIZE}")
    endif()
# GNU -----------------------------------------------------------------------------------------------------------------
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    string(APPEND CMAKE_CXX_FLAGS " -fopenmp -Wall -Wextra -Werror")
    if (WIN32)
        string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wl,--stack,${STACK_SIZE}")
    else()
        string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wl,-zstack-size=${STACK_SIZE}")
    endif()
else()
    log_fatal("Unsupported compiler")
endif()

if(STDOUT_IS_TERMINAL)
    add_compile_options(
        $<$<CXX_COMPILER_ID:GNU>:-fdiagnostics-color=always>
        $<$<CXX_COMPILER_ID:Clang>:-fcolor-diagnostics>
        $<$<CXX_COMPILER_ID:MSVC>:/FC>
    )
endif()