# ==================================================================================================
# @file logging.cmake
# @brief Compiler configurations for cuda.
#
# @note This file shoule NEVER include any other file (to avoid circular dependencies).
#
# @note Values MUST be set BEFORE including this file:
#         - `STDOUT_IS_TERMINAL`: Whether stdout is a terminal.
# ==================================================================================================

if(STDOUT_IS_TERMINAL)
    string(ASCII 27 Esc)
    set(LOG_RED "${Esc}[0;31m")
    set(LOG_GREEN "${Esc}[0;32m")
    set(LOG_YELLOW "${Esc}[0;33m")
    set(LOG_BLUE "${Esc}[0;34m")
    set(LOG_PURPLE "${Esc}[0;35m")
    set(LOG_CYAN "${Esc}[0;36m")
    set(LOG_WHITE "${Esc}[0;37m")
    set(LOG_RESET "${Esc}[m")
else()
    set(LOG_RED "")
    set(LOG_GREEN "")
    set(LOG_YELLOW "")
    set(LOG_BLUE "")
    set(LOG_PURPLE "")
    set(LOG_CYAN "")
    set(LOG_WHITE "")
    set(LOG_RESET "")
endif()

if(NOT DEFINED LOG_PREFIX)
    set(LOG_PREFIX "playground")
endif()

function(log_info msg)
    message(STATUS "[${LOG_PURPLE}${LOG_PREFIX}${LOG_RESET}|${LOG_GREEN}INFO${LOG_RESET}] >>> ${msg}")
endfunction()

function(log_warning msg)
    message(WARNING "[${LOG_PREFIX}|WARNING] >>> ${msg}")
endfunction()

function(log_error msg)
    message(SEND_ERROR "[${LOG_PREFIX}|ERROR] >>> ${msg}")
endfunction()

function(log_fatal msg)
    message(FATAL_ERROR "[${LOG_PREFIX}|FATAL] >>> ${msg}")
endfunction()