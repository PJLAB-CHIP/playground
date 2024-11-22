include(${PROJECT_SOURCE_DIR}/cmake/utils/logging.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/utils/common.cmake)

# @brief Run python command and get result, output and error
function(run_python PY_COMMAND PY_RESULT PY_OUTPUT PY_ERROR)
    set_default_values(PYTHON_EXECUTABLE "python")
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "${PY_COMMAND}"
        RESULT_VARIABLE _PY_RESULT
        OUTPUT_VARIABLE _PY_OUTPUT
        ERROR_VARIABLE _PY_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    if(NOT _PY_RESULT EQUAL 0)
        log_fatal("run_python: PY_COMMAND failed:\nCOMMAND:\n${PY_COMMAND}\nERROR:\n${_PY_ERROR}")
    endif()
    set(${PY_RESULT} "${_PY_RESULT}" PARENT_SCOPE)
    set(${PY_OUTPUT} "${_PY_OUTPUT}" PARENT_SCOPE)
    set(${PY_ERROR} "${_PY_ERROR}" PARENT_SCOPE)
endfunction()