include(${CMAKE_CURRENT_LIST_DIR}/logging.cmake)

# Set default values for variables;
# User can override these values before or after calling this function
function(set_default_values)
    math(EXPR ARG_COUNT "${ARGC} % 2")
    if(NOT ${ARG_COUNT} EQUAL 0)
        log_error("`set_default_values()` requires pairs of VAR_NAME and DEFAULT_VALUE")
    endif()

    math(EXPR LAST_INDEX "${ARGC} - 1")
    foreach(IDX RANGE 0 ${LAST_INDEX} 2)
        math(EXPR VALUE_IDX "${IDX} + 1")
        list(GET ARGV ${IDX} VAR_NAME)
        list(GET ARGV ${VALUE_IDX} DEFAULT_VALUE)
        
        if(NOT DEFINED ${VAR_NAME})
            set(${VAR_NAME} ${DEFAULT_VALUE} PARENT_SCOPE)
        endif()
    endforeach()
endfunction()