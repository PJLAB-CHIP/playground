include(${CMAKE_CURRENT_LIST_DIR}/logging.cmake)

# @func  set_default_values(<var1> <value1> <var2> <value2> ...)
# @brief Set default values for variables;
#        The variable will not be overwrite if it already has a value.
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

# @func try_get_value(<var> HINTS <hint1> <hint2> ...)
# @brief Try to assign value to <var> from hints or env:hints.
#        $<var>_FOUND will be set to true if one of the hint found; Otherwise false.
function(try_get_value VAR)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "HINTS")
    
    # First try each hint as a CMake variable
    foreach(hint IN LISTS ARG_HINTS)
        if(DEFINED ${hint})
            set(${VAR} "${${hint}}" PARENT_SCOPE)
            set(${VAR}_FOUND TRUE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    
    # Then try each hint as an environment variable
    foreach(hint IN LISTS ARG_HINTS)
        if(DEFINED ENV{${hint}})
            set(${VAR} "$ENV{${hint}}" PARENT_SCOPE)
            set(${VAR}_FOUND TRUE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    
    # If nothing found
    set(${VAR} "" PARENT_SCOPE)
    set(${VAR}_FOUND FALSE PARENT_SCOPE)
endfunction()