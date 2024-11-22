include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# @breif Install target libraries
function(install_libraries LIB_COMPONENT LIB_NAMESPACE LIB_LIST)
    log_info("Libraries to install:")
    log_info("|- Component: ${LIB_COMPONENT}")
    log_info("|- Namespace: ${LIB_NAMESPACE}")
    log_info("|- Library Targets: [ ${LIB_LIST} ]")

    # Install libraries
    install(
        TARGETS ${LIB_LIST}
        COMPONENT ${LIB_COMPONENT}
        EXPORT ${LIB_COMPONENT}Targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION Lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )

    # Install headers
    install(
        DIRECTORY ${PROJECT_SOURCE_DIR}/include/
        COMPONENT ${LIB_COMPONENT}
        DESTINATION include
    )

    # Install cmake config files
    install(
        EXPORT ${LIB_COMPONENT}Targets
        COMPONENT ${LIB_COMPONENT}
        FILE ${LIB_COMPONENT}Targets.cmake
        NAMESPACE ${LIB_NAMESPACE}
        DESTINATION lib/cmake/${LIB_COMPONENT}
    )

    # Generate and install version file
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/${LIB_COMPONENT}ConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    # Generate and install config file
    configure_package_config_file(
        ${PROJECT_SOURCE_DIR}/cmake/config.cmake.in/${LIB_COMPONENT}Config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/${LIB_COMPONENT}Config.cmake
        INSTALL_DESTINATION lib/cmake/${LIB_COMPONENT}
    )

    # Install config files
    install(
        FILES
            ${CMAKE_CURRENT_BINARY_DIR}/${LIB_COMPONENT}Config.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/${LIB_COMPONENT}ConfigVersion.cmake
        COMPONENT ${LIB_COMPONENT}
        DESTINATION 
            lib/cmake/${LIB_COMPONENT}
    )
endfunction()