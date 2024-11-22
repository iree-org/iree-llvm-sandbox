add_custom_target(structured-tools)
add_dependencies(structured-all structured-tools)

add_subdirectory(structured-lsp-server)
add_subdirectory(structured-opt)
