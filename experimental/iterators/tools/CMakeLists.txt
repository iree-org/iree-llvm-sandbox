add_custom_target(iterators-tools)
add_dependencies(iterators-all iterators-tools)

add_subdirectory(iterators-lsp-server)
add_subdirectory(iterators-opt)
