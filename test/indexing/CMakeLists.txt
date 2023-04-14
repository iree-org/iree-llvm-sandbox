llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

# Passed to lit.site.cfg.py.in to set up the path where to find libraries.
set(MLIR_LIB_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(INDEXING_TEST_DEPENDS
        count
        FileCheck
        indexing-opt
        mlir-cpu-runner
        mlir_c_runner_utils
        mlir_runner_utils
        not
        )
list(APPEND INDEXING_TEST_DEPENDS IndexingPythonModules)

add_lit_testsuite(check-indexing "Running the Indexing regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${INDEXING_TEST_DEPENDS}
        )
set_target_properties(check-indexing PROPERTIES FOLDER "Tests")

add_lit_testsuites(Indexing ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${INDEXING_TEST_DEPENDS})

add_custom_target(check-indexing-dependencies)
add_dependencies(check-indexing-dependencies ${INDEXING_TEST_DEPENDS})
