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

set(STRUCTURED_TEST_DEPENDS
  count
  FileCheck
  structured-opt
  mlir_async_runtime
  mlir-cpu-runner
  mlir_c_runner_utils
  mlir_runner_utils
  not
  )
list(APPEND STRUCTURED_TEST_DEPENDS StructuredPythonModules)

add_lit_testsuite(check-structured "Running the Structured regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${STRUCTURED_TEST_DEPENDS}
  )
set_target_properties(check-structured PROPERTIES FOLDER "Tests")

add_lit_testsuites(Structured ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${STRUCTURED_TEST_DEPENDS})

add_custom_target(check-structured-dependencies)
add_dependencies(check-structured-dependencies ${STRUCTURED_TEST_DEPENDS})
add_dependencies(structured-all check-structured-dependencies)
