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

set(ITERATORS_TEST_DEPENDS
  count
  FileCheck
  iterators-opt
  mlir-cpu-runner
  mlir_c_runner_utils
  mlir_runner_utils
  not
  )
list(APPEND ITERATORS_TEST_DEPENDS IteratorsPythonModules)

add_lit_testsuite(check-iterators "Running the Iterators regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${ITERATORS_TEST_DEPENDS}
  )
set_target_properties(check-iterators PROPERTIES FOLDER "Tests")

add_lit_testsuites(Iterators ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${ITERATORS_TEST_DEPENDS})

add_custom_target(check-iterators-dependencies)
add_dependencies(check-iterators-dependencies ${ITERATORS_TEST_DEPENDS})
add_dependencies(iterators-all check-iterators-dependencies)
