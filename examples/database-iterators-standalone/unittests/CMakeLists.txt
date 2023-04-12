include(GoogleTest)

# Googletest-based unit tests for Operator implementations.
add_executable(DatabaseIteratorsTests
  ColumnScanOperatorTest.cpp
  FilterOperatorTest.cpp
  HashJoinOperatorTest.cpp
  MapOperatorTest.cpp
  ReduceByKeyOperatorTest.cpp
  ReduceOperatorTest.cpp
  UtilsTest.cpp
)
target_link_libraries(DatabaseIteratorsTests
  PUBLIC
  GTest::gtest_main
  PRIVATE
  DatabaseIterators
)
gtest_discover_tests(DatabaseIteratorsTests)

# Set up lit.
set(LITCFG_IN ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.in.py)
set(LITCFG_OUT ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py)

configure_file(${LITCFG_IN} ${LITCFG_OUT} @ONLY)
