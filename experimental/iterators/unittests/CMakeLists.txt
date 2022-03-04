add_custom_target(IteratorsUnitTests)
set_target_properties(IteratorsUnitTests PROPERTIES FOLDER "Iterators tests")

function(add_iterators_unittest test_dirname)
  add_unittest(IteratorsUnitTests ${test_dirname} ${ARGN})
endfunction()

add_iterators_unittest(IteratorsTests
  ColumnScanOperatorTest.cpp
  FilterOperatorTest.cpp
  HashJoinOperatorTest.cpp
  MapOperatorTest.cpp
  ReduceByKeyOperatorTest.cpp
  ReduceOperatorTest.cpp
  UtilsTest.cpp
)
target_link_libraries(IteratorsTests
  PRIVATE
    Iterators
    LLVMTestingSupport
)
