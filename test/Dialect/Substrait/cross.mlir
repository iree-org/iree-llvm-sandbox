// RUN: structured-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:           %[[V0:.*]] = named_table
// CHECK:           %[[V1:.*]] = named_table
// CHECK-NEXT:      %[[V2:.*]] = cross %[[V0]] x %[[V1]]
// CHECK-SAME:        : tuple<si32> x tuple<si1>
// CHECK-NEXT:      yield %[[V2]] : tuple<si32, si1>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si1>
    %2 = cross %0 x %1 : tuple<si32> x tuple<si1>
    yield %2 : tuple<si32, si1>
  }
}
