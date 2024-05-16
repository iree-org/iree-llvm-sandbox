// RUN: structured-opt -split-input-file %s -canonicalize \
// RUN: | FileCheck %s

// Check that identiy mapping is folded.

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      %[[V0:.*]] = named_table
// CHECK-NEXT:      yield %[[V0]]

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<si1, si32>
    %1 = emit [0, 1] from %0 : tuple<si1, si32> -> tuple<si1, si32>
    yield %1 : tuple<si1, si32>
  }
}

// -----

// Check that non-identiy mapping is not folded.

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      %[[V0:.*]] = named_table
// CHECK-NEXT:      %[[V1:.*]] = emit {{.*}} from %[[V0]]
// CHECK-NEXT:      yield %[[V1]]

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<si1, si32>
    %1 = emit [1, 0] from %0 : tuple<si1, si32> -> tuple<si32, si1>
    yield %1 : tuple<si32, si1>
  }
}

// -----

// Check that identiy prefix is not folded.

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      %[[V0:.*]] = named_table
// CHECK-NEXT:      %[[V1:.*]] = emit [0] from %[[V0]]
// CHECK-NEXT:      yield %[[V1]]

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<si1, si32>
    %1 = emit [0] from %0 : tuple<si1, si32> -> tuple<si1>
    yield %1 : tuple<si1>
  }
}
