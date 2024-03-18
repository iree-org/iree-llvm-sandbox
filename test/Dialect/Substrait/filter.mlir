// RUN: structured-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         %[[V0:.*]] = named_table
// CHECK-NEXT:    %[[V1:.*]] = filter %[[V0]] : tuple<si32> {
// CHECK-NEXT:    ^[[BB0:.*]](%[[ARG0:.*]]: tuple<si32>):
// CHECK-NEXT:      %[[V2:.*]] = literal -1 : si1
// CHECK-NEXT:      yield %[[V2]] : si1
// CHECK-NEXT:    }
// CHECK-NEXT:    yield %[[V1]] :

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = filter %0 : tuple<si32> {
    ^bb0(%arg : tuple<si32>):
      %2 = literal -1 : si1
      yield %2 : si1
    }
    yield %1 : tuple<si32>
  }
}
