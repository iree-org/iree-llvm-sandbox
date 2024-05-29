// RUN: structured-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      named_table
// CHECK-NEXT:      filter
// CHECK-NEXT:      (%[[ARG0:.*]]: tuple<si1>):
// CHECK-NEXT:        %[[V0:.*]] = field_reference %[[ARG0]]{{\[}}[0]] : tuple<si1>
// CHECK-NEXT:        yield %[[V0]] : si1

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si1>
    %1 = filter %0 : tuple<si1> {
    ^bb0(%arg : tuple<si1>):
      %2 = field_reference %arg[[0]] : tuple<si1>
      yield %2 : si1
    }
    yield %1 : tuple<si1>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      named_table
// CHECK-NEXT:      filter
// CHECK-NEXT:      (%[[ARG0:.*]]: tuple<si1, tuple<si1>>):
// CHECK-NEXT:        %[[V0:.*]] = field_reference %[[ARG0]]{{\[}}[1, 0]] : tuple<si1, tuple<si1>>
// CHECK-NEXT:        yield %[[V0]] : si1

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b", "c"] : tuple<si1, tuple<si1>>
    %1 = filter %0 : tuple<si1, tuple<si1>> {
    ^bb0(%arg : tuple<si1, tuple<si1>>):
      %2 = field_reference %arg[[1, 0]] : tuple<si1, tuple<si1>>
      yield %2 : si1
    }
    yield %1 : tuple<si1, tuple<si1>>
  }
}

// -----

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      named_table
// CHECK-NEXT:      filter
// CHECK-NEXT:      (%[[ARG0:.*]]: tuple<si1, tuple<si1>>):
// CHECK-NEXT:        %[[V0:.*]] = field_reference %[[ARG0]]{{\[}}[1]] : tuple<si1, tuple<si1>>
// CHECK-NEXT:        %[[V1:.*]] = field_reference %[[V0]]{{\[}}[0]] : tuple<si1>
// CHECK-NEXT:        yield %[[V1]] : si1

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b", "c"] : tuple<si1, tuple<si1>>
    %1 = filter %0 : tuple<si1, tuple<si1>> {
    ^bb0(%arg : tuple<si1, tuple<si1>>):
      %2 = field_reference %arg[[1]] : tuple<si1, tuple<si1>>
      %3 = field_reference %2[[0]] : tuple<si1>
      yield %3 : si1
    }
    yield %1 : tuple<si1, tuple<si1>>
  }
}
