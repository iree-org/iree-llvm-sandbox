// RUN: structured-opt -split-input-file %s -substrait-emit-deduplication \
// RUN: | FileCheck %s

// `cross` op with left `emit` input with duplicates.

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      %[[V0:.*]] = named_table
// CHECK-NEXT:      %[[V1:.*]] = emit [1, 0] from %[[V0]] :
// CHECK-NEXT:      %[[V2:.*]] = cross %[[V1]] x %[[V0]] :
// CHECK-NEXT:      %[[V3:.*]] = emit [0, 0, 1, 1, 0, 2, 3] from %[[V2]] :
// CHECK-NEXT:      yield %[[V3]] : tuple<si32, si32, si1, si1, si32, si1, si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<si1, si32>
    %1 = emit [1, 1, 0, 0, 1] from %0 : tuple<si1, si32> -> tuple<si32, si32, si1, si1, si32>
    %2 = cross %1 x %0 : tuple<si32, si32, si1, si1, si32> x tuple<si1, si32>
    yield %2 : tuple<si32, si32, si1, si1, si32, si1, si32>
  }
}

// -----

// `cross` op with left `emit` input without duplicates.

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      %[[V0:.*]] = named_table
// CHECK-NEXT:      %[[V1:.*]] = emit [1, 0] from %[[V0]] :
// CHECK-NEXT:      %[[V2:.*]] = cross %[[V0]] x %[[V1]] :
// CHECK-NEXT:      yield %[[V2]]

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<si1, si32>
    %1 = emit [1, 0] from %0 : tuple<si1, si32> -> tuple<si32, si1>
    %2 = cross %0 x %1 : tuple<si1, si32> x tuple<si32, si1>
    yield %2 : tuple<si1,si32, si32, si1>
  }
}

// -----

// `cross` op with right `emit` input with duplicates.

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      %[[V0:.*]] = named_table
// CHECK-NEXT:      %[[V1:.*]] = emit [1, 0] from %[[V0]] :
// CHECK-NEXT:      %[[V2:.*]] = cross %[[V0]] x %[[V1]] :
// CHECK-NEXT:      %[[V3:.*]] = emit [0, 1, 2, 2, 3, 3, 2] from %[[V2]] :
// CHECK-NEXT:      yield %[[V3]] : tuple<si1, si32, si32, si32, si1, si1, si32>

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<si1, si32>
    %1 = emit [1, 1, 0, 0, 1] from %0 : tuple<si1, si32> -> tuple<si32, si32, si1, si1, si32>
    %2 = cross %0 x %1 : tuple<si1, si32> x tuple<si32, si32, si1, si1, si32>
    yield %2 : tuple<si1, si32, si32, si32, si1, si1, si32>
  }
}

// -----

// `cross` op with right `emit` input without duplicates.

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      %[[V0:.*]] = named_table
// CHECK-NEXT:      %[[V1:.*]] = emit [1, 0] from %[[V0]] :
// CHECK-NEXT:      %[[V2:.*]] = cross %[[V1]] x %[[V0]] :
// CHECK-NEXT:      yield %[[V2]]

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<si1, si32>
    %1 = emit [1, 0] from %0 : tuple<si1, si32> -> tuple<si32, si1>
    %2 = cross %1 x %0 : tuple<si32, si1> x tuple<si1, si32>
    yield %2 : tuple<si32, si1, si1, si32>
  }
}

// -----

// `cross` op with two `emit` inputs with duplicates.

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      %[[V0:.*]] = named_table
// CHECK-DAG:       %[[V1:.*]] = emit [1] from %[[V0]] :
// CHECK-DAG:       %[[V2:.*]] = emit [0] from %[[V0]] :
// CHECK-NEXT:      %[[V3:.*]] = cross %[[V1]] x %[[V2]] :
// CHECK-NEXT:      %[[V4:.*]] = emit [0, 0, 1, 1] from %[[V3]] :
// CHECK-NEXT:      yield %[[V4]]

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<si1, si32>
    %1 = emit [1, 1] from %0 : tuple<si1, si32> -> tuple<si32, si32>
    %2 = emit [0, 0] from %0 : tuple<si1, si32> -> tuple<si1, si1>
    %3 = cross %1 x %2 : tuple<si32, si32> x tuple<si1, si1>
    yield %3 : tuple<si32, si32, si1, si1>
  }
}

// -----

// `cross` op with mixed `emit` duplicates/no duplicates inputs.

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      %[[V0:.*]] = named_table
// CHECK-DAG:       %[[V1:.*]] = emit [1, 0] from %[[V0]] :
// CHECK-DAG:       %[[V2:.*]] = emit [0] from %[[V0]] :
// CHECK-NEXT:      %[[V3:.*]] = cross %[[V1]] x %[[V2]] :
// CHECK-NEXT:      %[[V4:.*]] = emit [0, 1, 2, 2] from %[[V3]] :
// CHECK-NEXT:      yield %[[V4]]

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<si1, si32>
    %1 = emit [1, 0] from %0 : tuple<si1, si32> -> tuple<si32, si1>
    %2 = emit [0, 0] from %0 : tuple<si1, si32> -> tuple<si1, si1>
    %3 = cross %1 x %2 : tuple<si32, si1> x tuple<si1, si1>
    yield %3 : tuple<si32, si1, si1, si1>
  }
}

// -----

// `cross` op with mixed `emit` duplicates/no duplicates inputs.

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      %[[V0:.*]] = named_table
// CHECK-DAG:       %[[V1:.*]] = emit [1, 0] from %[[V0]] :
// CHECK-DAG:       %[[V2:.*]] = emit [1] from %[[V0]] :
// CHECK-NEXT:      %[[V3:.*]] = cross %[[V2]] x %[[V1]] :
// CHECK-NEXT:      %[[V4:.*]] = emit [0, 0, 1, 2] from %[[V3]] :
// CHECK-NEXT:      yield %[[V4]]

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<si1, si32>
    %1 = emit [1, 1] from %0 : tuple<si1, si32> -> tuple<si32, si32>
    %2 = emit [1, 0] from %0 : tuple<si1, si32> -> tuple<si32, si1>
    %3 = cross %1 x %2 : tuple<si32, si32> x tuple<si32, si1>
    yield %3 : tuple<si32, si32, si32, si1>
  }
}

// -----

// `filter` op (`PushDuplicatesThroughFilterPattern`).

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      %[[V0:.*]] = named_table
// CHECK-NEXT:      %[[V1:.*]] = emit [1, 2, 0] from %[[V0]] :
// CHECK-NEXT:      %[[V2:.*]] = filter %[[V1]] : {{.*}} {
// CHECK-NEXT:      ^{{.*}}(%[[ARG0:.*]]: [[TYPE:.*]]):
// CHECK-NEXT:        %[[V3:.*]] = field_reference %[[ARG0]]{{\[}}[0]] : [[TYPE]]
// CHECK-NEXT:        %[[V5:.*]] = field_reference %[[ARG0]]{{\[}}[1, 0]] : [[TYPE]]
// CHECK-NEXT:        %[[V6:.*]] = field_reference %[[ARG0]]{{\[}}[1]] : [[TYPE]]
// CHECK-NEXT:        %[[V7:.*]] = field_reference %[[V6]]{{\[}}[1]] :
// CHECK-NEXT:        %[[V9:.*]] = field_reference %[[ARG0]]{{\[}}[2]] : [[TYPE]]
// CHECK-NEXT:        %[[Va:.*]] = func.call @f(%[[V3]], %[[V3]], %[[V5]], %[[V7]], %[[V3]], %[[V9]])
// CHECK-NEXT:        yield %[[Va]] : si1
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[Vb:.*]] = emit [0, 0, 1, 0, 2] from %[[V2]]

func.func private @f(si1, si1, si1, si32, si1, si1) -> si1

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b", "c", "d", "e"] : tuple<si1, si1, tuple<si1, si32>>
    // Fields in position 1 and 3 are duplicates of field in position 0, so we
    // expect all references to the former to be replaced by the latter and an
    // `emit` re-establishing the original fields after the `filter`.
    %1 = emit [1, 1, 2, 1, 0] from %0
        : tuple<si1, si1, tuple<si1, si32>> -> tuple<si1, si1, tuple<si1, si32>, si1, si1>
    %2 = filter %1 : tuple<si1, si1, tuple<si1, si32>, si1, si1> {
    ^bb0(%arg0: tuple<si1, si1, tuple<si1, si32>, si1, si1>):
      %3 = field_reference %arg0[[0]] : tuple<si1, si1, tuple<si1, si32>, si1, si1>
      %4 = field_reference %arg0[[1]] : tuple<si1, si1, tuple<si1, si32>, si1, si1>
      %5 = field_reference %arg0[[2, 0]] : tuple<si1, si1, tuple<si1, si32>, si1, si1>
      %6 = field_reference %arg0[[2]] : tuple<si1, si1, tuple<si1, si32>, si1, si1>
      %7 = field_reference %6[[1]] : tuple<si1, si32>
      %8 = field_reference %arg0[[3]] : tuple<si1, si1, tuple<si1, si32>, si1, si1>
      %9 = field_reference %arg0[[4]] : tuple<si1, si1, tuple<si1, si32>, si1, si1>
      %a = func.call @f(%3, %4, %5, %7, %8, %9) : (si1, si1, si1, si32, si1, si1) -> si1
      yield %a : si1
    }
    yield %2 : tuple<si1, si1, tuple<si1, si32>, si1, si1>
  }
}

// -----

// `project` op (`PushDuplicatesThroughProjectPattern`).

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      %[[V0:.*]] = named_table
// CHECK-NEXT:      %[[V1:.*]] = emit [1] from %[[V0]] :
// CHECK-NEXT:      %[[V2:.*]] = project %[[V1]] : tuple<si32> -> tuple<si32, si1> {
// CHECK-NEXT:      ^{{.*}}(%[[ARG0:.*]]: [[TYPE:.*]]):
// CHECK-NEXT:        %[[V3:.*]] = field_reference %[[ARG0]]{{\[}}[0]] : [[TYPE]]
// CHECK-NEXT:        %[[V5:.*]] = func.call @f(%[[V3]], %[[V3]]) :
// CHECK-NEXT:        yield %[[V5]] : si1
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[V6:.*]] = emit [0, 0, 1] from %[[V2]]

func.func private @f(si32, si32) -> si1

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<si1, si32>
    %1 = emit [1, 1] from %0 : tuple<si1, si32> -> tuple<si32, si32>
    %2 = project %1 : tuple<si32, si32> -> tuple<si32, si32, si1> {
    ^bb0(%arg : tuple<si32, si32>):
      %3 = field_reference %arg[[0]] : tuple<si32, si32>
      %4 = field_reference %arg[[1]] : tuple<si32, si32>
      %5 = func.call @f(%3, %4) : (si32, si32) -> si1
      yield %5 : si1
    }
    yield %2 : tuple<si32, si32, si1>
  }
}

// -----

// `project` op (`EliminateDuplicateYieldsInProjectPattern`).

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      %[[V0:.*]] = named_table
// CHECK-NEXT:      %[[V1:.*]] = project %[[V0]] : {{.*}} {
// CHECK-NEXT:      ^{{.*}}(%[[ARG0:.*]]: [[TYPE:.*]]):
// CHECK-NEXT:        %[[V2:.*]] = field_reference %[[ARG0]]{{\[}}[0]] : [[TYPE]]
// CHECK-NEXT:        %[[V3:.*]] = func.call @f(%[[V2]]) :
// CHECK-NEXT:        yield %[[V3]] : si1
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[V4:.*]] = emit [0, 1, 1] from %[[V1]]

func.func private @f(si32) -> si1

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = project %0 : tuple<si32> -> tuple<si32, si1, si1> {
    ^bb0(%arg : tuple<si32>):
      %2 = field_reference %arg[[0]] : tuple<si32>
      %3 = func.call @f(%2) : (si32) -> si1
      // We yield two times the same value. This pattern should remove one of
      // the two and re-establish the duplicate with an `amit` after the
      // `project`.
      yield %3, %3 : si1, si1
    }
    yield %1 : tuple<si32, si1, si1>
  }
}

// -----

// `project` op (`EliminateIdentityYieldsInProjectPattern`).

// CHECK-LABEL: substrait.plan
// CHECK-NEXT:    relation
// CHECK-NEXT:      %[[V0:.*]] = named_table
// CHECK-NEXT:      %[[V1:.*]] = project %[[V0]] : {{.*}} {
// CHECK-NEXT:      ^{{.*}}(%[[ARG0:.*]]: [[TYPE:.*]]):
// CHECK-NEXT:        %[[V2:.*]] = field_reference %[[ARG0]]{{\[}}[0]] : [[TYPE]]
// CHECK-NEXT:        %[[V3:.*]] = func.call @f(%[[V2]]) :
// CHECK-NEXT:        yield %[[V3]] : si1
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[V4:.*]] = emit [0, 1, 0, 2] from %[[V1]]

func.func private @f(si32) -> si1

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<si32, si1>
    %1 = project %0 : tuple<si32, si1> -> tuple<si32, si1, si32, si1> {
    ^bb0(%arg0: tuple<si32, si1>):
      %2 = field_reference %arg0[[0]] : tuple<si32, si1>
      %3 = func.call @f(%2) : (si32) -> si1
      // `%2` yields an input field without modifications. This pattern removes
      // that yielding and re-establishes the duplicated field with an `emit`
      // following the `project` instead.
      yield %2, %3 : si32, si1
    }
    yield %1 : tuple<si32, si1, si32, si1>
  }
}
