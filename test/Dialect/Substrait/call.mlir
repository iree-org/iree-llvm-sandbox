// RUN: structured-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK-LABEL: substrait.plan
// CHECK:         relation
// CHECK:         named_table
// CHECK-NEXT:    filter
// CHECK-NEXT:    (%[[ARG0:.*]]: tuple<si32>)
// CHECK-NEXT:      %[[V0:.*]] = field_reference %[[ARG0]]
// CHECK-NEXT:      %[[V1:.*]] = call @function(%[[V0]]) : (si32) -> si1
// CHECK-NEXT:      yield
// CHECK-NEXT:    }

substrait.plan version 0 : 42 : 1 {
  extension_uri @extension at "http://some.url/with/extensions.yml"
  extension_function @function at @extension["somefunc"]
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = filter %0 : tuple<si32> {
    ^bb0(%arg : tuple<si32>):
      %2 = field_reference %arg[0] : tuple<si32>
      %3 = call @function(%2) : (si32) -> si1
      yield %3 : si1
    }
    yield %1 : tuple<si32>
  }
}
