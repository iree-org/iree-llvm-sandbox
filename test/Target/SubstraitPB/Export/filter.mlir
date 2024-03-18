// RUN: structured-translate -substrait-to-protobuf %s \
// RUN: | FileCheck %s

// RUN: structured-translate -substrait-to-protobuf %s \
// RUN: | structured-translate -protobuf-to-substrait \
// RUN: | structured-translate -substrait-to-protobuf \
// RUN: | FileCheck %s

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      filter {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK:             input {
// CHECK:             condition {
// CHECK-NEXT:          literal {
// CHECK-NEXT:            boolean: true

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
