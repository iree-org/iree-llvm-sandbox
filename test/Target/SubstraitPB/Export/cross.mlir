// RUN: structured-translate -substrait-to-protobuf %s \
// RUN: | FileCheck %s

// RUN: structured-translate -substrait-to-protobuf %s \
// RUN: | structured-translate -protobuf-to-substrait \
// RUN: | structured-translate -substrait-to-protobuf \
// RUN: | FileCheck %s

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      cross {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        left {
// CHECK-NEXT:          read {
// CHECK:             right {
// CHECK-NEXT:          read {

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = named_table @t2 as ["b"] : tuple<si32>
    %2 = cross %0 x %1 : tuple<si32> x tuple<si32>
    yield %2 : tuple<si32, si32>
  }
}
