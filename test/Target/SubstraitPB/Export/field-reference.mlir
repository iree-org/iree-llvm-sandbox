// RUN: structured-translate -substrait-to-protobuf --split-input-file %s \
// RUN: | FileCheck %s

// RUN: structured-translate -substrait-to-protobuf %s \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | structured-translate -protobuf-to-substrait \
// RUN:   --split-input-file="# -----" --output-split-marker="// ""-----" \
// RUN: | structured-translate -substrait-to-protobuf \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | FileCheck %s

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      filter {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK:             input {
// CHECK:             condition {
// CHECK-NEXT:          selection {
// CHECK-NEXT:            direct_reference {
// CHECK-NEXT:              struct_field {
// CHECK-NEXT:                field: 1
// CHECK-NEXT:                child {
// CHECK-NEXT:                  struct_field {
// CHECK:                 root_reference {
// CHECK-NEXT:            }

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

// CHECK-LABEL: relations {
// CHECK-NEXT:    rel {
// CHECK-NEXT:      filter {
// CHECK-NEXT:        common {
// CHECK-NEXT:          direct {
// CHECK:             input {
// CHECK:             condition {
// CHECK-NEXT:          selection {
// CHECK-NEXT:            direct_reference {
// CHECK-NEXT:              struct_field {
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:            expression {
// CHECK-NEXT:              selection {
// CHECK-NEXT:                direct_reference {
// CHECK-NEXT:                  struct_field {
// CHECK-NEXT:                    field: 1
// CHECK-NEXT:                  }
// CHECK-NEXT:                }
// CHECK-NEXT:                root_reference {
// CHECK-NEXT:                }
// CHECK-NEXT:              }

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
