// RUN: structured-translate -substrait-to-protobuf --split-input-file %s \
// RUN: | FileCheck %s

// RUN: structured-translate -substrait-to-protobuf %s \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | structured-translate -protobuf-to-substrait \
// RUN:   --split-input-file="# -----" --output-split-marker="// ""-----" \
// RUN: | structured-translate -substrait-to-protobuf \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | FileCheck %s

// CHECK-LABEL: version {
// CHECK-DAG:     minor_number: 42
// CHECK-DAG:     patch_number: 1
// CHECK-DAG:     git_hash: "hash"
// CHECK-DAG:     producer: "producer"
// CHECK-NEXT:  }
substrait.plan
  version 0 : 42 : 1
  git_hash "hash"
  producer "producer"
  {}

// -----

// CHECK:      relations {
// CHECK-NEXT:   root {
// CHECK-NEXT:     input {
// CHECK-NEXT:       read {
// CHECK:              named_table {
// CHECK-NEXT:           names
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     names: "x"
// CHECK-NEXT:     names: "y"
// CHECK-NEXT:     names: "z"
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: version

substrait.plan version 0 : 42 : 1 {
  relation as ["x", "y", "z"] {
    %0 = named_table @t as ["a", "b", "c"] : tuple<si32, tuple<si32>>
    yield %0 : tuple<si32, tuple<si32>>
  }
}
