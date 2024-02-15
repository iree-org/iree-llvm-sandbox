// RUN: structured-translate -substrait-to-protobuf %s \
// RUN: | FileCheck %s

// RUN: structured-translate -substrait-to-protobuf %s \
// RUN: | structured-translate -protobuf-to-substrait \
// RUN: | structured-translate -substrait-to-protobuf \
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
