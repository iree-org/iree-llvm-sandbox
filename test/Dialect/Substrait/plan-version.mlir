// RUN: structured-opt -split-input-file %s \
// RUN: | FileCheck %s

// CHECK:      substrait.plan version 0 : 42 : 1
// CHECK-SAME:   git_hash "hash" producer "producer" {
// CHECK-NEXT: }
substrait.plan
  version 0 : 42 : 1
  git_hash "hash"
  producer "producer"
  {}

// -----

// CHECK:      substrait.plan version 0 : 42 : 1 {
// CHECK-NEXT:   relation
// CHECK-NEXT: }
substrait.plan version 0 : 42 : 1 {
  relation
}

// -----

// CHECK:      substrait.plan version 0 : 42 : 1 {
// CHECK-NEXT:   relation
// CHECK-NEXT:   relation
// CHECK-NEXT: }
substrait.plan version 0 : 42 : 1 {
  relation
  relation
}
