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
// CHECK-NEXT:   relation {
// CHECK-NEXT:     named_table
// CHECK-NEXT:     yield
// CHECK-NEXT:   }
// CHECK-NEXT: }
substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @foo::@bar as ["a", "b"] : tuple<si32, si32>
    yield %0 : tuple<si32, si32>
  }
}

// -----

// CHECK:      substrait.plan version 0 : 42 : 1 {
// CHECK-NEXT:   relation {
// CHECK-NEXT:     named_table
// CHECK-NEXT:     yield
// CHECK-NEXT:   }
// CHECK-NEXT:   relation {
// CHECK-NEXT:     named_table
// CHECK-NEXT:     yield
// CHECK-NEXT:   }
// CHECK-NEXT: }
substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @foo::@bar as ["a", "b"] : tuple<si32, si32>
    yield %0 : tuple<si32, si32>
  }
  relation {
    %0 = named_table @foo::@bar as ["a", "b"] : tuple<si32, si32>
    yield %0 : tuple<si32, si32>
  }
}
