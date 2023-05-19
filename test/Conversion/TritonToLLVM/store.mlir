// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm \
// RUN: | FileCheck %s

// CHECK-LABEL: llvm.func @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: i32)
// CHECK-NEXT:    llvm.store %[[ARG1]], %[[ARG0]] : !llvm.ptr<i32, 1>
// CHECK-NEXT:    llvm.return
tt.func public @kernel(%arg0: !tt.ptr<i32>, %arg1: i32) {
  tt.store %arg0, %arg1 {cache = 1 : i32, evict = 1 : i32} : i32
  tt.return
}
