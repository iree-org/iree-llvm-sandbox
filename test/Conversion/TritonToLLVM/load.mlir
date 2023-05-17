// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm \
// RUN: | FileCheck %s

// CHECK-LABEL: llvm.func @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i32, 1>)
// CHECK-NEXT:    llvm.load %[[ARG0]] : !llvm.ptr<i32, 1>
// CHECK-NEXT:    llvm.return
tt.func public @kernel(%arg0: !tt.ptr<i32>) {
  %0 = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : i32
  tt.return
}