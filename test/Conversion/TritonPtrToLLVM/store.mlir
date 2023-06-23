// RUN: structured-opt %s \
// RUN:   -convert-triton-ptr-to-llvm -split-input-file \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: i32)
// CHECK-NEXT:    llvm.store %[[ARG1]], %[[ARG0]] : !llvm.ptr<i32, 1>
// CHECK-NEXT:    return
func.func public @kernel(%arg0: !tt.ptr<i32>, %arg1: i32) {
  tt.store %arg0, %arg1 {cache = 1 : i32, evict = 1 : i32} : i32
  return
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<f32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: f32)
// CHECK-NEXT:    llvm.store %[[ARG1]], %[[ARG0]] : !llvm.ptr<f32, 1>
// CHECK-NEXT:    return
func.func public @kernel(%arg0: !tt.ptr<f32>, %arg1: f32) {
  tt.store %arg0, %arg1 {cache = 1 : i32, evict = 1 : i32} : f32
  return
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<ptr<i32, 1>, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: !llvm.ptr<i32, 1>)
// CHECK-NEXT:    llvm.store %[[ARG1]], %[[ARG0]] : !llvm.ptr<ptr<i32, 1>, 1>
// CHECK-NEXT:    return
func.func public @kernel(%arg0: !tt.ptr<!tt.ptr<i32>>, %arg1: !tt.ptr<i32>) {
  tt.store %arg0, %arg1 {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<i32>
  return
}
