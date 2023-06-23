// RUN: structured-opt %s \
// RUN:   -convert-triton-ptr-to-llvm -split-input-file \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: i32)
// CHECK-NEXT:    %[[V0:.*]] = llvm.getelementptr %[[ARG0]][%[[ARG1]]] : (!llvm.ptr<i32, 1>, i32) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:    return
func.func public @kernel(%arg0: !tt.ptr<i32>, %arg1: i32) {
  %0 = tt.addptr %arg0, %arg1 : !tt.ptr<i32>, i32
  return
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<f32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: i32)
// CHECK-NEXT:    %[[V0:.*]] = llvm.getelementptr %[[ARG0]][%[[ARG1]]] : (!llvm.ptr<f32, 1>, i32) -> !llvm.ptr<f32, 1>
// CHECK-NEXT:    return
func.func public @kernel(%arg0: !tt.ptr<f32>, %arg1: i32) {
  %0 = tt.addptr %arg0, %arg1 : !tt.ptr<f32>, i32
  return
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<ptr<f32, 1>, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: i32)
// CHECK-NEXT:    %[[V0:.*]] = llvm.getelementptr %[[ARG0]][%[[ARG1]]] : (!llvm.ptr<ptr<f32, 1>, 1>, i32) -> !llvm.ptr<ptr<f32, 1>, 1>
// CHECK-NEXT:    return
func.func public @kernel(%arg0: !tt.ptr<!tt.ptr<f32>>, %arg1: i32) {
  %0 = tt.addptr %arg0, %arg1 : !tt.ptr<!tt.ptr<f32>>, i32
  return
}
