// RUN: structured-opt %s \
// RUN:   -convert-triton-ptr-to-llvm -split-input-file \
// RUN: | FileCheck %s

// CHECK: func.func private @f(!llvm.ptr<i32, 1>) -> !llvm.ptr<i32, 1>
// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i32, 1>) -> !llvm.ptr<i32, 1> {
// CHECK-DAG:     %[[V0:.*]] = call @f(%[[ARG0]]) : (!llvm.ptr<i32, 1>) -> !llvm.ptr<i32, 1>
// CHECK-DAG:     return %[[V0]] : !llvm.ptr<i32, 1>
func.func private @f(!tt.ptr<i32>) -> !tt.ptr<i32>
func.func public @kernel(%arg0: !tt.ptr<i32>) -> !tt.ptr<i32> {
  %0 = func.call @f(%arg0) : (!tt.ptr<i32>) -> !tt.ptr<i32>
  return %0 : !tt.ptr<i32>
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<ptr<i32, 1>, 1>) -> !llvm.ptr<ptr<i32, 1>, 1> {
// CHECK-NEXT:    return %[[ARG0]] : !llvm.ptr<ptr<i32, 1>, 1>
func.func public @kernel(%arg0: !tt.ptr<!tt.ptr<i32>>) -> !tt.ptr<!tt.ptr<i32>> {
  return %arg0 : !tt.ptr<!tt.ptr<i32>>
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: index) -> index {
// CHECK-NEXT:    return %[[ARG0]] : index
func.func public @kernel(%arg0: index) -> index {
  return %arg0 : index
}

// -----

// CHECK: func.func private @f(tensor<4x!llvm.ptr<i32, 1>>) -> tensor<4x!llvm.ptr<i32, 1>>
// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]:  tensor<4x!llvm.ptr<i32, 1>>) -> tensor<4x!llvm.ptr<i32, 1>> {
// CHECK-DAG:     %[[V0:.*]] = call @f(%[[ARG0]]) : (tensor<4x!llvm.ptr<i32, 1>>) -> tensor<4x!llvm.ptr<i32, 1>>
// CHECK-DAG:     return %[[V0]] : tensor<4x!llvm.ptr<i32, 1>>
func.func private @f(tensor<4x!tt.ptr<i32>>) -> tensor<4x!tt.ptr<i32>>
func.func public @kernel(%arg0: tensor<4x!tt.ptr<i32>>) -> tensor<4x!tt.ptr<i32>> {
  %0 = func.call @f(%arg0) : (tensor<4x!tt.ptr<i32>>) -> tensor<4x!tt.ptr<i32>>
  return %0 : tensor<4x!tt.ptr<i32>>
}

// -----

// CHECK: func.func private @f(tensor<4x!llvm.ptr<ptr<i32, 1>, 1>>) -> tensor<4x!llvm.ptr<ptr<i32, 1>, 1>>
// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]:  tensor<4x!llvm.ptr<ptr<i32, 1>, 1>>) -> tensor<4x!llvm.ptr<ptr<i32, 1>, 1>> {
// CHECK-DAG:     %[[V0:.*]] = call @f(%[[ARG0]]) : (tensor<4x!llvm.ptr<ptr<i32, 1>, 1>>) -> tensor<4x!llvm.ptr<ptr<i32, 1>, 1>>
// CHECK-DAG:     return %[[V0]] : tensor<4x!llvm.ptr<ptr<i32, 1>, 1>>
func.func private @f(tensor<4x!tt.ptr<!tt.ptr<i32>>>) -> tensor<4x!tt.ptr<!tt.ptr<i32>>>
func.func public @kernel(%arg0: tensor<4x!tt.ptr<!tt.ptr<i32>>>) -> tensor<4x!tt.ptr<!tt.ptr<i32>>> {
  %0 = func.call @f(%arg0) : (tensor<4x!tt.ptr<!tt.ptr<i32>>>) -> tensor<4x!tt.ptr<!tt.ptr<i32>>>
  return %0 : tensor<4x!tt.ptr<!tt.ptr<i32>>>
}
