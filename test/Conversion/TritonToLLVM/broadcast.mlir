// RUN: structured-opt %s -split-input-file \
// RUN:   -convert-triton-to-llvm \
// RUN: | FileCheck %s

// Check that canonicalization properly folds subsequent `expand_shape` and
// `collapse_shape`.
// RUN: structured-opt %s -split-input-file \
// RUN:   -convert-triton-to-llvm -canonicalize \
// RUN: | FileCheck --check-prefix=CHECK-CANON %s

// CHECK-LABEL: func.func public @kernel()
// CHECK-NEXT:    %[[V0:.*]] = arith.constant {{.*}} : tensor<4xi32>
// CHECK-DAG:     %[[V1:.*]] = tensor.expand_shape %[[V0]] {{\[}}[0, 1]] : tensor<4xi32> into tensor<1x4xi32>
// CHECK-DAG:     %[[V2:.*]] = tensor.collapse_shape %[[V1]] {{\[}}[0, 1]] : tensor<1x4xi32> into tensor<4xi32>
// CHECK-DAG:     %[[V3:.*]] = tensor.empty() : tensor<4x4xi32>
// CHECK-DAG:     %[[V4:.*]] = linalg.broadcast ins(%[[V2]] : tensor<4xi32>) outs(%[[V3]] : tensor<4x4xi32>) dimensions = [0]
// CHECK-NEXT:    return %[[V4]] : tensor<4x4xi32>

// CHECK-CANON-LABEL: func.func public @kernel()
// CHECK-CANON-NEXT:    %[[V0:.*]] = arith.constant {{.*}} : tensor<4xi32>
// CHECK-CANON-DAG:     %[[V1:.*]] = tensor.empty() : tensor<4x4xi32>
// CHECK-CANON-DAG:     %[[V2:.*]] = linalg.broadcast ins(%[[V0]] : tensor<4xi32>) outs(%[[V1]] : tensor<4x4xi32>) dimensions = [0]
// CHECK-CANON-NEXT:    return %[[V2]] : tensor<4x4xi32>
func.func public @kernel() -> tensor<4x4xi32> {
  %0 = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %1 = tt.expand_dims %0 {axis = 0 : i32} : (tensor<4xi32>) -> tensor<1x4xi32>
  %2 = tt.broadcast %1 : (tensor<1x4xi32>) -> tensor<4x4xi32>
  return %2 : tensor<4x4xi32>
}

// -----

// CHECK-LABEL: func.func public @kernel()
// CHECK-NEXT:    %[[V0:.*]] = arith.constant {{.*}} : tensor<4xi32>
// CHECK-DAG:     %[[V1:.*]] = tensor.expand_shape %[[V0]] {{\[}}[0, 1]] : tensor<4xi32> into tensor<1x4xi32>
// CHECK-DAG:     %[[V2:.*]] = tensor.expand_shape %[[V1]] {{\[}}[0, 1], [2]] : tensor<1x4xi32> into tensor<1x1x4xi32>
// CHECK-DAG:     %[[V3:.*]] = tensor.collapse_shape %[[V2]] {{\[}}[0, 1], [2]] : tensor<1x1x4xi32> into tensor<1x4xi32>
// CHECK-DAG:     %[[V4:.*]] = tensor.empty() : tensor<4x1x4xi32>
// CHECK-DAG:     %[[V5:.*]] = linalg.broadcast ins(%[[V3]] : tensor<1x4xi32>) outs(%[[V4]] : tensor<4x1x4xi32>) dimensions = [0]
// CHECK-DAG:     %[[V6:.*]] = tensor.collapse_shape %[[V2]] {{\[}}[0], [1, 2]] : tensor<1x1x4xi32> into tensor<1x4xi32>
// CHECK-DAG:     %[[V7:.*]] = tensor.empty() : tensor<1x4x4xi32>
// CHECK-DAG:     %[[V8:.*]] = linalg.broadcast ins(%[[V6]] : tensor<1x4xi32>) outs(%[[V7]] : tensor<1x4x4xi32>) dimensions = [1]
// CHECK-DAG:     %[[V9:.*]] = tensor.collapse_shape %[[V2]] {{\[}}[0, 1, 2]] : tensor<1x1x4xi32> into tensor<4xi32>
// CHECK-DAG:     %[[Va:.*]] = tensor.empty() : tensor<4x4x4xi32>
// CHECK-DAG:     %[[Vb:.*]] = linalg.broadcast ins(%[[V9]] : tensor<4xi32>) outs(%[[Va]] : tensor<4x4x4xi32>) dimensions = [0, 1]
// CHECK-NEXT:    return %[[Vb]] :  tensor<4x4x4xi32>

// CHECK-CANON-LABEL: func.func public @kernel()
// CHECK-CANON-NEXT:    %[[V0:.*]] = arith.constant {{.*}} : tensor<4xi32>
// CHECK-CANON-DAG:     %[[V1:.*]] = tensor.empty() : tensor<4x4x4xi32>
// CHECK-CANON-DAG:     %[[V2:.*]] = linalg.broadcast ins(%[[V0]] : tensor<4xi32>) outs(%[[V1]] : tensor<4x4x4xi32>) dimensions = [0, 1]
// CHECK-CANON-NEXT:    return %[[V2]] :  tensor<4x4x4xi32>
func.func public @kernel() -> tensor<4x4x4xi32> {
  %0 = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %1 = tt.expand_dims %0 {axis = 0 : i32} : (tensor<4xi32>) -> tensor<1x4xi32>
  %2 = tt.expand_dims %1 {axis = 0 : i32} : (tensor<1x4xi32>) -> tensor<1x1x4xi32>
  %3 = tt.broadcast %2 : (tensor<1x1x4xi32>) -> tensor<4x1x4xi32>
  %4 = tt.broadcast %2 : (tensor<1x1x4xi32>) -> tensor<1x4x4xi32>
  %5 = tt.broadcast %2 : (tensor<1x1x4xi32>) -> tensor<4x4x4xi32>
  return %5 : tensor<4x4x4xi32>
}

// -----

// CHECK-LABEL: func.func public @kernel()
// CHECK-NEXT:    %[[V0:.*]] = arith.constant {{.*}} : tensor<4xi32>
// CHECK-DAG:     %[[V1:.*]] = tensor.expand_shape %[[V0]] {{\[}}[0, 1]] : tensor<4xi32> into tensor<1x4xi32>
// CHECK-DAG:     %[[V2:.*]] = tensor.expand_shape %[[V1]] {{\[}}[0, 1], [2]] : tensor<1x4xi32> into tensor<1x1x4xi32>
// CHECK-NEXT:    return %[[V2]] : tensor<1x1x4xi32>

// CHECK-CANON-LABEL: func.func public @kernel()
// CHECK-CANON-NEXT:    %[[V0:.*]] = arith.constant {{.*}} : tensor<1x1x4xi32>
// CHECK-CANON-NEXT:    return %[[V0]] : tensor<1x1x4xi32>
func.func public @kernel() -> tensor<1x1x4xi32> {
  %0 = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %1 = tt.expand_dims %0 {axis = 0 : i32} : (tensor<4xi32>) -> tensor<1x4xi32>
  %2 = tt.expand_dims %1 {axis = 0 : i32} : (tensor<1x4xi32>) -> tensor<1x1x4xi32>
  %3 = tt.broadcast %2 : (tensor<1x1x4xi32>) -> tensor<1x1x4xi32>
  return %3 : tensor<1x1x4xi32>
}
