// RUN: structured-opt %s -split-input-file \
// RUN:   -convert-triton-to-llvm \
// RUN: | FileCheck %s

// Check that canonicalization properly folds subsequent `expand_shape` and
// `collapse_shape`.
// RUN: structured-opt %s -split-input-file \
// RUN:   -convert-triton-to-llvm -canonicalize \
// RUN: | FileCheck --check-prefix=CHECK-CANON %s

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x4xi32>) -> tensor<4x4xi32> {
// CHECK-DAG:     %[[V0:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]] : tensor<1x4xi32> into tensor<4xi32>
// CHECK-DAG:     %[[V1:.*]] = tensor.empty() : tensor<4x4xi32>
// CHECK-DAG:     %[[V2:.*]] = linalg.broadcast ins(%[[V0]] : tensor<4xi32>) outs(%[[V1]] : tensor<4x4xi32>) dimensions = [0]
// CHECK-NEXT:    return %[[V2]] : tensor<4x4xi32>
func.func public @kernel(%arg0: tensor<1x4xi32>) -> tensor<4x4xi32> {
  %0 = tt.broadcast %arg0 : (tensor<1x4xi32>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<4x1xi32>) -> tensor<4x4xi32> {
// CHECK-DAG:     %[[V0:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]] : tensor<4x1xi32> into tensor<4xi32>
// CHECK-DAG:     %[[V1:.*]] = tensor.empty() : tensor<4x4xi32>
// CHECK-DAG:     %[[V2:.*]] = linalg.broadcast ins(%[[V0]] : tensor<4xi32>) outs(%[[V1]] : tensor<4x4xi32>) dimensions = [1]
// CHECK-NEXT:    return %[[V2]] : tensor<4x4xi32>
func.func public @kernel(%arg0: tensor<4x1xi32>) -> tensor<4x4xi32> {
  %0 = tt.broadcast %arg0 : (tensor<4x1xi32>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x1x4xi32>) -> tensor<4x1x4xi32> {
// CHECK-DAG:     %[[V0:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1], [2]] : tensor<1x1x4xi32> into tensor<1x4xi32>
// CHECK-DAG:     %[[V1:.*]] = tensor.empty() : tensor<4x1x4xi32>
// CHECK-DAG:     %[[V2:.*]] = linalg.broadcast ins(%[[V0]] : tensor<1x4xi32>) outs(%[[V1]] : tensor<4x1x4xi32>) dimensions = [0]
// CHECK-NEXT:    return %[[V2]] : tensor<4x1x4xi32>
func.func public @kernel(%arg0: tensor<1x1x4xi32>) -> tensor<4x1x4xi32> {
  %0 = tt.broadcast %arg0 : (tensor<1x1x4xi32>) -> tensor<4x1x4xi32>
  return %0 : tensor<4x1x4xi32>
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x1x4xi32>) -> tensor<1x4x4xi32> {
// CHECK-DAG:     %[[V0:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0], [1, 2]] : tensor<1x1x4xi32> into tensor<1x4xi32>
// CHECK-DAG:     %[[V1:.*]] = tensor.empty() : tensor<1x4x4xi32>
// CHECK-DAG:     %[[V2:.*]] = linalg.broadcast ins(%[[V0]] : tensor<1x4xi32>) outs(%[[V1]] : tensor<1x4x4xi32>) dimensions = [1]
// CHECK-NEXT:    return %[[V2]] : tensor<1x4x4xi32>
func.func public @kernel(%arg0: tensor<1x1x4xi32>) -> tensor<1x4x4xi32> {
  %0 = tt.broadcast %arg0 : (tensor<1x1x4xi32>) -> tensor<1x4x4xi32>
  return %0 : tensor<1x4x4xi32>
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x1x4xi32>) -> tensor<4x4x4xi32> {
// CHECK-DAG:     %[[V0:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2]] : tensor<1x1x4xi32> into tensor<4xi32>
// CHECK-DAG:     %[[V1:.*]] = tensor.empty() : tensor<4x4x4xi32>
// CHECK-DAG:     %[[V2:.*]] = linalg.broadcast ins(%[[V0]] : tensor<4xi32>) outs(%[[V1]] : tensor<4x4x4xi32>) dimensions = [0, 1]
// CHECK-NEXT:    return %[[V2]] : tensor<4x4x4xi32>
func.func public @kernel(%arg0: tensor<1x1x4xi32>) -> tensor<4x4x4xi32> {
  %0 = tt.broadcast %arg0 : (tensor<1x1x4xi32>) -> tensor<4x4x4xi32>
  return %0 : tensor<4x4x4xi32>
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x1x4xi32>) -> tensor<1x1x4xi32> {
// CHECK-NEXT:    return %[[ARG0]] : tensor<1x1x4xi32>
func.func public @kernel(%arg0: tensor<1x1x4xi32>) -> tensor<1x1x4xi32> {
  %0 = tt.broadcast %arg0 : (tensor<1x1x4xi32>) -> tensor<1x1x4xi32>
  return %0 : tensor<1x1x4xi32>
}

// -----

// CHECK-CANON-LABEL: func.func public @canonicalize(
// CHECK-CANON-SAME:      %[[ARG0:.*]]: tensor<4xi32>) -> tensor<4x4xi32> {
// CHECK-CANON-DAG:     %[[V1:.*]] = tensor.empty() : tensor<4x4xi32>
// CHECK-CANON-DAG:     %[[V2:.*]] = linalg.broadcast ins(%[[ARG0]] : tensor<4xi32>) outs(%[[V1]] : tensor<4x4xi32>) dimensions = [0]
// CHECK-CANON-NEXT:    return %[[V2]] : tensor<4x4xi32>
func.func public @canonicalize(%arg0: tensor<4xi32>) -> tensor<4x4xi32> {
  %0 = tt.expand_dims %arg0 {axis = 0 : i32} : (tensor<4xi32>) -> tensor<1x4xi32>
  %1 = tt.broadcast %0 : (tensor<1x4xi32>) -> tensor<4x4xi32>
  return %1 : tensor<4x4xi32>
}

// -----

// CHECK-CANON-LABEL: func.func public @canonicalize(
// CHECK-CANON-SAME:      %[[ARG0:.*]]: tensor<4xi32>) -> tensor<4x4x4xi32> {
// CHECK-CANON-DAG:     %[[V1:.*]] = tensor.empty() : tensor<4x4x4xi32>
// CHECK-CANON-DAG:     %[[V2:.*]] = linalg.broadcast ins(%[[ARG0]] : tensor<4xi32>) outs(%[[V1]] : tensor<4x4x4xi32>) dimensions = [0, 1]
// CHECK-CANON-NEXT:    return %[[V2]] :  tensor<4x4x4xi32>
func.func public @canonicalize(%arg0: tensor<4xi32>) -> tensor<4x4x4xi32> {
  %0 = tt.expand_dims %arg0 {axis = 0 : i32} : (tensor<4xi32>) -> tensor<1x4xi32>
  %1 = tt.expand_dims %0 {axis = 0 : i32} : (tensor<1x4xi32>) -> tensor<1x1x4xi32>
  %2 = tt.broadcast %1 : (tensor<1x1x4xi32>) -> tensor<4x4x4xi32>
  return %2 : tensor<4x4x4xi32>
}

// -----

// CHECK-CANON-LABEL: func.func public @canonicalize(
// CHECK-CANON-SAME:      %[[ARG0:.*]]: tensor<4xi32>) -> tensor<1x1x4xi32> {
// CHECK-CANON-DAG:     %[[V0:.*]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1, 2]] : tensor<4xi32> into tensor<1x1x4xi32>
// CHECK-CANON-NEXT:    return %[[V0]] : tensor<1x1x4xi32>
func.func public @canonicalize(%arg0: tensor<4xi32>) -> tensor<1x1x4xi32> {
  %0 = tt.expand_dims %arg0 {axis = 0 : i32} : (tensor<4xi32>) -> tensor<1x4xi32>
  %1 = tt.expand_dims %0 {axis = 0 : i32} : (tensor<1x4xi32>) -> tensor<1x1x4xi32>
  %2 = tt.broadcast %1 : (tensor<1x1x4xi32>) -> tensor<1x1x4xi32>
  return %2 : tensor<1x1x4xi32>
}
