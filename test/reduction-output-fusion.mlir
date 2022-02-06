// RUN: mlir-proto-opt --split-input-file %s\
// RUN: -linalg-fuse-fill-into-reduction="anchor-func=reduce anchor-op=linalg.generic" \
// RUN: -canonicalize -cse |\
// RUN: FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func @reduce(%in: tensor<8x16xf32>) -> tensor<8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %0 = linalg.init_tensor [8] : tensor<8xf32>
  %fill = linalg.fill(%cst, %0) : f32, tensor<8xf32> -> tensor<8xf32>
  %2 = scf.for %i = %c0 to %c8 step %c4
      iter_args (%fill0_ = %fill) -> (tensor<8xf32>) {
    %out = scf.for %j = %c0 to %c16 step %c2
           iter_args (%fill_ = %fill0_) -> (tensor<8xf32>) {
      %3 = tensor.extract_slice %in[%i, %j] [4, 2] [1, 1]
        : tensor<8x16xf32> to tensor<4x2xf32>
      %4 = tensor.extract_slice %fill_[%i] [4] [1]
        : tensor<8xf32> to tensor<4xf32>
      %5 = linalg.generic {
             indexing_maps = [#map0, #map1],
             iterator_types = ["parallel", "reduction"]}
             ins(%3 : tensor<4x2xf32>)
             outs(%4 : tensor<4xf32>) {
      ^bb0(%arg5: f32, %arg6: f32):
        %7 = arith.addf %arg5, %arg6 : f32
        linalg.yield %7 : f32
      } -> tensor<4xf32>
      %6 = tensor.insert_slice %5 into %fill_[%i] [4] [1]
        : tensor<4xf32> into tensor<8xf32>
      scf.yield %6 : tensor<8xf32>
    }
    scf.yield %out : tensor<8xf32>
  }
  return %2 : tensor<8xf32>
}

// CHECK-LABEL: #map0 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK:       #map1 = affine_map<(d0, d1) -> (d0)>
// CHECK:       #map2 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @reduce(
// CHECK-SAME:    %[[INPUT:.*]]: tensor<8x16xf32>) -> tensor<8xf32> {

// CHECK-DAG: %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index

// CHECK: %[[INIT8:.*]] = linalg.init_tensor [8] : tensor<8xf32>
// CHECK: %[[INIT4:.*]] = linalg.init_tensor [4] : tensor<4xf32>
// CHECK: %[[FILL8:.*]] = linalg.fill(%[[C0_F32]], %[[INIT8]]) : f32, tensor<8xf32> -> tensor<8xf32>

// CHECK: %[[OUTER:.*]]:2 = scf.for %[[I:.*]] = %[[C0]] to %[[C8]] step %[[C4]]
// CHECK-SAME: iter_args(%[[FILL8_:.*]] = %[[FILL8]],
// CHECK-SAME: %[[INIT4_:.*]] = %[[INIT4]]) -> (tensor<8xf32>, tensor<4xf32>) {

// CHECK:   %[[INNER:.*]]:2 = scf.for %[[J:.*]] = %[[C0]] to %[[C16]] step %[[C2]]
// CHECK-SAME: iter_args(%[[FILL8_BB:.*]] = %[[FILL8_]],
// CHECK-SAME: %[[INIT4_BB:.*]] = %[[INIT4_]]) -> (tensor<8xf32>, tensor<4xf32>) {

// CHECK:     %[[IN_SUB:.*]] = tensor.extract_slice %[[INPUT]][%[[I]], %[[J]]] [4, 2] [1, 1] : tensor<8x16xf32> to tensor<4x2xf32>
// CHECK:     %[[FILL8_SUB:.*]] = tensor.extract_slice %[[FILL8_BB]][%[[I]]] [4] [1] : tensor<8xf32> to tensor<4xf32>
// CHECK:     %[[FILL4:.*]] = linalg.fill(%[[C0_F32]], %[[INIT4_BB]]) : f32, tensor<4xf32> -> tensor<4xf32>

// CHECK:     %[[REDUCE_TILE:.*]] = linalg.generic {
// CHECK-SAME:  indexing_maps = [#map0, #map1],
// CHECK-SAME:  iterator_types = ["parallel", "reduction"]}
// CHECK-SAME:  ins(%[[IN_SUB]] : tensor<4x2xf32>)
// CHECK-SAME:  outs(%[[FILL4]] : tensor<4xf32>) {
// CHECK:     ^bb0(%[[VAL_22:.*]]: f32, %[[VAL_23:.*]]: f32):
// CHECK:       %[[VAL_24:.*]] = arith.addf %[[VAL_22]], %[[VAL_23]] : f32
// CHECK:       linalg.yield %[[VAL_24]] : f32
// CHECK:     } -> tensor<4xf32>

// CHECK:     %[[COMBINE:.*]] = linalg.generic {
// CHECK-SAME:  indexing_maps = [#map2, #map2],
// CHECK-SAME:  iterator_types = ["parallel"]}
// CHECK-SAME:  ins(%[[REDUCE_TILE]] : tensor<4xf32>)
// CHECK-SAME:  outs(%[[FILL8_SUB]] : tensor<4xf32>) {
// CHECK:     ^bb0(%[[VAL_27:.*]]: f32, %[[VAL_28:.*]]: f32):
// CHECK:       %[[VAL_29:.*]] = arith.addf %[[VAL_27]], %[[VAL_28]] : f32
// CHECK:       linalg.yield %[[VAL_29]] : f32
// CHECK:     } -> tensor<4xf32>
// CHECK:     %[[UPDATE:.*]] = tensor.insert_slice %[[COMBINE:.*]] into %[[FILL8_BB]][%[[I]]] [4] [1] : tensor<4xf32> into tensor<8xf32>
// CHECK:     scf.yield %[[UPDATE]], %[[FILL4]] : tensor<8xf32>, tensor<4xf32>
// CHECK:   }
// CHECK:   scf.yield %[[INNER]]#0, %[[INNER]]#1 : tensor<8xf32>, tensor<4xf32>
// CHECK: }
// CHECK: return %[[OUTER]]#0 : tensor<8xf32>
