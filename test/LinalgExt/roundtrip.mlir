// RUN: mlir-proto-opt %s | FileCheck %s

func @reverse_1d_tensor(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %reverse = iree_linalg_ext.reverse
      dimensions(dense<0> : tensor<1xi64>)
      ins(%arg0 : tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) : tensor<?x?xf32>
  return %reverse : tensor<?x?xf32>
}
//       CHECK: func @reverse_1d_tensor
//  CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//       CHECK:   %[[INIT:.+]] = linalg.init_tensor [%[[D0]], %[[D1]]] : tensor<?x?xf32>
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.reverse
//  CHECK-SAME:       dimensions(dense<0> : tensor<1xi64>)
//  CHECK-SAME:       ins(%[[ARG0]] : tensor<?x?xf32>)
//  CHECK-SAME:       outs(%[[INIT]] : tensor<?x?xf32>)
//       CHECK:   return %[[RESULT]]

// -----

func @reverse_2d_tensor(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %reverse = iree_linalg_ext.reverse
      dimensions(dense<[0, 1]> : tensor<2xi64>)
      ins(%arg0 : tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) : tensor<?x?xf32>
  return %reverse : tensor<?x?xf32>
}
//       CHECK: func @reverse_2d_tensor
//  CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//       CHECK:   %[[INIT:.+]] = linalg.init_tensor [%[[D0]], %[[D1]]] : tensor<?x?xf32>
//       CHECK:   %[[RESULT:.+]] = iree_linalg_ext.reverse
//  CHECK-SAME:       dimensions(dense<[0, 1]> : tensor<2xi64>)
//  CHECK-SAME:       ins(%[[ARG0]] : tensor<?x?xf32>)
//  CHECK-SAME:       outs(%[[INIT]] : tensor<?x?xf32>)
//       CHECK:   return %[[RESULT]]

// -----

// CHECK-LABEL: func @static_tile
func @static_tile(%chunk_size: index, %in: tensor<?xf32>, %out: tensor<?xf32>, %out2: tensor<?xf32>) -> (tensor<?xf32>) {
  %c0 = arith.constant 0: index
  //%d0 = tensor.dim %out, %c0: tensor<?xf32>

  // CHECK: iree_linalg_ext.tile %{{.*}} outs(%{{.*}}: tensor<?xf32>, %{{.*}}: tensor<?xf32>)
  // CHECK: ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: tensor<?xf32>, %{{.*}}: tensor<?xf32>):
  %0:2 = iree_linalg_ext.tile %chunk_size outs(%out: tensor<?xf32>, %out2: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    // TODO: one offset and one size per tensor?
    // If not necessary in the dense strided-array world, what about the rest?
    ^bb0(%offset: index, %size: index, %st1: tensor<?xf32>, %st2: tensor<?xf32>):
      // TODO: atm this is just 1-1: out-chunk-size -> in-size.
      %1 = tensor.extract_slice %in[%offset][%size][1] : tensor<?xf32> to tensor<?xf32>
      %3 = linalg.generic {
           indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
           iterator_types = ["parallel"]}
         ins(%1: tensor<?xf32>) outs(%st1: tensor<?xf32>) {
         ^bb0(%a: f32, %b:f32):  // no predecessors
           %f42 = arith.constant 42.0: f32
           %tmp = arith.mulf %a, %f42: f32
           linalg.yield %tmp: f32
      } -> tensor<?xf32>
      iree_linalg_ext.tile_yield %3, %st2: tensor<?xf32>, tensor<?xf32> // assumes dim is 0 and stacks
  }
  return %0#0: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @simple_example
func @simple_example(%in: tensor<100xf32>, %out: tensor<100xf32>) -> (tensor<100xf32>) {
  %num_threads = arith.constant 100 : index
  %result = iree_linalg_ext.in_parallel %num_threads -> tensor<100xf32> {
    ^bb0(%thread_idx : index):
      %0 = arith.constant 0 : index
      %1 = tensor.extract_slice %in[%thread_idx][1][1] : tensor<100xf32> to tensor<1xf32>
      iree_linalg_ext.perform_concurrently {
        iree_linalg_ext.parallel_insert_slice %1 into %out[%thread_idx][%0][%0] :
          tensor<1xf32> into tensor<100xf32>
      }
  }
  return %result : tensor<100xf32>
}

func @no_terminator() -> () {
  %num_threads = arith.constant 100 : index
  iree_linalg_ext.in_parallel %num_threads -> () {
    ^bb0(%thread_idx : index):
  }
  return
}
