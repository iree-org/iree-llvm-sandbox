// RUN: mlir-proto-opt %s | mlir-proto-opt | FileCheck %s

// CHECK-LABEL: func @simple_example
func @simple_example(%in: tensor<100xf32>, %out: tensor<100xf32>) -> (tensor<100xf32>) {
  %num_threads = arith.constant 100 : index
  %result = linalg_ext.in_parallel %num_threads -> tensor<100xf32> {
    ^bb0(%thread_idx : index):
      %0 = arith.constant 0 : index
      %1 = tensor.extract_slice %in[%thread_idx][1][1] : tensor<100xf32> to tensor<1xf32>
      linalg_ext.perform_concurrently {
        linalg_ext.parallel_insert_slice %1 into %out[%thread_idx][%0][%0] :
          tensor<1xf32> into tensor<100xf32>
      }
  }
  return %result : tensor<100xf32>
}

func @no_terminator() -> () {
  %num_threads = arith.constant 100 : index
  linalg_ext.in_parallel %num_threads -> () {
    ^bb0(%thread_idx : index):
  }
  return
}
