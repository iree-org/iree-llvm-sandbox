// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

func @not_enough_results() -> () {
  %num_threads = arith.constant 100 : index
  // expected-error@+1 {{'linalg_ext.in_parallel' op produces 1 results, but its terminator yields 0 values}}
  %result = linalg_ext.in_parallel %num_threads -> tensor<100xf32> {
    ^bb0(%thread_idx : index):
      linalg_ext.perform_concurrently {}
  }
}

// -----

func @too_many_results(%1 : tensor<1xf32>, %out : tensor<100xf32>) -> () {
  %num_threads = arith.constant 100 : index
  // expected-error@+1 {{'linalg_ext.in_parallel' op produces 1 results, but its terminator yields 2 values}}
  %result = linalg_ext.in_parallel %num_threads -> tensor<100xf32> {
    ^bb0(%thread_idx : index):
      %0 = arith.constant 1 : index
      linalg_ext.perform_concurrently {
        linalg_ext.parallel_insert_slice %1 into %out[%thread_idx][%0][%0] :
          tensor<1xf32> into tensor<100xf32>
        linalg_ext.parallel_insert_slice %1 into %out[%thread_idx][%0][%0] :
          tensor<1xf32> into tensor<100xf32>
      }
  }
}

// -----

func @type_mismatch(%1 : tensor<1xf32>, %out : tensor<200xf32>) -> () {
  %num_threads = arith.constant 100 : index
  // expected-error@+1 {{'linalg_ext.in_parallel' op type mismatch between 0th result of in_parallel ('tensor<200xf32>') and 0th result yielded by its terminator ('tensor<100xf32>')}}
  %result = linalg_ext.in_parallel %num_threads -> tensor<100xf32> {
    ^bb0(%thread_idx : index):
      %0 = arith.constant 1 : index
      linalg_ext.perform_concurrently {
        linalg_ext.parallel_insert_slice %1 into %out[%thread_idx][%0][%0] :
          tensor<1xf32> into tensor<200xf32>
      }
  }
}
