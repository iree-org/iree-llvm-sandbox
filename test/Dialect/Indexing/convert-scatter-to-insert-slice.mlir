// RUN: structured-opt -allow-unregistered-dialect -canonicalize -scatter-to-insert-slice -split-input-file %s | FileCheck %s

// CHECK-LABEL: module {
// CHECK:         func.func @test_convert_scatter_to_insert() {
// CHECK:           %[[VAL_0:.*]] = tensor.empty() : tensor<7x22x330x4400xf32>
// CHECK:           %[[VAL_1:.*]] = indexing.arange(start = 0, stop = 22, step = 2) : tensor<?xindex>
// CHECK:           %[[VAL_2:.*]] = indexing.meshgrid(%[[VAL_1]]) : (tensor<?xindex>) -> tensor<?x1xindex>
// CHECK:           %[[VAL_3:.*]] = indexing.gather %[[VAL_0]]{{\[}}%[[VAL_2]]] gather_dims([1]) unique : (tensor<7x22x330x4400xf32>, tensor<?x1xindex>) -> tensor<7x11x330x4400xf32>
// CHECK:           %[[VAL_4:.*]] = "use1"(%[[VAL_3]]) : (tensor<7x11x330x4400xf32>) -> tensor<7x11x330x4400xf32>
// CHECK:           %[[VAL_5:.*]] = tensor.insert_slice %[[VAL_4]] into %[[VAL_0]][0, 0, 0, 0] [7, 11, 330, 4400] [1, 2, 1, 1] : tensor<7x11x330x4400xf32> into tensor<7x22x330x4400xf32>
// CHECK:           "use2"(%[[VAL_5]]) : (tensor<7x22x330x4400xf32>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK:       }

module {
  func.func @test_convert_scatter_to_insert() {
    %0 = tensor.empty() : tensor<7x22x330x4400xf32>
    %c0 = arith.constant 0 : index
    %c22 = arith.constant 22 : index
    %c2 = arith.constant 2 : index
    %1 = indexing.arange(start = %c0, stop = %c22, step = %c2) : tensor<?xindex>
    %5 = indexing.meshgrid(%1) : (tensor<?xindex>) -> tensor<?x1xindex>
    %2 = indexing.gather %0[%5] gather_dims([1]) unique : (tensor<7x22x330x4400xf32>, tensor<?x1xindex>) -> tensor<7x11x330x4400xf32>
    %3 = "use1"(%2) : (tensor<7x11x330x4400xf32>) -> (tensor<7x11x330x4400xf32>)
    %4 = indexing.scatter %3 into %0[%5] scatter_dims([1]) unique : (tensor<7x11x330x4400xf32>, tensor<7x22x330x4400xf32>, tensor<?x1xindex>) -> tensor<7x22x330x4400xf32>
    "use2"(%4) : (tensor<7x22x330x4400xf32>) -> ()

    return
  }
}

// -----

// CHECK-LABEL: module {
// CHECK:         func.func @test_convert_gather_to_extract_with_concat() {
// CHECK:           %[[VAL_0:.*]] = tensor.empty() : tensor<7x22x330x4400xf32>
// CHECK:           %[[VAL_1:.*]] = indexing.arange(start = 0, stop = 22, step = 2) : tensor<?xindex>
// CHECK:           %[[VAL_2:.*]] = indexing.arange(start = 0, stop = 330, step = 30) : tensor<?xindex>
// CHECK:           %[[VAL_3:.*]] = indexing.meshgrid(%[[VAL_1]], %[[VAL_2]]) : (tensor<?xindex>, tensor<?xindex>) -> tensor<?x?x2xindex>
// CHECK:           %[[VAL_4:.*]] = indexing.gather %[[VAL_0]]{{\[}}%[[VAL_3]]] gather_dims([1, 2]) unique : (tensor<7x22x330x4400xf32>, tensor<?x?x2xindex>) -> tensor<7x?x?x4400xf32>
// CHECK:           %[[VAL_5:.*]] = "use1"(%[[VAL_4]]) : (tensor<7x?x?x4400xf32>) -> tensor<7x11x11x4400xf32>
// CHECK:           %[[VAL_6:.*]] = tensor.insert_slice %[[VAL_5]] into %[[VAL_0]][0, 0, 0, 0] [7, 11, 11, 4400] [1, 2, 30, 1] : tensor<7x11x11x4400xf32> into tensor<7x22x330x4400xf32>
// CHECK:           "use2"(%[[VAL_6]]) : (tensor<7x22x330x4400xf32>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK:       }

module {
  func.func @test_convert_gather_to_extract_with_concat() {
    %0 = tensor.empty() : tensor<7x22x330x4400xf32>
    %c0 = arith.constant 0 : index
    %c22 = arith.constant 22 : index
    %c2 = arith.constant 2 : index
    %1 = indexing.arange(start = %c0, stop = %c22, step = %c2) : tensor<?xindex>
    %c330 = arith.constant 330 : index
    %c30 = arith.constant 30 : index
    %4 = indexing.arange(start = %c0, stop = %c330, step = %c30) : tensor<?xindex>
    %10 = indexing.meshgrid(%1, %4) : (tensor<?xindex>, tensor<?xindex>) -> tensor<?x?x2xindex>
    %6 = indexing.gather %0[%10] gather_dims([1, 2]) unique : (tensor<7x22x330x4400xf32>, tensor<?x?x2xindex>) -> tensor<7x?x?x4400xf32>
    %7 = "use1"(%6) : (tensor<7x?x?x4400xf32>) -> (tensor<7x11x11x4400xf32>)
    %9 = indexing.scatter %7 into %0[%10] scatter_dims([1, 2]) unique : (tensor<7x11x11x4400xf32>, tensor<7x22x330x4400xf32>, tensor<?x?x2xindex>) -> tensor<7x22x330x4400xf32>
    "use2"(%9) : (tensor<7x22x330x4400xf32>) -> ()

    return
  }
}