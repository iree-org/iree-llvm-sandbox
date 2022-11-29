// RUN: iterators-opt %s \
// RUN: | FileCheck %s

func.func @main(%input : tensor<?xi32>) {
  // CHECK-LABEL: func.func @main(%{{arg.*}}: tensor<?xi32>) {
  %stream = iterators.tensor_to_stream %input :
                tensor<?xi32> to !iterators.stream<tensor<4xi32>>
// CHECK-NEXT:    %[[V0:fromtensor.*]] = iterators.tensor_to_stream %[[arg0:.*]] : tensor<?xi32> to !iterators.stream<tensor<4xi32>>
  return
// CHECK-NEXT:    return
}
// CHECK-NEXT:  }
