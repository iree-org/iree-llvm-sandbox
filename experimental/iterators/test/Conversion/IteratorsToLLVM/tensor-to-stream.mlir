// RUN: iterators-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN: | FileCheck --enable-var-scope %s

// CHECK-LABEL: func private @iterators.tensor_to_stream.close.{{[0-9]+}}(
// CHECK-SAME:        %[[ARG0:.*]]: !iterators.state<index, [[tensorType:.*]]>) ->
// CHECK-SAME:            !iterators.state<index, [[tensorType]]> {
// CHECK-NEXT:    return %[[arg0:.*]] : !iterators.state<index, [[tensorType]]>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @iterators.tensor_to_stream.next.{{[0-9]+}}(
// CHECK-SAME:        %[[ARG0:.*]]: !iterators.state<index, [[tensorType:.*]]>) ->
// CHECK-SAME:            (!iterators.state<index, [[tensorType]]>, i1, [[tensorSliceType:.*]]>) {
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[arg0:.*]][0] : !iterators.state<index, [[tensorType]]>
// CHECK-NEXT:    %[[V1:.*]] = iterators.extractvalue %[[arg0]][1] : !iterators.state<index, [[tensorType]]>
// CHECK-NEXT:    %[[Vx:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[V2:.*]] = tensor.dim %[[V1]], %[[Vx]] : [[tensorType]]
// CHECK-NEXT:    %[[V3:.*]] = arith.cmpi slt, %[[V0]], %[[V2]] : index
// CHECK-NEXT:    %[[V4:.*]]:2 = scf.if %[[V3]] -> (!iterators.state<index, [[tensorType]]>, [[tensorSliceType]]>) {
// CHECK-NEXT:      %[[C1:.*]] = arith.constant 2 : index
// CHECK-NEXT:      %[[V5:.*]] = arith.addi %[[C1]], %[[V0]] : index
// CHECK-NEXT:      %[[V6:.*]] = iterators.insertvalue %[[V5]] into %[[arg0]][0] : !iterators.state<index, [[tensorType]]>
// CHECK-NEXT:      %[[V7:.*]] = tensor.extract_slice %[[V1]][%[[V0]]] [2] [1] : [[tensorType]] to [[tensorSliceType]]>
// CHECK-NEXT:      scf.yield %[[V6]], %[[V7]] : !iterators.state<index, [[tensorType]]>, [[tensorSliceType]]>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[V5:.*]] = tensor.extract_slice %[[V1]][0] [2] [1] : [[tensorType]] to [[tensorSliceType]]>
// CHECK-NEXT:      scf.yield %[[arg0]], %[[V5]] : !iterators.state<index, [[tensorType]]>, [[tensorSliceType]]>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[V4]]#0, %[[V3]], %[[V4]]#1 : !iterators.state<index, [[tensorType]]>, i1, [[tensorSliceType]]>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @iterators.tensor_to_stream.open.{{[0-9]+}}(
// CHECK-SAME:        %[[ARG0:.*]]: !iterators.state<index, [[tensorType:.*]]>) ->
// CHECK-SAME:            !iterators.state<index, [[tensorType]]> {
// CHECK-NEXT:    %[[V0:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[V1:.*]] = iterators.insertvalue %[[V0]] into %[[ARG0]][0] : !iterators.state<index, [[tensorType]]>
// CHECK-NEXT:    return %[[V1]] : !iterators.state<index, [[tensorType]]>
// CHECK-NEXT:  }

func.func @main(%tensor : tensor<?xi32>) {
// CHECK-LABEL:  func.func @main(
// CHECK-SAME:      %[[arg0:.*]]: [[tensorType:.*]]) {
  %stream = iterators.tensor_to_stream %tensor :
                tensor<?xi32> to !iterators.stream<tensor<2xi32>>
  // CHECK-NEXT:   %[[V1:.*]] = arith.constant 0 : index
  // CHECK-NEXT:   %[[V2:.*]] = iterators.createstate(%[[V1]], %[[arg0]]) : !iterators.state<index, [[tensorType]]>
  return
  // CHECK-NEXT:   return
}
// CHECK-NEXT:   }
