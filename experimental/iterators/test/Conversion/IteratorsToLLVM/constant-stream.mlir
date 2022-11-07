// RUN: mlir-proto-opt %s -convert-iterators-to-llvm \
// RUN: | FileCheck --enable-var-scope %s

!element_type = !llvm.struct<(i32)>

// CHECK-LABEL: func private @iterators.constantstream.close.{{[0-9]+}}(%{{.*}}: !iterators.state<i32>) -> !iterators.state<i32>
// CHECK-NEXT:    return %[[arg0:.*]] : !iterators.state<i32>
// CHECK-NEXT:  }

// CHECK-LABEL:  llvm.mlir.global internal constant @iterators.constant_stream_data{{.*}}() : !llvm.array<4 x struct<(i32)>> {
// CHECK-NEXT:     %[[V0:.*]] = llvm.mlir.undef : !llvm.array<4 x struct<(i32)>>
// CHECK-NEXT:     %[[V1:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
// CHECK-NEXT:     %[[V2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:     %[[V3:.*]] = llvm.insertvalue %[[V2]], %[[V1]][0 : index] : !llvm.struct<(i32)>
// CHECK-NEXT:     %[[V4:.*]] = llvm.insertvalue %[[V3]], %[[V0]][0 : index] : !llvm.array<4 x struct<(i32)>>
// CHECK-NEXT:     %[[V5:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
// CHECK-NEXT:     %[[V6:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:     %[[V7:.*]] = llvm.insertvalue %[[V6]], %[[V5]][0 : index] : !llvm.struct<(i32)>
// CHECK-NEXT:     %[[V8:.*]] = llvm.insertvalue %[[V7]], %[[V4]][1 : index] : !llvm.array<4 x struct<(i32)>>
// CHECK-NEXT:     %[[V9:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
// CHECK-NEXT:     %[[V10:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:     %[[V11:.*]] = llvm.insertvalue %[[V10]], %[[V9]][0 : index] : !llvm.struct<(i32)>
// CHECK-NEXT:     %[[V12:.*]] = llvm.insertvalue %[[V11]], %[[V8]][2 : index] : !llvm.array<4 x struct<(i32)>>
// CHECK-NEXT:     %[[V13:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
// CHECK-NEXT:     %[[V14:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-NEXT:     %[[V15:.*]] = llvm.insertvalue %[[V14]], %[[V13]][0 : index] : !llvm.struct<(i32)>
// CHECK-NEXT:     %[[V16:.*]] = llvm.insertvalue %[[V15]], %[[V12]][3 : index] : !llvm.array<4 x struct<(i32)>>
// CHECK-NEXT:     llvm.return %[[V16]] : !llvm.array<4 x struct<(i32)>>
// CHECK-NEXT:   }

// CHECK-LABEL: func private @iterators.constantstream.next.{{[0-9]+}}(%{{.*}}: !iterators.state<i32>) -> (!iterators.state<i32>, i1, !llvm.struct<(i32)>)
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[arg0:.*]][0] : !iterators.state<i32>
// CHECK-NEXT:    %[[V1:.*]] = arith.constant 4 : i32
// CHECK-NEXT:    %[[V2:.*]] = arith.cmpi slt, %[[V0]], %[[V1]] : i32
// CHECK-NEXT:    %[[V3:.*]]:2 = scf.if %[[V2]] -> (!iterators.state<i32>, !llvm.struct<(i32)>) {
// CHECK-NEXT:      %[[V4:.*]] = arith.constant 1 : i32
// CHECK-NEXT:      %[[V5:.*]] = arith.addi %[[V0]], %[[V4]] : i32
// CHECK-NEXT:      %[[V6:.*]] = iterators.insertvalue %[[V5]] into %[[arg0]][0] : !iterators.state<i32>
// CHECK-NEXT:      %[[V7:.*]] = llvm.mlir.addressof @iterators.constant_stream_data{{.*}} : !llvm.ptr<array<4 x struct<(i32)>>>
// CHECK-NEXT:      %[[V8:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[V9:.*]] = llvm.getelementptr %[[V7]][%[[V8]], %[[V0]]] : (!llvm.ptr<array<4 x struct<(i32)>>>, i32, i32) -> !llvm.ptr<struct<(i32)>>
// CHECK-NEXT:      %[[Va:.*]] = llvm.load %[[V9]] : !llvm.ptr<struct<(i32)>>
// CHECK-NEXT:      scf.yield %[[V6]], %[[Va]] : !iterators.state<i32>, !llvm.struct<(i32)>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[Vb:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
// CHECK-NEXT:      scf.yield %[[arg0]], %[[Vb]] : !iterators.state<i32>, !llvm.struct<(i32)>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[V3]]#0, %[[V2]], %[[V3]]#1 : !iterators.state<i32>, i1, !llvm.struct<(i32)>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @iterators.constantstream.open.{{[0-9]+}}(%{{.*}}: !iterators.state<i32>) -> !iterators.state<i32>
// CHECK-NEXT:    %[[V0:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[V1:.*]] = iterators.insertvalue %[[V0]] into %[[arg0:.*]][0] : !iterators.state<i32>
// CHECK-NEXT:    return %[[V1]] : !iterators.state<i32>
// CHECK-NEXT:  }

func.func @main() {
  // CHECK-LABEL: func.func @main()
  %input = "iterators.constantstream"()
      { value = [[0 : i32], [1 : i32], [2 : i32], [3 : i32]] }
      : () -> (!iterators.stream<!element_type>)
  // CHECK-NEXT:   %[[V0:.*]] = iterators.undefstate : !iterators.state<i32>
  return
  // CHECK-NEXT:   return
}
// CHECK-NEXT:   }
// CHECK-NEXT: }
