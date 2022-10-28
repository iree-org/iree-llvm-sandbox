// RUN: mlir-proto-opt %s \
// RUN:   -convert-iterators-to-llvm -reconcile-unrealized-casts \
// RUN: | FileCheck --enable-var-scope %s

// CHECK-LABEL: func private @iterators.tabular_view_to_stream.close.{{[0-9]+}}(%{{.*}}: !iterators.state<i64, !llvm.struct<(i64, ptr<i32>)>>) -> !iterators.state<i64, !llvm.struct<(i64, ptr<i32>)>> {
// CHECK-NEXT:    return %[[arg0:.*]] : !iterators.state<i64, !llvm.struct<(i64, ptr<i32>)>>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @iterators.tabular_view_to_stream.next.{{[0-9]+}}(%{{.*}}: !iterators.state<i64, !llvm.struct<(i64, ptr<i32>)>>) -> (!iterators.state<i64, !llvm.struct<(i64, ptr<i32>)>>, i1, !llvm.struct<(i32)>)
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[arg0:.*]][0] : !iterators.state<i64, !llvm.struct<(i64, ptr<i32>)>>
// CHECK-NEXT:    %[[V1:.*]] = iterators.extractvalue %[[arg0]][1] : !iterators.state<i64, !llvm.struct<(i64, ptr<i32>)>>
// CHECK-NEXT:    %[[V2:.*]] = llvm.extractvalue %[[V1]][0 : index] : !llvm.struct<(i64, ptr<i32>)>
// CHECK-NEXT:    %[[V3:.*]] = arith.cmpi slt, %[[V0]], %[[V2]] : i64
// CHECK-NEXT:    %[[V4:.*]]:2 = scf.if %[[V3]] -> (!iterators.state<i64, !llvm.struct<(i64, ptr<i32>)>>, !llvm.struct<(i32)>) {
// CHECK-NEXT:      %[[C1:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[V5:.*]] = arith.addi %[[V0]], %[[C1]] : i64
// CHECK-NEXT:      %[[V6:.*]] = iterators.insertvalue %[[V5]] into %[[arg0]][0] : !iterators.state<i64, !llvm.struct<(i64, ptr<i32>)>>
// CHECK-NEXT:      %[[V7:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
// CHECK-NEXT:      %[[V8:.*]] = llvm.extractvalue %[[V1]][1 : index] : !llvm.struct<(i64, ptr<i32>)>
// CHECK-NEXT:      %[[V9:.*]] = llvm.getelementptr %[[V8]][%[[V0]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:      %[[Va:.*]] = llvm.load %[[V9]] : !llvm.ptr<i32>
// CHECK-NEXT:      %[[Vb:.*]] = llvm.insertvalue %[[Va]], %[[V7]][0 : index] : !llvm.struct<(i32)>
// CHECK-NEXT:      scf.yield %[[V6]], %[[Vb]] : !iterators.state<i64, !llvm.struct<(i64, ptr<i32>)>>, !llvm.struct<(i32)>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[V5:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
// CHECK-NEXT:      scf.yield %[[arg0]], %[[V5]] : !iterators.state<i64, !llvm.struct<(i64, ptr<i32>)>>, !llvm.struct<(i32)>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[V4]]#0, %[[V3]], %[[V4]]#1 : !iterators.state<i64, !llvm.struct<(i64, ptr<i32>)>>, i1, !llvm.struct<(i32)>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @iterators.tabular_view_to_stream.open.{{[0-9]+}}(%{{.*}}: !iterators.state<i64, !llvm.struct<(i64, ptr<i32>)>>) -> !iterators.state<i64, !llvm.struct<(i64, ptr<i32>)>>
// CHECK-NEXT:    %[[V0:.*]] = arith.constant 0 : i64
// CHECK-NEXT:    %[[V1:.*]] = iterators.insertvalue %[[V0]] into %[[arg0:.*]][0] : !iterators.state<i64, !llvm.struct<(i64, ptr<i32>)>>
// CHECK-NEXT:    return %[[V1]] : !iterators.state<i64, !llvm.struct<(i64, ptr<i32>)>>
// CHECK-NEXT:  }

func.func @main(%input : !tabular.tabular_view<i32>) {
  // CHECK-LABEL: func.func @main(%{{arg.*}}: !llvm.struct<(i64, ptr<i32>)>) {
  %stream = iterators.tabular_view_to_stream %input
                to !iterators.stream<!llvm.struct<(i32)>>
  // CHECK-NEXT:    %[[V1:.*]] = iterators.undefstate : !iterators.state<i64, !llvm.struct<(i64, ptr<i32>)>>
  // CHECK-NEXT:    %[[V2:.*]] = iterators.insertvalue %[[arg:.*]] into %[[V1]][1] : !iterators.state<i64, !llvm.struct<(i64, ptr<i32>)>>
  return
  // CHECK-NEXT:   return
}
// CHECK-NEXT:   }
