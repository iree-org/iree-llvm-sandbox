// RUN: mlir-proto-opt %s -convert-iterators-to-llvm \
// RUN: | FileCheck --enable-var-scope %s

!element_type = !llvm.struct<(i32)>

// CHECK-LABEL: func.func private @iterators.map.close.{{[0-9]+}}(%{{.*}}: !iterators.state<!iterators.state<i32>>) -> !iterators.state<!iterators.state<i32>> {
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[arg0:.*]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    %[[V1:.*]] = call @iterators.{{[a-zA-Z]+}}.close.{{[0-9]+}}(%[[V0]]) : (!iterators.state<i32>) -> !iterators.state<i32>
// CHECK-NEXT:    %[[V2:.*]] = iterators.insertvalue %[[V1]] into %[[arg0]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    return %[[V2]] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @iterators.map.next.{{[0-9]+}}(%{{.*}}: !iterators.state<!iterators.state<i32>>) -> (!iterators.state<!iterators.state<i32>>, i1, !llvm.struct<(i32)>)
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[arg0:.*]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    %[[V1:.*]]:3 = call @iterators.{{[a-zA-Z]+}}.next.0(%[[V0]]) : (!iterators.state<i32>) -> (!iterators.state<i32>, i1, !llvm.struct<(i32)>)
// CHECK-NEXT:    %[[V2:.*]] = scf.if %[[V1]]#1 -> (!llvm.struct<(i32)>) {
// CHECK-NEXT:      %[[V4:.*]] = func.call @double_struct(%[[V1]]#2) : (!llvm.struct<(i32)>) -> !llvm.struct<(i32)>
// CHECK-NEXT:      scf.yield %[[V4]] : !llvm.struct<(i32)>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[V4]] = llvm.mlir.undef : !llvm.struct<(i32)>
// CHECK-NEXT:      scf.yield %[[V4]] : !llvm.struct<(i32)>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[V3:.*]] = iterators.insertvalue %[[V1]]#0 into %arg0[0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    return %[[V3]], %[[V1]]#1, %[[V2]] : !iterators.state<!iterators.state<i32>>, i1, !llvm.struct<(i32)>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @iterators.map.open.{{[0-9]+}}(%{{.*}}: !iterators.state<!iterators.state<i32>>) -> !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[arg0:.*]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    %[[V1:.*]] = call @iterators.{{[a-zA-Z]+}}.open.{{[0-9]+}}(%[[V0]]) : (!iterators.state<i32>) -> !iterators.state<i32>
// CHECK-NEXT:    %[[V2:.*]] = iterators.insertvalue %[[V1]] into %[[arg0]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    return %[[V2]] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:  }

func.func private @double_struct(%struct : !element_type) -> !element_type {
// CHECK-LABEL: func.func private @double_struct(%{{.*}}: !llvm.struct<(i32)>) -> !llvm.struct<(i32)> {
  %i = llvm.extractvalue %struct[0 : index] : !element_type
// CHECK-NEXT:    %[[V0:.*]] = llvm.extractvalue %[[arg0:.*]][0 : index] : !llvm.struct<(i32)>
  %doubled = arith.addi %i, %i : i32
// CHECK-NEXT:    %[[V1:.*]] = arith.addi %[[V0]], %[[V0]] : i32
  %result = llvm.insertvalue %doubled, %struct[0 : index] : !element_type
// CHECK-NEXT:    %[[V2:.*]] = llvm.insertvalue %[[V1]], %[[arg0]][0 : index] : !llvm.struct<(i32)>
  return %result : !element_type
// CHECK-NEXT:    return %[[V2]] : !llvm.struct<(i32)>
}
// CHECK-NEXT:  }

// CHECK-LABEL: func.func @main() {
func.func @main() {
  %input = "iterators.constantstream"()
      { value = [[0 : i32], [1 : i32], [2 : i32], [3 : i32]] }
      : () -> (!iterators.stream<!element_type>)
  %reduce = "iterators.map"(%input) {mapFuncRef = @double_struct}
    : (!iterators.stream<!element_type>) -> (!iterators.stream<!element_type>)
  "iterators.sink"(%reduce) : (!iterators.stream<!element_type>) -> ()
  // CHECK:        %[[V1:.*]] = iterators.undefstate : !iterators.state<!iterators.state<i32>>
  // CHECK-NEXT:   %[[V2:.*]] = iterators.insertvalue %[[V0:.*]] into %[[V1]][0] : !iterators.state<!iterators.state<i32>>
  return
}
