// RUN: mlir-proto-opt %s -convert-iterators-to-llvm \
// RUN: | FileCheck --enable-var-scope %s

!element_type = !llvm.struct<(i32)>

// CHECK-LABEL: func.func private @iterators.filter.close.{{[0-9]+}}(%{{.*}}: !iterators.state<!iterators.state<i32>>) -> !iterators.state<!iterators.state<i32>> {
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[arg0:.*]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    %[[V1:.*]] = call @iterators.{{[a-zA-Z]+}}.close.{{[0-9]+}}(%[[V0]]) : ([[upstreamStateType:.*]]) -> [[upstreamStateType]]
// CHECK-NEXT:    %[[V2:.*]] = iterators.insertvalue %[[V1]] into %[[arg0]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    return %[[V2]] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @iterators.filter.next.{{[0-9]+}}(%{{.*}}: !iterators.state<!iterators.state<i32>>) -> (!iterators.state<!iterators.state<i32>>, i1, !llvm.struct<(i32)>)
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[arg0:.*]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    %[[V1:.*]]:3 = scf.while (%[[arg1:.*]] = %[[V0]]) : ([[upstreamStateType:.*]]) -> ([[upstreamStateType]], i1, !llvm.struct<(i32)>) {
// CHECK-NEXT:      %[[V3:.*]]:3 = func.call @iterators.{{[a-zA-Z]+}}.next.0(%[[arg1]]) : ([[upstreamStateType]]) -> ([[upstreamStateType]], i1, !llvm.struct<(i32)>)
// CHECK-NEXT:      %[[V4:.*]] = scf.if %[[V3]]#1 -> (i1) {
// CHECK-NEXT:        %[[V7:.*]] = func.call @is_positive_struct(%[[V3]]#2) : (!llvm.struct<(i32)>) -> i1
// CHECK-NEXT:        scf.yield %[[V7]] : i1
// CHECK-NEXT:      } else {
// CHECK-NEXT:        scf.yield %[[V3]]#1 : i1
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[Vtrue:.*]] = arith.constant true
// CHECK-NEXT:      %[[V5:.*]] = arith.xori %[[V4]], %[[Vtrue]] : i1
// CHECK-NEXT:      %[[V6:.*]] = arith.andi %[[V3]]#1, %[[V5]] : i1
// CHECK-NEXT:      scf.condition(%[[V6]]) %[[V3]]#0, %[[V3]]#1, %[[V3]]#2 : [[upstreamStateType]], i1, !llvm.struct<(i32)>
// CHECK-NEXT:    } do {
// CHECK-NEXT:    ^bb0(%[[arg2:.*]]: [[upstreamStateType]], %arg2: i1, %arg3: !llvm.struct<(i32)>):
// CHECK-NEXT:      scf.yield %[[arg2]] : [[upstreamStateType]]
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[V2:.*]] = iterators.insertvalue %[[V1]]#0 into %[[arg0]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    return %[[V2]], %[[V1]]#1, %[[V1]]#2 : !iterators.state<!iterators.state<i32>>, i1, !llvm.struct<(i32)>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @iterators.filter.open.{{[0-9]+}}(%{{.*}}: !iterators.state<!iterators.state<i32>>) -> !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[arg0:.*]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    %[[V1:.*]] = call @iterators.{{[a-zA-Z]+}}.open.{{[0-9]+}}(%[[V0]]) : ([[upstreamStateType:.*]]) -> [[upstreamStateType]]
// CHECK-NEXT:    %[[V2:.*]] = iterators.insertvalue %[[V1]] into %[[arg0]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    return %[[V2]] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:  }

func.func private @is_positive_struct(%struct : !element_type) -> i1 {
// CHECK-LABEL: func.func private @is_positive_struct(%{{.*}}: !llvm.struct<(i32)>) -> i1 {
  %i = llvm.extractvalue %struct[0 : index] : !element_type
// CHECK-NEXT:    %[[i:.*]] = llvm.extractvalue %[[struct:.*]][0 : index] : !llvm.struct<(i32)>
  %zero = arith.constant 0 : i32
// CHECK-NEXT:    %[[zero:.*]] = arith.constant 0 : i32
  %cmp = arith.cmpi "sgt", %i, %zero : i32
// CHECK-NEXT:    %[[cmp:.*]] = arith.cmpi sgt, %[[i]], %[[zero]] : i32
  return %cmp : i1
// CHECK-NEXT:    return %[[cmp]] : i1
}
// CHECK-NEXT:  }

func.func @main() {
  // CHECK-LABEL: func.func @main()
  %input = "iterators.constantstream"() { value = [] } : () -> (!iterators.stream<!element_type>)
  %filter = "iterators.filter"(%input) {predicateRef = @is_positive_struct}
    : (!iterators.stream<!element_type>) -> (!iterators.stream<!element_type>)
  // CHECK:        %[[V1:.*]] = iterators.undefstate : !iterators.state<!iterators.state<i32>>
  // CHECK-NEXT:   %[[V2:.*]] = iterators.insertvalue %[[V0:.*]] into %[[V1]][0] : !iterators.state<!iterators.state<i32>>
  return
  // CHECK-NEXT:   return
}
// CHECK-NEXT:   }
// CHECK-NEXT: }
