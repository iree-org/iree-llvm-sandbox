// RUN: mlir-proto-opt %s -convert-iterators-to-llvm \
// RUN: | FileCheck --enable-var-scope %s

!element_type = !llvm.struct<(i32)>

// CHECK-LABEL: func private @iterators.reduce.close.{{[0-9]+}}(%{{.*}}: !llvm.struct<"iterators.reduce_state{{.*}}", ({{.*}})>) -> !llvm.struct<"iterators.reduce_state{{.*}}", ({{.*}})> {
// CHECK-NEXT:    %[[V0:.*]] = llvm.extractvalue %[[arg0:.*]][0 : index] : !llvm.struct<"[[reduceStateName:iterators\.reduce_state.*]]", ([[nestedUpstreamStateType:.*]])>
// CHECK-NEXT:    %[[V1:.*]] = call @iterators.{{[a-zA-Z]+}}.close.{{[0-9]+}}(%[[V0]]) : ([[upstreamStateType:.*]]) -> [[upstreamStateType]]
// CHECK-NEXT:    %[[V2:.*]] = llvm.insertvalue %[[V1]], %[[arg0]][0 : index] : !llvm.struct<"[[reduceStateName]]", ([[nestedUpstreamStateType]])>
// CHECK-NEXT:    return %[[V2]] : !llvm.struct<"[[reduceStateName]]", ([[nestedUpstreamStateType]])>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @iterators.reduce.next.{{[0-9]+}}(%{{.*}}: !llvm.struct<"iterators.reduce_state{{.*}}", ({{.*}})>) -> (!llvm.struct<"iterators.reduce_state{{.*}}", ({{.*}})>, i1, !llvm.struct<(i32)>)
// CHECK-NEXT:    %[[V0:.*]] = llvm.extractvalue %[[arg0:.*]][0 : index] : !llvm.struct<"[[reduceStateName:iterators\.reduce_state.*]]", ([[nestedUpstreamStateType:.*]])>
// CHECK-NEXT:    %[[V1:.*]]:3 = call @iterators.{{[a-zA-Z]+}}.next.{{[0-9]+}}(%[[V0]]) : ([[upstreamStateType:.*]]) -> ([[upstreamStateType]], i1, !llvm.struct<(i32)>)
// CHECK-NEXT:    %[[V2:.*]]:3 = scf.if %[[V1]]#1 -> ([[upstreamStateType]], i1, !llvm.struct<(i32)>) {
// CHECK-NEXT:      %[[V4:.*]]:3 = scf.while (%[[arg1:.*]] = %[[V1]]#0, %[[arg2:.*]] = %[[V1]]#2) : ([[upstreamStateType]], !llvm.struct<(i32)>) -> ([[upstreamStateType]], !llvm.struct<(i32)>, !llvm.struct<(i32)>) {
// CHECK-NEXT:        %[[V5:.*]]:3 = func.call @iterators.{{[a-zA-Z]+}}.next.{{[0-9]+}}(%[[arg1]]) : ([[upstreamStateType]]) -> ([[upstreamStateType]], i1, !llvm.struct<(i32)>)
// CHECK-NEXT:        scf.condition(%[[V5]]#1) %[[V5]]#0, %[[arg2]], %[[V5]]#2 : [[upstreamStateType]], !llvm.struct<(i32)>, !llvm.struct<(i32)>
// CHECK-NEXT:      } do {
// CHECK-NEXT:      ^[[bb0:.*]](%[[arg1:.*]]: [[upstreamStateType]], %[[arg2:.*]]: !llvm.struct<(i32)>, %[[arg3:.*]]: !llvm.struct<(i32)>):
// CHECK-NEXT:        %[[V5:.*]] = func.call @sum_struct(%[[arg2]], %[[arg3]]) : (!llvm.struct<(i32)>, !llvm.struct<(i32)>) -> !llvm.struct<(i32)>
// CHECK-NEXT:        scf.yield %[[arg1]], %[[V5]] : [[upstreamStateType]], !llvm.struct<(i32)>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[true:.*]] = arith.constant true
// CHECK-NEXT:      scf.yield %[[V4]]#0, %[[true]], %[[V4]]#1 : [[upstreamStateType]], i1, !llvm.struct<(i32)>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %[[V1]]#0, %[[V1]]#1, %[[V1]]#2 : [[upstreamStateType]], i1, !llvm.struct<(i32)>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[V3:.*]] = llvm.insertvalue %[[V2]]#0, %[[arg0]][0 : index] : !llvm.struct<"[[reduceStateName]]", ([[nestedUpstreamStateType]])>
// CHECK-NEXT:    return %[[V3]], %[[V2]]#1, %[[V2]]#2 : !llvm.struct<"[[reduceStateName]]", ([[nestedUpstreamStateType]])>, i1, !llvm.struct<(i32)>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @iterators.reduce.open.{{[0-9]+}}(%{{.*}}: !llvm.struct<"iterators.reduce_state{{.*}}", ({{.*}})>) -> !llvm.struct<"iterators.reduce_state{{.*}}", ({{.*}})>
// CHECK-NEXT:    %[[V0:.*]] = llvm.extractvalue %[[arg0:.*]][0 : index] : !llvm.struct<"[[reduceStateName:iterators\.reduce_state.*]]", ([[nestedUpstreamStateType:.*]])>
// CHECK-NEXT:    %[[V1:.*]] = call @iterators.{{[a-zA-Z]+}}.open.{{[0-9]+}}(%[[V0]]) : ([[upstreamStateType:.*]]) -> [[upstreamStateType]]
// CHECK-NEXT:    %[[V2:.*]] = llvm.insertvalue %[[V1]], %[[arg0]][0 : index] : !llvm.struct<"[[reduceStateName]]", ([[nestedUpstreamStateType]])>
// CHECK-NEXT:    return %[[V2]] : !llvm.struct<"[[reduceStateName]]", ([[nestedUpstreamStateType]])>
// CHECK-NEXT:  }

func.func private @sum_struct(%lhs : !element_type, %rhs : !element_type) -> !element_type {
// CHECK-LABEL: func private @sum_struct(%{{.*}}: !llvm.struct<(i32)>, %{{.*}}: !llvm.struct<(i32)>) -> !llvm.struct<(i32)>
  %lhsi = llvm.extractvalue %lhs[0 : index] : !element_type
// CHECK-NEXT:    %[[lhsi:.*]] = llvm.extractvalue %[[lhs:.*]][0 : index] : !llvm.struct<(i32)>
  %rhsi = llvm.extractvalue %rhs[0 : index] : !element_type
// CHECK-NEXT:    %[[rhsi:.*]] = llvm.extractvalue %[[rhs:.*]][0 : index] : !llvm.struct<(i32)>
  %i = arith.addi %lhsi, %rhsi : i32
// CHECK-NEXT:    %[[i:.*]] = arith.addi %[[lhsi]], %[[rhsi]] : i32
  %result = llvm.insertvalue %i, %lhs[0 : index] : !element_type
// CHECK-NEXT:    %[[result:.*]] = llvm.insertvalue %[[i]], %[[lhs]][0 : index] : !llvm.struct<(i32)>
  return %result : !element_type
// CHECK-NEXT:    return %[[result]] : !llvm.struct<(i32)>
}
// CHECK-NEXT:  }

func.func @main() {
  // CHECK-LABEL: func.func @main()
  %input = "iterators.constantstream"() { value = [] } : () -> (!iterators.stream<!element_type>)
  %reduce = "iterators.reduce"(%input) {reduceFuncRef = @sum_struct}
    : (!iterators.stream<!element_type>) -> (!iterators.stream<!element_type>)
  // CHECK:        %[[V1:.*]] = llvm.mlir.undef : !llvm.struct<"[[reduceStateName:iterators\.reduce_state.*]]", ([[nestedUpstreamStateType:.*]])>
  // CHECK-NEXT:   %[[V2:.*]] = llvm.insertvalue %[[V0:.*]], %[[V1]][0 : index] : !llvm.struct<"[[reduceStateName]]", ([[nestedUpstreamStateType]])>
  return
  // CHECK-NEXT:   return
}
// CHECK-NEXT:   }
// CHECK-NEXT: }
