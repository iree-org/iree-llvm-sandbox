// RUN: structured-opt %s -convert-iterators-to-llvm \
// RUN: | FileCheck --enable-var-scope %s

// CHECK-LABEL: func.func private @iterators.map.close.{{[0-9]+}}(%{{.*}}: !iterators.state<!iterators.state<i32>>) -> !iterators.state<!iterators.state<i32>> {
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[arg0:.*]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    %[[V1:.*]] = call @iterators.{{[a-zA-Z]+}}.close.{{[0-9]+}}(%[[V0]]) : (!iterators.state<i32>) -> !iterators.state<i32>
// CHECK-NEXT:    %[[V2:.*]] = iterators.insertvalue %[[V1]] into %[[arg0]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    return %[[V2]] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @iterators.map.next.{{[0-9]+}}(%{{.*}}: !iterators.state<!iterators.state<i32>>) -> (!iterators.state<!iterators.state<i32>>, i1, tuple<i32>)
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[arg0:.*]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    %[[V1:.*]]:3 = call @iterators.{{[a-zA-Z]+}}.next.0(%[[V0]]) : (!iterators.state<i32>) -> (!iterators.state<i32>, i1, tuple<i32>)
// CHECK-NEXT:    %[[V2:.*]] = scf.if %[[V1]]#1 -> (tuple<i32>) {
// CHECK-NEXT:      %[[V4:.*]] = func.call @double_tuple(%[[V1]]#2) : (tuple<i32>) -> tuple<i32>
// CHECK-NEXT:      scf.yield %[[V4]] : tuple<i32>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[V5:.*]] = llvm.mlir.undef : i32
// CHECK-NEXT:      %[[V6:.*]] = tuple.from_elements %[[V5]] : tuple<i32>
// CHECK-NEXT:      scf.yield %[[V6]] : tuple<i32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[V3:.*]] = iterators.insertvalue %[[V1]]#0 into %arg0[0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    return %[[V3]], %[[V1]]#1, %[[V2]] : !iterators.state<!iterators.state<i32>>, i1, tuple<i32>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @iterators.map.open.{{[0-9]+}}(%{{.*}}: !iterators.state<!iterators.state<i32>>) -> !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[arg0:.*]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    %[[V1:.*]] = call @iterators.{{[a-zA-Z]+}}.open.{{[0-9]+}}(%[[V0]]) : (!iterators.state<i32>) -> !iterators.state<i32>
// CHECK-NEXT:    %[[V2:.*]] = iterators.insertvalue %[[V1]] into %[[arg0]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    return %[[V2]] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @double_tuple(%{{.*}}: tuple<i32>) -> tuple<i32> {
// CHECK-NEXT:    %[[V0:.*]] = tuple.to_elements %[[arg0:.*]] : tuple<i32>
// CHECK-NEXT:    %[[V1:.*]] = arith.addi %[[V0]], %[[V0]] : i32
// CHECK-NEXT:    %[[V2:.*]] = tuple.from_elements %[[V1]] : tuple<i32>
// CHECK-NEXT:    return %[[V2]] : tuple<i32>
func.func private @double_tuple(%tuple : tuple<i32>) -> tuple<i32> {
  %i = tuple.to_elements %tuple : tuple<i32>
  %doubled = arith.addi %i, %i : i32
  %result = tuple.from_elements %doubled : tuple<i32>
  return %result : tuple<i32>
}

func.func @main() {
// CHECK-LABEL:  func.func @main() {
  %input = "iterators.constantstream"()
      { value = [[0 : i32], [1 : i32], [2 : i32], [3 : i32]] }
      : () -> (!iterators.stream<tuple<i32>>)
  // CHECK:        %[[V0:.*]] = iterators.createstate({{.*}}) : [[upstreamStateType:.*]]
  %reduce = "iterators.map"(%input) {mapFuncRef = @double_tuple}
    : (!iterators.stream<tuple<i32>>) -> (!iterators.stream<tuple<i32>>)
  // CHECK-NEXT:   %[[V1:.*]] = iterators.createstate(%[[V0]]) : !iterators.state<[[upstreamStateType]]>
  return
  // CHECK-NEXT:   return
}
// CHECK-NEXT:   }
