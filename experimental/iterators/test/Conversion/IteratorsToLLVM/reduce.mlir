// RUN: iterators-opt %s -convert-iterators-to-llvm \
// RUN: | FileCheck --enable-var-scope %s

// CHECK-LABEL: func private @iterators.reduce.close.{{[0-9]+}}(%{{.*}}: !iterators.state<!iterators.state<i32>>) -> !iterators.state<!iterators.state<i32>> {
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[arg0:.*]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    %[[V1:.*]] = call @iterators.{{[a-zA-Z]+}}.close.{{[0-9]+}}(%[[V0]]) : (!iterators.state<i32>) -> !iterators.state<i32>
// CHECK-NEXT:    %[[V2:.*]] = iterators.insertvalue %[[V1]] into %[[arg0]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    return %[[V2]] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @iterators.reduce.next.{{[0-9]+}}(%{{.*}}: !iterators.state<!iterators.state<i32>>) -> (!iterators.state<!iterators.state<i32>>, i1, tuple<i32>)
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[arg0:.*]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    %[[V1:.*]]:3 = call @iterators.{{[a-zA-Z]+}}.next.{{[0-9]+}}(%[[V0]]) : (!iterators.state<i32>) -> (!iterators.state<i32>, i1, tuple<i32>)
// CHECK-NEXT:    %[[V2:.*]]:3 = scf.if %[[V1]]#1 -> (!iterators.state<i32>, i1, tuple<i32>) {
// CHECK-NEXT:      %[[V4:.*]]:3 = scf.while (%[[arg1:.*]] = %[[V1]]#0, %[[arg2:.*]] = %[[V1]]#2) : (!iterators.state<i32>, tuple<i32>) -> (!iterators.state<i32>, tuple<i32>, tuple<i32>) {
// CHECK-NEXT:        %[[V5:.*]]:3 = func.call @iterators.{{[a-zA-Z]+}}.next.{{[0-9]+}}(%[[arg1]]) : (!iterators.state<i32>) -> (!iterators.state<i32>, i1, tuple<i32>)
// CHECK-NEXT:        scf.condition(%[[V5]]#1) %[[V5]]#0, %[[arg2]], %[[V5]]#2 : !iterators.state<i32>, tuple<i32>, tuple<i32>
// CHECK-NEXT:      } do {
// CHECK-NEXT:      ^[[bb0:.*]](%[[arg1:.*]]: !iterators.state<i32>, %[[arg2:.*]]: tuple<i32>, %[[arg3:.*]]: tuple<i32>):
// CHECK-NEXT:        %[[V5:.*]] = func.call @sum_tuple(%[[arg2]], %[[arg3]]) : (tuple<i32>, tuple<i32>) -> tuple<i32>
// CHECK-NEXT:        scf.yield %[[arg1]], %[[V5]] : !iterators.state<i32>, tuple<i32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[true:.*]] = arith.constant true
// CHECK-NEXT:      scf.yield %[[V4]]#0, %[[true]], %[[V4]]#1 : !iterators.state<i32>, i1, tuple<i32>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %[[V1]]#0, %[[V1]]#1, %[[V1]]#2 : !iterators.state<i32>, i1, tuple<i32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[V3:.*]] = iterators.insertvalue %[[V2]]#0 into %[[arg0]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    return %[[V3]], %[[V2]]#1, %[[V2]]#2 : !iterators.state<!iterators.state<i32>>, i1, tuple<i32>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @iterators.reduce.open.{{[0-9]+}}(%{{.*}}: !iterators.state<!iterators.state<i32>>) -> !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[arg0:.*]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    %[[V1:.*]] = call @iterators.{{[a-zA-Z]+}}.open.{{[0-9]+}}(%[[V0]]) : (!iterators.state<i32>) -> !iterators.state<i32>
// CHECK-NEXT:    %[[V2:.*]] = iterators.insertvalue %[[V1]] into %[[arg0]][0] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:    return %[[V2]] : !iterators.state<!iterators.state<i32>>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @sum_tuple(
// CHECK-SAME:                          %[[ARG0:[^:]*]]: tuple<i32>,
// CHECK-SAME:                          %[[ARG1:[^:]*]]: tuple<i32>) -> tuple<i32>
// CHECK-NEXT:    %[[V0:.*]] = tuple.to_elements %[[ARG0]] : tuple<i32>
// CHECK-NEXT:    %[[V1:.*]] = tuple.to_elements %[[ARG1]] : tuple<i32>
// CHECK-NEXT:    %[[V2:.*]] = arith.addi %[[V0]], %[[V1]] : i32
// CHECK-NEXT:    %[[V3:.*]] = tuple.from_elements %[[V2]] : tuple<i32>
// CHECK-NEXT:    return %[[V3]] : tuple<i32>
func.func private @sum_tuple(%lhs : tuple<i32>, %rhs : tuple<i32>) -> tuple<i32> {
  %lhsi = tuple.to_elements %lhs : tuple<i32>
  %rhsi = tuple.to_elements %rhs : tuple<i32>
  %i = arith.addi %lhsi, %rhsi : i32
  %result = tuple.from_elements %i : tuple<i32>
  return %result : tuple<i32>
}

func.func @main() {
// CHECK-LABEL:  func.func @main()
  %input = "iterators.constantstream"() { value = [] } : () -> (!iterators.stream<tuple<i32>>)
  // CHECK:        %[[V0:.*]] = iterators.createstate({{.*}}) : [[upstreamStateType:.*]]
  %reduce = "iterators.reduce"(%input) {reduceFuncRef = @sum_tuple}
    : (!iterators.stream<tuple<i32>>) -> (!iterators.stream<tuple<i32>>)
  // CHECK-NEXT:   %[[V1:.*]] = iterators.createstate(%[[V0]]) : !iterators.state<[[upstreamStateType]]>
  return
  // CHECK-NEXT:   return
}
// CHECK-NEXT:   }
