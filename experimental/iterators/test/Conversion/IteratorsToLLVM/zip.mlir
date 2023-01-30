// RUN: iterators-opt %s -convert-iterators-to-llvm \
// RUN: | FileCheck --enable-var-scope %s

// CHECK-LABEL: func.func private @iterators.zip.close.{{[0-9]+}}(
// CHECK-SAME:      %[[arg0:.*]]: !iterators.state<[[lhsUpstreamStateType:!iterators\.state.*]], [[rhsUpstreamStateType:!iterators.state.*]]>) ->
// CHECK-SAME:      !iterators.state<[[lhsUpstreamStateType]], [[rhsUpstreamStateType]]> {
// CHECK-NEXT:     %[[V0:.*]] = iterators.extractvalue %arg0[0] : !iterators.state<[[lhsUpstreamStateType]], [[rhsUpstreamStateType]]>
// CHECK-NEXT:     %[[V1:.*]] = call @iterators.{{.*}}.close.{{[0-9]+}}(%[[V0]]) : ([[lhsUpstreamStateType]]) -> [[lhsUpstreamStateType]]
// CHECK-NEXT:     %[[V2:.*]] = iterators.insertvalue %[[V1]] into %arg0[0] : !iterators.state<[[lhsUpstreamStateType]], [[rhsUpstreamStateType]]>
// CHECK-NEXT:     %[[V3:.*]] = iterators.extractvalue %[[V2]][1] : !iterators.state<[[lhsUpstreamStateType]], [[rhsUpstreamStateType]]>
// CHECK-NEXT:     %[[V4:.*]] = call @iterators.{{.*}}.close.{{[0-9]+}}(%[[V3]]) : ([[rhsUpstreamStateType]]) -> [[rhsUpstreamStateType]]
// CHECK-NEXT:     %[[V5:.*]] = iterators.insertvalue %[[V4]] into %[[V2]][1] : !iterators.state<[[lhsUpstreamStateType]], [[rhsUpstreamStateType]]>
// CHECK-NEXT:     return %[[V5]] : !iterators.state<[[lhsUpstreamStateType]], [[rhsUpstreamStateType]]>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @iterators.zip.next.{{[0-9]+}}(
// CHECK-SAME:      %[[arg0:.*]]: !iterators.state<[[lhsUpstreamStateType:!iterators\.state.*]], [[rhsUpstreamStateType:!iterators.state.*]]>) ->
// CHECK-SAME:      (!iterators.state<[[lhsUpstreamStateType]], [[rhsUpstreamStateType]]>, i1, tuple<i32, i32>) {
// CHECK-NEXT:     %[[V0:.*]] = arith.constant true
// CHECK-NEXT:     %[[V2:.*]] = iterators.extractvalue %arg0[0] : !iterators.state<[[lhsUpstreamStateType]], [[rhsUpstreamStateType]]>
// CHECK-NEXT:     %[[V3:.*]]:3 = call @iterators.{{.*}}.next.{{[0-9]+}}(%[[V2]]) : ([[lhsUpstreamStateType]]) -> ([[lhsUpstreamStateType]], i1, i32)
// CHECK-NEXT:     %[[V4:.*]] = arith.andi %[[V0]], %[[V3]]#1 : i1
// CHECK-NEXT:     %[[V6:.*]] = iterators.insertvalue %[[V3]]#0 into %arg0[0] : !iterators.state<[[lhsUpstreamStateType]], [[rhsUpstreamStateType]]>
// CHECK-NEXT:     %[[V7:.*]] = iterators.extractvalue %[[V6]][1] : !iterators.state<[[lhsUpstreamStateType]], [[rhsUpstreamStateType]]>
// CHECK-NEXT:     %[[V8:.*]]:3 = call @iterators.{{.*}}.next.{{[0-9]+}}(%[[V7]]) : ([[rhsUpstreamStateType]]) -> ([[rhsUpstreamStateType]], i1, i32)
// CHECK-NEXT:     %[[V9:.*]] = arith.andi %[[V4]], %[[V8]]#1 : i1
// CHECK-NEXT:     %[[Vb:.*]] = iterators.insertvalue %[[V8]]#0 into %[[V6]][1] : !iterators.state<[[lhsUpstreamStateType]], [[rhsUpstreamStateType]]>
// CHECK-NEXT:     %[[Vc:.*]] = tuple.from_elements %[[V3]]#2, %[[V8]]#2 : tuple<i32, i32>
// CHECK-NEXT:     return %[[Vb]], %[[V9]], %[[Vc]] : !iterators.state<[[lhsUpstreamStateType]], [[rhsUpstreamStateType]]>, i1, tuple<i32, i32>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @iterators.zip.open.{{[0-9]+}}(
// CHECK-SAME:      %[[arg0:.*]]: !iterators.state<[[lhsUpstreamStateType:!iterators\.state.*]], [[rhsUpstreamStateType:!iterators.state.*]]>) ->
// CHECK-SAME:      !iterators.state<[[lhsUpstreamStateType]], [[rhsUpstreamStateType]]> {
// CHECK-NEXT:     %[[V0:.*]] = iterators.extractvalue %arg0[0] : !iterators.state<[[lhsUpstreamStateType]], [[rhsUpstreamStateType]]>
// CHECK-NEXT:     %[[V1:.*]] = call @iterators.{{.*}}.open.{{[0-9]+}}(%[[V0]]) : ([[lhsUpstreamStateType]]) -> [[lhsUpstreamStateType]]
// CHECK-NEXT:     %[[V2:.*]] = iterators.insertvalue %[[V1]] into %arg0[0] : !iterators.state<[[lhsUpstreamStateType]], [[rhsUpstreamStateType]]>
// CHECK-NEXT:     %[[V3:.*]] = iterators.extractvalue %[[V2]][1] : !iterators.state<[[lhsUpstreamStateType]], [[rhsUpstreamStateType]]>
// CHECK-NEXT:     %[[V4:.*]] = call @iterators.{{.*}}.open.{{[0-9]+}}(%[[V3]]) : ([[rhsUpstreamStateType]]) -> [[rhsUpstreamStateType]]
// CHECK-NEXT:     %[[V5:.*]] = iterators.insertvalue %[[V4]] into %[[V2]][1] : !iterators.state<[[lhsUpstreamStateType]], [[rhsUpstreamStateType]]>
// CHECK-NEXT:     return %[[V5]] : !iterators.state<[[lhsUpstreamStateType]], [[rhsUpstreamStateType]]>
// CHECK-NEXT:  }

func.func private @unpack_i32(%input : tuple<i32>) -> i32 {
  %i = tuple.to_elements %input : tuple<i32>
  return %i : i32
}

func.func @main() {
// CHECK-LABEL:   func.func @main() {
  // Left-hand stream of numbers.
  %zero_to_three = "iterators.constantstream"()
      { value = [[0 : i32], [1 : i32], [2 : i32], [3 : i32]] }
      : () -> (!iterators.stream<tuple<i32>>)
  // CHECK:         %[[lhsInnerState:.*]] = iterators.createstate
  %unpacked_lhs = "iterators.map"(%zero_to_three) {mapFuncRef = @unpack_i32}
    : (!iterators.stream<tuple<i32>>) -> (!iterators.stream<i32>)
  // CHECK:         %[[lhsOuterState:.*]] = iterators.createstate(%[[lhsInnerState]]) : [[lhsStateType:.*]]

  // Right-hand stream of numbers.
  %four_to_seven = "iterators.constantstream"()
      { value = [[4 : i32], [5 : i32], [6 : i32], [7 : i32]] }
      : () -> (!iterators.stream<tuple<i32>>)
  // CHECK:         %[[rhsInnerState:.*]] = iterators.createstate
  %unpacked_rhs = "iterators.map"(%four_to_seven) {mapFuncRef = @unpack_i32}
    : (!iterators.stream<tuple<i32>>) -> (!iterators.stream<i32>)
  // CHECK:         %[[rhsOuterState:.*]] = iterators.createstate(%[[rhsInnerState]]) : [[rhsStateType:.*]]

  // Zip.
  %zipped = iterators.zip %unpacked_lhs, %unpacked_rhs :
                (!iterators.stream<i32>, !iterators.stream<i32>)
                  -> (!iterators.stream<tuple<i32, i32>>)
  // CHECK-NEXT:    %[[state:.*]] = iterators.createstate(%[[lhsOuterState]], %[[rhsOuterState]]) : !iterators.state<[[lhsStateType]], [[rhsStateType]]>
  return
  // CHECK-NEXT:    return
}
// CHECK-NEXT:    }
