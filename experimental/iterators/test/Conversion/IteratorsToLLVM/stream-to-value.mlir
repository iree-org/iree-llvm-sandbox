// RUN: iterators-opt %s -convert-iterators-to-llvm \
// RUN: | FileCheck --enable-var-scope %s

// CHECK-LABEL: func.func private @iterators.value_to_stream.close.{{[0-9]+}}(%{{.*}}: !iterators.state<i1, i32>) -> !iterators.state<i1, i32> {

// CHECK-LABEL: func.func private @iterators.value_to_stream.next.{{[0-9]+}}(%{{.*}}: !iterators.state<i1, i32>) -> (!iterators.state<i1, i32>, i1, i32) {

// CHECK-LABEL: func.func private @iterators.value_to_stream.open.{{[0-9]+}}(%{{.*}}: !iterators.state<i1, i32>) -> !iterators.state<i1, i32> {

func.func @main() {
// CHECK-LABEL:  func.func @main() {
  %value = arith.constant 42 : i32
  %stream = iterators.value_to_stream %value : !iterators.stream<i32>
  // CHECK:        %[[V0:.*]] = iterators.createstate({{.*}}) : [[upstreamStateType:.*]]
  %result:2 = iterators.stream_to_value %stream : !iterators.stream<i32>
  // CHECK-NEXT:   %[[V1:.*]] = call @iterators.value_to_stream.open.0(%[[V0]]) : ([[upstreamStateType]]) -> [[upstreamStateType]]
  // CHECK-NEXT:   %[[V2:.*]]:3 = call @iterators.value_to_stream.next.0(%[[V1]]) : ([[upstreamStateType]]) -> ([[upstreamStateType]], i1, i32)
  // CHECK-NEXT:   %[[V3:.*]] = call @iterators.value_to_stream.close.0(%[[V2]]#0) : ([[upstreamStateType]]) -> [[upstreamStateType]]
  return
  // CHECK-NEXT:   return
}
// CHECK-NEXT:   }
