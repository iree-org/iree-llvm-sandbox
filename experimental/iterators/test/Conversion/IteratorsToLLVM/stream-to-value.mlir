// RUN: mlir-proto-opt %s -convert-iterators-to-llvm \
// RUN: | FileCheck --enable-var-scope %s

// CHECK-LABEL: func.func private @iterators.value_to_stream.close.{{[0-9]+}}(%{{.*}}: !iterators.state<i1, i32>) -> !iterators.state<i1, i32> {

// CHECK-LABEL: func.func private @iterators.value_to_stream.next.{{[0-9]+}}(%{{.*}}: !iterators.state<i1, i32>) -> (!iterators.state<i1, i32>, i1, i32) {

// CHECK-LABEL: func.func private @iterators.value_to_stream.open.{{[0-9]+}}(%{{.*}}: !iterators.state<i1, i32>) -> !iterators.state<i1, i32> {

func.func @main() {
// CHECK-LABEL: func.func @main() {
  %value = arith.constant 42 : i32
// CHECK-NEXT:    %[[V0:.*]] = arith.constant 42 : i32
  %stream = iterators.value_to_stream %value : !iterators.stream<i32>
// CHECK-NEXT:    %[[V1:.*]] = iterators.undefstate : !iterators.state<i1, i32>
// CHECK-NEXT:    %[[V2:.*]] = iterators.insertvalue %[[V0]] into %[[V1]][1] : !iterators.state<i1, i32>
  %result:2 = iterators.stream_to_value %stream : !iterators.stream<i32>
// CHECK-NEXT:    %[[V3:.*]] = call @iterators.value_to_stream.open.{{[0-9]+}}(%[[V2]]) : (!iterators.state<i1, i32>) -> !iterators.state<i1, i32>
// CHECK-NEXT:    %[[V4:.*]]:3 = call @iterators.value_to_stream.next.{{[0-9]+}}(%[[V3]]) : (!iterators.state<i1, i32>) -> (!iterators.state<i1, i32>, i1, i32)
// CHECK-NEXT:    %[[V5:.*]] = call @iterators.value_to_stream.close.{{[0-9]+}}(%[[V4]]#0) : (!iterators.state<i1, i32>) -> !iterators.state<i1, i32>
  return
// CHECK-NEXT:    return
}
// CHECK-NEXT:  }
