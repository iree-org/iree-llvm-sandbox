// RUN: mlir-proto-opt %s \
// RUN: | FileCheck %s

func.func @main() {
  // CHECK-LABEL: func.func @main() {
  %value = arith.constant 42 : i32
  // CHECK-NEXT:   %[[V0:.*]] = arith.constant 42 : i32
  %stream = iterators.value_to_stream %value : !iterators.stream<i32>
  // CHECK-NEXT:   %[[V1:.*]] = iterators.value_to_stream %[[V0]] : !iterators.stream<i32>
  %result:2 = iterators.stream_to_value %stream : !iterators.stream<i32>
  // CHECK-NEXT:   %[[V2:result.*]], %[[V3:hasResult.*]] = iterators.stream_to_value %[[V1]] : !iterators.stream<i32>
  return
// CHECK-NEXT:    return
}
// CHECK-NEXT:  }
