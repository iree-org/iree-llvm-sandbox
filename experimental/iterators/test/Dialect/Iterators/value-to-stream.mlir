// RUN: mlir-proto-opt %s \
// RUN: | FileCheck %s

func.func @main() {
  // CHECK-LABEL: func.func @main() {
  %value = arith.constant 42 : i32
  // CHECK-NEXT:   %[[V0:.*]] = arith.constant 42 : i32
  %stream = iterators.value_to_stream %value : !iterators.stream<i32>
  // CHECK-NEXT:   %[[V1:singletonstream.*]] = iterators.value_to_stream %[[V0]] : !iterators.stream<i32>
  return
// CHECK-NEXT:    return
}
// CHECK-NEXT:  }
