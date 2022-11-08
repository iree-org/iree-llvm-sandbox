// RUN: mlir-proto-opt %s -convert-iterators-to-llvm \
// RUN: | FileCheck --enable-var-scope %s
// CHECK-LABEL: func.func private @iterators.value_to_stream.close.{{[0-9]+}}(%{{.*}}: !iterators.state<i1, i32>) -> !iterators.state<i1, i32> {
// CHECK-NEXT:    return %[[arg0:.*]] : !iterators.state<i1, i32>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @iterators.value_to_stream.next.{{[0-9]+}}(%{{.*}}: !iterators.state<i1, i32>) -> (!iterators.state<i1, i32>, i1, i32) {
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[arg0:.*]][0] : !iterators.state<i1, i32>
// CHECK-NEXT:    %[[V1:.*]] = arith.constant true
// CHECK-NEXT:    %[[V2:.*]] = arith.xori %[[V1]], %[[V0]] : i1
// CHECK-NEXT:    %[[V3:.*]] = iterators.extractvalue %[[arg0]][1] : !iterators.state<i1, i32>
// CHECK-NEXT:    %[[V4:.*]] = iterators.insertvalue %[[V1]] into %[[arg0]][0] : !iterators.state<i1, i32>
// CHECK-NEXT:    return %[[V4]], %[[V2]], %[[V3]] : !iterators.state<i1, i32>, i1, i32
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @iterators.value_to_stream.open.{{[0-9]+}}(%{{.*}}: !iterators.state<i1, i32>) -> !iterators.state<i1, i32> {
// CHECK-NEXT:    %[[V0:.*]] = arith.constant false
// CHECK-NEXT:    %[[V1:.*]] = iterators.insertvalue %[[V0]] into %[[arg0:.*]][0] : !iterators.state<i1, i32>
// CHECK-NEXT:    return %[[V1]] : !iterators.state<i1, i32>
// CHECK-NEXT:  }

func.func @main() {
// CHECK-LABEL:  func.func @main() {
  %value = arith.constant 42 : i32
  // CHECK-NEXT:   %[[V0:.*]] = arith.constant 42 : i32
  %stream = iterators.value_to_stream %value : !iterators.stream<i32>
  // CHECK-NEXT:   %[[V1:.*]] = iterators.undefstate : !iterators.state<i1, i32>
  // CHECK-NEXT:   %[[V2:.*]] = iterators.insertvalue %[[V0]] into %[[V1]][1] : !iterators.state<i1, i32>
  return
  // CHECK-NEXT:   return
}
// CHECK-NEXT:   }
