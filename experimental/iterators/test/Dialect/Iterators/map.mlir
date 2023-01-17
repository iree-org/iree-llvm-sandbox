// RUN: mlir-proto-opt %s \
// RUN: | FileCheck %s

!i32_struct = !llvm.struct<(i32)>

func.func private @unpack_i32(%input : !i32_struct) -> i32 {
  %i = llvm.extractvalue %input[0] : !i32_struct
  return %i : i32
}

func.func @main() {
// CHECK-LABEL: func.func @main() {
  %input = "iterators.constantstream"() { value = [] } :
               () -> (!iterators.stream<!i32_struct>)
// CHECK-NEXT:    %[[V0:.*]] = "iterators.constantstream"{{.*}}
  %unpacked = "iterators.map"(%input) {mapFuncRef = @unpack_i32} :
                  (!iterators.stream<!i32_struct>) -> (!iterators.stream<i32>)
// CHECK-NEXT:    %[[V1:mapped.*]] = "iterators.map"(%[[V0]]) {mapFuncRef = @unpack_i32} : (!iterators.stream<!llvm.struct<(i32)>>) -> !iterators.stream<i32>
  return
// CHECK-NEXT:    return
}
// CHECK-NEXT:  }
