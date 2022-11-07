// RUN: mlir-proto-opt %s \
// RUN: | FileCheck %s

!i32_struct = !llvm.struct<(i32)>

func.func private @is_positive_struct(%struct : !i32_struct) -> i1 {
  %i = llvm.extractvalue %struct[0 : index] : !i32_struct
  %zero = arith.constant 0 : i32
  %cmp = arith.cmpi "sgt", %i, %zero : i32
  return %cmp : i1
}

func.func @main() {
// CHECK-LABEL: func.func @main() {
  %input = "iterators.constantstream"() { value = [] } :
               () -> (!iterators.stream<!i32_struct>)
// CHECK-NEXT:    %[[V0:.*]] = "iterators.constantstream"{{.*}}
  %filtered = "iterators.filter"(%input) {predicateRef = @is_positive_struct} :
                  (!iterators.stream<!i32_struct>) ->
                      (!iterators.stream<!i32_struct>)
// CHECK-NEXT:    %[[V1:filtered.*]] = "iterators.filter"(%[[V0]]) {predicateRef = @is_positive_struct} : (!iterators.stream<!llvm.struct<(i32)>>) -> !iterators.stream<!llvm.struct<(i32)>>
  return
// CHECK-NEXT:    return
}
// CHECK-NEXT:  }
