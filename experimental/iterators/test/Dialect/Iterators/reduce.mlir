// RUN: iterators-opt %s \
// RUN: | FileCheck %s

!i32_struct = !llvm.struct<(i32)>

func.func private @sum_struct(%lhs : !i32_struct, %rhs : !i32_struct) -> !i32_struct {
  %lhsi = llvm.extractvalue %lhs[0] : !i32_struct
  %rhsi = llvm.extractvalue %rhs[0] : !i32_struct
  %i = arith.addi %lhsi, %rhsi : i32
  %result = llvm.insertvalue %i, %lhs[0] : !i32_struct
  return %result : !i32_struct
}

func.func @main() {
// CHECK-LABEL: func.func @main() {
  %input = "iterators.constantstream"() { value = [] } :
               () -> (!iterators.stream<!i32_struct>)
// CHECK-NEXT:    %[[V0:.*]] = "iterators.constantstream"{{.*}}
  %reduced = "iterators.reduce"(%input) {reduceFuncRef = @sum_struct} :
                 (!iterators.stream<!i32_struct>) ->
                     (!iterators.stream<!i32_struct>)
// CHECK-NEXT:    %[[V1:reduced.*]] = "iterators.reduce"(%[[V0]]) {reduceFuncRef = @sum_struct} : (!iterators.stream<!llvm.struct<(i32)>>) -> !iterators.stream<!llvm.struct<(i32)>>
  return
// CHECK-NEXT:    return
}
// CHECK-NEXT:  }
