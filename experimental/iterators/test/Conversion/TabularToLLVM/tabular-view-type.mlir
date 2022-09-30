// RUN: mlir-proto-opt %s -convert-tabular-to-llvm \
// RUN: | FileCheck --enable-var-scope %s

func.func @func_call_return(%view1 : !tabular.tabular_view<i32>) -> !tabular.tabular_view<i32> {
// CHECK:      func.func @func_call_return(%[[arg0:.*]]: !llvm.struct<(i64, ptr<i32>)>) -> !llvm.struct<(i64, ptr<i32>)> {
  %view2 = func.call @func_call_return(%view1)
    : (!tabular.tabular_view<i32>) -> !tabular.tabular_view<i32>
// CHECK-NEXT:   %[[V0:.*]] = call @func_call_return(%[[arg0]]) : (!llvm.struct<(i64, ptr<i32>)>) -> !llvm.struct<(i64, ptr<i32>)>
  return %view2 : !tabular.tabular_view<i32>
// CHECK-NEXT:   return %[[V0]] : !llvm.struct<(i64, ptr<i32>)>
}
// CHECK-NEXT: }
