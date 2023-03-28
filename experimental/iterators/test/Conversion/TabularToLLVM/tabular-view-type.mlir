// RUN: iterators-opt %s -convert-tabular-to-llvm \
// RUN: | FileCheck --enable-var-scope %s

// CHECK-LABEL: func.func @func_call_return(
// CHECK-SAME:                              %[[ARG0:.*]]: !llvm.struct<(i64, ptr<i32>)>) -> !llvm.struct<(i64, ptr<i32>)> {
// CHECK-NEXT:    %[[V0:.*]] = call @func_call_return(%[[ARG0]]) : (!llvm.struct<(i64, ptr<i32>)>) -> !llvm.struct<(i64, ptr<i32>)>
// CHECK-NEXT:    return %[[V0]] : !llvm.struct<(i64, ptr<i32>)>
func.func @func_call_return(%view1 : !tabular.tabular_view<i32>) -> !tabular.tabular_view<i32> {
  %view2 = func.call @func_call_return(%view1)
    : (!tabular.tabular_view<i32>) -> !tabular.tabular_view<i32>
  return %view2 : !tabular.tabular_view<i32>
}

// CHECK-LABEL: func.func @scf_if(
// CHECK-SAME:                    %[[ARG0:[^:]*]]: !llvm.struct<(i64, ptr<i32>)>,
// CHECK-SAME:                    %[[ARG1:[^:]*]]: !llvm.struct<(i64, ptr<i32>)>,
// CHECK-SAME:                    %[[ARG2:[^:]*]]: i1) -> !llvm.struct<(i64, ptr<i32>)> {
// CHECK-NEXT:    %[[V0:.*]] = scf.if %arg2 -> (!llvm.struct<(i64, ptr<i32>)>) {
// CHECK-NEXT:      scf.yield %[[ARG0]] : !llvm.struct<(i64, ptr<i32>)>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %[[ARG1]] : !llvm.struct<(i64, ptr<i32>)>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[V0]] : !llvm.struct<(i64, ptr<i32>)>
func.func @scf_if(%view1 : !tabular.tabular_view<i32>,
                  %view2 : !tabular.tabular_view<i32>,
                  %cmp : i1) -> !tabular.tabular_view<i32> {
  %result = scf.if %cmp -> (!tabular.tabular_view<i32>) {
    scf.yield %view1 : !tabular.tabular_view<i32>
  } else {
    scf.yield %view2 : !tabular.tabular_view<i32>
  }
  return %result : !tabular.tabular_view<i32>
}
