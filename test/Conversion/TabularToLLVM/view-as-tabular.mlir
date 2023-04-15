// RUN: structured-opt %s -convert-tabular-to-llvm \
// RUN: | FileCheck --enable-var-scope %s

func.func @main(%memref : memref<3xi32>) {
  // CHECK-LABEL: func.func @main(%{{arg.*}}: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>) {
  %view = tabular.view_as_tabular %memref
    : (memref<3xi32>) -> !tabular.tabular_view<i32>
  // CHECK-NEXT:    %[[V0:.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
  // CHECK-NEXT:    %[[V1:.*]] = llvm.extractvalue %[[arg:.*]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK-NEXT:    %[[V2:.*]] = llvm.extractvalue %[[arg]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK-NEXT:    %[[V3:.*]] = llvm.insertvalue %[[V1]], %[[V0]][1] : !llvm.struct<(i64, ptr)>
  // CHECK-NEXT:    %[[V4:.*]] = llvm.insertvalue %[[V2]], %[[V3]][0] : !llvm.struct<(i64, ptr)>
  return
  // CHECK-NEXT:    return
}
// CHECK-NEXT:    }
