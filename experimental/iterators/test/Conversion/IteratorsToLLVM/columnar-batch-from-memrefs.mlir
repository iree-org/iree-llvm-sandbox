// RUN: mlir-proto-opt %s -convert-iterators-to-llvm \
// RUN: | FileCheck --enable-var-scope %s

func.func @main(%memref : memref<3xi32>) {
  // CHECK-LABEL: func.func @main(%{{arg.*}}: !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>) {
  %batch = "iterators.view_as_columnar_batch"(%memref)
    : (memref<3xi32>) -> !iterators.columnar_batch<tuple<i32>>
  // CHECK-NEXT:    %[[V0:.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr<i32>)>
  // CHECK-NEXT:    %[[V1:.*]] = llvm.extractvalue %[[arg:.*]][1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK-NEXT:    %[[V2:.*]] = llvm.extractvalue %[[arg]][3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK-NEXT:    %[[V3:.*]] = llvm.insertvalue %[[V1]], %[[V0]][1 : index] : !llvm.struct<(i64, ptr<i32>)>
  // CHECK-NEXT:    %[[V4:.*]] = llvm.insertvalue %[[V2]], %[[V3]][0 : index] : !llvm.struct<(i64, ptr<i32>)>
  return
  // CHECK-NEXT:    return
}
// CHECK-NEXT:    }
