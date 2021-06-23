// RUN: export M=192 && export N=192 && export K=192 && export ITERS=1 &&\
// RUN: cat %p/matmul_f32_base.mlir | sed 's@${M}@'"$M"'@g'| sed 's@${K}@'"$K"'@g' | sed 's@${N}@'"$N"'@g'| sed 's@${ITERS}@'"$ITERS"'@g' |\

// RUN: mlir-proto-opt -canonicalize -mlir-disable-threading \
// RUN: -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul distribute distribute-tile-sizes=24,16" |\
// RUN: tee | FileCheck %s

// CHECK-LABEL: func @init_and_matmul(
// CHECK-SAME:    %[[A:.*]]: [[TENSOR_TY:.*]], %[[B:.*]]: [[TENSOR_TY]],
// CHECK-SAME:    %[[C:.*]]: [[TENSOR_TY]]) -> [[TENSOR_TY]] {

//  CHECK-DAG:   %[[C0_F32:.*]] = constant 0.000000e+00 : f32
//  CHECK-DAG:   %[[C24:.*]] = constant 24 : index
//  CHECK-DAG:   %[[C16:.*]] = constant 16 : index
//  CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//  CHECK-DAG:   %[[C192:.*]] = constant 192 : index

// CHECK:   %[[RESULT:.*]] = linalg.tiled_loop (%[[I:.*]], %[[J:.*]]) =
// CHECK-SAME: (%[[C0]], %[[C0]]) to (%[[C192]], %[[C192]])
// CHECK-SAME: step (%[[C24]], %[[C16]])
// CHECK-SAME: ins (%[[A_:.*]] = %[[A]]: [[TENSOR_TY]], %[[B_:.*]] = %[[B]]: [[TENSOR_TY]])
// CHECK-SAME: outs (%[[C_:.*]] = %[[C]]: [[TENSOR_TY]]) {

// CHECK:     %[[A_SUB:.*]] = subtensor %[[A_]][%[[I]], 0]
// CHECK:     %[[B_SUB:.*]] = subtensor %[[B_]][0, %[[J]]]
// CHECK:     %[[C_SUB:.*]] = subtensor %[[C_]][%[[I]], %[[J]]]
// CHECK:     %[[C_INIT:.*]] = linalg.fill(%[[C0_F32]], %[[C_SUB]])
// CHECK-SAME:  tensor<24x16xf32>, f32 -> tensor<24x16xf32>
// CHECK:     %[[PROD:.*]] = linalg.matmul ins(%[[A_SUB]], %[[B_SUB]]
// CHECK-SAME:  outs(%[[C_INIT]] : tensor<24x16xf32>) -> tensor<24x16xf32>

// TODO: one canonicalization is missing here which fails at cleaning up tensor_cast to tensor<?x?xf32>
// CHECK:     %[[PROD_SUB:.*]] = subtensor_insert %{{.*}} into %[[C_]]
// CHECK:     linalg.yield %[[PROD_SUB]] : [[TENSOR_TY]]

