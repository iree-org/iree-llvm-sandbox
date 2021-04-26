// RUN: export M=192 && export N=192 && export K=192 && export ITERS=1 &&\
// RUN: cat %p/matmul_f32_base.mlir | sed 's@${M}@'"$M"'@g'| sed 's@${K}@'"$K"'@g' | sed 's@${N}@'"$N"'@g'| sed 's@${ITERS}@'"$ITERS"'@g' |\

// RUN: mlir-proto-opt -canonicalize -mlir-disable-threading \
// RUN: -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul distribute distribute-tile-sizes=24,16" | FileCheck %s

// CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0)[s0] -> (24, -d0 + s0)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (16, -d0 + s0)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1) -> (24, d0 - d1)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0, d1) -> (16, d0 - d1)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0, d1)[s0] -> (d1, -d0 + s0)>

// CHECK-LABEL: func @init_and_matmul(
// CHECK-SAME:    %[[A:.*]]: [[TENSOR_TY:.*]], %[[B:.*]]: [[TENSOR_TY]],
// CHECK-SAME:    %[[C:.*]]: [[TENSOR_TY]]) -> [[TENSOR_TY]] {

// CHECK:   %[[C0_F32:.*]] = constant 0.000000e+00 : f32
// CHECK:   %[[C24:.*]] = constant 24 : index
// CHECK:   %[[C16:.*]] = constant 16 : index
// CHECK:   %[[C0:.*]] = constant 0 : index
// CHECK:   %[[C192:.*]] = constant 192 : index

// CHECK:   %[[RESULT:.*]] = linalg.tiled_loop (%[[I:.*]], %[[J:.*]]) =
// CHECK-SAME: (%[[C0]], %[[C0]]) to (%[[C192]], %[[C192]])
// CHECK-SAME: step (%[[C24]], %[[C16]])
// CHECK-SAME: ins (%[[A_:.*]] = %[[A]]: [[TENSOR_TY]], %[[B_:.*]] = %[[B]]: [[TENSOR_TY]])
// CHECK-SAME: outs (%[[C_:.*]] = %[[C]]: [[TENSOR_TY]]) {

// CHECK:     %[[A_SUB:.*]] = subtensor %[[A_]][%[[I]], 0]
// CHECK:     %[[B_SUB:.*]] = subtensor %[[B_]][0, %[[J]]]
// CHECK:     %[[C_SUB:.*]] = subtensor %[[C_]][%[[I]], %[[J]]]
// CHECK:     %[[C_INIT:.*]] = linalg.fill(%[[C_SUB]], %[[C0_F32]])
// CHECK-SAME:  tensor<?x?xf32>, f32 -> tensor<?x?xf32>
// CHECK:     %[[PROD:.*]] = linalg.matmul ins(%[[A_SUB]], %[[B_SUB]]
// CHECK-SAME:  outs(%[[C_INIT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// CHECK:     %[[PROD_SUB:.*]] = subtensor_insert %[[PROD]] into %[[C_]]
// CHECK:     linalg.yield %[[PROD_SUB]] : [[TENSOR_TY]]

