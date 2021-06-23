// RUN: export M=192 && export N=192 && export K=192 && export ITERS=1 &&\
// RUN: cat %p/matmul_f32_base.mlir | sed 's@${M}@'"$M"'@g'| sed 's@${K}@'"$K"'@g' | sed 's@${N}@'"$N"'@g'| sed 's@${ITERS}@'"$ITERS"'@g' |\

// RUN: mlir-proto-opt -canonicalize -mlir-disable-threading \
// RUN: -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul distribute distribute-tile-sizes=24,16" \
// RUN: -linalg-comprehensive-bufferize-inplace |\
// RUN: tee | FileCheck %s

// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1)[s0] -> (d0 * 192 + s0 + d1)>

// CHECK-LABEL: func @init_and_matmul(
// CHECK-SAME:    %[[A:.*]]: memref<192x192xf32>, %[[B:.*]]: memref<192x192xf32>,
// CHECK-SAME:    %[[C:.*]]: memref<192x192xf32>) {
//  CHECK-DAG:  %[[C0_F32:.*]] = constant 0.000000e+00 : f32
//  CHECK-DAG:  %[[C24:.*]] = constant 24 : index
//  CHECK-DAG:  %[[C16:.*]] = constant 16 : index
//  CHECK-DAG:  %[[C0:.*]] = constant 0 : index
//  CHECK-DAG:  %[[C192:.*]] = constant 192 : index

// CHECK:  linalg.tiled_loop (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME: to (%[[C192]], %[[C192]]) step (%[[C24]], %[[C16]])
// CHECK-SAME: ins (%[[A_:.*]] = %[[A]]: memref<192x192xf32>, %[[B_:.*]] = %[[B]]: memref<192x192xf32>)
// CHECK-SAME: outs (%[[C_:.*]] = %[[C]]: memref<192x192xf32>) {

// CHECK:    %[[A_sub:.*]] = memref.subview %[[A_]]{{\[}}%[[I]], 0]
// CHECK:    %[[B_sub:.*]] = memref.subview %[[B_]][0, %[[J]]]
// CHECK:    %[[C_sub:.*]] = memref.subview %[[C_]]{{\[}}%[[I]], %[[J]]]

// CHECK:    linalg.fill(%[[C0_F32]], %[[C_sub]])
// CHECK:    linalg.matmul ins(%[[A_sub]], %[[B_sub]]
// CHECK-SAME:  memref<24x192xf32, #[[$MAP1]]>, memref<192x16xf32, #[[$MAP1]]>)
// CHECK-SAME:             outs(%[[C_sub]] : memref<24x16xf32, #[[$MAP1]]>)

// CHECK:    linalg.yield
