// RUN: export M=192 && export N=192 && export K=192 && export ITERS=1 &&\
// RUN: cat %p/matmul_f32_base.mlir | sed 's@${M}@'"$M"'@g'| sed 's@${K}@'"$K"'@g' | sed 's@${N}@'"$N"'@g'| sed 's@${ITERS}@'"$ITERS"'@g' |\

// RUN: mlir-proto-opt -canonicalize -mlir-disable-threading \
// RUN: -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul distribute distribute-tile-sizes=24,16" \
// RUN: -linalg-comprehensive-bufferize-inplace | FileCheck %s

// CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0) -> (24, -d0 + 192)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1)[s0] -> (d0 * 192 + s0 + d1)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> (16, -d0 + 192)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0) -> (-d0 + 192, 24)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0) -> (-d0 + 192, 16)>

// CHECK-LABEL: func @init_and_matmul(
// CHECK-SAME:    %[[A:.*]]: memref<192x192xf32>, %[[B:.*]]: memref<192x192xf32>,
// CHECK-SAME:    %[[C:.*]]: memref<192x192xf32>) {
// CHECK:  %[[C0_F32:.*]] = constant 0.000000e+00 : f32
// CHECK:  %[[C24:.*]] = constant 24 : index
// CHECK:  %[[C16:.*]] = constant 16 : index
// CHECK:  %[[C0:.*]] = constant 0 : index
// CHECK:  %[[C192:.*]] = constant 192 : index

// CHECK:  linalg.tiled_loop (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME: to (%[[C192]], %[[C192]]) step (%[[C24]], %[[C16]])
// CHECK-SAME: ins (%[[A]], %[[B]]: memref<192x192xf32>, memref<192x192xf32>)
// CHECK-SAME: outs (%[[C]]:memref<192x192xf32>) {

// CHECK:    %[[A_sub:.*]] = memref.subview %[[A]]{{\[}}%[[I]], 0]
// CHECK:    %[[B_sub:.*]] = memref.subview %[[B]][0, %[[J]]]
// CHECK:    %[[C_sub:.*]] = memref.subview %[[C]]{{\[}}%[[I]], %[[J]]]

// CHECK:    linalg.fill(%[[C_sub]], %[[C0_F32]])
// CHECK:    linalg.matmul ins(%[[A_sub]], %[[B_sub]]
// CHECK-SAME:  memref<?x192xf32, #[[$MAP1]]>, memref<192x?xf32, #[[$MAP1]]>)
// CHECK-SAME:             outs(%[[C_sub]] : memref<?x?xf32, #[[$MAP1]]>)

// CHECK:    linalg.yield
