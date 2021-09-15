// RUN: export M=32 && export N=64 && export K=128 && export ITERS=10 && \
// RUN: cat %p/matmul_f32_base.mlir | sed 's@${M}@'"$M"'@g'| sed 's@${K}@'"$K"'@g' | sed 's@${N}@'"$N"'@g'| sed 's@${ITERS}@'"$ITERS"'@g' |\

// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul tile-sizes=5,9,17 pad hoist-padding=1" -canonicalize -cse |\
// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul vectorize vector-contract-lowering=false vectorize-padding" |\
// RUN: mlir-proto-opt -linalg-tensor-codegen-driver="bufferize=true" |\
// RUN: tee | FileCheck %s

// CHECK-LABEL: func @init_and_matmul(
//  CHECK-SAME:       %[[A:[0-9a-zA-Z]+]]: memref<
//  CHECK-SAME:       %[[B:[0-9a-zA-Z]+]]: memref<
//  CHECK-SAME:       %[[C:[0-9a-zA-Z]+]]: memref<
//       CHECK:   constant 0.0
//   CHECK-NOT:   memref.alloc
// At function boundary we are still pessimistic, so spurious memref.cast to most dynamic strided memrefs are introduced.
// These will go away in the future.
//       CHECK:   linalg.fill(%{{.*}}, %[[C]]) : f32, memref<32x64xf32{{.*}}>
//   CHECK-DAG:   %[[PACKED_B:.*]] = memref.alloc() {{.*}} : memref<8x17x9xf32>
//   CHECK-DAG:   %[[PACKED_A:.*]] = memref.alloc() {{.*}} : memref<8x5x17xf32>
//   CHECK-NOT:   copy
//       CHECK:   scf.for %[[I:.*]] =
//       CHECK:     scf.for %[[J:.*]] =
//       CHECK:       scf.for %[[K1:.*]] =
//       CHECK:         %[[PACKED_IDX_B:.*]] = affine.apply
//       CHECK:         memref.subview %[[B]][%[[K1]], %[[J]]] [{{.*}}] [1, 1] : memref<128x64xf32> to memref<?x?xf32
//       CHECK:         %[[VREAD1:.*]] = vector.transfer_read {{.*}} vector<17x9xf32>
//       CHECK:         vector.transfer_write %[[VREAD1]], {{.*}} vector<17x9xf32>
//   CHECK-NOT:         linalg.copy
//       CHECK:       scf.for %[[K2:.*]] =
//       CHECK:         %[[PACKED_IDX_A:.*]] = affine.apply
//       CHECK:         memref.subview %[[A]][%[[I]], %[[K2]]] [{{.*}}] [1, 1] : memref<32x128xf32> to memref<?x?xf32
//       CHECK:         %[[VREAD2:.*]] = vector.transfer_read {{.*}} vector<5x17xf32>
//       CHECK:         vector.transfer_write %[[VREAD2]], {{.*}} vector<5x17xf32>
//   CHECK-NOT:         linalg.copy
//       CHECK:       %[[SVC:.*]] = memref.subview %[[C]]{{.*}} : memref<32x64xf32> to memref<?x?xf32
//       CHECK:       %[[VC:.*]] = vector.transfer_read %[[SVC]]{{.*}} : memref<?x?xf32{{.*}}>, vector<5x9xf32>
//       CHECK:       scf.for %[[K:.*]] = {{.*}} iter_args(%{{.*}} = %[[VC]]) -> (vector<5x9xf32>)
//       CHECK:         %[[PACKED_IDX:.*]] = affine.apply
//       CHECK:         %[[SVA:.*]] = memref.subview %[[PACKED_A]][%[[PACKED_IDX]], 0, 0] [1, 5, 17] [1, 1, 1] : memref<8x5x17xf32> to memref<5x17xf32
//       CHECK:         %[[SVB:.*]] = memref.subview %[[PACKED_B]][%[[PACKED_IDX]], 0, 0] [1, 17, 9] [1, 1, 1] : memref<8x17x9xf32> to memref<17x9xf32
//       CHECK:         vector.transfer_read %[[SVA]]{{.*}} {in_bounds = [true, true]} : memref<5x17xf32{{.*}}>, vector<5x17xf32>
//       CHECK:         vector.transfer_read %[[SVB]]{{.*}}, %cst {in_bounds = [true, true], permutation_map = {{.*}}} : memref<17x9xf32{{.*}}>, vector<9x17xf32>
//       CHECK:         vector.contract
//       CHECK:         scf.yield %{{.*}} : vector<5x9xf32>
//   CHECK-NOT:         copy
//       CHECK:       }
//       CHECK:       vector.transfer_write %{{.*}}, %[[SVC]]{{.*}} : vector<5x9xf32>, memref<?x?xf32
//   CHECK-NOT:       copy
//       CHECK:     }
//       CHECK:   }
//   CHECK-NOT:   copy
//   CHECK-DAG:   memref.dealloc %[[PACKED_A]]
//   CHECK-DAG:   memref.dealloc %[[PACKED_B]]
