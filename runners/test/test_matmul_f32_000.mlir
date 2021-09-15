// RUN: export M=32 && export N=64 && export K=128 && export ITERS=10 && \
// RUN: cat %p/matmul_f32_base.mlir | sed 's@${M}@'"$M"'@g'| sed 's@${K}@'"$K"'@g' | sed 's@${N}@'"$N"'@g'| sed 's@${ITERS}@'"$ITERS"'@g' |\

// RUN: mlir-opt -linalg-comprehensive-module-bufferize |\
// RUN: tee | FileCheck %s

// CHECK-LABEL: func @init_and_matmul(
//  CHECK-SAME:       %[[A:[0-9a-zA-Z]+]]: memref<
//  CHECK-SAME:       %[[B:[0-9a-zA-Z]+]]: memref<
//  CHECK-SAME:       %[[C:[0-9a-zA-Z]+]]: memref<
//       CHECK:   constant 0.0

// Analysis kicks in, we can write in %[[C]] and no spurious memref.alloc/copies are inserted.
//  CHECK-NEXT:   linalg.fill(%{{.*}}, %[[C]]) : f32, memref<32x64xf32{{.*}}>
//  CHECK-NEXT:   linalg.matmul ins(%[[A]], %[[B]] : memref<32x128xf32{{.*}}>, memref<128x64xf32{{.*}}>) outs(%[[C]] : memref<32x64xf32{{.*}}>)
//  CHECK-NEXT:   return

// CHECK-LABEL: func @exec(
//   CHECK-DAG:   %[[f0:.*]] = constant 0.0
//   CHECK-DAG:   %[[f1:.*]] = constant 1.0
//   CHECK-DAG:   %[[f2:.*]] = constant 2.0
//   CHECK-DAG:   memref.alloc() {{.*}} : memref<32x128xf32>
//   CHECK-DAG:   memref.alloc() {{.*}} : memref<128x64xf32>
//   CHECK-DAG:   memref.alloc() {{.*}} : memref<32x64xf32>
//   CHECK-DAG:   linalg.fill(%[[f1]], %[[A:.*]]) : f32, memref<32x128xf32>
//   CHECK-DAG:   linalg.fill(%[[f2]], %[[B:.*]]) : f32, memref<128x64xf32>
//   CHECK-DAG:   linalg.fill(%[[f0]], %[[C:.*]]) : f32, memref<32x64xf32>

//       CHECK:   call @rtclock() : () -> f64
//       CHECK:   scf.for %{{.*}} {
// At function boundary we are still pessimistic, so spurious memref.cast to most dynamic strided memrefs are introduced.
// These will go away in the future.
//  CHECK-NEXT:     call @init_and_matmul({{.*}}) : (memref<32x128xf32{{.*}}>, memref<128x64xf32{{.*}}>, memref<32x64xf32{{.*}}>) -> ()
//  CHECK-NEXT:   }
//       CHECK:   call @rtclock() : () -> f64
//       CHECK:   vector.transfer_read %[[C]]
