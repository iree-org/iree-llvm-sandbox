// RUN: export M=128 && export N=128 && export K=128 && export ITERS=1 &&\
// RUN: cat %p/../matmul_f16_base.mlir | sed 's@${M}@'"$M"'@g'| sed 's@${K}@'"$K"'@g' | sed 's@${N}@'"$N"'@g'| sed 's@${ITERS}@'"$ITERS"'@g' |\

// RUN: mlir-proto-opt -canonicalize -mlir-disable-threading \
// RUN: -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul distribute distribute-tile-sizes=16,16" |\
// Tile K loop with scf::ForOp as Tiledloop doesn't support reduction yet.
// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul tile-sizes=0,0,16 vectorize vector-contract-lowering=false vector-to-scf-conversion=false" |\
// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.fill vectorize vector-contract-lowering=false vector-to-scf-conversion=false" |\
// RUN: mlir-opt -canonicalize |\
// RUN: mlir-opt -linalg-comprehensive-module-bufferize |\
// RUN: mlir-opt -fold-memref-subview-ops -convert-vector-to-gpu -canonicalize -convert-linalg-to-loops  |\
// RUN: mlir-proto-opt -canonicalize -linalg-tensor-codegen-strategy="convert-to-gpu num-gpu-workgrpoups=1,1 distribute-to-gpu-ids tiled-loop-to-scf" |\
// RUN: mlir-opt -gpu-kernel-outlining |\
// RUN: mlir-opt -lower-affine -convert-scf-to-std -convert-vector-to-llvm |\
// RUN: mlir-opt -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm{index-bitwidth=32},gpu-to-cubin{chip=sm_70})' \
// RUN: -gpu-to-llvm |\

// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%linalg_test_lib_dir/libmlir_cuda_runtime%shlibext \
// RUN:   -shared-libs=%iree_runners_test_dir/libruntime-support%shlibext | \
// RUN: tee | FileCheck %s

// CHECK: ( 256 )
