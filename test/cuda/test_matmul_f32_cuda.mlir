// RUN: export M=128 && export N=128 && export K=128 && export ITERS=1 &&\
// RUN: cat %p/../matmul_f32_base.mlir | sed 's@${M}@'"$M"'@g'| sed 's@${K}@'"$K"'@g' | sed 's@${N}@'"$N"'@g'| sed 's@${ITERS}@'"$ITERS"'@g' |\

// RUN: mlir-proto-opt -canonicalize -mlir-disable-threading \
// R-UN: -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul distribute distribute-tile-sizes=16,16" |\
// R-UN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul distribute distribute-tile-sizes=4,4" |\
// Tile K loop with scf::ForOp as Tiledloop doesn't support reduction yet.
// R-UN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul tile-sizes=0,0,4 vectorize" |\
// R-UN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.fill vectorize" |\
// R-UN: mlir-opt --linalg-comprehensive-module-bufferize |\
// R-UN: mlir-opt -fold-memref-subview-ops -canonicalize |\
// R-UN: mlir-proto-opt -canonicalize -linalg-tensor-codegen-strategy="convert-to-gpu num-gpu-workgrpoups=2,2 distribute-to-gpu-ids tiled-loop-to-scf" |\
// R-UN: mlir-opt -convert-vector-to-scf=full-unroll=true -convert-linalg-to-loops |\
// R-UN: mlir-opt -gpu-kernel-outlining |\
// R-UN: mlir-opt -lower-affine -convert-scf-to-std |\
// R-UN: mlir-opt -convert-vector-to-llvm -convert-memref-to-llvm -canonicalize -cse |\
// R-UN: mlir-opt -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,canonicalize,gpu-to-cubin)' \
// R-UN: -gpu-to-llvm |\

// R-UN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// R-UN:   -shared-libs=%linalg_test_lib_dir/libmlir_cuda_runtime%shlibext \
// R-UN:   -shared-libs=%iree_runners_test_dir/libruntime-support%shlibext | \
// RUN: tee
//| FileCheck %s

// CHECK: ( ( 256 ) )
