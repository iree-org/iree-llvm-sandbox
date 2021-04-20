// RUN: export M=1024 && export N=1024 && export K=1024 && export ITERS=25 &&\
// RUN: cat %p/matmul_f32_base.mlir | sed 's@${M}@'"$M"'@g'| sed 's@${K}@'"$K"'@g' | sed 's@${N}@'"$N"'@g'| sed 's@${ITERS}@'"$ITERS"'@g' |\

// RUN: mlir-proto-opt -canonicalize -mlir-disable-threading \
// RUN: -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul distribute distribute-tile-sizes=24,16" |\
// RUN: mlir-proto-opt -linalg-comprehensive-bufferize-inplace |\
// RUN: mlir-proto-opt -convert-to-async |\
// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="tiled-loop-to-scf" |\

// RUN: mlir-opt -async-to-async-runtime | \
// RUN: mlir-opt -async-runtime-ref-counting | \
// RUN: mlir-opt -async-runtime-ref-counting-opt | \
// RUN: mlir-opt -canonicalize -convert-vector-to-scf -convert-linalg-to-loops |\
// RUN: mlir-opt -canonicalize -lower-affine -convert-scf-to-std -convert-vector-to-llvm | \
// RUN: mlir-opt -canonicalize -convert-async-to-llvm -convert-std-to-llvm |\


// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%iree_runners_test_dir/libruntime-support%shlibext | \
// RUN: tee | FileCheck %s

// CHECK: ( ( 2048 ) )
