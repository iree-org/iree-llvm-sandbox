// RUN: export M=1024 && export N=1024 && export K=1024 && export ITERS=25 &&\
// RUN: cat %p/matmul_f32_base.mlir | sed 's@${M}@'"$M"'@g'| sed 's@${K}@'"$K"'@g' | sed 's@${N}@'"$N"'@g'| sed 's@${ITERS}@'"$ITERS"'@g' |\

// RUN: mlir-proto-opt -canonicalize -mlir-disable-threading \
// R-UN: -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul distribute distribute-tile-sizes=24,16" |\

// TODO: bufferization bug
// R-UN: mlir-proto-opt -linalg-tensor-codegen-driver="bufferize lower-vector" |\
// R-UN: mlir-proto-opt -linalg-tensor-codegen-strategy="convert-to-async tiled-loop-to-scf" |\

// R-UN: mlir-opt -async-to-async-runtime | \
// R-UN: mlir-opt -async-runtime-ref-counting | \
// R-UN: mlir-opt -async-runtime-ref-counting-opt | \
// R-UN: mlir-opt -canonicalize -convert-vector-to-scf -convert-linalg-to-loops |\
// R-UN: mlir-opt -canonicalize -lower-affine -convert-scf-to-std -convert-vector-to-llvm | \
// R-UN: mlir-opt -canonicalize -convert-async-to-llvm -convert-std-to-llvm |\


// R-UN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// R-UN:   -shared-libs=%iree_runners_test_dir/libruntime-support%shlibext | \
// RUN: tee
//| FileCheck %s

// C-HECK: ( ( 2048 ) )
