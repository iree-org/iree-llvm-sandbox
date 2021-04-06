// RUN: export M=128 && export N=128 && export K=128 && export ITERS=1 &&\
// RUN: cat %p/matmul_f32_base.mlir | sed 's@${M}@'"$M"'@g'| sed 's@${K}@'"$K"'@g' | sed 's@${N}@'"$N"'@g'| sed 's@${ITERS}@'"$ITERS"'@g' |\

// RUN: mlir-proto-opt -linalg-comprehensive-bufferize-inplace |\
// RUN: mlir-proto-opt -canonicalize -convert-to-gpu="gpu-func-name=init_and_matmul" |\

// RUN: mlir-opt -canonicalize -convert-vector-to-scf -lower-affine -convert-linalg-to-loops |\
// RUN: mlir-opt -canonicalize -convert-scf-to-std |\
// RUN: mlir-opt -gpu-kernel-outlining \
// RUN:   -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)' \
// RUN:   -gpu-to-llvm |\

// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%linalg_test_lib_dir/libmlir_cuda_runtime%shlibext \
// RUN:   -shared-libs=%iree_runners_test_dir/libruntime-support%shlibext \
// RUN: | FileCheck %s

// CHECK: ( ( 256 ) )
