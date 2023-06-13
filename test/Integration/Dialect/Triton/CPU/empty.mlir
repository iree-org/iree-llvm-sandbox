// Run through CPU runner to make sure the output of the full pipeline is valid.

// Pipeline with original passes from Triton.
// RUN: structured-opt %s \
// RUN:   -convert-triton-to-tritongpu \
// RUN:   -convert-triton-gpu-to-llvm \
// RUN: | mlir-cpu-runner -e kernel -entry-point-result=void

// Pipeline with our own passes.
// RUN: structured-opt %s \
// RUN:   -convert-triton-func-to-func \
// RUN:   -convert-triton-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN: | mlir-cpu-runner -e kernel -entry-point-result=void

tt.func public @kernel() {
  tt.return
}
