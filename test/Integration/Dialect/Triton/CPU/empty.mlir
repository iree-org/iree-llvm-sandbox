// Run through CPU runner to make sure the output of the full pipeline is valid.
// RUN: structured-opt %s \
// RUN:   -convert-triton-to-tritongpu \
// RUN:   -convert-triton-gpu-to-llvm \
// RUN: | mlir-cpu-runner -e kernel -entry-point-result=void

tt.func public @kernel() {
  tt.return
}
