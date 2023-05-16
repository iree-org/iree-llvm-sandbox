// Check that transforms on triton_gpu dialect are available in structured-opt.
// RUN: structured-opt -tritongpu-pipeline %s | FileCheck %s

// CHECK-LABEL: module
// CHECK-SAME:     triton_gpu.num-warps
module attributes {"triton_gpu.num-warps" = 4 : i32} {}
