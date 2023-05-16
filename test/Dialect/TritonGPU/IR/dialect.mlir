// Check that triton_gpu dialect is available in structured-opt.
// RUN: structured-opt %s | FileCheck %s

// CHECK-LABEL: @module
// CHECK-SAME:     triton_gpu.num-warps
module attributes {"triton_gpu.num-warps" = 4 : i32} {}
