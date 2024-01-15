// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Transform script for GPU post-bufferization codegen.
transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  // Introduce gpu.launch ops for every linalg op.
  %linalg_ops = transform.structured.match interface {LinalgOp} in %arg0
    : (!transform.any_op) -> !transform.any_op
  transform.jasc.wrap_in_gpu_launch %linalg_ops
    : (!transform.any_op) -> !transform.op<"gpu.launch">
}