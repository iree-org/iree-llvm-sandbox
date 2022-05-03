//===-- Passes.td - Driver pass definition file -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IREE_LLVM_SANDBOX_PASSES
#define IREE_LLVM_SANDBOX_PASSES

include "mlir/Pass/PassBase.td"

def UnrollOneVectorOp : Pass<"unroll-one-vector-op", "FuncOp"> {
  let summary = "Pass to unroll a vector op to a target size.";
  let constructor = "mlir::createUnrollOneVectorOpPass()";
  let options = [
    Option<"anchorFuncOpName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
    Option<"anchorOpName", "anchor-op", "std::string", /*default=*/"",
      "Which unique op within the func is the anchor to latch on.">,

    // UnrollOneVectorOp options.
    ListOption<"sourceShape", "source-shape", "int64_t", "Source vector shape",
               "llvm::cl::ZeroOrMore">,
    ListOption<"targetShape", "target-shape", "int64_t", "Target vector shape",
               "llvm::cl::ZeroOrMore">,
  ];
  let dependentDialects = [
    "::mlir::arith::ArithmeticDialect", "::mlir::AffineDialect",
    "::mlir::linalg::LinalgDialect", "::mlir::scf::SCFDialect",
    "::mlir::func::FuncDialect", "::mlir::tensor::TensorDialect",
    "::mlir::vector::VectorDialect"
  ];
}

#endif // IREE_LLVM_SANDBOX_PASSES
