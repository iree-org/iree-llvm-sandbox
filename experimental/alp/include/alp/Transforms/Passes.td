//===-- Passes.td - Alp pass definition file -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ALP_LLVM_SANDBOX_PASSES
#define ALP_LLVM_SANDBOX_PASSES

include "mlir/Pass/PassBase.td"

def ExtractKernelPass: Pass<"alp-extract-kernel", "ModuleOp"> {
  let summary = "Pass to extract the kernel in a separate function.";
  let constructor = "mlir::createExtractKernelPass()";
  let options = [
    Option<"anchorFuncOpName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
    //Option<"anchorOpName", "anchor-op", "std::string", /*default=*/"",
    //  "Which linalg op within the func is the anchor to latch on.">,
  ];
}

def ModuloSchedulingPass: FunctionPass<"alp-modulo-scheduling"> {
  let summary = "Pass to modulo-schedule a loop.";
  let constructor = "mlir::createModuloSchedulingPass()";
  let options = [

    Option<"unrolling", "unrolling", "int", /*default=*/"",
      "Unrolling level before scheduling the loop.">,
  ];
}

#endif // ALP_LLVM_SANDBOX_PASSES
