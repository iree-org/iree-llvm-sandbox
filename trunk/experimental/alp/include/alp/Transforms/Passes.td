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
  ];
}

def ExtractKernelTailPass: Pass<"alp-extract-kernel-tail", "ModuleOp"> {
  let summary = "Pass to extract the kernel in a separate function.";
  let constructor = "mlir::createExtractKernelTailPass()";
  let options = [
    Option<"anchorFuncOpName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
  ];
}

def ModuloSchedulingPass: Pass<"alp-modulo-scheduling", "FuncOp"> {
  let summary = "Pass to modulo-schedule a loop.";
  let constructor = "mlir::createModuloSchedulingPass()";
  let options = [

    Option<"unrolling", "unrolling", /*type*/"int", /*default=*/"2",
      "Unrolling level before scheduling the loop.">,
    Option<"distance", "distance", /*type*/"int", /*default=*/"1",
      "Unrolling level before scheduling the loop.">,
    Option<"interleave", "interleave", /*type*/"bool", /*default=*/"false",
      "interleave the kernel computation while modulo scheduling.">,
  ];
}

def LegalizePass: Pass<"alp-legalize", "FuncOp"> {
  let summary = "Pass to legalize vector operations.";
  let constructor = "mlir::createLegalizePass()";
  let options = [
    Option<"anchorFuncOpName", "anchor-func", "std::string", /*default=*/"\"kernel\"",
      "Which func op is the anchor to latch on.">,
    Option<"vectorLength", "vector-len", "int", /*default=*/"4",
      "Size of the vector we want to legalize on.">,
  ];
}

def ForToDoWhileLoop: Pass<"alp-for-to-dowhile", "FuncOp"> {
  let summary = "Pass to legalize vector operations.";
  let constructor = "mlir::createForToDoWhileLoopPass()";
  let options = [
    Option<"anchorFuncOpName", "anchor-func", "std::string", /*default=*/"\"kernel\"",
      "Which func op is the anchor to latch on.">,
  ];
}


#endif // ALP_LLVM_SANDBOX_PASSES
