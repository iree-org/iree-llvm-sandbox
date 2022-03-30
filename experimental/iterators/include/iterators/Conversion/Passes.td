//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_CONVERSION_PASSES
#define ITERATORS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// IteratorsToStandard
//===----------------------------------------------------------------------===//

def ConvertIteratorsToStandard : Pass<"convert-iterators-to-std", "ModuleOp"> {
  let summary = "Convert the operations from the iterators dialect into the "
                "Standard dialect";
  let constructor = "mlir::createConvertIteratorsToStandardPass()";
  let dependentDialects = ["func::FuncDialect", "memref::MemRefDialect"];
}

#endif // ITERATORS_CONVERSION_PASSES
