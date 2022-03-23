//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//

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
