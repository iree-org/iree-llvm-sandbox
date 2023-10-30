#ifndef JASC_DIALECT_DIALECT
#define JASC_DIALECT_DIALECT

include "mlir/IR/DialectBase.td"

def Jasc_Dialect : Dialect {
  let name = "jasc";
  let cppNamespace = "::jasc";
  let dependentDialects = [
    "::mlir::transform::TransformDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
  ];
}

#endif  // JASC_DIALECT_DIALECT