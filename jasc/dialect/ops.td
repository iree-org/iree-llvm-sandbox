#ifndef JASC_DIALECT_OPS
#define JASC_DIALECT_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"

include "dialect.td"

def Jasc_TagRegionOp : Op<Jasc_Dialect, "tag_region", [
  NoRegionArguments,
  SingleBlockImplicitTerminator<"::jasc::ReturnOp">
]> {
  let summary = "Tags a region for matching it with the transform dialect.";

  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs  Variadic<AnyType>:$results);

  let assemblyFormat = "$name $body attr-dict `:` type($results)";
}

def Jasc_ReturnOp : Op<Jasc_Dialect, "return", [Terminator]> {
  let summary = "Terminates a tag region.";

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict `:` type($operands)";

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

#endif  // JASC_DIALECT_OPS