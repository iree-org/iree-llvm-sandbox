// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef JASC_JASC_TRANSFORM_OPS_TD
#define JASC_JASC_TRANSFORM_OPS_TD

include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class Jasc_TransformOp<string mnemonic, list<Trait> traits = []> :
    Op<Transform_Dialect, "jasc." # mnemonic, traits> {
  let cppNamespace = "jasc";
}

def Jasc_MatchTagOp : Jasc_TransformOp<"match_tag", [
  MemoryEffectsOpInterface,
  NavigationTransformOpTrait,
  DeclareOpInterfaceMethods<TransformOpInterface>
]> {
  let summary = "Matches operations with the given tag";

  let arguments = (ins
    TransformHandleTypeInterface:$target,
    StrArrayAttr:$tags);

  let results = (outs TransformHandleTypeInterface:$matched_ops);

  let assemblyFormat = [{
    $tags `in` $target attr-dict `:` functional-type($target, results)
  }];
}

def ApplyFoldFillIntoPadPatternsOp : Op<Transform_Dialect,
    "apply_patterns.fold_fill_into_pad",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Populates a pattern that folds
    "tensor.pad(cst, tensor.extract*(linalg.fill(cst)))" into
    "linalg.fill(cst, empty)" when the padding constant and the fill constant
    are the same.
  }];

  let assemblyFormat = "attr-dict";
  let cppNamespace = "jasc";
}

def ApplyTuningConfigOp : Jasc_TransformOp<"apply_tuning_config", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformEachOpTrait,
  TransformOpInterface,
  ReportTrackingListenerFailuresOpTrait
]> {
  let summary = "Specializes parametric transform IR.";
  let description = [{
    Specializes all TuningParamOps nested under `target` using the values in
    `config`. This means all unknown parameters will have an explicit value
    after applying this transform op. The number of elements in `config` have to
    match the number of tunable parameters nested under `target`. If there are
    no nested tunable parameters this will not perform any modifications and
    return success.
  }];
  let arguments = (ins TransformHandleTypeInterface:$target, ArrayAttr:$config);
  let assemblyFormat = "$target attr-dict `:` type($target)";
  let cppNamespace = "jasc";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *operation,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def TuningParamOp : Jasc_TransformOp<"tuning_param", [
  MemoryEffectsOpInterface,
  DeclareOpInterfaceMethods<TransformOpInterface>,
  ParamProducerTransformOpTrait
]> {
  let summary = "A tunable parameter";
  let description = [{
    Represents an unknown parameter that is to be specified before execution of
    the transform script. Requires a default_value that will be used as value in
    the case no tuning is performed.
  }];
  let arguments = (ins AnyAttr:$default_value, OptionalAttr<AnyAttr>:$tuned_value);
  let results = (outs TransformParamTypeInterface:$param);
  let assemblyFormat = "attr-dict `->` type($param)";
  let cppNamespace = "jasc";
}

def SynchronizeOp : Op<
    Transform_Dialect, "jasc.synchronize", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      TransformEachOpTrait,
      TransformOpInterface,
      ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Inserts a gpu.barrier after a given operation.";

  let arguments = (
      ins TransformHandleTypeInterface:$op);
  let results = (outs TransformHandleTypeInterface:$barrier);
  let assemblyFormat = [{
    $op
    attr-dict
    `:` functional-type(operands, results)}];

  let cppNamespace = "jasc";
    let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *operation,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def Jasc_WrapInGpuLaunchOp : Jasc_TransformOp<"wrap_in_gpu_launch", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformEachOpTrait,
  TransformOpInterface,
]> {
  let summary = "Wraps operations in a gpu.launch region.";

  let arguments = (ins TransformHandleTypeInterface:$ops);
  let results = (outs Transform_ConcreteOpType<"gpu.launch">:$gpu_launch);

  let assemblyFormat = "$ops attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *operation,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // JASC_JASC_TRANSFORM_OPS_TD
