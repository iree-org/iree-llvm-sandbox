//===-- TritonSPMDToFuncArgs.h - Triton SPMD ops to func args ---*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_CONVERSION_TRITONSPMDTOFUNCARGS_TRITONSPMDTOFUNCARGS_H
#define STRUCTURED_CONVERSION_TRITONSPMDTOFUNCARGS_TRITONSPMDTOFUNCARGS_H

#include <memory>

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;

/// Create a pass to convert Triton func ops to the func dialect.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonSPMDToFuncArgsPass();

} // namespace mlir

#endif // STRUCTURED_CONVERSION_TRITONSPMDTOFUNCARGS_TRITONSPMDTOFUNCARGS_H
