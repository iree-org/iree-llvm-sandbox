//===-- mlir_lowering.h - Passws for lowering Jasc --------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef JASC_MLIR_LOWERING_H_
#define JASC_MLIR_LOWERING_H_

#include "absl/status/status.h"
#include "mlir/IR/BuiltinOps.h"

namespace jasc {

absl::Status ApplyTransformScript(mlir::ModuleOp module, bool dump_ir);

absl::Status LowerStableHloToCpuLLVM(mlir::ModuleOp module, bool dump_ir);

absl::Status LowerStableHloToGpuLLVM(mlir::ModuleOp module, bool dump_ir);

absl::Status LowerStableHloToLinalg(mlir::ModuleOp module, bool dump_ir);

void registerMLIRLoweringPasses();

}  // namespace jasc

#endif  // JASC_MLIR_LOWERING_H_
