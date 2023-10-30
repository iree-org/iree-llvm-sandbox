#ifndef THIRD_PARTY_MLIR_EDGE_JASC_MLIR_LOWERING_H_
#define THIRD_PARTY_MLIR_EDGE_JASC_MLIR_LOWERING_H_

#include "absl/status/status.h"
#include "mlir/IR/BuiltinOps.h"

namespace jasc {

absl::Status ApplyTransformScript(mlir::ModuleOp module, bool dump_ir);

absl::Status LowerStableHloToCpuLLVM(mlir::ModuleOp module, bool dump_ir);

absl::Status LowerStableHloToGpuLLVM(mlir::ModuleOp module, bool dump_ir);

absl::Status LowerStableHloToLinalg(mlir::ModuleOp module, bool dump_ir);

void registerMLIRLoweringPasses();

}  // namespace jasc

#endif  // THIRD_PARTY_MLIR_EDGE_JASC_MLIR_LOWERING_H_
