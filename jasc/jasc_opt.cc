#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "dialect/dialect.h"
#include "gpu_lowering_passes.h"
#include "mlir_lowering.h"
#include "transform_ops/dialect_extension.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  jasc::registerGPULoweringPasses();
  jasc::registerMLIRLoweringPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllToLLVMIRTranslations(registry);
  jasc::registerTransformDialectExtension(registry);

  registry.insert<
      // clang-format off
      jasc::JascDialect
      // clang-format on
      >();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR modular optimizer driver\n", registry));
}
