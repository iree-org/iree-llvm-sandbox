//===- Registration.cpp - Centralize the registration mess ----------------===//
//
// Isolate the registration pieces that should never be duplicated by clients.
//
//===----------------------------------------------------------------------===//

#include "Registration.h"
#include "Dialect/VectorExt/VectorExtDialect.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Arithmetic/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::linalg;

#include "Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "Dialect/LinalgExt/LinalgExtBufferization.h"
#include "Dialect/LinalgExt/Passes/Passes.h"
#include "Dialect/LinalgTransform/LinalgTransformOps.h"
#include "Dialect/LinalgTransform/Passes.h"

using namespace mlir::iree_compiler::IREE;

//===----------------------------------------------------------------------===//
// Optional dialects and projects.
//===----------------------------------------------------------------------===//

#ifdef SANDBOX_ENABLE_ITERATORS
#include "iterators/Conversion/Passes.h"
#include "iterators/Dialect/Iterators/IR/Iterators.h"

static void registerIteratorDialects(DialectRegistry &registry) {
  registry.insert<mlir::iterators::IteratorsDialect>();
  registerIteratorsConversionPasses();
}
#else
static void registerIteratorDialects(DialectRegistry &registry) {}
#endif

#ifdef SANDBOX_ENABLE_ALP
#include "alp/Transforms/Passes.h"
#endif

static void registerExperimentalPasses() {
#ifdef SANDBOX_ENABLE_ALP
  registerALPPasses();
#endif
}

//===----------------------------------------------------------------------===//
// Non-optional registrations
//===----------------------------------------------------------------------===//

void mlir::registerOutsideOfDialectRegistry() {
  registerDriverPasses();
  registerExperimentalPasses();
}

void mlir::registerIntoDialectRegistry(DialectRegistry &registry) {
  registerAllDialects(registry);
  registry.insert<vector_ext::VectorExtDialect>();
  registerIteratorDialects(registry);

  // Tiling external models.
  LinalgExt::registerTilingInterfaceExternalModels(registry);
  // Bufferize external models.
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  vector::registerBufferizableOpInterfaceExternalModels(registry);
  LinalgExt::registerBufferizableOpInterfaceExternalModels(registry);
}
