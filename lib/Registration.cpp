//===- Registration.cpp - Centralize the registration mess ----------------===//
//
// Isolate the registration pieces that should never be duplicated by clients.
//
//===----------------------------------------------------------------------===//

#include "Registration.h"
#include "Dialects/LinalgTransform/LinalgTransformOps.h"
#include "Dialects/LinalgTransform/Passes.h"
#include "Dialects/VectorExt/VectorExtDialect.h"
#include "Transforms/Passes.h"

#include "mlir/Dialect/Arithmetic/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/AffineInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ModuleBufferization.h"
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

//===----------------------------------------------------------------------===//
// Optional dialects and projects.
//===----------------------------------------------------------------------===//

#ifdef SANDBOX_ENABLE_IREE_DIALECTS
#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/LinalgExtBufferization.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"

using namespace mlir::iree_compiler::IREE;

static void registerIreeDialects(DialectRegistry &registry) {
  registry.insert<mlir::iree_compiler::IREE::Input::IREEInputDialect>();
  registry.insert<mlir::iree_compiler::IREE::LinalgExt::IREELinalgExtDialect>();
  mlir::iree_compiler::IREE::LinalgExt::registerPasses();
}
#else
static void registerIreeDialects(DialectRegistry &registry) {}
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
// Test passes.
//===----------------------------------------------------------------------===//
namespace mlir {
namespace test_ext {
void registerTestStagedPatternRewriteDriver();
void registerTestVectorMaskingUtils();
void registerTestListenerPasses();
void registerTestLinalgTransformWrapScope();
void registerTestVectorWarps();
} // namespace test_ext
} // namespace mlir

void registerTestPasses() {
  mlir::test_ext::registerTestStagedPatternRewriteDriver();
  mlir::test_ext::registerTestVectorMaskingUtils();
  mlir::test_ext::registerTestListenerPasses();
  mlir::test_ext::registerTestLinalgTransformWrapScope();
  mlir::test_ext::registerTestVectorWarps();
}

//===----------------------------------------------------------------------===//
// Non-optional registrations
//===----------------------------------------------------------------------===//

void mlir::registerOutsideOfDialectRegistry() {
  registerDriverPasses();
  registerExperimentalPasses();
  registerTestPasses();
  transform::registerLinalgTransformInterpreterPass();
  transform::registerDropScheduleFromModulePass();
  transform::registerLinalgTransformExpertExpansionPass();
}

void mlir::registerIntoDialectRegistry(DialectRegistry &registry) {
  registerAllDialects(registry);
  registerIreeDialects(registry);
  registry.insert<linalg::transform::LinalgTransformDialect,
                  vector_ext::VectorExtDialect>();

  // Tiling external models.
  LinalgExt::registerTilingInterfaceExternalModels(registry);

  // Bufferization external models.
  linalg::comprehensive_bufferize::affine_ext::
      registerBufferizableOpInterfaceExternalModels(registry);
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::comprehensive_bufferize::std_ext::
      registerModuleBufferizationExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  vector::registerBufferizableOpInterfaceExternalModels(registry);
  LinalgExt::registerBufferizableOpInterfaceExternalModels(registry);
}
