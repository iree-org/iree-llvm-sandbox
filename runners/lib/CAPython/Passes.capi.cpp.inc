// TODO: auto-gen this once in a stable place.

namespace mlir {
namespace linalg {
  void registerLinalgTensorCodegenStrategyPass();
}
}

void mlirRegisterLinalgTensorCodegenStrategyPass() {
  mlir::linalg::registerLinalgTensorCodegenStrategyPass();
}
