namespace mlir {
namespace linalg {
namespace transform {

void registerLinalgTransformInterpreterPass();
void registerLinalgTransformExpertExpansionPass();

} // namespace transform
} // namespace linalg
} // namespace mlir

namespace mlir {
class Pass;
std::unique_ptr<Pass> createLinalgTransformInterpreterPass();
} // namespace mlir
