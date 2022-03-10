namespace mlir {
namespace linalg {
namespace transform {

void registerLinalgTransformInterpreterPass();
void registerLinalgTransformExpertExpansionPass();
void registerDropScheduleFromModulePass();

} // namespace transform
} // namespace linalg
} // namespace mlir

namespace mlir {
class Pass;
std::unique_ptr<Pass> createLinalgTransformInterpreterPass();
std::unique_ptr<Pass> createDropScheduleFromModulePass();
} // namespace mlir
