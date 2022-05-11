#include "IteratorAnalysis.h"
#include "iterators/Dialect/Iterators/IR/Iterators.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace iterators {

IteratorAnalysis::IteratorAnalysis(Operation *parentOp) : parentOp(parentOp) {
  parentOp->walk([&](Operation *op) {
    llvm::TypeSwitch<Operation *, void>(op).Case<SampleInputOp, ReduceOp>(
        [&](auto op) { buildIteratorInfo(op); });
  });
}

llvm::Optional<IteratorAnalysis::IteratorInfo>
IteratorAnalysis::getIteratorInfo(Operation *op) const {
  auto it = opMap.find(op);
  if (it == opMap.end())
    return llvm::None;
  return it->getSecond();
}

llvm::SmallVector<SymbolRefAttr, 3>
IteratorAnalysis::createFunctionNames(Operation *op) {
  ModuleOp module = op->template getParentOfType<ModuleOp>();

  llvm::SmallVector<SymbolRefAttr, 3> symbols;
  for (auto const *suffix :
       std::array<const char *, 3>{"Open", "Next", "Close"}) {
    // Construct base name from op type and Open/Next/Close.
    auto baseName = StringAttr::get(
        op->getContext(),
        (op->getName().getStringRef() + Twine(".") + suffix).str());

    // Make name unique. This may increment uniqueNumber.
    StringAttr uniqueName = createUniqueFunctionName(baseName, module);

    auto symbol = SymbolRefAttr::get(op->getContext(), uniqueName);
    symbols.push_back(symbol);
  }

  // Increment such that subsequent calls get a different value.
  uniqueNumber++;

  return symbols;
}

StringAttr IteratorAnalysis::createUniqueFunctionName(StringRef prefix,
                                                      ModuleOp module) {
  llvm::SmallString<64> candidateName;
  while (true) {
    (prefix + Twine(".") + Twine(uniqueNumber)).toStringRef(candidateName);
    if (!module.lookupSymbol<func::FuncOp>(candidateName)) {
      break;
    }
    uniqueNumber++;
  }
  return StringAttr::get(module.getContext(), candidateName);
}

/// The state of SampleInputOp consists of a single number that corresponds
/// to the next number returned by the iterator.
LLVM::LLVMStructType IteratorAnalysis::computeStateType(SampleInputOp op) {
  auto resultType = op.result().getType().dyn_cast<StreamType>();
  assert(resultType);
  auto elementType =
      resultType.getElementType().dyn_cast<LLVM::LLVMStructType>();
  assert(elementType && elementType.getBody().size() == 1);
  auto counterType = elementType.getBody().front();
  return LLVM::LLVMStructType::getNewIdentified(
      op->getContext(), "iterators.sampleInputState", {counterType});
}

/// The state of ReduceOp only consists of the state of its upstream.
LLVM::LLVMStructType IteratorAnalysis::computeStateType(ReduceOp op) {
  auto maybeUpstreamInfo = getIteratorInfo(op.input().getDefiningOp());
  assert(maybeUpstreamInfo.hasValue());
  auto upstreamInfo = maybeUpstreamInfo.getValue();
  return LLVM::LLVMStructType::getNewIdentified(
      op->getContext(), "iterators.reduceState", {upstreamInfo.stateType});
}

} // namespace iterators
} // namespace mlir
