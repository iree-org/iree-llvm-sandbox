#include "IteratorAnalysis.h"

#include "iterators/Dialect/Iterators/IR/Iterators.h"
#include "iterators/Utils/NameAssigner.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace iterators {

IteratorAnalysis::IteratorAnalysis(Operation *parentOp, ModuleOp module)
    : parentOp(parentOp), nameAssigner(module) {
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

void IteratorAnalysis::buildIteratorInfo(Operation *op) {
  llvm::SmallVector<SymbolRefAttr, 3> symbols = assignFunctionNames(op);
  SymbolRefAttr openFuncSymbol = symbols[0];
  SymbolRefAttr nextFuncSymbol = symbols[1];
  SymbolRefAttr closeFuncSymbol = symbols[2];

  auto stateType = llvm::TypeSwitch<Operation *, LLVM::LLVMStructType>(op)
                       .Case<ReduceOp, SampleInputOp>(
                           [&](auto op) { return computeStateType(op); })
                       .Default(LLVM::LLVMStructType());

  opMap.try_emplace(op, IteratorInfo{stateType, openFuncSymbol, nextFuncSymbol,
                                     closeFuncSymbol});
}

llvm::SmallVector<SymbolRefAttr, 3>
IteratorAnalysis::assignFunctionNames(Operation *op) {
  llvm::SmallVector<SymbolRefAttr, 3> symbols;
  for (auto const *suffix :
       std::array<const char *, 3>{"open", "next", "close"}) {
    // Construct base name from op type and Open/Next/Close.
    auto baseName = StringAttr::get(
        op->getContext(),
        (op->getName().getStringRef() + Twine(".") + suffix).str());

    // Make name unique. This may increment uniqueNumber.
    StringAttr uniqueName = nameAssigner.assignName(baseName);

    auto symbol = SymbolRefAttr::get(op->getContext(), uniqueName);
    symbols.push_back(symbol);
  }

  return symbols;
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
      op->getContext(), "iterators.sample_input_state", {counterType});
}

/// The state of ReduceOp only consists of the state of its upstream iterator,
/// i.e., the state of the iterator that produces its input stream.
LLVM::LLVMStructType IteratorAnalysis::computeStateType(ReduceOp op) {
  auto maybeUpstreamInfo = getIteratorInfo(op.input().getDefiningOp());
  assert(maybeUpstreamInfo.hasValue());
  auto upstreamInfo = maybeUpstreamInfo.getValue();
  return LLVM::LLVMStructType::getNewIdentified(
      op->getContext(), "iterators.reduce_state", {upstreamInfo.stateType});
}

} // namespace iterators
} // namespace mlir
