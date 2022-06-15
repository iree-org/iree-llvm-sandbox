#include "IteratorAnalysis.h"

#include "iterators/Dialect/Iterators/IR/Iterators.h"
#include "iterators/Utils/NameAssigner.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::iterators;

using SymbolTriple = std::tuple<SymbolRefAttr, SymbolRefAttr, SymbolRefAttr>;

/// Pre-assigns names for the Open/Next/Close functions of the given iterator
/// op. The conversion is expected to create these names in the lowering of
/// the corresponding op and can look them up in the lowering of downstream
/// iterators.
static SymbolTriple assignFunctionNames(Operation *op,
                                        NameAssigner &nameAssigner) {
  SmallVector<SymbolRefAttr, 3> symbols;
  for (auto suffix : {"open", "next", "close"}) {
    // Construct base name from op type and Open/Next/Close.
    auto baseName = StringAttr::get(
        op->getContext(),
        (op->getName().getStringRef() + Twine(".") + suffix).str());

    // Make name unique. This may increment uniqueNumber.
    StringAttr uniqueName = nameAssigner.assignName(baseName);

    auto symbol = SymbolRefAttr::get(op->getContext(), uniqueName);
    symbols.push_back(symbol);
  }
  return std::make_tuple(symbols[0], symbols[1], symbols[2]);
}

/// The state of SampleInputOp consists of a single number that corresponds
/// to the next number returned by the iterator.
static LLVM::LLVMStructType computeStateType(SampleInputOp op) {
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
static LLVM::LLVMStructType
computeStateType(ReduceOp op, LLVM::LLVMStructType upstreamStateType) {
  return LLVM::LLVMStructType::getNewIdentified(
      op->getContext(), "iterators.reduce_state", {upstreamStateType});
}

/// Build IteratorInfo, assigning new unique names as needed.
/// Takes the `LLVM::LLVMStructType` as a parameter, to ensure proper build
/// order (all uses are visited before any def).
mlir::iterators::IteratorInfo::IteratorInfo(IteratorOpInterface op,
                                            NameAssigner &nameAssigner,
                                            LLVM::LLVMStructType t) {
  std::tie(openFunc, nextFunc, closeFunc) =
      assignFunctionNames(op, nameAssigner);
  stateType = t;
}

IteratorInfo mlir::iterators::IteratorAnalysis::getExpectedIteratorInfo(
    IteratorOpInterface op) const {
  auto it = opMap.find(op);
  assert(it != opMap.end() && "analysis does not contain this op");
  return it->getSecond();
}

void mlir::iterators::IteratorAnalysis::setIteratorInfo(
    IteratorOpInterface op, const IteratorInfo &info) {
  assert(info.stateType && "state type must be computed");
  auto inserted = opMap.insert({op, info});
  assert(inserted.second && "IteratorInfo already present");
}

template <typename OpTy>
static OpTy getSelfOrParentOfType(Operation *op) {
  auto maybe = dyn_cast<OpTy>(op);
  return maybe ? maybe : op->getParentOfType<OpTy>();
}

mlir::iterators::IteratorAnalysis::IteratorAnalysis(Operation *rootOp)
    : rootOp(rootOp), nameAssigner(getSelfOrParentOfType<ModuleOp>(rootOp)) {
  /// This needs to be built in use-def order so that all uses are visited
  /// before any def.
  rootOp->walk([&](IteratorOpInterface iteratorOp) {
    llvm::TypeSwitch<Operation *, void>(iteratorOp)
        /// The state of SampleInputOp consists of a single number that
        /// corresponds to the next number returned by the iterator.
        .Case<SampleInputOp>([&](auto op) {
          auto stateType = computeStateType(op);
          setIteratorInfo(op, IteratorInfo(op, nameAssigner, stateType));
        })
        /// The state of ReduceOp only consists of the state of its upstream
        /// iterator, i.e., the state of the iterator that produces its input
        /// stream.
        // TODO: ReduceOp verifier that op.input does not come from a bbArg.
        .Case<ReduceOp>([&](auto op) {
          Operation *def = op.input().getDefiningOp();
          auto stateType =
              computeStateType(op, getExpectedIteratorInfo(def).stateType);
          setIteratorInfo(op, IteratorInfo(op, nameAssigner, stateType));
        })
        .Default([&](auto op) { assert(false && "Unexpected op"); });
  });
}
