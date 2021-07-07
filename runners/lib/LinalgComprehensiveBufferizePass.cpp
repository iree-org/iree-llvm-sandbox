//===- LinalgComprehensiveBufferizePass.cpp - Bufferize Linalg on tensors -===//
//
// Convert from Linalg ops on tensors to Linalg ops on buffers in a single pass.
// Aggressively try to perform inPlace bufferization and fail if any allocation
// tries to cross function boundaries or if the pattern
// `memref.tensor_load(tensor_memref(x))` is deemed unsafe (very conservative
// impl for now).
//
//===----------------------------------------------------------------------===//

#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/BufferUtils.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "linalg-comprehensive-bufferize-inplace"

using namespace mlir;
using namespace linalg;

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

/// Comprehensive Linalg bufferize pass that aims at avoiding phase-ordering and
/// safety + optimization issues that are present in upstream bufferization.
/// At the same time, this pass only cares about enabling aggressive inPlace
/// bufferization for linalg ops and scf.for, **including across function
/// boundaries**.
/// In particular no branching behavior is supported atm besides function calls
/// and scf.for.
/// This ModulePass consists in the following steps:
/// 1. perform a `funcArgumentsInPlaceAnalysis` which traverses all CallOps and
///    determine whether any tensor operand could potentially bufferize to a
///    buffer that can be updated inPlace (i.e. an in-out buffer).
///    Such operands are ones whose value is not read in any other op at the
///    caller site.
///    As a result of this analysis, CallOp operands are marked with
///    `kInPlaceResultsAttrName`. The "meet" of all `kInPlaceResultsAttrName`
///    for all `callOp` to a given FuncOp determines the
///    `kInPlaceResultsAttrName` for that FuncOp.
/// 2. traverse each FuncOp and perform bufferization within the function
///    boundaries. Bufferization occurs by:
///    a. performing an inPlace analysis `inPlaceAnalysisFuncOpInternals`
///       which marks each operation within the function with the
///       `kInPlaceResultsAttrName` attribute.
///    b. traversing each operation in the function and rewriting it in
///       buffer form and keeping a BlockAndValueMapping mapping of the
///       rewrites.
///       New allocations are introduced during this step.
///       TODO: Allocation + depending op hoisting to outermost enclosing
///       sequential scope.
/// 3. once bufferization within function boundaries is done, the next step
///    runs `bufferizeFunctionsAndCalls`, which involves:
///    a. detecting `function_arg -> memref.buffer_cast -> memref.tensor_load ->
///    return`
///       patterns for each FuncOp, which determines the `tiedResultMap` between
///       function args and results.
///    b. rewrite function arguments and returns in buffer forms, skipping the
///       tensors that appear in the `tiedResultMap`.
///    c. bufferize the CallOps using the callee's `tiedResultMap`.
///
/// TensorToMemRefOps are only ever inserted as a transient abstraction for
/// function arguments that have not yet been bufferized.
/// All other places either allocate or forward existing buffers.
///
/// TensorLoadOps are only even inserted as a transient abstraction for
/// terminators (return, scf.yield).
/// The `function_arg -> memref.buffer_cast -> memref.tensor_load -> return` is
/// used to analyze which function result ties to a function operand.
namespace {
struct LinalgComprehensiveBufferizePass
    : public PassWrapper<LinalgComprehensiveBufferizePass,
                         OperationPass<ModuleOp>> {
  LinalgComprehensiveBufferizePass()
      : enablingPassPipeline(OpPassManager("func")) {
    enablingPassPipeline.addPass(createCanonicalizerPass());
    enablingPassPipeline.addPass(createCSEPass());
    enablingPassPipeline.addPass(createLoopInvariantCodeMotionPass());
  }
  LinalgComprehensiveBufferizePass(const LinalgComprehensiveBufferizePass &pass)
      : enablingPassPipeline(pass.enablingPassPipeline) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LinalgDialect, memref::MemRefDialect, scf::SCFDialect,
                    StandardOpsDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;

  void runEnablingTransforms(FuncOp funcOp);
  void bufferizeFuncOpInternals(
      FuncOp funcOp, BlockAndValueMapping &bvm, GlobalCreator &globals,
      const DenseMap<FuncOp, SmallVector<int64_t>> &tiedResultsMap);
  void inPlaceAnalysisFuncOpInternals(FuncOp funcOp,
                                      const DominanceInfo &domInfo);

  /// Dynamic pass pipeline of transformations that enable better inPlace
  /// bufferization.
  OpPassManager enablingPassPipeline;
};
}  // namespace

//===----------------------------------------------------------------------===//
// Bufferization-specific attribute manipulation.
//===----------------------------------------------------------------------===//

/// Attribute marker to specify results that can be bufferized inPlace.
constexpr StringLiteral kInPlaceResultsAttrName = "__inplace_results_attr__";

/// Attribute marker to specify func/call arguments that can be written inPlace
/// from the perspective of the caller.
constexpr StringLiteral kWriteableFuncBufferArgsAttrName =
    "__writeable_func_buffer_args_attr__";

// default clause
enum class InPlaceSpec {
  False,
  True,
  None,
};

static StringRef stringify(InPlaceSpec val) {
  switch (val) {
    case InPlaceSpec::False:
      return "false";
    case InPlaceSpec::True:
      return "true";
    case InPlaceSpec::None:
      return "none";
  }
  return "";
}

static Optional<InPlaceSpec> symbolize(StringRef str) {
  return StringSwitch<Optional<InPlaceSpec>>(str)
      .Case("false", InPlaceSpec::False)
      .Case("true", InPlaceSpec::True)
      .Case("none", InPlaceSpec::None)
      .Default(None);
}

static FuncOp getCalledFunction(CallOpInterface callOp) {
  SymbolRefAttr sym = callOp.getCallableForCallee().dyn_cast<SymbolRefAttr>();
  if (!sym) return nullptr;
  return dyn_cast_or_null<FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

/// Factor out the logic that matches tied OpResult to BlockArgument.
/// For FuncOp the analysis is dependent on the result of bufferization so we
/// always return null.
static OpResult getTiedOpResult(BlockArgument &bbArg) {
  Operation *op = bbArg.getOwner()->getParentOp();
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return forOp->getResult(bbArg.getArgNumber() - /*#iv=*/1);

  if (auto tiledLoop = dyn_cast<TiledLoopOp>(op)) {
    // Tiledloop operands = [lbs, ubs, steps, inputs, outputs],
    // Tiledloop bbArgs = [ivs, inputs, outputs], i.e.
    // input and output indices are shifted by numControlOperands - numLoops.
    OpOperand &operand = tiledLoop->getOpOperand(
        bbArg.getArgNumber() + tiledLoop.getNumControlOperands() -
        tiledLoop.getNumLoops());
    return tiledLoop.getTiedOpResult(operand);
  }
  if (auto funcOp = dyn_cast<FuncOp>(op)) return OpResult();
  op->getParentOfType<FuncOp>()->dump();
  op->dump();
  llvm_unreachable("Unsupported getTiedOpResult(BlockArgument) op");
}

/// Factor out the logic that matches tied OpResult to OpOperand.
/// For CallOp, the analysis is dependent on the result of bufferization of the
/// callee, so we always return null.
/// For terminators there is no possible operand/result tie, so we always return
/// null.
/// Other ops are enumerated on a case-by-case basis for now.
/// TODO: we should really have a TiedOpInterface for this.
static OpResult getTiedOpResult(OpOperand &opOperand) {
  Operation *op = opOperand.getOwner();
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return forOp->getResult(opOperand.getOperandNumber() -
                            forOp.getNumControlOperands());
  if (auto linalgOp = dyn_cast<LinalgOp>(op)) {
    if (opOperand.getOperandNumber() < linalgOp.getNumInputs())
      return OpResult();
    return linalgOp->getResult(opOperand.getOperandNumber() -
                               linalgOp.getNumInputs());
  }
  if (auto tiledLoopOp = dyn_cast<TiledLoopOp>(op))
    return tiledLoopOp.getTiedOpResult(opOperand);
  if (isa<linalg::PadTensorOp, tensor::ExtractSliceOp, tensor::InsertSliceOp,
          tensor::CastOp, vector::TransferReadOp, vector::TransferWriteOp,
          tensor::ExtractOp>(op))
    return op->getResult(0);
  if (op->hasTrait<mlir::OpTrait::IsTerminator>()) return OpResult();
  if (isa<CallOpInterface, vector::PrintOp, vector::ContractionOp>(op))
    return OpResult();
  op->getParentOfType<FuncOp>()->dump();
  op->dump();
  llvm_unreachable("Unsupported getTiedOpResult(OpOperand) op");
}

namespace detail {
static void setInPlaceFuncOrCallArgument(
    Operation *op, unsigned idx, InPlaceSpec inPlace = InPlaceSpec::True) {
  auto funcOp = dyn_cast<FuncOp>(op);
  auto callOp = dyn_cast<CallOpInterface>(op);
  assert((funcOp || callOp) && "must be func or call");

  unsigned numArgs =
      funcOp ? funcOp.getNumArguments() : callOp->getNumOperands();
  auto attr = op->getAttr(kWriteableFuncBufferArgsAttrName)
                  .dyn_cast_or_null<ArrayAttr>();
  SmallVector<StringRef> inPlaceVector =
      attr ? SmallVector<StringRef>(
                 llvm::to_vector<4>(attr.getAsValueRange<StringAttr>()))
           : SmallVector<StringRef>(numArgs, stringify(InPlaceSpec::None));
  LLVM_DEBUG(DBGS() << "Set inPlace=" << stringify(inPlace) << ": " << *op
                    << " @idx=" << idx << "\n");
  inPlaceVector[idx] = stringify(inPlace);
  op->setAttr(kWriteableFuncBufferArgsAttrName,
              OpBuilder(op).getStrArrayAttr(inPlaceVector));
}
}  // namespace detail

static void setInPlaceFuncArgument(BlockArgument arg,
                                   InPlaceSpec inPlace = InPlaceSpec::True) {
  ::detail::setInPlaceFuncOrCallArgument(arg.getOwner()->getParentOp(),
                                         arg.getArgNumber(), inPlace);
}

static void setInPlaceCallArgument(OpOperand &operand,
                                   InPlaceSpec inPlace = InPlaceSpec::True) {
  ::detail::setInPlaceFuncOrCallArgument(operand.getOwner(),
                                         operand.getOperandNumber(), inPlace);
}

static void setInPlaceOpResult(OpResult opResult,
                               InPlaceSpec inPlace = InPlaceSpec::True) {
  if (!opResult) return;

  Operation *op = opResult.getOwner();
  auto attr =
      op->getAttr(kInPlaceResultsAttrName).dyn_cast_or_null<ArrayAttr>();
  SmallVector<StringRef> inPlaceVector =
      attr ? SmallVector<StringRef>(
                 llvm::to_vector<4>(attr.getAsValueRange<StringAttr>()))
           : SmallVector<StringRef>(op->getNumResults(),
                                    stringify(InPlaceSpec::None));
  LLVM_DEBUG(DBGS() << "Set inPlace=" << stringify(inPlace) << ": " << *op
                    << " @idx=" << opResult.getResultNumber() << "\n");
  inPlaceVector[opResult.getResultNumber()] = stringify(inPlace);
  op->setAttr(kInPlaceResultsAttrName,
              OpBuilder(op).getStrArrayAttr(inPlaceVector));
}

/// Get the attribute entry `kInPlaceResultsAttrName`@`idx` corresponding to a
/// tied operand/result pair. If `idx` is llvm::None, this means the `op` has
/// only a single relevant tensor operand/result and that its position is not
/// important. In such cases, we just get the single entry string array
/// attribute @0. If the attribute does not exist yet, return InPlaceSpec::None.
static InPlaceSpec getInPlace(OpResult opResult) {
  if (!opResult) return InPlaceSpec::None;

  Operation *op = opResult.getOwner();
  auto attr =
      op->getAttr(kInPlaceResultsAttrName).dyn_cast_or_null<ArrayAttr>();
  if (!attr) return InPlaceSpec::None;

  // Must return a proper value.
  return *symbolize(*(attr.getAsValueRange<StringAttr>().begin() +
                      opResult.getResultNumber()));
}

namespace detail {
static InPlaceSpec getInPlaceFuncOrCallBufferArg(Operation *op, unsigned idx) {
  auto funcOp = dyn_cast<FuncOp>(op);
  auto callOp = dyn_cast<CallOpInterface>(op);
  assert((funcOp || callOp) && "must be func or call");
  auto attr = op->getAttr(kWriteableFuncBufferArgsAttrName)
                  .dyn_cast_or_null<ArrayAttr>();
  if (!attr) return InPlaceSpec::None;
  // Must return a proper value.
  return *symbolize(*(attr.getAsValueRange<StringAttr>().begin() + idx));
}
}  // namespace detail

/// Get inPlace information depending on the owner of `bbArg`:
///   1. if not a FuncOp, get the information from `kInPlaceResultsAttrName`
///      for the tied op result.
///   2. otherwise, get the information from `kWriteableFuncBufferArgsAttrName`
static InPlaceSpec getInPlace(BlockArgument bbArg) {
  if (!isa<FuncOp>(bbArg.getOwner()->getParentOp()))
    return getInPlace(getTiedOpResult(bbArg));
  return ::detail::getInPlaceFuncOrCallBufferArg(
      bbArg.getOwner()->getParentOp(), bbArg.getArgNumber());
}

//===----------------------------------------------------------------------===//
// Bufferization-specific BlockAndValueMapping support with debugging.
//===----------------------------------------------------------------------===//

/// Wrapper for better debugging.
static void map(BlockAndValueMapping &bvm, ValueRange key, ValueRange value) {
  if (key.empty()) return;
  LLVM_DEBUG(DBGS() << "Map: " << key.front() << " to " << value.front()
                    << "\n");
  return bvm.map(key, value);
}

/// Wrapper for better debugging.
static void map(BlockAndValueMapping &bvm, Value key, Value value) {
  LLVM_DEBUG(DBGS() << "Map: " << key << " to " << value << "\n");
  return bvm.map(key, value);
}

/// Wrapper for better debugging.
static Value lookup(BlockAndValueMapping &bvm, Value key) {
  // TODO: if key comes from bbArg, forward.
  assert(key.getType().isa<TensorType>());
  if (!bvm.lookupOrNull(key)) {
    if (auto bbArg = key.dyn_cast<BlockArgument>()) {
      if (isa<FuncOp>(key.getParentBlock()->getParentOp()))
        key.getParentBlock()->getParentOp()->dump();
      else
        key.getParentBlock()->getParentOp()->getParentOfType<FuncOp>()->dump();
      bbArg.getOwner()->getParentOp()->dump();
    } else {
      key.getDefiningOp()->getParentOfType<FuncOp>()->dump();
    }
    llvm::errs() << "NO VALUE FOR KEY: " << key << "\n";
    abort();
  }
  return bvm.lookup(key);
}

//===----------------------------------------------------------------------===//
// Bufferization-specific support.
//===----------------------------------------------------------------------===//

/// Store all functions of the `moduleOp` in a `worklist` and sort the list
/// topolocigally such that every function only calls subsequent functions in
/// the worklist.
static LogicalResult computeOrderedWorklist(ModuleOp moduleOp,
                                            SmallVectorImpl<FuncOp> &worklist) {
  // Compute for every function the set of functions it is called by and the
  // number of functions it calls.
  DenseMap<FuncOp, llvm::SmallSet<FuncOp, 4>> calledBy;
  DenseMap<FuncOp, unsigned> callCounter;
  moduleOp.walk([&](FuncOp funcOp) {
    callCounter[funcOp] = 0;
    funcOp.walk([&](CallOp callOp) {
      SymbolRefAttr sym =
          callOp.getCallableForCallee().dyn_cast<SymbolRefAttr>();
      if (!sym) return;
      auto callable = dyn_cast_or_null<FuncOp>(
          SymbolTable::lookupNearestSymbolFrom(callOp, sym));
      if (calledBy[callable].count(funcOp) == 0) {
        calledBy[callable].insert(funcOp);
        callCounter[funcOp]++;
      }
    });
  });
  // Iteratively remove function operation that do not call any of the functions
  // remaining in the callCounter map and add them to the worklist.
  while (!callCounter.empty()) {
    auto it = llvm::find_if(callCounter,
                            [](auto entry) { return entry.getSecond() == 0; });
    if (it == callCounter.end())
      return moduleOp.emitOpError(
          "expected callgraph to be free of circular dependencies.");
    worklist.push_back(it->getFirst());
    for (auto callee : calledBy[it->getFirst()]) callCounter[callee]--;
    callCounter.erase(it);
  }
  return success();
}

/// For now, assume any use is a read.
/// Write-only is a non-problem: will represent with shapes in the future.
/// If any use of the tensor does not properly dominate `opOperand.getOwner()`,
/// then the tensor cannot be bufferized inPlace.
static bool hasInterferingTensorRead(OpOperand &opOperand,
                                     const DominanceInfo &domInfo) {
  if (!opOperand.get().getType().isa<RankedTensorType>()) return false;
  for (auto &use : opOperand.get().getUses()) {
    Operation *user = use.getOwner();
    // Tolerate forwarding-only of the value through an scf::ForOp: if the
    // corresponding bbArg has no use, we can just ignore it.
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      BlockArgument bbArg = forOp.getRegionIterArgForOpOperand(use);
      if (bbArg.use_empty()) continue;
    }

    // If properly dominate, there  is a clear sequence point and we can dismiss
    // read.
    if (domInfo.properlyDominates(user, opOperand.getOwner())) continue;
    // Otherwise, we need to analyze self-dependencies, for now just let it go.
    // TODO: proper self-dependence analysis.
    if (domInfo.dominates(user, opOperand.getOwner())) continue;
    if (user == opOperand.getOwner() &&
        use.getOperandNumber() == opOperand.getOperandNumber())
      continue;
    LLVM_DEBUG(DBGS() << "found interfering read operand #"
                      << opOperand.getOperandNumber()
                      << " in op: " << *opOperand.getOwner() << "\n");
    return true;
  }
  LLVM_DEBUG(DBGS() << "no interfering read\n");
  return false;
}

/// Return false if either:
/// 1. `opOperand` is produced by a constant op. For now this is assumed to be
///    bufferized to a GlobalMemrefOp that cannot be written. Generalize in the
///    future.
/// 2.`opOperand` is a BlockArgument of a FuncOp that is not known to be
///    bufferizable inplace.
/// 3.`opOperand` has an interfering tensor read.
/// Return true otherwise.
static bool isBufferizableInPlace(OpOperand &opOperand,
                                  const DominanceInfo &domInfo) {
  // Constant tensors are deemed not bufferizable for now.
  if (auto constantOp =
          dyn_cast_or_null<ConstantOp>(opOperand.get().getDefiningOp()))
    return !constantOp.getResult().getType().isa<RankedTensorType>();
  if (auto bbArg = opOperand.get().dyn_cast<BlockArgument>()) {
    // Function argument that may not be written-to needs to be copied by
    // user(s).
    // TODO: better propagate the fact that we want a single clone inside the
    // function. Atm every user that wants to write inplace will create its own
    // alloc, irrespective of whether or not interfering reads occur.
    if (isa<FuncOp>(bbArg.getOwner()->getParentOp()))
      return getInPlace(bbArg) == InPlaceSpec::True;
    // ForOp must additionally check whether there is an interfering read to
    // their original iter operand.
    if (auto forOp = dyn_cast<scf::ForOp>(bbArg.getOwner()->getParentOp()))
      if (hasInterferingTensorRead(forOp.getOpOperandForRegionIterArg(bbArg),
                                   domInfo))
        return false;
  }
  return !hasInterferingTensorRead(opOperand, domInfo);
}

//===----------------------------------------------------------------------===//
// Bufferization-specific MemRefType support.
//===----------------------------------------------------------------------===//

/// Return a contiguous MemRefType (i.e. with canonical/empty layout map) with
/// the same shape as `shapedType` and specified `layout` and `addressSpace`.
static MemRefType getContiguousMemRefType(ShapedType shapedType,
                                          ArrayRef<AffineMap> layout = {},
                                          unsigned addressSpace = 0) {
  if (RankedTensorType tensorType = shapedType.dyn_cast<RankedTensorType>())
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                           layout, addressSpace);
  MemRefType memrefType = shapedType.cast<MemRefType>();
  return MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                         layout, addressSpace);
}

/// Return a contiguous MemRefType (i.e. with canonical/empty layout map) with
/// the same shape as `shapedType` and specified `layout` and `addressSpace` or
/// an UnrankedMemRefType otherwise.
static Type getContiguousOrUnrankedMemRefType(Type type,
                                              ArrayRef<AffineMap> layout = {},
                                              unsigned addressSpace = 0) {
  if (type.isa<RankedTensorType, MemRefType>())
    return getContiguousMemRefType(type.cast<ShapedType>(), layout,
                                   addressSpace);
  assert(layout.empty() && "expected empty layout with UnrankedMemRefType");
  return UnrankedMemRefType::get(getElementTypeOrSelf(type), addressSpace);
}

/// Return a MemRefType to which the `tensorType` can be bufferized in a
/// composable fashion. The layout must be the most dynamic possible and
/// canonicalize away once bufferization is finished.
static MemRefType getDynamicMemRefType(RankedTensorType tensorType,
                                       unsigned addressSpace = 0) {
  // TODO: address space decisions to connect with the actual alloc.
  int64_t dynamicOffset = ShapedType::kDynamicStrideOrOffset;
  SmallVector<int64_t> dynamicStrides(tensorType.getRank(),
                                      ShapedType::kDynamicStrideOrOffset);
  AffineMap stridedLayout = makeStridedLinearLayoutMap(
      dynamicStrides, dynamicOffset, tensorType.getContext());
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                         stridedLayout, addressSpace);
}

// Transfer all `dim` ops on `tensor` to `memref`.
static void transferDimOpsToMemref(Value tensor, Value memref) {
  for (OpOperand &opOperand : llvm::make_early_inc_range(tensor.getUses())) {
    if (isa<memref::DimOp>(opOperand.getOwner())) {
      opOperand.set(memref);
    }
  }
}

//===----------------------------------------------------------------------===//
// Bufferization-specific inPlace pattern matching support.
//===----------------------------------------------------------------------===//

/// First assign `op` if `slice.back()` isa `T`, then check condition.
/// If anything fails just return failure. Otherwise update `sliceRef` by
/// dropping `sliceRef.back()`, then return success().
template <typename T>
static LogicalResult matchAndDropBack(
    ArrayRef<Operation *> &sliceRef, T &op,
    llvm::function_ref<LogicalResult(T)> condition = nullptr) {
  if (sliceRef.empty()) return failure();
  op = dyn_cast<T>(sliceRef.back());
  if (!op || (condition && failed(condition(op)))) return failure();
  sliceRef = sliceRef.drop_back();
  return success();
}

/// First assign `op1`/`op2` if `slice.front()`/`slice.back()` isa `T1`/`T2`,
/// respectively. Then check condition. If anything fails just return failure.
/// Otherwise update `sliceRef` by dropping `sliceRef.front()` and
/// `sliceRef.back()`, then return success().
template <typename T1, typename T2>
static LogicalResult matchAndDropEnclosingPair(
    ArrayRef<Operation *> &sliceRef, T1 &op1, T2 &op2,
    llvm::function_ref<LogicalResult(T1, T2)> condition = nullptr) {
  if (sliceRef.size() < 2) return failure();
  op1 = dyn_cast<T1>(sliceRef.front());
  op2 = dyn_cast<T2>(sliceRef.back());
  if (!op1 || !op2 || (condition && failed(condition(op1, op2))))
    return failure();
  sliceRef = sliceRef.drop_front().drop_back();
  return success();
}

//===----------------------------------------------------------------------===//
// Bufferization-specific scoped alloc/dealloc insertion support.
//===----------------------------------------------------------------------===//

/// Create an Allocop/DeAllocOp pair, where the AllocOp is after
/// `shapedValue.getDefiningOp` (or at the top of the block in case of a bbArg)
/// and the DeallocOp is at the end of the block.
/// Since this may insert **after** the op definining `shapedValue`, there is
/// a risk of abstraction gap with what the caller may legitimately expect.
/// As a consequence, this function should not be called with `b` rooted around
/// `shapedValue.getDefiningOp()`, as the insertion point may shift.
// TODO: need a better API to make things less surprising while avoiding
// implicit state passed across function boundaries: this still significantly
// beats mutating the insertion point for `b`.
// TODO: need to hoist this across function boundaries. Maybe by using
// init_tensor + subtensor_insert before bufferization.
static Value createNewAllocDeallocPairForShapedValue(
    OpBuilder &b, Location loc, Value shapedValue,
    SmallVector<Value, 4> dynOperands = {}) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);

  // TODO: non-zero address space.
  // TODO: layout information if relevant.
  // Cannot allocate an unranked memref so just always go for the contiguous
  // form.
  MemRefType allocMemRefType =
      getContiguousMemRefType(shapedValue.getType().cast<ShapedType>());
  assert(shapedValue.getType().isa<ShapedType>());
  MemRefType memRefType = shapedValue.getType().dyn_cast<MemRefType>();
  memRefType = memRefType ? memRefType : allocMemRefType;

  if (auto bbArg = shapedValue.dyn_cast<BlockArgument>()) {
    b.setInsertionPointToStart(bbArg.getOwner());
    loc = bbArg.getOwner()->getParentOp()->getLoc();
  } else {
    b.setInsertionPointAfter(shapedValue.getDefiningOp());
    loc = shapedValue.getDefiningOp()->getLoc();
  }

  // If the dynOperands are not passed explicity, copmpute them.
  // This circumvents currently missing dim(init_tensor) canonicalizations.
  if (dynOperands.empty()) {
    for (auto dim : llvm::enumerate(memRefType.getShape()))
      if (dim.value() == ShapedType::kDynamicSize)
        dynOperands.push_back(
            b.create<memref::DimOp>(loc, shapedValue, dim.index()));
  }
  Value allocated =
      b.create<memref::AllocOp>(loc, allocMemRefType, dynOperands);
  if (memRefType != allocMemRefType)
    allocated = b.create<memref::CastOp>(loc, memRefType, allocated);
  b.setInsertionPoint(allocated.getParentBlock()->getTerminator());
  b.create<memref::DeallocOp>(loc, allocated);
  return allocated;
}

//===----------------------------------------------------------------------===//
// Bufferization-specific inPlace analysis support.
//===----------------------------------------------------------------------===//

/// Return true is all offsets, sizes and strides are equal.
static LogicalResult sameOffsetsSizesAndStrides(
    OffsetSizeAndStrideOpInterface op1, OffsetSizeAndStrideOpInterface op2) {
  if (op1.static_offsets().size() != op2.static_offsets().size())
    return failure();
  if (op1.static_sizes().size() != op2.static_sizes().size()) return failure();
  if (op1.static_strides().size() != op2.static_strides().size())
    return failure();
  for (auto it : llvm::zip(op1.getMixedOffsets(), op2.getMixedOffsets()))
    if (!isEqualConstantIntOrValue(std::get<0>(it), std::get<1>(it)))
      return failure();
  for (auto it : llvm::zip(op1.getMixedSizes(), op2.getMixedSizes()))
    if (!isEqualConstantIntOrValue(std::get<0>(it), std::get<1>(it)))
      return failure();
  for (auto it : llvm::zip(op1.getMixedStrides(), op2.getMixedStrides()))
    if (!isEqualConstantIntOrValue(std::get<0>(it), std::get<1>(it)))
      return failure();
  return success();
}

static LogicalResult matchingVectorTransfersAtSource(
    vector::TransferReadOp read, vector::TransferWriteOp write,
    Value subtensor) {
  // Either we have a pair of matching transfer read/write or none.
  if (read && !write) {
    LLVM_DEBUG(DBGS() << "Slice has transferReadOp but no transferWriteOp"
                         "\nDestructive update chain FAIL\n");
    return failure();
  }
  if (!read && write) {
    LLVM_DEBUG(DBGS() << "Slice has transferWriteOp but no transferReadOp"
                         "\nDestructive update chain FAIL\n");
    return failure();
  }
  if (read && write) {
    // If we have a pair of mathing read/write, the tensor and vector shape
    // must exactly match (i.e. this is a vectorization).
    if (read.source() != subtensor) {
      LLVM_DEBUG(DBGS() << "transferReadOp.source() != subTensor.result()"
                           "\nDestructive update chain FAIL\n");
      return failure();
    }
    if (write.source() != subtensor) {
      LLVM_DEBUG(DBGS() << "transferWriteOp.source() != subTensor.result()"
                           "\nDestructive update chain FAIL\n");
      return failure();
    }
    if (read.getShapedType().getShape() != read.getVectorType().getShape()) {
      LLVM_DEBUG(DBGS() << "transferReadOp source and result shapes mismatch"
                           "\nDestructive update chain FAIL\n");
      return failure();
    }
    if (write.getShapedType().getShape() != write.getVectorType().getShape()) {
      LLVM_DEBUG(DBGS() << "transferWriteOp source and result shapes mismatch"
                           "\nDestructive update chain FAIL\n");
      return failure();
    }
  }
  return success();
}

/// Detect whether `v` has a single user that is exactly `user`.
static LogicalResult isInPlaceSingleUseOp(Value v, Operation *user) {
  if (!v.hasOneUse() || *v.getUsers().begin() != user) return failure();
  return success();
}

/// Detect whether `v` has a single user that is exactly `terminatorOp`.
/// If `bbArg` comes from an scf::ForOp, additionally check the operand index
/// is exactly `bbArg.getArgumentNumber`.
static LogicalResult isInPlaceSingleUseTerminatorValue(Value v,
                                                       Operation *terminatorOp,
                                                       BlockArgument bbArg) {
  if (failed(isInPlaceSingleUseOp(v, terminatorOp))) return failure();
  // Check terminatorOp is indeed a terminator and that it terminates the same
  // block as `bbArg`.
  if (!terminatorOp->hasTrait<OpTrait::IsTerminator>()) return failure();
  if (bbArg.getOwner() != terminatorOp->getBlock()) return failure();
  // If the parent is a scf::ForOp, we must additionally check the
  // operand/terminator numbers agree.
  if (isa<scf::ForOp>(bbArg.getOwner()->getParentOp()))
    return (getTiedOpResult(bbArg).getResultNumber() ==
            v.getUses().begin()->getOperandNumber())
               ? success()
               : failure();
  // If the parent is a linalg::TiledLoopOp, we must additionally check the
  // operand/terminator numbers agree.
  if (auto tiledLoop = dyn_cast<TiledLoopOp>(bbArg.getOwner()->getParentOp())) {
    if (bbArg.getArgNumber() <
        tiledLoop.getNumLoops() + tiledLoop.inputs().size())
      return failure();
    return (getTiedOpResult(bbArg).getResultNumber() ==
            v.getUses().begin()->getOperandNumber())
               ? success()
               : failure();
  }
  if (isa<FuncOp>(bbArg.getOwner()->getParentOp())) return success();
  llvm_unreachable("isInPlaceSingleUseOperand: unsupported op");
}

/// Detect the simple overwrite pattern:
/// ```
///    candidate -> vector.transfer_write(**) -> subtensor_insert(**) -> term
/// ```
///
/// (**) represents an optional op in the chain, at least one must be present
template <typename ContainerOp, typename TerminatorOp>
static LogicalResult detectOverWritePattern(Operation *parentOp,
                                            BlockArgument candidate,
                                            ArrayRef<Operation *> &sliceRef) {
  if (!parentOp || !isa<ContainerOp>(parentOp)) return failure();

  ArrayRef<Operation *> tmpSliceRef = sliceRef;
  TerminatorOp terminatorOp;
  // Match terminator and update tmpSliceRef.
  if (failed(matchAndDropBack(tmpSliceRef, terminatorOp))) {
    LLVM_DEBUG(DBGS() << "FAIL: partial overwrite pattern -> must end with "
                         "known terminator\n");
    return failure();
  }
  tensor::InsertSliceOp subTensorInsertOp;
  vector::TransferWriteOp vectorTransferWriteOp;
  // Maybe match subTensorInsertOp and update tmpSliceRef.
  (void)matchAndDropBack(tmpSliceRef, subTensorInsertOp);
  // Maybe match vectorTransferWriteOp and update tmpSliceRef.
  (void)matchAndDropBack(tmpSliceRef, vectorTransferWriteOp);

  if (!subTensorInsertOp && !vectorTransferWriteOp) {
    LLVM_DEBUG(DBGS() << "FAIL: partial overwrite pattern -> need at least "
                         "a subtensor_insert or a vector.transfer_write\n");
    return failure();
  }

  if (subTensorInsertOp) {
    // term(subTensorInsertOp(vectorTransferWriteOp))
    if (vectorTransferWriteOp &&
        failed(isInPlaceSingleUseOp(vectorTransferWriteOp.result(),
                                    subTensorInsertOp))) {
      LLVM_DEBUG(DBGS() << "FAIL: partial overwrite pattern -> expected "
                           "term(subTensorInsertOp(vectorTransferWriteOp))\n");
      return failure();
    }
    // term(subTensorInsertOp)
    if (failed(isInPlaceSingleUseTerminatorValue(subTensorInsertOp.result(),
                                                 terminatorOp, candidate))) {
      LLVM_DEBUG(DBGS() << "FAIL: partial overwrite pattern -> expected "
                           "term(subTensorInsertOp)\n");
      return failure();
    }
  } else if (vectorTransferWriteOp) {
    // term(vectorTransferWriteOp)
    if (failed(isInPlaceSingleUseTerminatorValue(vectorTransferWriteOp.result(),
                                                 terminatorOp, candidate))) {
      LLVM_DEBUG(DBGS() << "FAIL: partial overwrite pattern -> expected "
                           "term(vectorTransferWriteOp)\n");
      return failure();
    }
  }

  // Commit what has been detected.
  if (vectorTransferWriteOp)
    setInPlaceOpResult(vectorTransferWriteOp->getResult(0));
  if (subTensorInsertOp) setInPlaceOpResult(subTensorInsertOp->getResult(0));
  // No action for the terminator.
  tmpSliceRef = sliceRef;

  LLVM_DEBUG(DBGS() << "SUCCESS: partial overwrite pattern\n");
  return success();
}

/// Detect the simple terminator pattern:
/// ```
///    candidate -> single-result with single-use linalg op -> term
/// ```
template <typename ContainerOp, typename TerminatorOp>
static LogicalResult detectLinalgToTerminator(Operation *parentOp,
                                              BlockArgument candidate,
                                              ArrayRef<Operation *> &sliceRef) {
  if (!parentOp || !isa<ContainerOp>(parentOp)) return failure();

  ArrayRef<Operation *> tmpSliceRef = sliceRef;

  TerminatorOp terminatorOp;
  // Match returnOp and update tmpSliceRef.
  if (failed(matchAndDropBack(tmpSliceRef, terminatorOp))) {
    LLVM_DEBUG(DBGS() << "FAIL: linalg to term pattern -> slice must end "
                         "with a known terminator\n");
    return failure();
  }

  LinalgOp linalgOp;
  // Match linalgOp with a single output tensor for now.
  // TODO: support more than single result.
  if (failed(matchAndDropBack(tmpSliceRef, linalgOp)) ||
      linalgOp.getOutputTensorOperands().size() != 1 ||
      failed(isInPlaceSingleUseOp(linalgOp->getResult(0), terminatorOp))) {
    LLVM_DEBUG(DBGS() << "FAIL: linalg to term pattern -> slice must end "
                         "with single-result linalg op\n");
    return failure();
  }

  // Commit what has been detected.
  // TODO: support more than single result.
  setInPlaceOpResult(linalgOp->getResult(0));
  tmpSliceRef = sliceRef;
  LLVM_DEBUG(DBGS() << "SUCCESS: linalg to term pattern\n");

  return success();
}

template <typename ContainerOp, typename TerminatorOp>
static LogicalResult detectForOpToTerminator(Operation *parentOp,
                                             BlockArgument candidate,
                                             ArrayRef<Operation *> &sliceRef) {
  if (!parentOp || !isa<ContainerOp>(parentOp)) return failure();

  ArrayRef<Operation *> tmpSliceRef = sliceRef;

  TerminatorOp terminatorOp;
  // Match returnOp and update tmpSliceRef.
  if (failed(matchAndDropBack(tmpSliceRef, terminatorOp))) {
    LLVM_DEBUG(DBGS() << "FAIL: forop to term pattern -> slice must end "
                         "with a known terminator\n");
    return failure();
  }

  scf::ForOp forOp;
  // Match forOp with a single output tensor for now.
  // TODO: support more than single result.
  if (failed(matchAndDropBack(tmpSliceRef, forOp)) ||
      forOp->getNumResults() != 1 ||
      failed(isInPlaceSingleUseOp(forOp->getResult(0), terminatorOp))) {
    LLVM_DEBUG(DBGS() << "FAIL: forop to term pattern -> slice must end "
                         "with single-result forOp op\n");
    return failure();
  }

  // Commit what has been detected.
  // TODO: support more than single result.
  setInPlaceOpResult(forOp->getResult(0));
  tmpSliceRef = sliceRef;
  LLVM_DEBUG(DBGS() << "SUCCESS: forop to term pattern\n");

  return success();
}

/// Detect the simple terminator pattern:
/// ```
///    candidate -> single-result with single-use linalg op -> term
/// ```
template <typename ContainerOp, typename TerminatorOp>
static LogicalResult detectTiledLoopToTerminator(
    Operation *parentOp, BlockArgument candidate,
    ArrayRef<Operation *> &sliceRef) {
  if (!parentOp || !isa<ContainerOp>(parentOp)) return failure();

  ArrayRef<Operation *> tmpSliceRef = sliceRef;

  TerminatorOp terminatorOp;
  // Match returnOp and update tmpSliceRef.
  if (failed(matchAndDropBack(tmpSliceRef, terminatorOp))) {
    LLVM_DEBUG(DBGS() << "FAIL: tiled loop return pattern -> slice must end "
                         "with a known terminator\n");
    return failure();
  }

  TiledLoopOp tiledLoop;
  // Match tiled loop  with a single output tensor for now and update
  // tmpSliceRef.
  if (failed(matchAndDropBack(tmpSliceRef, tiledLoop)) ||
      tiledLoop->getNumResults() != 1 ||
      failed(isInPlaceSingleUseOp(tiledLoop->getResult(0), terminatorOp))) {
    LLVM_DEBUG(DBGS() << "FAIL: tiled loop return pattern -> slice must end "
                         "with a single-result tiled loop\n");
    return failure();
  }

  // Commit what has been detected.
  // TODO: support more than single result.
  if (tiledLoop) setInPlaceOpResult(tiledLoop->getResult(0));
  tmpSliceRef = sliceRef;
  LLVM_DEBUG(DBGS() << "SUCCESS: tiled loop return pattern\n");

  return success();
}

template <typename T>
static unsigned sizeOfRange(T range) {
  unsigned inc = 0;
  for (const auto &_ : range) {
    (void)_;
    ++inc;
  }
  return inc;
}

/// In the case of an scf::ForOp, we look for:
///   `candidate -> subtensor -> vector.transfer_read(*) -> ...
///      vector.transfer_write(*) -> subtensor_insert -> yield`.
/// sliceRef is automaticaly updated to match `...`.
///
/// (*) represents an optional op in the chain, if a subtensor or
/// vector.transfer is included, the matching op must be included too.
template <typename ContainerOp, typename TerminatorOp>
static LogicalResult detectDestructiveUpdatePattern(
    Operation *parentOp, BlockArgument candidate,
    ArrayRef<Operation *> &sliceRef) {
  if (!parentOp || !isa<ContainerOp>(parentOp)) return failure();

  ArrayRef<Operation *> tmpSliceRef = sliceRef;

  // bbArg must be used exactly by one subtensor / subtensor_insert pair.
  if (sizeOfRange(candidate.getUses()) != 2) {
    LLVM_DEBUG(DBGS() << "FAIL: destr. updates -> bbArg with != 2 uses\n");
    return failure();
  }
  if (tmpSliceRef.size() < 3) {
    LLVM_DEBUG(DBGS() << "FAIL: destr. updates -> slice must have >= 3 ops\n");
    return failure();
  }

  // Match yieldOp and update tmpSliceRef.
  TerminatorOp terminatorOp;
  if (failed(matchAndDropBack(tmpSliceRef, terminatorOp))) {
    LLVM_DEBUG(DBGS() << "FAIL: destr. updates -> slice unknown terminator\n");
    return failure();
  }

  // Match subtensor pair and update tmpSliceRef.
  tensor::ExtractSliceOp subTensorOp;
  tensor::InsertSliceOp subTensorInsertOp;
  if (failed(matchAndDropEnclosingPair<tensor::ExtractSliceOp,
                                       tensor::InsertSliceOp>(
          tmpSliceRef, subTensorOp, subTensorInsertOp,
          [](tensor::ExtractSliceOp st, tensor::InsertSliceOp sti) {
            // subtensor / subtensor_insert must match.
            return sameOffsetsSizesAndStrides(st, sti);
          }))) {
    LLVM_DEBUG(DBGS() << "FAIL: destr. updates -> subtensor ops don't match\n");
    return failure();
  }

  // term(subTensorInsertOp).
  if (failed(isInPlaceSingleUseTerminatorValue(subTensorInsertOp.result(),
                                               terminatorOp, candidate))) {
    LLVM_DEBUG(DBGS() << "FAIL: destr. updates -> tensor::InsertSliceOp "
                         "does not have a single terminator use "
                         "at the right index\n");
    return failure();
  }

  // Maybe match vector transfer pair and update tmpSliceRef.
  // If we find one, the other must be present and match too.
  vector::TransferReadOp vectorTransferReadOp;
  vector::TransferWriteOp vectorTransferWriteOp;
  if (failed(matchAndDropEnclosingPair<vector::TransferReadOp,
                                       vector::TransferWriteOp>(
          tmpSliceRef, vectorTransferReadOp, vectorTransferWriteOp,
          [&](vector::TransferReadOp read, vector::TransferWriteOp write) {
            return matchingVectorTransfersAtSource(read, write,
                                                   subTensorOp.result());
          }))) {
    // If we found a TransferReadOp or a TransferWriteOp but they don't match,
    // fail.
    if (vectorTransferReadOp || vectorTransferWriteOp) {
      LLVM_DEBUG(DBGS() << "FAIL: destr. updates -> non-matching transfers\n");
      return failure();
    }
  }

  // Commit what has been detected.
  setInPlaceOpResult(subTensorOp->getResult(0));
  if (vectorTransferReadOp)
    setInPlaceOpResult(vectorTransferReadOp->getResult(0));
  if (vectorTransferWriteOp)
    setInPlaceOpResult(vectorTransferWriteOp->getResult(0));
  setInPlaceOpResult(subTensorInsertOp->getResult(0));
  // No action for the terminator.
  tmpSliceRef = sliceRef;

  LLVM_DEBUG(DBGS() << "SUCCESS: destr. updates pattern\n");
  return success();
}

namespace detail {
// TODO: generalize and refactor.
// TODO: do we need more safeguards for setting ops inPlace ? In particular, do
// we want to explicitly disallow subtensor/subtensor_insert and
// vector.transfer_read/write that should have been already captured in
// destructive update patterns ?
// The following uses internal knowledge of the position of tied operand /
// results. A proper TieOperandInterface would be much better.
static void propagateInPlace(const SmallVector<OpOperand *> &initalWorklist,
                             const DominanceInfo &domInfo) {
  LLVM_DEBUG(DBGS() << "Start propagateInPlace from initial WL\n");
  for (OpOperand *operand : initalWorklist)
    LLVM_DEBUG(DBGS() << "WL item: " << operand->get() << " used by "
                      << *operand->getOwner() << "\n");
  SmallVector<OpOperand *> worklist(initalWorklist);
  for (unsigned idx = 0; idx < worklist.size(); ++idx) {
    // TODO: bail on subtensor/subtensor_insert and vector.transfer_read/write
    // that should have been already captured in destructive update patterns?
    OpOperand &operand = *worklist[idx];
    LLVM_DEBUG(DBGS() << "WL item: " << *operand.getOwner() << "\n");
    // If the owner turns out to be a CallOp without
    // `kWriteableFuncBufferArgsAttrName` this will be a noop.
    if (operand.get().getType().isa<TensorType>() &&
        isBufferizableInPlace(operand, domInfo)) {
      LLVM_DEBUG(DBGS() << "bufferizable inplace\n");
      setInPlaceOpResult(getTiedOpResult(operand));
    }
    LLVM_DEBUG(DBGS() << "propagatedInPlace: " << *operand.getOwner() << "\n");
    // use can have interfering reads that prevent it from being written inPlace
    // but the values it produces are still themselves candidates for inPlace at
    // their point of use.
    for (Value v : operand.getOwner()->getResults()) {
      LLVM_DEBUG(DBGS() << "propagate result: " << v << "\n");
      for (auto &use : v.getUses()) {
        LLVM_DEBUG(DBGS() << "add use to WL: " << use.get() << "\n");
        worklist.push_back(&use);
      }
    }
  }
}
}  // namespace detail

static void propagateInPlace(OpOperand &opOperand,
                             const DominanceInfo &domInfo) {
  SmallVector<OpOperand *> worklist{&opOperand};
  ::detail::propagateInPlace(worklist, domInfo);
}

static void propagateInPlace(BlockArgument &bbArg,
                             const DominanceInfo &domInfo) {
  SmallVector<OpOperand *> worklist;
  for (auto &use : bbArg.getUses()) worklist.push_back(&use);
  ::detail::propagateInPlace(worklist, domInfo);
}

/// Iterate over bbArgs of `parentOp` and determine if they are the root of a
/// known destructive update chain. Such a destructive update is related to
/// traditional loop nest + memory analysis but provides a simpler
/// abstraction. In traditional memory-based dependence analysis, one would
/// need to analyze all possible interleavings of possibly aliasing loads and
/// stores in the context of the k-common surrounding loops. With scf.for +
/// subtensor + subtensor_insert + yield, more ordering semantics are
/// available as well as dealiasing thanks to SSA use-def chains.
static void destructiveUpdateAnalysis(Block *block,
                                      const DominanceInfo &domInfo) {
  Operation *parentOp = block->getParentOp();
  // In this loop, we do not check whether `candidate` can itself be bufferized
  // inPlace: this is not a consideration for the inside of `block`.
  for (BlockArgument candidate : block->getArguments()) {
    LLVM_DEBUG(llvm::dbgs() << "\n\n");
    LLVM_DEBUG(DBGS() << "Destructive update analysis on candidate: "
                      << candidate << "\nof:\n"
                      << *parentOp << "\n");

    if (!candidate.getType().isa<ShapedType>()) {
      LLVM_DEBUG(DBGS() << "Not a tensor\n");
      continue;
    }

    llvm::SetVector<Operation *> slice;
    getForwardSlice(candidate, &slice, [&](Operation *op) {
      // Skip any extra nesting between parentOp and op.
      return op == parentOp || op->getBlock()->getParentOp() == parentOp;
    });

    LLVM_DEBUG(DBGS() << "Slice:\n");
    for (auto *op : slice) LLVM_DEBUG(DBGS() << *op << "\n");

    ArrayRef<Operation *> sliceRef = slice.getArrayRef();
    bool failedDetectingDestructiveUpdate =
        // linalg.tiled_loop / linalg.yield inplace patterns
        failed(detectDestructiveUpdatePattern<TiledLoopOp, linalg::YieldOp>(
            parentOp, candidate, sliceRef)) &&
        failed(detectLinalgToTerminator<TiledLoopOp, linalg::YieldOp>(
            parentOp, candidate, sliceRef)) &&
        failed(detectOverWritePattern<TiledLoopOp, linalg::YieldOp>(
            parentOp, candidate, sliceRef)) &&
        failed(detectTiledLoopToTerminator<FuncOp, ReturnOp>(
            parentOp, candidate, sliceRef)) &&
        // scf.for / scf.yield inplace patterns.
        failed(detectDestructiveUpdatePattern<scf::ForOp, scf::YieldOp>(
            parentOp, candidate, sliceRef)) &&
        failed(detectOverWritePattern<scf::ForOp, scf::YieldOp>(
            parentOp, candidate, sliceRef)) &&
        failed(detectLinalgToTerminator<scf::ForOp, scf::YieldOp>(
            parentOp, candidate, sliceRef)) &&
        failed(detectForOpToTerminator<scf::ForOp, scf::YieldOp>(
            parentOp, candidate, sliceRef)) &&
        // func / return inplace patterns.
        failed(detectDestructiveUpdatePattern<FuncOp, ReturnOp>(
            parentOp, candidate, sliceRef)) &&
        failed(detectOverWritePattern<FuncOp, ReturnOp>(parentOp, candidate,
                                                        sliceRef)) &&
        failed(detectLinalgToTerminator<FuncOp, ReturnOp>(parentOp, candidate,
                                                          sliceRef)) &&
        failed(detectForOpToTerminator<FuncOp, ReturnOp>(parentOp, candidate,
                                                         sliceRef));
    if (failedDetectingDestructiveUpdate) {
      LLVM_DEBUG(DBGS() << "Failed to detect a destructive update pattern\n");
      continue;
    }

    propagateInPlace(candidate, domInfo);
  }
}

void LinalgComprehensiveBufferizePass::inPlaceAnalysisFuncOpInternals(
    FuncOp funcOp, const DominanceInfo &domInfo) {
  if (!funcOp || funcOp->getNumRegions() == 0 || funcOp.body().empty()) return;

  // Start propagating from ConstantOps.
  funcOp.walk<WalkOrder::PreOrder>([&](ConstantOp constantOp) {
    if (!constantOp.getResult().getType().isa<TensorType>()) return;
    for (auto &use : constantOp->getUses()) propagateInPlace(use, domInfo);
  });

  // Start propagating from InitTensorOps.
  funcOp.walk<WalkOrder::PreOrder>([&](InitTensorOp initTensorOp) {
    for (auto &use : initTensorOp->getUses()) propagateInPlace(use, domInfo);
  });

  // Start propagating from FuncOp bbArgs.
  destructiveUpdateAnalysis(&funcOp.body().front(), domInfo);

  // Start propagating from scf::ForOps.
  funcOp.walk<WalkOrder::PreOrder>([&](scf::ForOp forOp) {
    destructiveUpdateAnalysis(&forOp.region().front(), domInfo);
  });

  // Start propagating from linalg::TiledLoopOps.
  funcOp.walk<WalkOrder::PreOrder>([&](TiledLoopOp tiledLoop) {
    destructiveUpdateAnalysis(tiledLoop.getBody(), domInfo);
  });
}

/// Analyze a `callOp` to a FuncOp and determine whether any of its tensor
/// operand could be safely written inPlace after it is converted to buffer
/// form by a bufferization process. Iterate on the uses of callOp's operands
/// to determine whether all such uses dominate callOp. If any use of an
/// operand does not dominate `callOp`, this means that the operand tensor
/// value may be needed somewhere else and it is illegal to update in-place
/// after bufferization. Add a `kInPlaceResultsAttrName` string attribute to
/// `callOp` to carry the result of this analysis until bufferization is
/// completed. The "meet" of all `kInPlaceResultsAttrName` for all `callOp` to a
/// given FuncOp determines the `kInPlaceResultsAttrName` for that FuncOp.
static void funcArgumentsInPlaceAnalysis(CallOpInterface callOp,
                                         const DominanceInfo &domInfo) {
  FuncOp funcOp = getCalledFunction(callOp);
  if (!funcOp || funcOp.body().empty()) return;

  if (llvm::none_of(callOp->getOperandTypes(),
                    [](Type t) { return t.isa<TensorType>(); }))
    return;

  LLVM_DEBUG(DBGS() << "Begin funcArgumentsInPlaceAnalysis within:\n"
                    << *callOp->getParentOfType<FuncOp>()
                    << "callOp: " << *callOp << "\n";);

  for (OpOperand &opOperand : callOp->getOpOperands()) {
    Value tensor = opOperand.get();
    if (!tensor.getType().isa<TensorType>()) continue;

    unsigned idx = opOperand.getOperandNumber();
    LLVM_DEBUG(DBGS() << "tensor @idx=" << idx << ": " << tensor << "\n");

    // FuncOp inPlace is the meet of all the calls. If we already know it
    // cannot be bufferized inPlace, just skip. Can't easily connect arguments
    // to results in FuncOp: use explicit idx.
    InPlaceSpec funcInPlace = getInPlace(funcOp.getArgument(idx));
    if (funcInPlace == InPlaceSpec::False) continue;

    InPlaceSpec callInPlace = isBufferizableInPlace(opOperand, domInfo)
                                  ? InPlaceSpec::True
                                  : InPlaceSpec::False;
    setInPlaceCallArgument(opOperand, callInPlace);
    setInPlaceFuncArgument(funcOp.getArgument(idx), callInPlace);
  }

  LLVM_DEBUG(DBGS() << "End funcArgumentsInPlaceAnalysis within:\n"
                    << *callOp->getParentOfType<FuncOp>()
                    << "callOp: " << *callOp << "\n";);
}

//===----------------------------------------------------------------------===//
// Bufferization as simple BlockAndValueMapping rewrites / without
// conversions.
//===----------------------------------------------------------------------===//

/// Non-conversion equivalent of the core MLIR Linalg bufferization patterns.
/// This works on mixed tensor + buffer Linalg ops: some results may have been
/// already bufferized by a previous destructive update bufferization.
/// Allocate the output buffers for the remaining tensor output operands of
/// the Linalg op. If the tensor is an "init" tensor (i.e. its value is
/// actually used in the payload region), we additionally copy the original
/// value into the newly allocated buffer.
static LogicalResult allocateBuffersForResults(
    OpBuilder &b, Location loc, LinalgOp op,
    SmallVectorImpl<Value> &resultBuffers, BlockAndValueMapping &bvm) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);

  // Lazily compute loopRanges.
  SmallVector<Range, 4> loopRanges;

  // Linalg invariant: output tensors and result match 1-1.
  assert(op.getOutputTensorOperands().size() == op->getNumResults());
  for (OpOperand *opOperand : op.getOutputOperands()) {
    Value output = opOperand->get();
    if (output.getType().isa<MemRefType>()) {
      resultBuffers.push_back(output);
      continue;
    }

    // If output tensor is marked inPlace, just use the buffer.
    // The following uses internal knowledge of the position of tied operand /
    // results. A proper TieOperandInterface would be much better.
    if (getInPlace(getTiedOpResult(*opOperand)) == InPlaceSpec::True) {
      resultBuffers.push_back(lookup(bvm, output));
      continue;
    }

    Value dimTensor = bvm.lookupOrDefault(output);
    Value alloc = createNewAllocDeallocPairForShapedValue(b, loc, dimTensor);
    b.setInsertionPointAfter(alloc.getDefiningOp());
    resultBuffers.push_back(alloc);

    // Additionally, if the output buffer is used, clone its value for now.
    if (op.payloadUsesValueFromOperand(opOperand))
      b.create<CopyOp>(loc, lookup(bvm, output), alloc);
  }
  map(bvm, op->getResults(), resultBuffers);
  for (auto it : llvm::zip(op->getResults(), resultBuffers)) {
    transferDimOpsToMemref(std::get<0>(it), std::get<1>(it));
  }
  return success();
}

// Non-conversion equivalent of the core MLIR Linalg bufferization patterns.
static void finalizeBufferAllocation(OpBuilder &b, LinalgOp op,
                                     ValueRange inputs, ValueRange outputs,
                                     BlockAndValueMapping &bvm) {
  SmallVector<Value, 8> newOperands = inputs;
  newOperands.append(outputs.begin(), outputs.end());
  Location loc = op.getLoc();
  op.clone(b, loc, /*resultTypes=*/TypeRange{}, newOperands);

  // Replace the results of the old op with the new output buffers.
  map(bvm, op.getOperation()->getResults(), outputs);
  for (auto it : llvm::zip(op.getOperation()->getResults(), outputs)) {
    transferDimOpsToMemref(std::get<0>(it), std::get<1>(it));
  }

  if (!op.hasTensorSemantics()) op->erase();
}

/// Generic conversion pattern that matches any LinalgOp. This avoids
/// template instantiating one pattern for each LinalgOp.
/// This works on mixed tensor + buffer Linalg ops: some results may have been
/// already bufferized by a previous destructive update bufferization.
static LogicalResult convertAnyLinalgOp(OpBuilder &b, LinalgOp op,
                                        BlockAndValueMapping &bvm) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);

  if (op.hasBufferSemantics()) return failure();

  LLVM_DEBUG(DBGS() << "convert: " << *op << "\n");

  b.setInsertionPoint(op);
  Location loc = op.getLoc();
  SmallVector<Value, 2> newInputBuffers;
  newInputBuffers.reserve(op.getNumInputs());
  for (OpOperand *opOperand : op.getInputOperands()) {
    if (!opOperand->get().getType().isa<TensorType>())
      newInputBuffers.push_back(opOperand->get());
    else
      newInputBuffers.push_back(lookup(bvm, opOperand->get()));
  }
  SmallVector<Value, 2> newOutputBuffers;
  if (failed(allocateBuffersForResults(b, loc, op, newOutputBuffers, bvm)))
    assert(false);

  // Delegate to the linalg generic pattern.
  if (auto genericOp = dyn_cast<GenericOp>(op.getOperation())) {
    finalizeBufferAllocation(b, genericOp, newInputBuffers, newOutputBuffers,
                             bvm);
    return success();
  }

  finalizeBufferAllocation(b, op, newInputBuffers, newOutputBuffers, bvm);

  return success();
}

static LogicalResult convertTransferOp(OpBuilder &b,
                                       VectorTransferOpInterface op,
                                       BlockAndValueMapping &bvm) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  Location loc = op.getLoc();

  if (op.getShapedType().isa<MemRefType>()) return failure();

  LLVM_DEBUG(DBGS() << "convert: " << *op << "\n");

  /// transfer_read from buffer
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op.getOperation())) {
    readOp.sourceMutable().assign(lookup(bvm, op.source()));
    return success();
  }

  auto inPlace = getInPlace(op->getResult(0));
  auto writeOp = cast<vector::TransferWriteOp>(op.getOperation());

  // If transfer_write is not inPlace, allocate a new buffer.
  Value newInputBuffer;
  if (inPlace != InPlaceSpec::True) {
    newInputBuffer =
        createNewAllocDeallocPairForShapedValue(b, loc, writeOp.result());
    b.setInsertionPointAfter(newInputBuffer.getDefiningOp());
    map(bvm, writeOp.result(), newInputBuffer);
    transferDimOpsToMemref(writeOp.result(), newInputBuffer);
  } else {
    // InPlace write will result in memref.tensor_load(x) which must
    // canonicalize away with one of it uses.
    newInputBuffer = lookup(bvm, writeOp.source());
  }

  // Create a new transfer_write on buffer that doesn't have a return value.
  // Leave the previous transfer_write to dead code as it still has uses at
  // this point.
  b.create<vector::TransferWriteOp>(
      loc, writeOp.vector(), newInputBuffer, writeOp.indices(),
      writeOp.permutation_map(),
      writeOp.in_bounds() ? *writeOp.in_bounds() : ArrayAttr());

  map(bvm, op->getResult(0), newInputBuffer);

  return success();
}

/// FuncOp always creates TensorToMemRef ops.
static LogicalResult convertFuncOp(OpBuilder &b, FuncOp funcOp,
                                   BlockAndValueMapping &bvm) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(&funcOp.body().front());
  for (auto bbArg : funcOp.getArguments()) {
    auto tensorType = bbArg.getType().dyn_cast<TensorType>();
    auto rankedTensorType = bbArg.getType().dyn_cast<RankedTensorType>();
    if (!tensorType) continue;
    // Cast the tensor to the most dynamic buffer possible. Further
    // canonicalizations will clean up.
    Type memRefType = rankedTensorType
                          ? getDynamicMemRefType(rankedTensorType)
                          : getContiguousOrUnrankedMemRefType(tensorType);
    Value tensorToMemref =
        b.create<memref::BufferCastOp>(funcOp.getLoc(), memRefType, bbArg);
    map(bvm, bbArg, tensorToMemref);
  }
  return success();
}

static LogicalResult convertScfForOp(OpBuilder &b, scf::ForOp forOp,
                                     BlockAndValueMapping &bvm) {
  LLVM_DEBUG(DBGS() << "convert: " << *forOp << "\n");

  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(forOp.getBody());

  // If inPlace, just forward the buffer.
  // Otherwise alloc and copy.
  b.setInsertionPointAfter(forOp);
  for (auto it : llvm::zip(forOp.getRegionIterArgs(), forOp->getResults())) {
    BlockArgument bbArg = std::get<0>(it);
    // TODO: Atm we bail on unranked TensorType because we don't know how to
    // alloc an UnrankedMemRefType + its underlying ranked MemRefType.
    if (!bbArg.getType().isa<RankedTensorType>()) continue;
    OpResult opResult = std::get<1>(it);
    Value operand = forOp.getIterOperands()[opResult.getResultNumber()];
    Value operandBuffer = lookup(bvm, operand);
    if (getInPlace(bbArg) != InPlaceSpec::True) {
      Value alloc =
          createNewAllocDeallocPairForShapedValue(b, forOp.getLoc(), operand);
      // If the tensor comes from `linalg::InitTensorOp`, the value is
      // unitialized and we do not need to copy.
      if (!operand.getDefiningOp<linalg::InitTensorOp>())
        b.create<linalg::CopyOp>(forOp.getLoc(), operandBuffer, alloc);
      operandBuffer = alloc;
    }
    map(bvm, bbArg, operandBuffer);
    map(bvm, opResult, operandBuffer);
  }

  return success();
}

/// TiledLoopOp maps `inputs` args to the corresponding buffers and
/// allocates the buffers for the `outputs` args depending on the in-place
/// analysis. The `outputs` args will contain both tensors and memrefs after
/// this conversion. The terminator is updated to yield the operand. This
/// simple forwarding pattern is canonicalized away later.
static LogicalResult convertTiledLoopOp(OpBuilder &b, TiledLoopOp tiledLoop,
                                        BlockAndValueMapping &bvm) {
  LLVM_DEBUG(DBGS() << "convert: " << *tiledLoop << "\n");

  // Allocate output buffers if needed, forward output tensor args to the
  // terminator.
  Operation *yieldOp = tiledLoop.getBody()->getTerminator();

  // Replace tensor `inputs` args with the buffer ones.
  int numLoops = tiledLoop.getNumLoops();
  int numInputs = tiledLoop.inputs().size();
  int numOutputs = tiledLoop.outputs().size();
  int afterInputs = tiledLoop.getNumControlOperands() + numInputs;
  int bbAfterInputs = tiledLoop.getNumLoops() + numInputs;

  auto oldInputs = llvm::to_vector<4>(tiledLoop.inputs());
  auto oldOutputs = llvm::to_vector<4>(tiledLoop.outputs());

  // Add buffers for inputs and the corresponding block arguments.
  Block *body = tiledLoop.getBody();
  for (auto en : llvm::enumerate(oldInputs)) {
    if (!en.value().getType().isa<TensorType>()) continue;
    Value inputBuffer = lookup(bvm, en.value());
    auto index = en.index();
    tiledLoop->insertOperands(afterInputs + index, {inputBuffer});
    auto bbArg =
        body->insertArgument(bbAfterInputs + index, inputBuffer.getType());
    map(bvm, body->getArgument(tiledLoop.getNumLoops() + index), bbArg);
    numInputs++;
  }

  // Add buffers for outputs and the corresponding block arguments.
  int newBuffersCount = 0;
  int outputsEndIndex =
      tiledLoop.getNumControlOperands() + numInputs + numOutputs;
  int bbAfterOutputs = tiledLoop.getNumLoops() + numInputs + numOutputs;

  for (auto en : llvm::enumerate(llvm::zip(oldOutputs, tiledLoop->getResults(),
                                           yieldOp->getOpOperands()))) {
    Value outputTensor = std::get<0>(en.value());
    // TODO: Atm we bail on TensorType because we don't know how to alloc an
    // UnrankedMemRefType + its underlying ranked MemRefType.
    if (!outputTensor.getType().isa<RankedTensorType>()) continue;
    Value operandBuffer = lookup(bvm, outputTensor);

    const OpResult &opResult = std::get<1>(en.value());
    OpOperand &yieldOperand = std::get<2>(en.value());
    if (getInPlace(opResult) != InPlaceSpec::True) {
      auto loc = tiledLoop.getLoc();
      Value alloc =
          createNewAllocDeallocPairForShapedValue(b, loc, outputTensor);
      // If the tensor comes from `linalg::InitTensorOp`, the value is
      // unitialized and we do not need to copy.
      if (!outputTensor.getDefiningOp<linalg::InitTensorOp>()) {
        b.setInsertionPointAfter(alloc.getDefiningOp());
        b.create<linalg::CopyOp>(loc, operandBuffer, alloc);
      }
      operandBuffer = alloc;
    }
    map(bvm, opResult, operandBuffer);

    tiledLoop->insertOperands(outputsEndIndex + newBuffersCount, operandBuffer);
    auto newBbArg = body->insertArgument(bbAfterOutputs + newBuffersCount,
                                         operandBuffer.getType());
    map(bvm, body->getArgument(bbAfterOutputs + newBuffersCount - 1), newBbArg);

    // Set operand of `linalg.yield` to the output tensor.
    yieldOperand.set(body->getArgument(bbAfterOutputs + newBuffersCount - 1));
    ++newBuffersCount;
  }

  // Update segment sizes.
  tiledLoop->setAttr(
      TiledLoopOp::getOperandSegmentSizeAttr(),
      b.getI32VectorAttr({numLoops, numLoops, numLoops, numInputs,
                          numOutputs + newBuffersCount}));
  return success();
}

static LogicalResult convertScfYieldOp(OpBuilder &b, scf::YieldOp yieldOp,
                                       BlockAndValueMapping &bvm) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(yieldOp);

  scf::ForOp forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
  assert(forOp && "only support scf::ForOp parent for scf::YieldOp");
  for (OpOperand &operand : yieldOp->getOpOperands()) {
    auto tensorType = operand.get().getType().dyn_cast<TensorType>();
    if (!tensorType) continue;
    auto bbArg = forOp.getRegionIterArgs()[operand.getOperandNumber()];
    if (getInPlace(bbArg) == InPlaceSpec::True)
      operand.set(bbArg);
    else
      operand.set(
          b.create<memref::TensorLoadOp>(yieldOp.getLoc(), lookup(bvm, bbArg)));
  }
  return success();
}

static LogicalResult convertCallOp(
    OpBuilder &b, CallOpInterface callOp, BlockAndValueMapping &bvm,
    SmallVector<Operation *> &convertedCallOps,
    const DenseMap<FuncOp, SmallVector<int64_t>> &tiedResultsMap) {
  LLVM_DEBUG(DBGS() << "convert: " << *callOp << "\n");

  FuncOp funcOp = getCalledFunction(callOp);
  if (!funcOp) return success();

  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(callOp);

  // 1. Rewrite tensor operands as memrefs. For now, only allow either using:
  //   a. a memref from the `bvm`, or
  //   b. the memref fed to a memref.tensor_load, if it does not itself come
  //   from a memref.buffer_cast.
  SmallVector<Value> newOperands(callOp->getOperands());
  for (unsigned idx = 0, e = newOperands.size(); idx < e; ++idx) {
    Value v = newOperands[idx];
    if (!v.getType().isa<TensorType>()) continue;
    if (auto mappedV = bvm.lookupOrNull(v)) {
      auto funcOp = getCalledFunction(callOp);
      auto memRefType = funcOp.getType().getInput(idx).dyn_cast<MemRefType>();
      if (memRefType && mappedV.getType() != memRefType) {
        // Since we don't yet have a clear layout story, buffer_cast may
        // conservatively turn tensors into more dynamic memref than necessary.
        // If the memref type of the callee fails, introduce an extra
        // memref.cast that will either canonicalize away or fail compilation
        // until we can do something better.
        Value newV =
            b.create<memref::CastOp>(callOp.getLoc(), memRefType, mappedV);
        bvm.map(v, newV);
        bvm.map(mappedV, newV);
        mappedV = newV;
      }
      newOperands[idx] = mappedV;
      continue;
    }
    // TODO: how dangerous is this at this point in spacetime ?
    if (auto tensorLoadOp = v.getDefiningOp<memref::TensorLoadOp>()) {
      if (!isa<memref::BufferCastOp>(tensorLoadOp.memref().getDefiningOp())) {
        Value newV = tensorLoadOp.memref();
        bvm.map(tensorLoadOp.memref(), newV);
        newOperands[idx] = newV;
        continue;
      }
    }
    llvm::errs() << "operand: " << v << "\n";
    llvm_unreachable("Operand does not come from a memref.tensor_load");
  }

  // 2. CallOp bufferization.
  // 2.a. if CallOp a FuncOp without a body, only the arguments have been
  // bufferized. Such a function cannot return a tensor and its results do not
  // change as a result of bufferization.
  if (funcOp.body().empty()) {
    if (llvm::any_of(funcOp.getType().getResults(),
                     [](Type t) { return t.isa<TensorType>(); })) {
      funcOp->dump();
      llvm_unreachable("Unexpected: Function declaration returns a tensor");
    }
    callOp->setOperands(newOperands);
    LLVM_DEBUG(DBGS() << "Map: inplace CallOp " << *callOp << "\n");
    return success();
  }

  // 2.b. Clone the CallOp with its attributes, its results may change as a
  // result of bufferization.
  assert(isa<CallOp>(callOp.getOperation()) && "expected a CallOp");
  Operation *newCallOp = b.create<CallOp>(
      callOp.getLoc(), funcOp.sym_name(),
      funcOp.type().cast<FunctionType>().getResults(), newOperands);
  newCallOp->setAttrs(callOp->getAttrs());

  // 3. Prepare replacements for the old CallOp results.
  auto tiedResults = tiedResultsMap.lookup(funcOp);
  unsigned newCallOpResultIndex = 0;
  SmallVector<Value> replacements;
  replacements.reserve(callOp->getNumResults());
  for (OpResult oldRes : callOp->getResults()) {
    // If not a tensor, no changes, just replace the new result.
    if (!oldRes.getType().isa<TensorType>()) {
      replacements.push_back(newCallOp->getResult(newCallOpResultIndex++));
      continue;
    }

    // Disallow memref returns for now as they are generally ambiguous. This
    // means we must have a non-negative `operandIndex`.
    // TODO: when such cases occur, add an Alloc hoisting pass and create new
    // inPlace function arguments.
    int64_t operandIndex = tiedResults[oldRes.getResultNumber()];
    if (operandIndex < 0) {
      callOp->getParentOfType<FuncOp>().dump();
      llvm_unreachable("Unsupported result memref");
    }

    // If the old callOp result is a tensor that does not fold on some input,
    // then there must be an allocated return value.
    // That value should be deallocated by the caller.
    // TODO: That value should be lifted out of the callee at the first
    // enclosing parallel scope. This lifting should be done to (the meet of)
    // all callers before we can hoist the alloc out of the funcOp.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointAfter(callOp);
    replacements.push_back(b.create<memref::TensorLoadOp>(
        callOp.getLoc(), newOperands[operandIndex]));

    map(bvm, oldRes, newOperands[operandIndex]);
  }
  convertedCallOps.push_back(callOp);

  return success();
}

static LogicalResult convertReturnOp(OpBuilder &b, ReturnOp returnOp,
                                     BlockAndValueMapping &bvm) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(returnOp);

  FuncOp funcOp = cast<FuncOp>(returnOp->getParentOp());
  assert(funcOp && "only support FuncOp parent for ReturnOp");
  for (OpOperand &operand : returnOp->getOpOperands()) {
    auto tensorType = operand.get().getType().dyn_cast<TensorType>();
    if (!tensorType) continue;
    operand.set(b.create<memref::TensorLoadOp>(returnOp.getLoc(),
                                               lookup(bvm, operand.get())));
  }
  return success();
}

/// InitTensor always allocates.
/// TODO: hoist across function boundaries prior to bufferization.
static LogicalResult convertInitTensorOp(OpBuilder &b,
                                         InitTensorOp initTensorOp,
                                         BlockAndValueMapping &bvm) {
  LLVM_DEBUG(DBGS() << "convert: " << *initTensorOp << "\n");

  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(initTensorOp);

  Value alloc = createNewAllocDeallocPairForShapedValue(
      b, initTensorOp->getLoc(), initTensorOp.result(), initTensorOp.sizes());
  map(bvm, initTensorOp.result(), alloc);
  return success();
}

// This implementation is a shortcut that assumes the tile size divides the
// problem size and is generally incorrect.
// TODO: revisit this (e.g. if needs_padding then linalg.fill; linalg.copy).
static LogicalResult convertPadTensorOp(OpBuilder &b, PadTensorOp padTensorOp,
                                        BlockAndValueMapping &bvm) {
  LLVM_DEBUG(DBGS() << "convert: " << *padTensorOp << "\n");

  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(padTensorOp);

  auto tensorType = padTensorOp.result().getType().cast<TensorType>();
  auto sourceMemRef = lookup(bvm, padTensorOp.source());
  auto sourceMemRefType = sourceMemRef.getType().cast<MemRefType>();
  auto memRefType = getContiguousOrUnrankedMemRefType(
      tensorType, sourceMemRefType.getAffineMaps(),
      sourceMemRefType.getMemorySpaceAsInt());
  Value res =
      b.create<memref::CastOp>(padTensorOp.getLoc(), memRefType, sourceMemRef);
  map(bvm, padTensorOp.result(), res);
  return success();
}

/// tensor::InsertSliceOp never allocates but may copy if it is not marked
/// inPlace.
static LogicalResult convertSubTensorInsertOp(
    OpBuilder &b, tensor::InsertSliceOp subTensorInsertOp,
    BlockAndValueMapping &bvm) {
  LLVM_DEBUG(DBGS() << "convert: " << *subTensorInsertOp << "\n");

  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(subTensorInsertOp);
  Location loc = subTensorInsertOp.getLoc();

  Value dstMemref;
  auto inPlace = getInPlace(subTensorInsertOp->getResult(0));
  // subtensor_insert must be inPlace, otherwise this is considered a bug.
  if (inPlace != InPlaceSpec::True) {
    llvm_unreachable("tensor::InsertSliceOp must be inPlace");
  } else {
    // InPlace write will result in memref.tensor_load(x) which must
    // canonicalize away with one of it uses.
    dstMemref = lookup(bvm, subTensorInsertOp.dest());
  }
  auto dstMemrefType = dstMemref.getType().cast<MemRefType>();

  Value srcMemref = lookup(bvm, subTensorInsertOp.source());
  auto subviewMemRefType =
      memref::SubViewOp::inferRankReducedResultType(
          subTensorInsertOp.getSourceType().getRank(), dstMemrefType,
          subTensorInsertOp.getMixedOffsets(),
          subTensorInsertOp.getMixedSizes(),
          subTensorInsertOp.getMixedStrides())
          .cast<MemRefType>();

  // Take a subview of the dst.
  Value subView = b.create<memref::SubViewOp>(
      loc, subviewMemRefType, dstMemref, subTensorInsertOp.getMixedOffsets(),
      subTensorInsertOp.getMixedSizes(), subTensorInsertOp.getMixedStrides());

  // Linalg op and vector.transfer_write producers directly write their output
  // buffer. If the producer is not one of these ops, we need to copy.
  Value source = subTensorInsertOp.source();
  InPlaceSpec inPlaceProducer = InPlaceSpec::None;
  if (auto opResult = source.dyn_cast<OpResult>())
    inPlaceProducer = getInPlace(opResult);
  else
    inPlaceProducer = getInPlace(source.cast<BlockArgument>());
  if (inPlaceProducer != InPlaceSpec::True) {
    LLVM_DEBUG(DBGS() << "source not inplace: " << source << " -> copy\n");
    b.create<CopyOp>(subTensorInsertOp.getLoc(), srcMemref, subView);
  }

  map(bvm, subTensorInsertOp.result(), subView);

  return success();
}

/// tensor::ExtractSliceOp never allocates or copies.
static LogicalResult convertSubTensorOp(OpBuilder &b,
                                        tensor::ExtractSliceOp subTensor,
                                        BlockAndValueMapping &bvm) {
  LLVM_DEBUG(DBGS() << "convert: " << *subTensor << "\n");

  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(subTensor);

  Location loc = subTensor.getLoc();
  Value srcMemref = lookup(bvm, subTensor.source());
  auto srcMemrefType = srcMemref.getType().cast<MemRefType>();
  auto dstTensorType = subTensor.result().getType().cast<RankedTensorType>();

  auto subviewMemRefType =
      memref::SubViewOp::inferRankReducedResultType(
          dstTensorType.getRank(), srcMemrefType, subTensor.getMixedOffsets(),
          subTensor.getMixedSizes(), subTensor.getMixedStrides())
          .cast<MemRefType>();

  Value subView = b.create<memref::SubViewOp>(
      loc, subviewMemRefType, srcMemref, subTensor.getMixedOffsets(),
      subTensor.getMixedSizes(), subTensor.getMixedStrides());
  map(bvm, subTensor.result(), subView);
  return success();
}

/// tensor::ExtractOp never allocates or copies.
static LogicalResult convertExtractOp(OpBuilder &b, tensor::ExtractOp extract,
                                      BlockAndValueMapping &bvm) {
  LLVM_DEBUG(DBGS() << "convert: " << *extract << "\n");

  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(extract);

  Location loc = extract.getLoc();
  Value srcMemref = lookup(bvm, extract.tensor());
  Value l = b.create<memref::LoadOp>(loc, srcMemref, extract.indices());
  extract.replaceAllUsesWith(l);
  return success();
}

/// TensorCastOp just lowers to MemRefCastOp.
static LogicalResult convertTensorCastOp(OpBuilder &b, tensor::CastOp castOp,
                                         BlockAndValueMapping &bvm) {
  LLVM_DEBUG(DBGS() << "convert: " << *castOp << "\n");

  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(castOp);

  Type sourceType = lookup(bvm, castOp.source()).getType();
  auto rankedMemRefType = sourceType.dyn_cast<MemRefType>();
  auto unrankedMemRefType = sourceType.dyn_cast<UnrankedMemRefType>();
  assert(rankedMemRefType || unrankedMemRefType);
  unsigned memorySpace = rankedMemRefType
                             ? rankedMemRefType.getMemorySpaceAsInt()
                             : unrankedMemRefType.getMemorySpaceAsInt();
  ArrayRef<AffineMap> affineMaps = rankedMemRefType
                                       ? rankedMemRefType.getAffineMaps()
                                       : ArrayRef<AffineMap>{};
  TensorType tensorType = castOp.getResult().getType().cast<TensorType>();
  Type memRefType =
      tensorType.isa<RankedTensorType>()
          ? getContiguousOrUnrankedMemRefType(castOp.getResult().getType(),
                                              affineMaps, memorySpace)
          : getContiguousOrUnrankedMemRefType(castOp.getResult().getType(), {},
                                              memorySpace);
  Value res = b.create<memref::CastOp>(castOp.getLoc(), memRefType,
                                       lookup(bvm, castOp.source()));
  map(bvm, castOp.getResult(), res);
  return success();
}

static LogicalResult convertConstantOp(OpBuilder &b, ConstantOp constantOp,
                                       BlockAndValueMapping &bvm,
                                       GlobalCreator &globals) {
  if (!constantOp.getType().dyn_cast<RankedTensorType>()) return failure();

  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(constantOp);

  auto globalMemref = globals.getGlobalFor(constantOp);
  Value memref = b.create<memref::GetGlobalOp>(
      constantOp.getLoc(), globalMemref.type(), globalMemref.getName());
  map(bvm, constantOp, memref);

  return success();
}

//===----------------------------------------------------------------------===//
// Functions and calls bufferization support.
//===----------------------------------------------------------------------===//

/// Return a FuncOp block argument if the `returnOperand` is produced by an
/// inPlace update pattern. Return the function argument that `returnOperand`
/// traces back to, if the following pattern is detected:
/// ```
///    func @foo(%A: tensor<...>, ..., %Z: tensor<...>) ->
///      (tensor<...>, ..., tensor<...>)
///      #inPlace_attr_specification
///    {
///       %p = memref.buffer_cast(%some_arg): ...
///       ... // uses of %p (read or writes)
///       %t = memref.tensor_load %p: ...
///       return ..., %t, ...: ..., tensor<...>, ...
///    }
/// ```
/// Otherwise return nullptr.
static BlockArgument analyzeTiedFuncOpResults(OpOperand &returnOperand) {
  assert(isa<ReturnOp>(returnOperand.getOwner()));
  FuncOp funcOp =
      cast<FuncOp>(returnOperand.get().getParentBlock()->getParentOp());
  Value returnValue = returnOperand.get();

  // Only consider tensors for folding.
  if (!returnValue.getType().isa<TensorType>()) return BlockArgument();

  // If returned value is a bbArg, it folds iff it is a function argument.
  if (auto bbArg = returnValue.dyn_cast<BlockArgument>())
    return (bbArg == funcOp.getArgument(bbArg.getArgNumber()))
               ? bbArg
               : BlockArgument();

  // Otherwise we look for either:
  //   1. memref.tensor_load(memref.buffer_cast(bbArg)), or
  //   2. memref.tensor_load(memref.cast(memref.buffer_cast(bbArg))).
  auto tensorLoadOp = returnValue.getDefiningOp<memref::TensorLoadOp>();
  if (!tensorLoadOp) return BlockArgument();
  auto memrefCastOp = tensorLoadOp.memref().getDefiningOp<memref::CastOp>();
  auto bufferCastOp =
      tensorLoadOp.memref().getDefiningOp<memref::BufferCastOp>();
  if (!memrefCastOp && !bufferCastOp) return BlockArgument();
  if (memrefCastOp)
    bufferCastOp = memrefCastOp.source().getDefiningOp<memref::BufferCastOp>();
  if (!bufferCastOp) return BlockArgument();

  // If returned value is a bbArg, it only folds if it is a function
  // argument.
  if (auto bbArg = bufferCastOp.tensor().dyn_cast<BlockArgument>())
    return (bbArg == funcOp.getArgument(bbArg.getArgNumber()))
               ? bbArg
               : BlockArgument();
  return BlockArgument();
}

static bool hasOnlyTensorToMemRefUses(Value v) {
  for (auto &use : v.getUses())
    if (!isa<memref::BufferCastOp>(use.getOwner())) return false;
  return true;
}

/// Search `funcOp` for the following pattern for each result to determine
/// whether it can fold onto an argument:
/// ```
///    func @foo(%A: tensor<...>, ..., %Z: tensor<...>) ->
///      (tensor<...>, ..., tensor<...>)
///      #inPlace_attr_specification
///    {
///       %p = memref.buffer_cast(%some_arg): ...
///       ... // uses of %p (read or writes)
///       %t = memref.tensor_load %p: ...
///       return ..., %t, ...: ..., tensor<...>, ...
///    }
/// ```
/// Information for such inPlace-bufferizable operands and the corresponding
/// result is added to `tiedResultsMap`.
/// Rewrite the `funcOp` arguments analysis return values and terminator into
/// buffer form (using the canonical memref layout for now), according to the
/// inPlace-bufferizable information added to `tiedResultsMap`.
static void bufferizeFuncOpBoundary(
    FuncOp funcOp, DenseMap<FuncOp, SmallVector<int64_t>> &tiedResultsMap) {
  // TODO: Generalize the use of contiguous MemRef at the function boundary.
  auto bufferizeFunctionArgsAndReturns = [](FuncOp funcOp,
                                            TypeRange returnTypes) {
    auto argTypes = llvm::to_vector<4>(
        llvm::map_range(funcOp.getType().getInputs(), [](Type t) -> Type {
          // TODO: non-zero address space.
          // TODO: layout information if relevant.
          if (auto tensorType = t.dyn_cast<TensorType>())
            return getContiguousOrUnrankedMemRefType(tensorType);
          return t;
        }));
    funcOp.setType(
        FunctionType::get(funcOp->getContext(), argTypes, returnTypes));
  };

  LLVM_DEBUG(DBGS() << "Begin bufferizeFuncOpBoundary:\n" << funcOp << "\n");

  // 0. Only convert arguments for function declarations. Such functions cannot
  // return a tensor and its results do not change as a result of bufferization.
  if (funcOp.getBody().empty()) {
    if (llvm::any_of(funcOp.getType().getResults(),
                     [](Type t) { return t.isa<TensorType>(); })) {
      funcOp->dump();
      llvm_unreachable("Unexpected: Function declaration returns a tensor");
    }
    bufferizeFunctionArgsAndReturns(funcOp, funcOp.getType().getResults());
    LLVM_DEBUG(DBGS() << "End bufferizeFuncOpBoundary no fun body: " << funcOp);
    return;
  }

  // 1. Analyze inplace return patterns and set an entry in `tiedResultsMap`.
  // Assume the last block terminator is the funcOp return.
  // TODO: Double-check this.
  auto returnOp = cast<ReturnOp>(funcOp.body().back().getTerminator());
  SmallVector<int64_t> resultArgumentFolding(
      funcOp.type().cast<FunctionType>().getNumResults(), -1);
  for (OpOperand &returnOperand : returnOp->getOpOperands()) {
    BlockArgument bbArg = analyzeTiedFuncOpResults(returnOperand);
    if (!bbArg) continue;
    // If the bbArg is not null, we still need to check the func arg is inPlace
    // writeable.
    if (getInPlace(bbArg) != InPlaceSpec::True) continue;
    // Mark bbArg as inPlace bufferizable.
    unsigned returnIndex = returnOperand.getOperandNumber();
    resultArgumentFolding[returnIndex] = bbArg.getArgNumber();
  }
  tiedResultsMap.insert(std::make_pair(funcOp, resultArgumentFolding));

  LLVM_DEBUG(
      DBGS() << "Computed tiedResultsMap:"
             << OpBuilder(funcOp).getIndexArrayAttr(resultArgumentFolding));

  // 2. Traverse terminator, skip return values that are inPlace bufferizable.
  OpBuilder b(returnOp);
  SmallVector<Value> returnValues;
  for (auto en : enumerate(resultArgumentFolding)) {
    LLVM_DEBUG(DBGS() << "return idx: " << en.index()
                      << " inPlace bufferizable on  input " << en.value()
                      << "\n");
    // Return value folds on some input.
    if (en.value() >= 0) continue;

    // Return value does not fold, add it to the new return op.
    Value unfolded = returnOp->getOperand(en.index());
    if (auto tensorLoadOp = unfolded.getDefiningOp<memref::TensorLoadOp>()) {
      unfolded = tensorLoadOp.memref();
      for (Operation *user : llvm::make_early_inc_range(unfolded.getUsers()))
        if (isa<memref::DeallocOp>(user)) user->erase();
    }
    returnValues.push_back(unfolded);
    if (unfolded.getType().isa<MemRefType>()) {
      funcOp->dump();
      llvm::errs() << "return val is not inPlace bufferizable: "
                   << returnValues.back() << "\n";
      abort();
    }
  }

  // 3. Rewrite the terminator without the inPlace bufferizable values.
  b.create<ReturnOp>(returnOp.getLoc(), returnValues);
  returnOp->erase();

  // 4. Rewrite the FuncOp type to buffer form.
  bufferizeFunctionArgsAndReturns(funcOp, ValueRange{returnValues}.getTypes());

  // 5. Rewrite the bbArgs.
  Block &frontBlock = funcOp.body().front();
  unsigned numArgs = frontBlock.getNumArguments();
  // Iterate on the original `numArgs` and replace them in order.
  // This guarantees the argument order still matches after the rewrite.
  for (unsigned idx = 0; idx < numArgs; ++idx) {
    auto bbArg = frontBlock.getArgument(0);
    auto tensorType = bbArg.getType().dyn_cast<TensorType>();
    if (!tensorType) {
      frontBlock.addArgument(bbArg.getType());
      bbArg.replaceAllUsesWith(frontBlock.getArguments().back());
    } else {
      // TODO: non-zero address space.
      // TODO: layout information if relevant.
      Value memref =
          frontBlock.addArgument(getContiguousOrUnrankedMemRefType(tensorType));
      OpBuilder b(funcOp->getContext());
      // No InsertionGuard needed here.
      b.setInsertionPointToStart(&frontBlock);
      // If the bbArg is only used by TensorToMemRef, we can directly replace
      // them by a simple MemRefCastOp.
      if (hasOnlyTensorToMemRefUses(bbArg)) {
        for (auto &use : llvm::make_early_inc_range(bbArg.getUses())) {
          Value bufferCast = use.getOwner()->getResult(0);
          bufferCast.replaceAllUsesWith(b.create<memref::CastOp>(
              funcOp.getLoc(), bufferCast.getType(), memref));
          use.getOwner()->erase();
        }
      } else {
        // Otherwise, there are uses that are not TensorToMemRefOp, we need to
        // insert a TensorLoadOp. Subsequent canonicalizations that perform:
        // `memref.buffer_cast(memref.tensor_load(x)) -> x` will later occur.
        Value tensor = b.create<memref::TensorLoadOp>(funcOp->getLoc(), memref);
        bbArg.replaceAllUsesWith(tensor);
      }
    }
    frontBlock.eraseArgument(0);
  }

  LLVM_DEBUG(DBGS() << "End bufferizeFuncOpBoundary:\n" << funcOp);
}

//===----------------------------------------------------------------------===//
// Bufferization passes.
//===----------------------------------------------------------------------===//

// Transformations that run iteratively with bufferization.
void LinalgComprehensiveBufferizePass::runEnablingTransforms(FuncOp funcOp) {
  if (failed(runPipeline(enablingPassPipeline, funcOp)))
    return signalPassFailure();
  (void)runPipeline(enablingPassPipeline, funcOp);
  linalg::hoistRedundantVectorTransfers(funcOp);
  linalg::hoistRedundantVectorTransfersOnTensor(funcOp);
}

void LinalgComprehensiveBufferizePass::bufferizeFuncOpInternals(
    FuncOp funcOp, BlockAndValueMapping &bvm, GlobalCreator &globals,
    const DenseMap<FuncOp, SmallVector<int64_t>> &tiedResultsMap) {
  if (!funcOp || funcOp->getNumRegions() == 0 || funcOp.body().empty()) return;

  LLVM_DEBUG(DBGS() << "Begin BufferizeFuncOpInternals:\n" << funcOp << "\n");

  OpBuilder b(funcOp->getContext());
  auto guard = llvm::make_scope_exit([&] {
    funcOp.walk(
        [&](Operation *op) { op->removeAttr(kInPlaceResultsAttrName); });
  });
  /// Start by converting `funcOp` arguments.
  SmallVector<Operation *> convertedCallOps;
  (void)succeeded(convertFuncOp(b, funcOp, bvm));
  funcOp.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    llvm::TypeSwitch<Operation *, void>(operation)
        .Case([&](scf::ForOp op) {
          (void)succeeded(convertScfForOp(b, op, bvm));
        })
        .Case([&](TiledLoopOp op) {
          (void)succeeded(convertTiledLoopOp(b, op, bvm));
        })
        .Case([&](ConstantOp op) {
          (void)succeeded(convertConstantOp(b, op, bvm, globals));
        })
        .Case([&](InitTensorOp op) {
          (void)succeeded(convertInitTensorOp(b, op, bvm));
        })
        .Case([&](tensor::ExtractSliceOp op) {
          (void)succeeded(convertSubTensorOp(b, op, bvm));
        })
        .Case([&](tensor::ExtractOp op) {
          (void)succeeded(convertExtractOp(b, op, bvm));
        })
        .Case([&](tensor::InsertSliceOp op) {
          (void)succeeded(convertSubTensorInsertOp(b, op, bvm));
        })
        .Case([&](tensor::CastOp op) {
          (void)succeeded(convertTensorCastOp(b, op, bvm));
        })
        .Case([&](PadTensorOp op) {
          (void)succeeded(convertPadTensorOp(b, op, bvm));
        })
        .Case([&](LinalgOp op) {
          (void)succeeded(convertAnyLinalgOp(b, op, bvm));
        })
        .Case([&](VectorTransferOpInterface op) {
          (void)succeeded(convertTransferOp(b, op, bvm));
        })
        .Case([&](scf::YieldOp op) {
          (void)succeeded(convertScfYieldOp(b, op, bvm));
        })
        .Case([&](CallOpInterface op) {
          (void)succeeded(
              convertCallOp(b, op, bvm, convertedCallOps, tiedResultsMap));
        })
        .Case(
            [&](ReturnOp op) { (void)succeeded(convertReturnOp(b, op, bvm)); });
  });
  // Delete all the converted call operations.
  for (Operation *op : convertedCallOps) op->erase();

  LLVM_DEBUG(DBGS() << "End BufferizeFuncOpInternals:\n" << funcOp << "\n");
}

namespace mlir {
std::unique_ptr<Pass> createLinalgComprehensiveBufferizePass() {
  return std::make_unique<LinalgComprehensiveBufferizePass>();
}
namespace linalg {
void registerLinalgComprehensiveBufferizePass() {
  PassRegistration<LinalgComprehensiveBufferizePass> pass(
      "linalg-comprehensive-bufferize-inplace",
      "Perform all required bufferization incantations to convert code with "
      "Linalg ops on tensors to buffers with inPlace optimizations.");
}
}  // namespace linalg
}  // namespace mlir

static void postTransformSanityChecks(ModuleOp moduleOp,
                                      BlockAndValueMapping &bvm) {
  moduleOp.walk([&](Operation *op) {
    op->removeAttr(kInPlaceResultsAttrName);
    op->removeAttr(kWriteableFuncBufferArgsAttrName);

    assert(!isa<memref::BufferCastOp>(op));
    if (auto tensorLoadOp = dyn_cast<memref::TensorLoadOp>(op)) {
      if (tensorLoadOp.memref().getDefiningOp<memref::BufferCastOp>()) {
        op->getParentOfType<ModuleOp>()->dump();
        op->emitWarning(
            "Most likely incorrect pattern: "
            "memref.tensor_load(memref.buffer_cast)");
        abort();
      }
      return;
    }

    if (llvm::any_of(op->getOperands(),
                     [](Value v) { return v.getType().isa<TensorType>(); })) {
      op->getParentOfType<ModuleOp>()->dump();
      op->emitWarning("Remaining op with tensor operands");
      abort();
    }

    if (auto callOp = dyn_cast<CallOpInterface>(op)) {
      for (auto result : callOp->getResults()) {
        if (result.getType().isa<MemRefType>()) {
          op->getParentOfType<ModuleOp>()->dump();
          op->emitWarning(
              "Most likely incorrect pattern: function returning memref -> "
              "alloc needs to be hoisted out of function boundary");
          abort();
        }
      }
      return;
    }
  });
}

/// Special patterns for which dealiasing is more difficult to detect when
/// operating on tensors but easier on buffers.
/// Detect and remove exactly the following patterns (where `|` denotes the
/// "next operation" and -> denoted a simple "copy" operation):
///  - alloc with no use
///  - alloc with a single use that is a dealloc
///  - alloc with 2 uses: (%x -> %alloc) | dealloc(%alloc)
///  - alloc with 3 uses:  (%x -> %alloc) | (%alloc -> %x) | dealloc(%alloc)
static void performLateCleanups(FuncOp funcOp) {
  SmallVector<Operation *> toErase;
  funcOp.walk([&](memref::AllocOp alloc) {
    LLVM_DEBUG(DBGS() << "alloc with " << sizeOfRange(alloc->getUses())
                      << " uses: " << alloc.memref() << "\n");

    // Look for useless alloc.
    if (alloc.use_empty()) {
      LLVM_DEBUG(DBGS() << "useless alloc: " << alloc.memref() << "\n");
      alloc->erase();
      return;
    }

    memref::DeallocOp dealloc;
    for (auto &use : alloc->getUses()) {
      if (auto candidate = dyn_cast<memref::DeallocOp>(use.getOwner())) {
        assert(!dealloc && "double free of AllocOp");
        dealloc = candidate;
      }
    }

    if (!dealloc) return;
    LLVM_DEBUG(DBGS() << "dealloc: " << *dealloc.getOperation() << "\n");

    // Look for trivial alloc | dealloc.
    if (alloc->hasOneUse()) {
      LLVM_DEBUG(DBGS() << "trivial alloc | dealloc\n");
      dealloc->erase();
      alloc->erase();
      return;
    }

    // Look for copy to nowhere:
    //   (%x -> %alloc) | dealloc(%alloc)
    LLVM_DEBUG(DBGS() << "prev(dealloc): " << *dealloc->getPrevNode() << "\n");
    auto lastCopy = dyn_cast_or_null<linalg::CopyOp>(dealloc->getPrevNode());
    // Bail on permutations.
    if (!lastCopy || lastCopy.inputPermutation() ||
        lastCopy.outputPermutation())
      return;

    LLVM_DEBUG(DBGS() << "lastCopy: " << *lastCopy.getOperation() << "\n");
    if (sizeOfRange(alloc->getUses()) == 2) {
      if (lastCopy.output() == alloc.memref()) {
        LLVM_DEBUG(DBGS() << "copy to nowhere\n");
        dealloc->erase();
        lastCopy->erase();
        alloc->erase();
      }
      return;
    }

    // Look for copy to copy to nowhere:
    //   (%x -> %alloc) | (%alloc -> %x) | dealloc(%alloc))
    LLVM_DEBUG(DBGS() << "prev(lastCopy): " << *lastCopy->getPrevNode()
                      << "\n");
    auto prevCopy = dyn_cast_or_null<linalg::CopyOp>(lastCopy->getPrevNode());
    // Bail on permutations.
    if (!prevCopy || prevCopy.inputPermutation() ||
        prevCopy.outputPermutation())
      return;

    LLVM_DEBUG(DBGS() << "prevCopy: " << *prevCopy.getOperation() << "\n");
    if (sizeOfRange(alloc->getUses()) == 3) {
      // lastCopy is the last one in program order.
      if (prevCopy.output() == alloc.memref() &&
          lastCopy.input() == alloc.memref() &&
          prevCopy.input() == lastCopy.output()) {
        LLVM_DEBUG(DBGS() << "copy to copy to nowhere\n");
        dealloc->erase();
        prevCopy->erase();
        lastCopy->erase();
        alloc->erase();
      }
      return;
    }
  });
}

void LinalgComprehensiveBufferizePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  // 0. Perform a bunch of enabling transformations related to canonicalizations
  // CSE and hoisting.
  moduleOp.walk([&](FuncOp funcOp) { runEnablingTransforms(funcOp); });

  // 1. Perform inPlace analysis to mark the arguments/operands of all calls and
  // functions that can be performed inPlace. The information set on the FuncOp
  // is the meet of the information set on the all CallOp calling that FuncOp.
  DominanceInfo domInfo(moduleOp);
  moduleOp.walk([&](CallOpInterface callOp) {
    funcArgumentsInPlaceAnalysis(callOp, domInfo);
  });

  // 2. Bufferize destructive update patterns within function boundaries.
  GlobalCreator globals(moduleOp);
  BlockAndValueMapping bvm;
  SmallVector<FuncOp> worklist;
  if (failed(computeOrderedWorklist(moduleOp, worklist)))
    return signalPassFailure();

  DenseMap<FuncOp, SmallVector<int64_t>> tiedResultsMap;
  for (auto funcOp : worklist) {
    // Perform bufferization within the funcOp boundary. This produces IR
    // in a form on which `bufferizeFuncOpBoundary` can decide whether return
    // values can fold onto operands.
    inPlaceAnalysisFuncOpInternals(funcOp, domInfo);
    bufferizeFuncOpInternals(funcOp, bvm, globals, tiedResultsMap);
    // Fold results on arguments if possible.
    bufferizeFuncOpBoundary(funcOp, tiedResultsMap);
  };

  // 3. Run cleanup pipeline.
  moduleOp.walk([&](FuncOp funcOp) { runEnablingTransforms(funcOp); });

  // 4. Sanity checks.
  postTransformSanityChecks(moduleOp, bvm);

  // 5. Late cleanups.
  moduleOp.walk([](FuncOp funcOp) { performLateCleanups(funcOp); });
}
