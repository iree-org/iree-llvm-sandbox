//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ModelBuilder/ModelBuilder.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

using namespace mlir;
using namespace mlir::edsc;

mlir::edsc::ScopedContext::ScopedContext(OpBuilder &b)
    : ScopedContext(b, b.getInsertionPoint()->getLoc()) {}

mlir::edsc::ScopedContext::ScopedContext(OpBuilder &b, Location location)
    : builder(b),
      guard(builder),
      location(location),
      enclosingScopedContext(ScopedContext::getCurrentScopedContext()) {
  getCurrentScopedContext() = this;
}

/// Sets the insertion point of the builder to 'newInsertPt' for the duration
/// of the scope. The existing insertion point of the builder is restored on
/// destruction.
mlir::edsc::ScopedContext::ScopedContext(OpBuilder &b,
                                         OpBuilder::InsertPoint newInsertPt,
                                         Location location)
    : builder(b),
      guard(builder),
      location(location),
      enclosingScopedContext(ScopedContext::getCurrentScopedContext()) {
  getCurrentScopedContext() = this;
  builder.restoreInsertionPoint(newInsertPt);
}

mlir::edsc::ScopedContext::~ScopedContext() {
  getCurrentScopedContext() = enclosingScopedContext;
}

ScopedContext *&mlir::edsc::ScopedContext::getCurrentScopedContext() {
  thread_local ScopedContext *context = nullptr;
  return context;
}

OpBuilder &mlir::edsc::ScopedContext::getBuilderRef() {
  assert(ScopedContext::getCurrentScopedContext() &&
         "Unexpected Null ScopedContext");
  return ScopedContext::getCurrentScopedContext()->builder;
}

Location mlir::edsc::ScopedContext::getLocation() {
  assert(ScopedContext::getCurrentScopedContext() &&
         "Unexpected Null ScopedContext");
  return ScopedContext::getCurrentScopedContext()->location;
}

MLIRContext *mlir::edsc::ScopedContext::getContext() {
  return getBuilderRef().getContext();
}

Block *mlir::edsc::createBlock(TypeRange argTypes) {
  assert(ScopedContext::getContext() != nullptr && "ScopedContext not set up");
  OpBuilder &builder = ScopedContext::getBuilderRef();
  Block *block = builder.getInsertionBlock();
  assert(block != nullptr &&
         "insertion point not set up in the builder within ScopedContext");

  return createBlockInRegion(*block->getParent(), argTypes);
}

Block *mlir::edsc::createBlockInRegion(Region &region, TypeRange argTypes) {
  assert(ScopedContext::getContext() != nullptr && "ScopedContext not set up");
  OpBuilder &builder = ScopedContext::getBuilderRef();

  OpBuilder::InsertionGuard guard(builder);
  return builder.createBlock(&region, {}, argTypes);
}

void mlir::edsc::appendToBlock(Block *block,
                               function_ref<void(ValueRange)> builderFn) {
  assert(ScopedContext::getContext() != nullptr && "ScopedContext not set up");
  OpBuilder &builder = ScopedContext::getBuilderRef();

  OpBuilder::InsertionGuard guard(builder);
  if (block->empty() || !block->back().mightHaveTrait<OpTrait::IsTerminator>())
    builder.setInsertionPointToEnd(block);
  else
    builder.setInsertionPoint(&block->back());
  builderFn(block->getArguments());
}

Block *mlir::edsc::buildInNewBlock(TypeRange argTypes,
                                   function_ref<void(ValueRange)> builderFn) {
  assert(ScopedContext::getContext() != nullptr && "ScopedContext not set up");
  OpBuilder &builder = ScopedContext::getBuilderRef();
  Block *block = builder.getInsertionBlock();
  assert(block != nullptr &&
         "insertion point not set up in the builder within ScopedContext");
  return buildInNewBlock(*block->getParent(), argTypes, builderFn);
}

Block *mlir::edsc::buildInNewBlock(Region &region, TypeRange argTypes,
                                   function_ref<void(ValueRange)> builderFn) {
  assert(ScopedContext::getContext() != nullptr && "ScopedContext not set up");
  OpBuilder &builder = ScopedContext::getBuilderRef();

  Block *block = createBlockInRegion(region, argTypes);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);
  builderFn(block->getArguments());
  return block;
}

BranchOp mlir::std_br(Block *block, ValueRange operands) {
  return OperationBuilder<BranchOp>(block, operands);
}

CondBranchOp mlir::std_cond_br(Value cond, Block *trueBranch,
                               ValueRange trueOperands, Block *falseBranch,
                               ValueRange falseOperands) {
  return OperationBuilder<CondBranchOp>(cond, trueBranch, trueOperands,
                                        falseBranch, falseOperands);
}

thread_local MLIRContext mlir::ModelBuilder::ctx;

void ModelBuilder::registerAllDialects() {
  // TODO: remove.
}

mlir::ModelBuilder::ModelBuilder()
    : OpBuilder(&ctx),
      module(mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx))),
      symbolTable(*module),
      loc(module->getLoc()),
      i8(IntegerType::get(&ctx, 8)),
      f32(FloatType::getF32(&ctx)),
      f64(FloatType::getF64(&ctx)) {
  ctx.getOrLoadDialect<AffineDialect>();
  ctx.getOrLoadDialect<gpu::GPUDialect>();
  ctx.getOrLoadDialect<LLVM::LLVMDialect>();
  ctx.getOrLoadDialect<linalg::LinalgDialect>();
  ctx.getOrLoadDialect<memref::MemRefDialect>();
  ctx.getOrLoadDialect<scf::SCFDialect>();
  ctx.getOrLoadDialect<omp::OpenMPDialect>();
  ctx.getOrLoadDialect<spirv::SPIRVDialect>();
  ctx.getOrLoadDialect<StandardOpsDialect>();
  ctx.getOrLoadDialect<vector::VectorDialect>();
  registerLLVMDialectTranslation(ctx);
}

Value mlir::ModelBuilder::constant_f32(float v) {
  return std_constant_float(llvm::APFloat(v),
                            FloatType::getF32(ScopedContext::getContext()));
}

Value mlir::ModelBuilder::constant_f64(double v) {
  return std_constant_float(llvm::APFloat(v),
                            FloatType::getF64(ScopedContext::getContext()));
}

Value mlir::ModelBuilder::constant_index(int64_t v) {
  return std_constant_index(v);
}

FuncOp mlir::ModelBuilder::makeFunction(StringRef name, ArrayRef<Type> results,
                                        ArrayRef<Type> args,
                                        MLIRFuncOpConfig config) {
  FunctionType ft = FunctionType::get(&ctx, args, results);
  auto function = FuncOp::create(loc, name, ft);
  config.apply(function);
  module->push_back(function);
  return function;
}
FuncOp mlir::ModelBuilder::makeFunction(
    std::function<std::string(FunctionType)> nameBuilder,
    ArrayRef<Type> results, ArrayRef<Type> args, MLIRFuncOpConfig config) {
  FunctionType ft = FunctionType::get(&ctx, args, results);
  return makeFunction(nameBuilder(ft), results, args, config);
}

static spirv::TargetEnvAttr getTargetEnv(MLIRContext *context) {
  auto triple = spirv::VerCapExtAttr::get(
      spirv::Version::V_1_0,
      {spirv::Capability::Shader, spirv::Capability::CooperativeMatrixNV,
       spirv::Capability::Int8, spirv::Capability::Float16,
       spirv::Capability::StorageUniform16,
       spirv::Capability::StorageBuffer8BitAccess,
       spirv::Capability::Float16Buffer},
      {spirv::Extension::SPV_KHR_storage_buffer_storage_class,
       spirv::Extension::SPV_NV_cooperative_matrix,
       spirv::Extension::SPV_KHR_8bit_storage,
       spirv::Extension::SPV_KHR_16bit_storage},
      context);
  return spirv::TargetEnvAttr::get(triple, spirv::Vendor::Unknown,
                                   spirv::DeviceType::Unknown,
                                   spirv::TargetEnvAttr::kUnknownDeviceID,
                                   spirv::getDefaultResourceLimits(context));
}

gpu::GPUModuleOp mlir::ModelBuilder::makeGPUModule(StringRef name) {
  // Add module attributes required first.
  addGPUAttr();
  OpBuilder b(&module->getBodyRegion());
  auto kernelModule = b.create<gpu::GPUModuleOp>(loc, name);
  return kernelModule;
}

void mlir::ModelBuilder::addGPUAttr() {
  // Add module attributes required first.
  (*module)->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                     UnitAttr::get(module->getContext()));
  spirv::TargetEnvAttr targetEnv = getTargetEnv(module->getContext());
  (*module)->setAttr(spirv::getTargetEnvAttrName(), targetEnv);
}

gpu::GPUFuncOp mlir::ModelBuilder::makeGPUKernel(
    StringRef name, gpu::GPUModuleOp GPUModule, ArrayRef<int32_t> workgroupSize,
    ArrayRef<Type> args, ArrayRef<Type> results) {
  auto fnType = FunctionType::get(module->getContext(), args, results);
  OpBuilder b(&GPUModule.body());
  auto kernelFunc = b.create<gpu::GPUFuncOp>(loc, name, fnType);
  kernelFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                      b.getUnitAttr());
  kernelFunc->setAttr(
      spirv::getEntryPointABIAttrName(),
      spirv::getEntryPointABIAttr(workgroupSize, module->getContext()));
  return kernelFunc;
}

VectorType mlir::ModelBuilder::getVectorType(ArrayRef<int64_t> shape,
                                             Type elementalType) {
  return VectorType::get(shape, elementalType);
}

MemRefType mlir::ModelBuilder::getMemRefType(ArrayRef<int64_t> shape,
                                             Type elementType,
                                             unsigned addressSpace) {
  return MemRefType::get(shape, elementType, {}, addressSpace);
}

RankedTensorType mlir::ModelBuilder::getRankedTensorType(
    ArrayRef<int64_t> shape, Type elementType) {
  return RankedTensorType::get(shape, elementType);
}

Value mlir::ModelBuilder::fusedBiasTanh(Value x, Value bias) {
  using edsc::op::operator+;
  using edsc::op::operator*;
  assert(x.getType().isF32() && bias.getType().isF32() && "f32 expected");
  Value half = constant_f32(0.5f);
  return x + half * call_tanhf((x + bias) * half) + half;
}

Value mlir::ModelBuilder::FCBiasTanh(std::array<Value, 3> fcArgs,
                                     Value biasValueArg) {
  //==========================================================================//
  // Layer 1: FC
  //==========================================================================//
  Value I = fcArgs[0], W = fcArgs[1], O = fcArgs[2];
  // Emit a linalg.generic op that implements matmul:
  linalg_generic_matmul(I, W, O);

  //==========================================================================//
  // Layer 2: BiasAddTanh Block
  //==========================================================================//
  // Build and capture AffineExpr i and j for building index expressions.
  AffineExpr i, j;
  bindDims(&ctx, i, j);

  // Emit a linalg.generic op that implements pointwise with `opBuilder` for:
  //   `0.5f * tanh(0.5f * (x + bias)) + 0.5f`
  //
  // This performs the (inplace) computation:
  //   `o[i, j] <- pointwise(bias[j], o[i, j])`
  //
  // in which bias is broadcast along `i`.
  StructuredIndexed o(O), bias(biasValueArg);
  linalg_generic_pointwise(fusedBiasTanh, o({i, j}), bias({j}), o({i, j}));

  return O;
}

Value ModelBuilder::FCBiasTanhTensors(RankedTensorType outputTensorType,
                                      std::array<Value, 2> fcArgs,
                                      Value fcInitTensor, Value biasValueArg) {
  //==========================================================================//
  // Layer 1: FC
  //==========================================================================//
  Value I = fcArgs[0], W = fcArgs[1];
  Value O2 =
      linalg_generic_matmul(I, W, fcInitTensor, outputTensorType)->getResult(0);

  //==========================================================================//
  // Layer 2: BiasAddTanh Block
  //==========================================================================//
  AffineExpr i, j;
  bindDims(&ctx, i, j);
  // in-place with explicit bias broacast
  StructuredIndexed o2(O2), bias(biasValueArg);
  return linalg_generic_pointwise(fusedBiasTanh, o2({i, j}), bias({j}),
                                  o2({i, j}))
      ->getResult(0);
}

Value ModelBuilder::call_tanhf(Value v) {
  assert(v.getType().isF32() && "f32 expected");
  return emitCallToRegisteredSymbol("tanhf", v.getType(), v)->getResult(0);
}

void ModelBuilder::call_print_memref_f32(Value v) {
  auto &builder = ScopedContext::getBuilderRef();
  auto loc = builder.getInsertionBlock()
                 ->getParent()
                 ->getParentOfType<FuncOp>()
                 .getLoc();
  auto elementType = v.getType().cast<MemRefType>().getElementType();
  auto unrankedType = UnrankedMemRefType::get(elementType, 0);
  auto castMemRef = builder.create<memref::CastOp>(loc, v, unrankedType);
  if (elementType.isF32())
    emitCallToRegisteredSymbol("print_memref_f32", {}, {castMemRef});
  else
    llvm_unreachable("Incorrect argument type for print_memref_f32");
}

Operation *ModelBuilder::emitCallToRegisteredSymbol(StringRef functionName,
                                                    ArrayRef<Type> returnTypes,
                                                    ValueRange values) {
  auto &builder = ScopedContext::getBuilderRef();
  auto callerFunc =
      builder.getInsertionBlock()->getParent()->getParentOfType<FuncOp>();
  FuncOp calleeFunc = SymbolTable::lookupNearestSymbolFrom<FuncOp>(
      callerFunc, builder.getStringAttr(functionName));
  if (!calleeFunc) {
    OpBuilder::InsertionGuard insertGuard(builder);
    auto module = callerFunc->getParentOfType<ModuleOp>();
    builder.setInsertionPointToStart(module.getBody());
    calleeFunc = builder.create<FuncOp>(
        module.getLoc(), functionName,
        FunctionType::get(builder.getContext(), values.getTypes(),
                          returnTypes));
    calleeFunc.setPrivate();
  }
  return std_call(calleeFunc, values);
}

MLIRFuncOpConfig &MLIRFuncOpConfig::setNoInline(bool v) {
  noInline = v;
  return *this;
}
MLIRFuncOpConfig &MLIRFuncOpConfig::setPreferAvx512(bool v) {
  preferAvx512 = v;
  return *this;
}
MLIRFuncOpConfig &MLIRFuncOpConfig::setTargetCpu(StringRef s) {
  targetCpu = std::string(s);
  return *this;
}
MLIRFuncOpConfig &MLIRFuncOpConfig::setDeclOnly(bool v) {
  declOnly = v;
  return *this;
}
MLIRFuncOpConfig &MLIRFuncOpConfig::setEmitCInterface(bool v) {
  emitCInterface = v;
  return *this;
}

void MLIRFuncOpConfig::apply(FuncOp &f) {
  MLIRContext *ctx = f.getContext();
  SmallVector<Attribute, 8> attrs;
  if (noInline) attrs.push_back(StringAttr::get(ctx, "noinline"));
  if (preferAvx512)
    attrs.push_back(
        ArrayAttr::get(ctx, {StringAttr::get(ctx, "prefer-vector-width"),
                             StringAttr::get(ctx, "512")}));
  if (!targetCpu.empty())
    attrs.push_back(ArrayAttr::get(ctx, {StringAttr::get(ctx, "target-cpu"),
                                         StringAttr::get(ctx, targetCpu)}));
  if (!attrs.empty()) f->setAttr("passthrough", ArrayAttr::get(ctx, attrs));

  if (emitCInterface)
    f->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(ctx));

  if (!declOnly)
    f.addEntryBlock();
  else
    f.setPrivate();
}

// -----------------------------------------------------------------------------
// EDSC extensions.
// -----------------------------------------------------------------------------
template <typename Lambda>
static SmallVector<Value, 4> valueRangeOperatorImpl(Lambda fun, ValueRange a,
                                                    ValueRange b) {
  SmallVector<Value, 4> res;
  res.reserve(std::min(a.size(), b.size()));
  for (auto it : llvm::zip(a, b))
    res.push_back(fun(std::get<0>(it), std::get<1>(it)));
  return res;
}
SmallVector<Value, 4> mlir::edsc::extensions::operator-(ValueRange a,
                                                        ValueRange b) {
  return valueRangeOperatorImpl(edsc::op::operator-, a, b);
}
SmallVector<Value, 4> mlir::edsc::extensions::operator+(ValueRange a,
                                                        ValueRange b) {
  return valueRangeOperatorImpl(edsc::op::operator+, a, b);
}
SmallVector<Value, 4> mlir::edsc::extensions::std_max(ValueRange a,
                                                      ValueRange b) {
  using edsc::op::slt;
  auto fun = [](Value va, Value vb) { return slt(va, vb) ? vb : va; };
  return valueRangeOperatorImpl(fun, a, b);
}
SmallVector<Value, 4> mlir::edsc::extensions::std_min(ValueRange a,
                                                      ValueRange b) {
  using edsc::op::slt;
  auto fun = [](Value va, Value vb) { return slt(va, vb) ? va : vb; };
  return valueRangeOperatorImpl(fun, a, b);
}
SmallVector<Value, 4> mlir::edsc::extensions::affine_max(ValueRange a,
                                                         ValueRange b) {
  // TODO(ntv): cleanup when affine_max accepts has more idiomatic builders.
  MLIRContext *ctx = ScopedContext::getContext();
  auto map = AffineMap::get(
      2, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx)}, ctx);
  auto fun = [&](Value va, Value vb) {
    return mlir::affine_max(map, ValueRange{va, vb});
  };
  return valueRangeOperatorImpl(fun, a, b);
}
SmallVector<Value, 4> mlir::edsc::extensions::affine_min(ValueRange a,
                                                         ValueRange b) {
  // TODO(ntv): cleanup when affine_min accepts has more idiomatic builders.
  MLIRContext *ctx = ScopedContext::getContext();
  auto map = AffineMap::get(
      2, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx)}, ctx);
  auto fun = [&](Value va, Value vb) {
    return mlir::affine_min(map, ValueRange{va, vb});
  };
  return valueRangeOperatorImpl(fun, a, b);
}

void mlir::edsc::affineLoopNestBuilder(
    ValueRange lbs, ValueRange ubs, ArrayRef<int64_t> steps,
    function_ref<void(ValueRange)> bodyBuilderFn) {
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");

  // Wrap the body builder function into an interface compatible with the main
  // builder.
  auto wrappedBuilderFn = [&](OpBuilder &nestedBuilder, Location nestedLoc,
                              ValueRange ivs) {
    ScopedContext context(nestedBuilder, nestedLoc);
    bodyBuilderFn(ivs);
  };
  function_ref<void(OpBuilder &, Location, ValueRange)> wrapper;
  if (bodyBuilderFn) wrapper = wrappedBuilderFn;

  // Extract the builder, location and construct the loop nest.
  OpBuilder &builder = ScopedContext::getBuilderRef();
  Location loc = ScopedContext::getLocation();
  buildAffineLoopNest(builder, loc, lbs, ubs, steps, wrapper);
}

void mlir::edsc::affineLoopBuilder(ValueRange lbs, ValueRange ubs, int64_t step,
                                   function_ref<void(Value)> bodyBuilderFn) {
  // Fetch the builder and location.
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  OpBuilder &builder = ScopedContext::getBuilderRef();
  Location loc = ScopedContext::getLocation();

  // Create the actual loop and call the body builder, if provided, after
  // updating the scoped context.
  builder.create<AffineForOp>(
      loc, lbs, builder.getMultiDimIdentityMap(lbs.size()), ubs,
      builder.getMultiDimIdentityMap(ubs.size()), step, llvm::None,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange itrArgs) {
        if (bodyBuilderFn) {
          ScopedContext nestedContext(nestedBuilder, nestedLoc);
          OpBuilder::InsertionGuard guard(nestedBuilder);
          bodyBuilderFn(iv);
        }
        nestedBuilder.create<AffineYieldOp>(nestedLoc);
      });
}

void mlir::edsc::affineLoopBuilder(
    ValueRange lbs, ValueRange ubs, int64_t step, ValueRange iterArgs,
    function_ref<void(Value, ValueRange)> bodyBuilderFn) {
  // Fetch the builder and location.
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  OpBuilder &builder = ScopedContext::getBuilderRef();
  Location loc = ScopedContext::getLocation();

  // Create the actual loop and call the body builder, if provided, after
  // updating the scoped context.
  builder.create<AffineForOp>(
      loc, lbs, builder.getMultiDimIdentityMap(lbs.size()), ubs,
      builder.getMultiDimIdentityMap(ubs.size()), step, iterArgs,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange itrArgs) {
        if (bodyBuilderFn) {
          ScopedContext nestedContext(nestedBuilder, nestedLoc);
          OpBuilder::InsertionGuard guard(nestedBuilder);
          bodyBuilderFn(iv, itrArgs);
        } else if (itrArgs.empty())
          nestedBuilder.create<AffineYieldOp>(nestedLoc);
      });
}

static std::pair<AffineExpr, Value> categorizeValueByAffineType(
    MLIRContext *context, Value val, unsigned &numDims, unsigned &numSymbols) {
  AffineExpr d;
  Value resultVal = nullptr;
  if (auto constant = val.getDefiningOp<ConstantIndexOp>()) {
    d = getAffineConstantExpr(constant.getValue(), context);
  } else if (isValidSymbol(val) && !isValidDim(val)) {
    d = getAffineSymbolExpr(numSymbols++, context);
    resultVal = val;
  } else {
    d = getAffineDimExpr(numDims++, context);
    resultVal = val;
  }
  return std::make_pair(d, resultVal);
}

static Value createBinaryIndexHandle(
    Value lhs, Value rhs,
    function_ref<AffineExpr(AffineExpr, AffineExpr)> affCombiner) {
  MLIRContext *context = ScopedContext::getContext();
  unsigned numDims = 0, numSymbols = 0;
  AffineExpr d0, d1;
  Value v0, v1;
  std::tie(d0, v0) =
      categorizeValueByAffineType(context, lhs, numDims, numSymbols);
  std::tie(d1, v1) =
      categorizeValueByAffineType(context, rhs, numDims, numSymbols);
  SmallVector<Value, 2> operands;
  if (v0) operands.push_back(v0);
  if (v1) operands.push_back(v1);
  auto map = AffineMap::get(numDims, numSymbols, affCombiner(d0, d1));

  // TODO: createOrFold when available.
  Operation *op =
      makeComposedAffineApply(ScopedContext::getBuilderRef(),
                              ScopedContext::getLocation(), map, operands)
          .getOperation();
  assert(op->getNumResults() == 1 && "Expected single result AffineApply");
  return op->getResult(0);
}

template <typename IOp, typename FOp>
static Value createBinaryHandle(
    Value lhs, Value rhs,
    function_ref<AffineExpr(AffineExpr, AffineExpr)> affCombiner) {
  auto thisType = lhs.getType();
  auto thatType = rhs.getType();
  assert(thisType == thatType && "cannot mix types in operators");
  (void)thisType;
  (void)thatType;
  if (thisType.isIndex()) {
    return createBinaryIndexHandle(lhs, rhs, affCombiner);
  } else if (thisType.isSignlessInteger()) {
    return ValueBuilder<IOp>(lhs, rhs);
  } else if (thisType.isa<FloatType>()) {
    return ValueBuilder<FOp>(lhs, rhs);
  } else if (thisType.isa<VectorType, TensorType>()) {
    auto aggregateType = thisType.cast<ShapedType>();
    if (aggregateType.getElementType().isSignlessInteger())
      return ValueBuilder<IOp>(lhs, rhs);
    else if (aggregateType.getElementType().isa<FloatType>())
      return ValueBuilder<FOp>(lhs, rhs);
  }
  llvm_unreachable("failed to create a Value");
}

Value mlir::edsc::op::operator+(Value lhs, Value rhs) {
  return createBinaryHandle<AddIOp, AddFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0 + d1; });
}

Value mlir::edsc::op::operator-(Value lhs, Value rhs) {
  return createBinaryHandle<SubIOp, SubFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0 - d1; });
}

Value mlir::edsc::op::operator*(Value lhs, Value rhs) {
  return createBinaryHandle<MulIOp, MulFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0 * d1; });
}

Value mlir::edsc::op::operator/(Value lhs, Value rhs) {
  return createBinaryHandle<SignedDivIOp, DivFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) -> AffineExpr {
        llvm_unreachable("only exprs of non-index type support operator/");
      });
}

Value mlir::edsc::op::operator%(Value lhs, Value rhs) {
  return createBinaryHandle<SignedRemIOp, RemFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0 % d1; });
}

Value mlir::edsc::op::floorDiv(Value lhs, Value rhs) {
  return createBinaryIndexHandle(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0.floorDiv(d1); });
}

Value mlir::edsc::op::ceilDiv(Value lhs, Value rhs) {
  return createBinaryIndexHandle(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0.ceilDiv(d1); });
}

Value mlir::edsc::op::negate(Value value) {
  assert(value.getType().isInteger(1) && "expected boolean expression");
  return ValueBuilder<ConstantIntOp>(1, 1) - value;
}

Value mlir::edsc::op::operator&&(Value lhs, Value rhs) {
  assert(lhs.getType().isInteger(1) && "expected boolean expression on LHS");
  assert(rhs.getType().isInteger(1) && "expected boolean expression on RHS");
  return ValueBuilder<AndOp>(lhs, rhs);
}

Value mlir::edsc::op::operator||(Value lhs, Value rhs) {
  assert(lhs.getType().isInteger(1) && "expected boolean expression on LHS");
  assert(rhs.getType().isInteger(1) && "expected boolean expression on RHS");
  return ValueBuilder<OrOp>(lhs, rhs);
}

static Value createIComparisonExpr(CmpIPredicate predicate, Value lhs,
                                   Value rhs) {
  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();
  (void)lhsType;
  (void)rhsType;
  assert(lhsType == rhsType && "cannot mix types in operators");
  assert((lhsType.isa<IndexType>() || lhsType.isSignlessInteger()) &&
         "only integer comparisons are supported");

  return ScopedContext::getBuilderRef().create<CmpIOp>(
      ScopedContext::getLocation(), predicate, lhs, rhs);
}

static Value createFComparisonExpr(CmpFPredicate predicate, Value lhs,
                                   Value rhs) {
  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();
  (void)lhsType;
  (void)rhsType;
  assert(lhsType == rhsType && "cannot mix types in operators");
  assert(lhsType.isa<FloatType>() && "only float comparisons are supported");

  return ScopedContext::getBuilderRef().create<CmpFOp>(
      ScopedContext::getLocation(), predicate, lhs, rhs);
}

// All floating point comparison are ordered through EDSL
Value mlir::edsc::op::eq(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OEQ, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::eq, lhs, rhs);
}
Value mlir::edsc::op::ne(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::ONE, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::ne, lhs, rhs);
}
Value mlir::edsc::op::slt(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OLT, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::slt, lhs, rhs);
}
Value mlir::edsc::op::sle(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OLE, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::sle, lhs, rhs);
}
Value mlir::edsc::op::sgt(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OGT, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::sgt, lhs, rhs);
}
Value mlir::edsc::op::sge(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OGE, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::sge, lhs, rhs);
}
Value mlir::edsc::op::ult(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OLT, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::ult, lhs, rhs);
}
Value mlir::edsc::op::ule(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OLE, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::ule, lhs, rhs);
}
Value mlir::edsc::op::ugt(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OGT, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::ugt, lhs, rhs);
}
Value mlir::edsc::op::uge(Value lhs, Value rhs) {
  auto type = lhs.getType();
  return type.isa<FloatType>()
             ? createFComparisonExpr(CmpFPredicate::OGE, lhs, rhs)
             : createIComparisonExpr(CmpIPredicate::uge, lhs, rhs);
}
