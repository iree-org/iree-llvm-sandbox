//===- IteratorsToStandard.h - Utils to convert from the Iterators dialect ===//

#ifndef ITERATORS_CONVERSION_ITERATORSTOSTANDARD_ITERATORSTOSTANDARD_H
#define ITERATORS_CONVERSION_ITERATORSTOSTANDARD_ITERATORSTOSTANDARD_H

#include "iterators/Dialect/Iterators/IR/Iterators.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;

namespace iterators {

/// Populate the given list with patterns that convert from Iterators to
/// Standard.
void populateIteratorsToStandardConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter);

} // namespace iterators

/// Create a pass to convert Iterators operations to the Standard dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertIteratorsToStandardPass();

} // namespace mlir

#endif // ITERATORS_CONVERSION_ITERATORSTOSTANDARD_ITERATORSTOSTANDARD_H
