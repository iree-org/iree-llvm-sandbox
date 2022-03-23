//===- Passes.h - Conversion Pass Construction and Registration -----------===//

#ifndef ITERATORS_CONVERSION_PASSES_H
#define ITERATORS_CONVERSION_PASSES_H

#include "iterators/Conversion/IteratorsToStandard/IteratorsToStandard.h"

namespace mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "iterators/Conversion/Passes.h.inc"

} // namespace mlir

#endif // ITERATORS_CONVERSION_PASSES_H
