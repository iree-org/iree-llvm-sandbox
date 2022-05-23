#include "iterators/Utils/NameAssigner.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"

#include <assert.h>

namespace mlir {
namespace iterators {

NameAssigner::NameAssigner(ModuleOp module) : module(module) { assert(module); }

StringAttr NameAssigner::assignName(StringRef prefix) {
  llvm::SmallString<64> candidateName;
  decltype(names)::iterator existingName;
  while (true) {
    (prefix + Twine(".") + Twine(uniqueNumber)).toStringRef(candidateName);
    existingName = names.find(candidateName);
    if (!module.lookupSymbol(candidateName) && existingName == names.end()) {
      break;
    }
    uniqueNumber++;
  }
  auto attr = StringAttr::get(module.getContext(), candidateName);
  names.insert(attr.getValue());
  return attr;
}

} // namespace iterators
} // namespace mlir
