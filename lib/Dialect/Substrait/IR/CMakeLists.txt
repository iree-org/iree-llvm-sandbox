add_mlir_dialect_library(MLIRSubstraitDialect
  Substrait.cpp

  LINK_LIBS PUBLIC
  MLIRInferTypeOpInterface
  MLIRIR

  DEPENDS
  MLIRSubstraitOpsIncGen
)
