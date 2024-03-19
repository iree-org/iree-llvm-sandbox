add_mlir_dialect_library(MLIRSubstraitDialect
  Substrait.cpp

  LINK_LIBS PUBLIC
  MLIRIR

  DEPENDS
  MLIRSubstraitOpsIncGen
)
