add_mlir_dialect_library(MLIRIndexingTransforms
  ScatterGatherToInsertExtractSlice.cpp
  RefBackend.cpp

  DEPENDS
  MLIRIndexingPassIncGen

  LINK_LIBS PUBLIC
  # generic
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRTransforms
  # specific
  MLIRIndexing
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRMemRefDialect
)
